# infopilot.py
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import csv


# 모듈 임포트
from filefinder import FileFinder
from pipeline import run_step2, TrainConfig, DEFAULT_N_COMPONENTS
from lnp_chat import LNPChat # 새로운 LNP Chat 클래스를 임포트


NORMALIZED_ALIASES = {
    "path": ("path", "filepath", "file_path", "fullpath", "full_path", "absolute_path"),
    "size": ("size", "filesize", "file_size", "bytes"),
    "mtime": ("mtime", "modified", "modified_time", "lastmodified", "timestamp"),
    "ext": ("ext", "extension", "suffix"),
    "drive": ("drive", "volume", "root"),
}


def _normalize_key(name: str) -> str:
    """Normalize header names by stripping non-alphanumerics and lowering case."""
    return "".join(ch for ch in (name or "").lower() if ch.isalnum())


def _pick_value(row: Dict[str, str], aliases) -> str:
    normalized = {_normalize_key(k): (k, v) for k, v in row.items() if k}
    for alias in aliases:
        alias_norm = _normalize_key(alias)
        data = normalized.get(alias_norm)
        if data:
            value = (data[1] or "").strip()
            if value:
                return value
    return ""


def _normalize_scan_row(raw: Dict[str, str], *, context: str = "") -> Dict[str, Any] | None:
    path = _pick_value(raw, NORMALIZED_ALIASES["path"])
    if not path:
        columns = ", ".join(k for k in raw.keys() if k)
        location = f" ({context})" if context else ""
        print(f"⚠️ 경고: 'path' 값을 찾지 못해 행을 건너뜁니다{location}. (감지한 열: {columns or '없음'})")
        return None

    size_raw = _pick_value(raw, NORMALIZED_ALIASES["size"])
    mtime_raw = _pick_value(raw, NORMALIZED_ALIASES["mtime"])
    ext = _pick_value(raw, NORMALIZED_ALIASES["ext"])
    drive = _pick_value(raw, NORMALIZED_ALIASES["drive"])

    def to_int(value: str) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0

    def to_float(value: str) -> float:
        try:
            out = float(value)
            if math.isnan(out) or math.isinf(out):
                return 0.0
            return out
        except (TypeError, ValueError):
            return 0.0

    normalized = dict(raw)
    normalized["path"] = path
    normalized["size"] = to_int(size_raw)
    normalized["mtime"] = to_float(mtime_raw)
    if ext:
        normalized["ext"] = ext
    if drive:
        normalized["drive"] = drive
    return normalized


def _parse_roots(raw_roots: List[str] | None) -> List[Path] | None:
    if not raw_roots:
        return None
    roots: List[Path] = []
    for raw in raw_roots:
        p = Path(raw).expanduser().resolve()
        if not p.exists():
            print(f"⚠️ 경고: 지정한 루트 '{p}'이(가) 존재하지 않아 건너뜁니다.")
            continue
        roots.append(p)
    if not roots:
        print("⚠️ 경고: 사용할 수 있는 루트가 없어 기본 전체 스캔을 수행합니다.")
        return None
    return roots


def _run_scan(out: Path, roots: List[Path] | None = None) -> List[Dict[str, Any]]:
    finder = FileFinder(
        exts=FileFinder.DEFAULT_EXTS,
        scan_all_drives=True,
        start_from_current_drive_only=False,
        follow_symlinks=False,
        max_depth=None,
        show_progress=True,
        progress_update_secs=0.5,
        estimate_total_dirs=False,
        startup_banner=True,
    )
    files = finder.find(roots=roots, run_async=False)
    FileFinder.to_csv(files, out)
    print(f"📦 스캔 결과 저장: {out}")
    return files


def cmd_scan(args):
    roots = _parse_roots(args.roots)
    _run_scan(Path(args.out), roots)


def _resolve_scan_csv(path: Path) -> Path:
    if path.exists():
        return path

    search_root = path.parent if path.parent else Path(".")
    candidates = []
    for candidate in sorted(search_root.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with candidate.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames or []
        except OSError:
            continue
        header_norm = {_normalize_key(h) for h in headers}
        if any(_normalize_key(alias) in header_norm for alias in NORMALIZED_ALIASES["path"]):
            candidates.append(candidate)

    if candidates:
        picked = candidates[0]
        print(f"⚠️ '{path}' 파일이 없어 '{picked}'을(를) 사용합니다.")
        return picked

    raise FileNotFoundError(f"스캔 CSV를 찾을 수 없습니다: {path}")


def _load_scan_rows(scan_csv: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with scan_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, raw in enumerate(reader, start=2):
            normalized = _normalize_scan_row(raw, context=f"{scan_csv}:{idx}")
            if normalized:
                rows.append(normalized)
    return rows


def _build_train_config(args) -> TrainConfig:
    return TrainConfig(
        max_features=args.max_features,
        n_components=args.n_components,
        n_clusters=args.n_clusters,
        ngram_range=(1, 2),
        min_df=args.min_df,
        max_df=args.max_df,
    )


def _maybe_limit_rows(rows: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if limit and len(rows) > limit:
        print(f"⚡ 테스트 모드: 상위 {limit}개 파일만 사용합니다.")
        return rows[:limit]
    return rows


def _default_train_config() -> TrainConfig:
    return TrainConfig(
        max_features=50000,
        n_components=DEFAULT_N_COMPONENTS,
        n_clusters=25,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
    )


def _ensure_chat_artifacts(
    scan_csv: Path,
    corpus: Path,
    model: Path,
    *,
    translate: bool,
    auto_train: bool,
) -> bool:
    """Ensure chat artifacts exist and are up to date. Returns True if training ran."""

    def mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    resolved_scan: Optional[Path] = None
    if scan_csv:
        try:
            resolved_scan = _resolve_scan_csv(scan_csv)
        except FileNotFoundError:
            resolved_scan = None

    artifacts_exist = corpus.exists() and model.exists()
    needs_train = not artifacts_exist

    if not needs_train and resolved_scan:
        scan_mtime = mtime(resolved_scan)
        artifacts_mtime = min(mtime(corpus), mtime(model))
        if scan_mtime > artifacts_mtime:
            needs_train = True

    if not needs_train:
        print("🔄 인덱스 최신성 확인 완료.")
        return False

    if resolved_scan is None:
        msg = (
            "⚠️ 학습 산출물이 없거나 오래되었지만 사용할 스캔 CSV를 찾지 못했습니다."
            " `--scan_csv` 경로를 확인하거나 'scan' 명령을 다시 실행해주세요."
        )
        raise FileNotFoundError(msg)

    if not auto_train:
        raise RuntimeError(
            "학습 산출물이 최신이 아닙니다. 'python infopilot.py train --scan_csv "
            f"{resolved_scan}'를 실행한 뒤 다시 시도해주세요."
        )

    print("⚠️ 스캔 결과가 모델보다 최신입니다. 자동으로 train 단계를 실행합니다.")
    rows = _load_scan_rows(resolved_scan)
    if not rows:
        raise ValueError("자동 학습을 위한 유효한 행이 없습니다. 스캔 결과를 확인해주세요.")

    cfg = _default_train_config()
    run_step2(
        rows,
        out_corpus=corpus,
        out_model=model,
        cfg=cfg,
        use_tqdm=True,
        translate=translate,
    )
    print("✅ 자동 학습 완료")
    return True


def cmd_train(args):
    scan_csv = _resolve_scan_csv(Path(args.scan_csv))
    rows = _load_scan_rows(scan_csv)
    rows = _maybe_limit_rows(rows, args.limit_files)

    if not rows:
        raise ValueError("유효한 학습 대상 행이 없습니다. 스캔 CSV를 확인해주세요.")

    cfg = _build_train_config(args)
    out_corpus = Path(args.corpus)
    out_model = Path(args.model)
    df, tm = run_step2(rows, out_corpus=out_corpus, out_model=out_model, cfg=cfg, use_tqdm=True, translate=args.translate)
    print("✅ 학습 완료")


def cmd_pipeline(args):
    out = Path(args.out)
    roots = _parse_roots(args.roots)
    _run_scan(out, roots)
    rows = _load_scan_rows(out)
    rows = _maybe_limit_rows(rows, args.limit_files)

    if not rows:
        raise ValueError("유효한 학습 대상 행이 없습니다. 스캔 결과를 확인해주세요.")

    cfg = _build_train_config(args)
    out_corpus = Path(args.corpus)
    out_model = Path(args.model)
    df, tm = run_step2(rows, out_corpus=out_corpus, out_model=out_model, cfg=cfg, use_tqdm=True, translate=args.translate)
    print("✅ 파이프라인 완료")


def cmd_chat(args):
    """대화형 검색 모드 (LNPChat 사용)"""
    auto_trained = _ensure_chat_artifacts(
        scan_csv=Path(args.scan_csv),
        corpus=Path(args.corpus),
        model=Path(args.model),
        translate=args.translate,
        auto_train=args.auto_train,
    )

    # LNPChat 클래스 인스턴스 생성 및 준비
    chat_session = LNPChat(
        model_path=Path(args.model),
        corpus_path=Path(args.corpus),
        cache_dir=Path(args.cache),
        topk=args.topk,
        translate=args.translate # 번역 옵션 전달
    )
    chat_session.ready(rebuild=auto_trained)

    print("\n💬 InfoPilot Chat 모드입니다. (종료하려면 'exit' 또는 '종료' 입력)")
    while True:
        try:
            query = input("질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 종료합니다.")
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit", "종료"}:
            print("👋 종료합니다.")
            break
        
        # LNPChat의 ask 메소드 호출
        result = chat_session.ask(query)
        print(result["answer"])
        if result.get("suggestions"):
            print("\n💡 이런 질문은 어떠세요?")
            for s in result["suggestions"]:
                print(f"   - {s}")
        print("-" * 80)


def main():
    ap = argparse.ArgumentParser(prog="infopilot", description="InfoPilot CLI - 다국어 문서 검색기")
    sp = ap.add_subparsers(dest="cmd", required=True)

    # scan
    ap_scan = sp.add_parser("scan", help="드라이브 스캔하여 파일 목록 수집")
    ap_scan.add_argument("--out", default="./data/found_files.csv")
    ap_scan.add_argument(
        "--root",
        "--roots",
        dest="roots",
        action="append",
        help="스캔할 루트 디렉터리. 여러 번 지정 가능. 미지정 시 전체 스캔.",
    )
    ap_scan.set_defaults(func=cmd_scan)

    # train
    ap_train = sp.add_parser("train", help="코퍼스 생성 + 모델 학습 (기본: 번역 활성)")
    ap_train.add_argument("--scan_csv", default="./data/found_files.csv")
    ap_train.add_argument("--corpus", default="./data/corpus.parquet")
    ap_train.add_argument("--model", default="./data/topic_model.joblib")
    ap_train.add_argument("--max_features", type=int, default=50000)
    ap_train.add_argument("--n_components", type=int, default=DEFAULT_N_COMPONENTS)
    ap_train.add_argument("--n_clusters", type=int, default=25)
    ap_train.add_argument("--min_df", type=int, default=2)
    ap_train.add_argument("--max_df", type=float, default=0.85)
    ap_train.add_argument(
        "--limit",
        "--limit-files",
        dest="limit_files",
        type=int,
        default=0,
        help="테스트용으로 상위 N개 파일만 사용합니다 (0=전체).",
    )
    ap_train.add_argument("--no-translate", dest="translate", action="store_false", help="번역 기능을 비활성화하고 원문으로 학습합니다.")
    ap_train.set_defaults(translate=True)
    ap_train.set_defaults(func=cmd_train)

    # pipeline
    ap_pipe = sp.add_parser("pipeline", help="스캔 후 바로 학습까지 진행")
    ap_pipe.add_argument("--out", default="./data/found_files.csv")
    ap_pipe.add_argument(
        "--root",
        "--roots",
        dest="roots",
        action="append",
        help="스캔할 루트 디렉터리. 여러 번 지정 가능. 미지정 시 전체 스캔.",
    )
    ap_pipe.add_argument("--corpus", default="./data/corpus.parquet")
    ap_pipe.add_argument("--model", default="./data/topic_model.joblib")
    ap_pipe.add_argument("--max_features", type=int, default=50000)
    ap_pipe.add_argument("--n_components", type=int, default=DEFAULT_N_COMPONENTS)
    ap_pipe.add_argument("--n_clusters", type=int, default=25)
    ap_pipe.add_argument("--min_df", type=int, default=2)
    ap_pipe.add_argument("--max_df", type=float, default=0.85)
    ap_pipe.add_argument(
        "--limit",
        "--limit-files",
        dest="limit_files",
        type=int,
        default=0,
        help="테스트용으로 상위 N개 파일만 사용합니다 (0=전체).",
    )
    ap_pipe.add_argument("--no-translate", dest="translate", action="store_false", help="번역 기능을 비활성화하고 원문으로 학습합니다.")
    ap_pipe.set_defaults(translate=True)
    ap_pipe.set_defaults(func=cmd_pipeline)

    # chat
    ap_chat = sp.add_parser("chat", help="대화형 질의 모드 (기본: 번역 활성)")
    ap_chat.add_argument("--model", default="./data/topic_model.joblib")
    ap_chat.add_argument("--corpus", default="./data/corpus.parquet")
    ap_chat.add_argument("--cache", default="./index_cache")
    ap_chat.add_argument("--scan_csv", default="./data/found_files.csv")
    ap_chat.add_argument("--topk", type=int, default=5)
    ap_chat.add_argument("--no-translate", dest="translate", action="store_false", help="질문 번역 기능을 비활성화합니다.")
    ap_chat.add_argument("--no-auto-train", dest="auto_train", action="store_false", help="자동 학습 갱신을 비활성화합니다.")
    ap_chat.set_defaults(translate=True)
    ap_chat.set_defaults(auto_train=True)
    ap_chat.set_defaults(func=cmd_chat)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
