# infopilot.py
from __future__ import annotations

import argparse
import itertools
import math
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

import csv
csv.field_size_limit(10**7)  # 10MB까지 허용

import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import joblib
except Exception:
    joblib = None

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
except Exception:
    FileSystemEventHandler = object  # type: ignore
    Observer = None


# 모듈 임포트
from filefinder import FileFinder
from pipeline import (
    run_step2,
    TrainConfig,
    DEFAULT_N_COMPONENTS,
    DEFAULT_EMBED_MODEL,
    update_corpus_file,
    remove_from_corpus,
    CorpusBuilder,
)
from lnp_chat import LNPChat  # 새로운 LNP Chat 클래스를 임포트
from retriever import VectorIndex, MODEL_TEXT_COLUMN, _split_tokens


NORMALIZED_ALIASES = {
    "path": ("path", "filepath", "file_path", "fullpath", "full_path", "absolute_path"),
    "size": ("size", "filesize", "file_size", "bytes"),
    "mtime": ("mtime", "modified", "modified_time", "lastmodified", "timestamp"),
    "ctime": ("ctime", "created", "created_time", "creation", "creation_time"),
    "ext": ("ext", "extension", "suffix"),
    "drive": ("drive", "volume", "root"),
    "owner": ("owner", "user", "username", "author", "created_by"),
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
    ctime_raw = _pick_value(raw, NORMALIZED_ALIASES["ctime"])
    owner = _pick_value(raw, NORMALIZED_ALIASES["owner"])

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
    normalized["ctime"] = to_float(ctime_raw)
    if ext:
        normalized["ext"] = ext
    if drive:
        normalized["drive"] = drive
    if owner:
        normalized["owner"] = owner
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


def _iter_scan_rows(scan_csv: Path) -> Iterator[Dict[str, Any]]:
    with scan_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, raw in enumerate(reader, start=2):
            normalized = _normalize_scan_row(raw, context=f"{scan_csv}:{idx}")
            if normalized:
                yield normalized


def _load_scan_rows(scan_csv: Path) -> Iterator[Dict[str, Any]]:
    return _iter_scan_rows(scan_csv)


def _sync_scan_csv(
    scan_csv: Path,
    rows_to_add: List[Dict[str, Any]],
    paths_to_remove: Set[str],
) -> None:
    if not rows_to_add and not paths_to_remove:
        return

    def _normalize_path(raw: Any) -> str:
        return str(raw or "").strip()

    fieldnames = ["path", "size", "mtime", "ctime", "ext", "drive", "owner"]
    additions: Dict[str, Dict[str, Any]] = {}
    for row in rows_to_add:
        path_key = _normalize_path(row.get("path"))
        if not path_key:
            continue
        additions[path_key] = {name: row.get(name) for name in fieldnames}

    removals = {_normalize_path(path) for path in paths_to_remove if _normalize_path(path)}
    removals.difference_update(additions.keys())

    scan_csv.parent.mkdir(parents=True, exist_ok=True)

    if not scan_csv.exists():
        with scan_csv.open("w", encoding="utf-8", newline="") as dst:
            writer = csv.DictWriter(dst, fieldnames=fieldnames)
            writer.writeheader()
            for record in additions.values():
                writer.writerow(record)
        return

    temp_path = scan_csv.with_suffix(scan_csv.suffix + ".tmp")
    with scan_csv.open("r", encoding="utf-8", newline="") as src, temp_path.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            path_key = _normalize_path(row.get("path"))
            if not path_key or path_key in removals or path_key in additions:
                continue
            writer.writerow({name: row.get(name) for name in fieldnames})

        for record in additions.values():
            writer.writerow(record)

    temp_path.replace(scan_csv)


def _load_sentence_encoder(model_path: Path) -> Tuple[Optional[SentenceTransformer], int, str]:
    model_name = DEFAULT_EMBED_MODEL
    batch_size = 32

    if joblib is not None and model_path.exists():
        try:
            payload = joblib.load(model_path)
            model_name = payload.get("model_name", model_name)
            cfg = payload.get("train_config")
            if cfg and hasattr(cfg, "embedding_batch_size"):
                batch_size = int(getattr(cfg, "embedding_batch_size", batch_size) or batch_size)
        except Exception as exc:
            print(f"⚠️ 임베딩 모델 메타 로드 실패 → 기본값 사용({model_name}): {exc}")

    if SentenceTransformer is None:
        print("⚠️ 'sentence-transformers' 패키지를 찾을 수 없어 임베딩을 계산할 수 없습니다.")
        return None, batch_size, model_name

    try:
        encoder = SentenceTransformer(model_name)
    except Exception as exc:
        print(f"⚠️ SentenceTransformer 모델 로드 실패({model_name}): {exc}")
        return None, batch_size, model_name
    return encoder, batch_size, model_name


def _load_vector_index(cache_dir: Path) -> VectorIndex:
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta = cache_dir / "doc_meta.json"
    emb = cache_dir / "doc_embeddings.npy"
    faiss_path = cache_dir / "doc_index.faiss"

    index = VectorIndex()
    if meta.exists():
        try:
            index.load(
                emb if emb.exists() else None,
                meta,
                faiss_path=faiss_path if faiss_path.exists() else None,
                use_mmap=False,
            )
        except Exception as exc:
            print(f"⚠️ 인덱스 로드 실패 → 새 인덱스를 생성합니다: {exc}")
            index = VectorIndex()
    return index


class WatchEventHandler(FileSystemEventHandler):
    def __init__(self, event_queue: "queue.Queue[Tuple[str, str]]", allowed_exts: Set[str]) -> None:
        super().__init__()
        self._queue = event_queue
        self._allowed_exts = {ext.lower() for ext in allowed_exts}

    def _should_process(self, path: str) -> bool:
        if not path:
            return False
        ext = Path(path).suffix.lower()
        return ext in self._allowed_exts

    def on_created(self, event):  # type: ignore[override]
        if getattr(event, "is_directory", False):
            return
        if self._should_process(event.src_path):
            self._queue.put(("created", event.src_path))

    def on_modified(self, event):  # type: ignore[override]
        if getattr(event, "is_directory", False):
            return
        if self._should_process(event.src_path):
            self._queue.put(("modified", event.src_path))

    def on_moved(self, event):  # type: ignore[override]
        if getattr(event, "is_directory", False):
            return
        if self._should_process(event.src_path):
            self._queue.put(("deleted", event.src_path))
        if self._should_process(event.dest_path):
            self._queue.put(("created", event.dest_path))

    def on_deleted(self, event):  # type: ignore[override]
        if getattr(event, "is_directory", False):
            return
        if self._should_process(event.src_path):
            self._queue.put(("deleted", event.src_path))


class IncrementalPipeline:
    def __init__(
        self,
        *,
        encoder: SentenceTransformer,
        batch_size: int,
        scan_csv: Path,
        corpus_path: Path,
        cache_dir: Path,
        translate: bool,
    ) -> None:
        self.encoder = encoder
        self.batch_size = max(1, int(batch_size))
        self.scan_csv = scan_csv
        self.corpus_path = corpus_path
        self.cache_dir = cache_dir
        self.translate = translate
        self.allowed_exts = {ext.lower() for ext in FileFinder.DEFAULT_EXTS}
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def process(self, add_paths: Set[str], remove_paths: Set[str]) -> None:
        if pd is None:
            raise RuntimeError("pandas 필요. pip install pandas")
        add_paths = {p for p in add_paths if Path(p).suffix.lower() in self.allowed_exts}
        remove_paths = {p for p in remove_paths if Path(p).suffix.lower() in self.allowed_exts}

        rows_to_add: List[Dict[str, Any]] = []
        for raw_path in sorted(add_paths):
            meta = FileFinder.collect_file_metadata(Path(raw_path), allowed_exts=self.allowed_exts)
            if meta:
                rows_to_add.append(meta)

        _sync_scan_csv(self.scan_csv, rows_to_add, {str(p) for p in remove_paths})

        if remove_paths:
            remove_from_corpus(list(remove_paths), self.corpus_path)

        new_records = None
        if rows_to_add:
            cb = CorpusBuilder(progress=False, translate=self.translate)
            new_records = cb.build(rows_to_add)
        else:
            new_records = None

        if new_records is not None and not new_records.empty:
            update_corpus_file(new_records, self.corpus_path)

        index = _load_vector_index(self.cache_dir)

        paths_to_remove = set(remove_paths)
        paths_to_remove.update(row["path"] for row in rows_to_add if "path" in row)
        if paths_to_remove:
            index.remove_paths(paths_to_remove)

        if new_records is None or new_records.empty:
            index.save(self.cache_dir)
            if rows_to_add or remove_paths:
                print(
                    f"⚡ watcher: removed {len(paths_to_remove)} 문서, 새 문서 없음.",
                    flush=True,
                )
            return

        valid_mask = new_records.get("ok", True)
        if pd is not None and isinstance(valid_mask, pd.Series):
            valid_df = new_records[valid_mask & (new_records[MODEL_TEXT_COLUMN].astype(str).str.len() > 0)].copy()
        else:
            valid_df = new_records.copy()

        if valid_df.empty:
            index.save(self.cache_dir)
            print(
                f"⚡ watcher: 갱신 {len(rows_to_add)}건 중 유효 텍스트가 없습니다.",
                flush=True,
            )
            return

        texts = valid_df[MODEL_TEXT_COLUMN].astype(str).tolist()
        embeddings = self.encoder.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)

        token_lists = [[tok for tok in _split_tokens(text.lower()) if tok] for text in texts]
        previews_series = valid_df["text_original"] if "text_original" in valid_df.columns else valid_df["text"]
        previews = previews_series.fillna("").astype(str).tolist()

        for idx, (_, row) in enumerate(valid_df.iterrows()):
            index.upsert(
                path=str(row.get("path", "")),
                ext=str(row.get("ext", "")),
                embedding=embeddings[idx],
                preview=previews[idx],
                size=int(row.get("size", 0) or 0),
                mtime=float(row.get("mtime", 0.0) or 0.0),
                ctime=float(row.get("ctime", 0.0) or 0.0),
                owner=str(row.get("owner", "") or ""),
                tokens=token_lists[idx],
            )

        index.save(self.cache_dir)
        print(
            f"⚡ watcher: 문서 {len(valid_df)}건 업데이트 (제거 {len(paths_to_remove)})",
            flush=True,
        )


def _watch_loop(
    event_queue: "queue.Queue[Tuple[str, str]]",
    pipeline_ctx: IncrementalPipeline,
    stop_event: threading.Event,
    debounce_sec: float,
) -> None:
    pending_add: Set[str] = set()
    pending_remove: Set[str] = set()
    last_event = 0.0

    def _log_throughput(add_count: int, remove_count: int, elapsed: float) -> None:
        total = add_count + remove_count
        if total <= 0:
            return
        rate = total / elapsed if elapsed > 0 else 0.0
        print(
            (
                "⚙️ watcher: processed add={add} remove={rem} in {secs:.2f}s "
                "(~{rate:.1f}/s)"
            ).format(add=add_count, rem=remove_count, secs=elapsed, rate=rate),
            flush=True,
        )

    while not stop_event.is_set():
        try:
            event_type, path = event_queue.get(timeout=0.5)
            path = str(path)
            if event_type == "deleted":
                pending_remove.add(path)
                pending_add.discard(path)
            else:
                pending_add.add(path)
                pending_remove.discard(path)
            last_event = time.time()
        except queue.Empty:
            pass

        now = time.time()
        if (pending_add or pending_remove) and (now - last_event) >= debounce_sec:
            to_add = set(pending_add)
            to_remove = set(pending_remove)
            pending_add.clear()
            pending_remove.clear()
            try:
                t0 = time.time()
                pipeline_ctx.process(to_add, to_remove)
                _log_throughput(len(to_add), len(to_remove), time.time() - t0)
            except Exception as exc:
                print(f"⚠️ 증분 파이프라인 처리 중 오류: {exc}")

    # Flush remaining events
    if pending_add or pending_remove:
        try:
            to_add = set(pending_add)
            to_remove = set(pending_remove)
            t0 = time.time()
            pipeline_ctx.process(to_add, to_remove)
            _log_throughput(len(to_add), len(to_remove), time.time() - t0)
        except Exception as exc:
            print(f"⚠️ 증분 파이프라인 종료 처리 중 오류: {exc}")


def _build_train_config(args) -> TrainConfig:
    return TrainConfig(
        max_features=args.max_features,
        n_components=args.n_components,
        n_clusters=args.n_clusters,
        ngram_range=(1, 2),
        min_df=args.min_df,
        max_df=args.max_df,
        use_sentence_transformer=getattr(args, "use_embedding", True),
        embedding_model=getattr(args, "embedding_model", DEFAULT_EMBED_MODEL),
        embedding_batch_size=getattr(args, "embedding_batch_size", 32),
    )


def _maybe_limit_rows(rows: Iterable[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    iterator = iter(rows)
    if limit and limit > 0:
        limited = list(itertools.islice(iterator, limit))
        if next(iterator, None) is not None:
            print(f"⚡ 테스트 모드: 상위 {limit}개 파일만 사용합니다.")
        return limited
    return list(iterator)


def _default_train_config() -> TrainConfig:
    return TrainConfig(
        max_features=50000,
        n_components=DEFAULT_N_COMPONENTS,
        n_clusters=25,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
        use_sentence_transformer=True,
        embedding_model=DEFAULT_EMBED_MODEL,
        embedding_batch_size=32,
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
    rows = list(_load_scan_rows(resolved_scan))
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
    row_iter = _load_scan_rows(scan_csv)
    rows = _maybe_limit_rows(row_iter, args.limit_files)

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
    scan_rows = _run_scan(out, roots)
    rows = _maybe_limit_rows(scan_rows, args.limit_files)

    if not rows:
        raise ValueError("유효한 학습 대상 행이 없습니다. 스캔 결과를 확인해주세요.")

    cfg = _build_train_config(args)
    out_corpus = Path(args.corpus)
    out_model = Path(args.model)
    df, tm = run_step2(rows, out_corpus=out_corpus, out_model=out_model, cfg=cfg, use_tqdm=True, translate=args.translate)
    print("✅ 파이프라인 완료")

    cache_dir = Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(
        "ℹ️ 파이프라인은 scan/train 단계까지만 자동 실행되며 chat 모드는 별도 실행이 필요합니다.\n"
        f"   → python infopilot.py chat --model {out_model} --corpus {out_corpus} --cache {cache_dir}"
    )

    if getattr(args, "launch_chat", False):
        print("\n💬 바로 chat 모드를 실행합니다. (종료하려면 'exit')")
        chat_args = argparse.Namespace(
            model=str(out_model),
            corpus=str(out_corpus),
            cache=str(cache_dir),
            scan_csv=str(out),
            topk=5,
            translate=args.translate,
            auto_train=True,
            rerank=True,
            rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            rerank_depth=80,
            rerank_batch_size=16,
            rerank_device=None,
            rerank_min_score=0.35,
            lexical_weight=0.0,
            show_translation=False,
            translation_lang="en",
            min_similarity=0.35,
        )
        cmd_chat(chat_args)


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
        translate=args.translate, # 번역 옵션 전달
        rerank=args.rerank,
        rerank_model=args.rerank_model,
        rerank_depth=args.rerank_depth,
        rerank_batch_size=args.rerank_batch_size,
        rerank_device=args.rerank_device or None,
        rerank_min_score=args.rerank_min_score,
        lexical_weight=args.lexical_weight,
        show_translation=args.show_translation,
        translation_lang=args.translation_lang,
        min_similarity=args.min_similarity,
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


def cmd_watch(args):
    if Observer is None:
        raise RuntimeError("watchdog 라이브러리가 필요합니다. pip install watchdog")

    encoder, batch_size, model_name = _load_sentence_encoder(Path(args.model))
    if encoder is None:
        raise RuntimeError("sentence-transformers 모델을 로드할 수 없어 watcher를 실행할 수 없습니다.")

    roots = _parse_roots(args.roots)
    if not roots:
        roots = [Path.cwd()]

    event_queue: "queue.Queue[Tuple[str, str]]" = queue.Queue()
    allowed_exts = {ext.lower() for ext in FileFinder.DEFAULT_EXTS}
    handler = WatchEventHandler(event_queue, allowed_exts)
    observer = Observer()
    for root in roots:
        observer.schedule(handler, str(root), recursive=True)

    pipeline_ctx = IncrementalPipeline(
        encoder=encoder,
        batch_size=batch_size,
        scan_csv=Path(args.scan_csv),
        corpus_path=Path(args.corpus),
        cache_dir=Path(args.cache),
        translate=args.translate,
    )

    debounce_sec = max(0.5, args.debounce_ms / 1000.0)
    stop_event = threading.Event()

    print(
        "👀 파일 변경 감시를 시작합니다. (Ctrl+C로 종료)"
        f"\n   roots: {', '.join(str(r) for r in roots)}"
        f"\n   embedding model: {model_name} (batch={batch_size})"
        f"\n   debounce: {debounce_sec:.2f}s"
    )

    observer.start()
    try:
        _watch_loop(event_queue, pipeline_ctx, stop_event, debounce_sec)
    except KeyboardInterrupt:
        print("\n👋 watcher를 종료합니다.")
    finally:
        stop_event.set()
        observer.stop()
        observer.join()


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
    ap_train = sp.add_parser(
        "train",
        help="코퍼스 생성 + 모델 학습 (기본: 번역 비활성, 다국어 Sentence-BERT)",
    )
    ap_train.add_argument("--scan_csv", default="./data/found_files.csv")
    ap_train.add_argument("--corpus", default="./data/corpus.parquet")
    ap_train.add_argument("--model", default="./data/topic_model.joblib")
    ap_train.add_argument("--max_features", type=int, default=50000)
    ap_train.add_argument("--n_components", type=int, default=DEFAULT_N_COMPONENTS)
    ap_train.add_argument("--n_clusters", type=int, default=25)
    ap_train.add_argument("--min_df", type=int, default=2)
    ap_train.add_argument("--max_df", type=float, default=0.85)
    ap_train.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL, help="Sentence-BERT 임베딩 모델 이름")
    ap_train.add_argument("--embedding-batch-size", type=int, default=32, help="Sentence-BERT 배치 크기")
    ap_train.add_argument(
        "--limit",
        "--limit-files",
        dest="limit_files",
        type=int,
        default=0,
        help="테스트용으로 상위 N개 파일만 사용합니다 (0=전체).",
    )
    translate_group = ap_train.add_mutually_exclusive_group()
    translate_group.add_argument(
        "--translate",
        dest="translate",
        action="store_true",
        help="deep-translator로 영어 번역을 강제 활성화합니다.",
    )
    translate_group.add_argument(
        "--no-translate",
        dest="translate",
        action="store_false",
        help="번역 기능을 비활성화하고 원문으로 학습합니다.",
    )
    ap_train.add_argument("--no-embedding", dest="use_embedding", action="store_false", help="Sentence-BERT 대신 TF-IDF 백업 경로를 사용합니다.")
    ap_train.set_defaults(translate=False)
    ap_train.set_defaults(use_embedding=True)
    ap_train.set_defaults(func=cmd_train)

    # pipeline
    ap_pipe = sp.add_parser(
        "pipeline",
        help="스캔 후 바로 학습까지 진행 (기본: 번역 비활성)",
    )
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
    ap_pipe.add_argument("--cache", default="./index_cache")
    ap_pipe.add_argument("--max_features", type=int, default=50000)
    ap_pipe.add_argument("--n_components", type=int, default=DEFAULT_N_COMPONENTS)
    ap_pipe.add_argument("--n_clusters", type=int, default=25)
    ap_pipe.add_argument("--min_df", type=int, default=2)
    ap_pipe.add_argument("--max_df", type=float, default=0.85)
    ap_pipe.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL, help="Sentence-BERT 임베딩 모델 이름")
    ap_pipe.add_argument("--embedding-batch-size", type=int, default=32, help="Sentence-BERT 배치 크기")
    ap_pipe.add_argument(
        "--limit",
        "--limit-files",
        dest="limit_files",
        type=int,
        default=0,
        help="테스트용으로 상위 N개 파일만 사용합니다 (0=전체).",
    )
    translate_group_pipe = ap_pipe.add_mutually_exclusive_group()
    translate_group_pipe.add_argument(
        "--translate",
        dest="translate",
        action="store_true",
        help="deep-translator로 영어 번역을 강제 활성화합니다.",
    )
    translate_group_pipe.add_argument(
        "--no-translate",
        dest="translate",
        action="store_false",
        help="번역 기능을 비활성화하고 원문으로 학습합니다.",
    )
    ap_pipe.add_argument("--no-embedding", dest="use_embedding", action="store_false", help="Sentence-BERT 대신 TF-IDF 백업 경로를 사용합니다.")
    ap_pipe.add_argument(
        "--launch-chat",
        action="store_true",
        help="파이프라인 완료 후 chat 모드를 바로 실행합니다.",
    )
    ap_pipe.set_defaults(translate=False)
    ap_pipe.set_defaults(use_embedding=True)
    ap_pipe.set_defaults(func=cmd_pipeline)

    # chat
    ap_chat = sp.add_parser(
        "chat",
        help="대화형 질의 모드 (기본: 번역 비활성, 다국어 Sentence-BERT)",
    )
    ap_chat.add_argument("--model", default="./data/topic_model.joblib")
    ap_chat.add_argument("--corpus", default="./data/corpus.parquet")
    ap_chat.add_argument("--cache", default="./index_cache")
    ap_chat.add_argument("--scan_csv", default="./data/found_files.csv")
    ap_chat.add_argument("--topk", type=int, default=5)
    translate_group_chat = ap_chat.add_mutually_exclusive_group()
    translate_group_chat.add_argument(
        "--translate",
        dest="translate",
        action="store_true",
        help="deep-translator로 질의를 영어 번역한 뒤 검색합니다.",
    )
    translate_group_chat.add_argument(
        "--no-translate",
        dest="translate",
        action="store_false",
        help="질문 번역 기능을 비활성화합니다.",
    )
    ap_chat.add_argument("--no-auto-train", dest="auto_train", action="store_false", help="자동 학습 갱신을 비활성화합니다.")
    rerank_group = ap_chat.add_mutually_exclusive_group()
    rerank_group.add_argument("--rerank", dest="rerank", action="store_true", help="Cross-Encoder 재랭킹을 사용합니다 (기본값).")
    rerank_group.add_argument("--no-rerank", dest="rerank", action="store_false", help="Cross-Encoder 재랭킹을 비활성화합니다.")
    ap_chat.add_argument("--rerank-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="재랭킹에 사용할 Cross-Encoder 모델 이름")
    ap_chat.add_argument("--rerank-depth", type=int, default=80, help="Cross-Encoder 재랭킹에 포함할 후보 문서 수 (50~100 권장)")
    ap_chat.add_argument("--rerank-batch-size", type=int, default=16, help="Cross-Encoder 추론 배치 크기 (CPU 환경은 8~16 권장)")
    ap_chat.add_argument("--rerank-device", default=None, help="재랭킹 모델을 로드할 디바이스(e.g. 'cuda', 'cuda:0', 'cpu')")
    ap_chat.add_argument(
        "--rerank-min-score",
        type=float,
        default=0.35,
        help="Cross-Encoder 점수가 이 값보다 낮은 문서는 제외합니다.",
    )
    ap_chat.add_argument(
        "--lexical-weight",
        type=float,
        default=0.0,
        help="BM25 가중치 (0=의미 검색 전용). 필요 시 수동 조정",
    )
    ap_chat.add_argument(
        "--min-similarity",
        type=float,
        default=0.35,
        help="이 값보다 낮은 유사도 문서는 제외합니다 (0.0~1.0).",
    )
    ap_chat.add_argument("--show-translation", action="store_true", help="검색 결과에 번역본을 함께 표시합니다.")
    ap_chat.add_argument("--translation-lang", default="en", help="번역 대상 언어 코드 (기본: en)")
    ap_chat.set_defaults(translate=False)
    ap_chat.set_defaults(auto_train=True)
    ap_chat.set_defaults(rerank=True)
    ap_chat.set_defaults(func=cmd_chat)

    # watch
    ap_watch = sp.add_parser("watch", help="파일 변경을 감지해 코퍼스/인덱스를 증분 갱신")
    ap_watch.add_argument("--root", "--roots", dest="roots", action="append", help="감시할 루트 디렉터리 (여러 번 지정 가능)")
    ap_watch.add_argument("--scan_csv", default="./data/found_files.csv")
    ap_watch.add_argument("--corpus", default="./data/corpus.parquet")
    ap_watch.add_argument("--model", default="./data/topic_model.joblib")
    ap_watch.add_argument("--cache", default="./index_cache")
    ap_watch.add_argument("--debounce-ms", type=int, default=2000, help="파일 이벤트 디바운스 시간(ms)")
    ap_watch.add_argument("--translate", action="store_true", help="증분 추출 시 번역을 포함합니다.")
    ap_watch.set_defaults(func=cmd_watch)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
