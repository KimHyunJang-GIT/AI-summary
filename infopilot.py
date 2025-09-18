# infopilot.py
from pathlib import Path
import argparse

# 모듈 임포트
from filefinder import FileFinder
from pipeline import run_step2, TrainConfig
from lnp_chat import LNPChat # 새로운 LNP Chat 클래스를 임포트


def cmd_scan(args):
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
    files = finder.find(run_async=False)
    out = Path(args.out)
    FileFinder.to_csv(files, out)
    print(f"📦 스캔 결과 저장: {out}")


def cmd_train(args):
    import csv
    rows = []
    with open(args.scan_csv, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            r["size"] = int(r.get("size") or 0)
            r["mtime"] = float(r.get("mtime") or 0.0)
            rows.append(r)

    cfg = TrainConfig(
        max_features=args.max_features,
        n_components=args.n_components,
        n_clusters=args.n_clusters,
        ngram_range=(1, 2),
        min_df=args.min_df,
        max_df=args.max_df,
    )
    out_corpus = Path(args.corpus)
    out_model = Path(args.model)
    # run_step2 호출 시 translate 플래그 전달
    df, tm = run_step2(rows, out_corpus=out_corpus, out_model=out_model, cfg=cfg, use_tqdm=True, translate=args.translate)
    print("✅ 학습 완료")


def cmd_chat(args):
    """대화형 검색 모드 (LNPChat 사용)"""
    # LNPChat 클래스 인스턴스 생성 및 준비
    chat_session = LNPChat(
        model_path=Path(args.model),
        corpus_path=Path(args.corpus),
        cache_dir=Path(args.cache),
        topk=args.topk,
        translate=args.translate # 번역 옵션 전달
    )
    chat_session.ready(rebuild=False)

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
    ap_scan.set_defaults(func=cmd_scan)

    # train
    ap_train = sp.add_parser("train", help="코퍼스 생성 + 모델 학습 (기본: 번역 활성)")
    ap_train.add_argument("--scan_csv", default="./data/found_files.csv")
    ap_train.add_argument("--corpus", default="./data/corpus.parquet")
    ap_train.add_argument("--model", default="./data/topic_model.joblib")
    ap_train.add_argument("--max_features", type=int, default=150000)
    ap_train.add_argument("--n_components", type=int, default=128)
    ap_train.add_argument("--n_clusters", type=int, default=25)
    ap_train.add_argument("--min_df", type=int, default=2)
    ap_train.add_argument("--max_df", type=float, default=0.85)
    ap_train.add_argument("--no-translate", dest="translate", action="store_false", help="번역 기능을 비활성화하고 원문으로 학습합니다.")
    ap_train.set_defaults(translate=True)
    ap_train.set_defaults(func=cmd_train)

    # chat
    ap_chat = sp.add_parser("chat", help="대화형 질의 모드 (기본: 번역 활성)")
    ap_chat.add_argument("--model", default="./data/topic_model.joblib")
    ap_chat.add_argument("--corpus", default="./data/corpus.parquet")
    ap_chat.add_argument("--cache", default="./index_cache")
    ap_chat.add_argument("--topk", type=int, default=5)
    ap_chat.add_argument("--no-translate", dest="translate", action="store_false", help="질문 번역 기능을 비활성화합니다.")
    ap_chat.set_defaults(translate=True)
    ap_chat.set_defaults(func=cmd_chat)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
