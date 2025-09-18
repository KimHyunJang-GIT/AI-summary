# infopilot.py (Main CLI)
from pathlib import Path
import argparse

# 업그레이드된 모듈 임포트
from filefinder import FileFinder
from pipeline import run_indexing
from lnp_chat import LNPChat

# 텍스트 추출을 위한 CorpusBuilder 임포트 (scan -> train 직접 연결 시 필요)
from pipeline import CorpusBuilder


def cmd_scan(args):
    """파일 시스템을 스캔하여 파일 목록을 생성합니다."""
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
    """스캔된 파일 목록을 기반으로 의미 기반 인덱스를 생성합니다."""
    import pandas as pd
    scan_csv_path = Path(args.scan_csv)
    corpus_path = Path(args.corpus)

    # 1. 스캔 결과(csv)를 읽어 코퍼스(parquet)로 변환
    print(f"📥 스캔 목록 로드: {scan_csv_path}")
    df_scan = pd.read_csv(scan_csv_path)
    file_rows = df_scan.to_dict('records')

    print("🛠️ 문서 텍스트 추출 시작...")
    # 번역 기능은 모델 자체의 다국어 성능을 활용하므로 False로 설정
    cb = CorpusBuilder(max_text_chars=200_000, progress=True, translate=False)
    df_corpus = cb.build(file_rows)
    cb.save(df_corpus, corpus_path)
    print(f"💾 코퍼스 저장 완료: {corpus_path}")

    # 2. 생성된 코퍼스를 기반으로 인덱싱 실행
    run_indexing(corpus_path=corpus_path, cache_dir=Path(args.cache))


def cmd_chat(args):
    """대화형 의미 기반 검색 모드를 시작합니다."""
    chat_session = LNPChat(
        corpus_path=Path(args.corpus),
        cache_dir=Path(args.cache),
        topk=args.topk,
    )
    chat_session.ready(rebuild=False)

    print("\n💬 InfoPilot Chat (의미 기반 검색) 모드입니다. (종료: 'exit' 또는 '종료')")
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

        result = chat_session.ask(query)
        print(result["answer"])
        if result.get("suggestions"):
            print("\n💡 이런 질문은 어떠세요?")
            for s in result["suggestions"]:
                print(f"   - {s}")
        print("-" * 80)


def main():
    ap = argparse.ArgumentParser(prog="infopilot", description="InfoPilot CLI - 의미 기반 문서 검색 엔진")
    sp = ap.add_subparsers(dest="cmd", required=True)

    # scan
    ap_scan = sp.add_parser("scan", help="드라이브를 스캔하여 파일 목록을 수집합니다.")
    ap_scan.add_argument("--out", default="./data/found_files.csv")
    ap_scan.set_defaults(func=cmd_scan)

    # train (이제 '인덱싱' 역할을 합니다)
    ap_train = sp.add_parser("train", help="스캔된 파일의 의미 벡터 인덱스를 생성합니다.")
    ap_train.add_argument("--scan_csv", default="./data/found_files.csv")
    ap_train.add_argument("--corpus", default="./data/corpus.parquet")
    ap_train.add_argument("--cache", default="./index_cache")
    ap_train.set_defaults(func=cmd_train)

    # chat
    ap_chat = sp.add_parser("chat", help="대화형 의미 기반 검색을 시작합니다.")
    ap_chat.add_argument("--corpus", default="./data/corpus.parquet")
    ap_chat.add_argument("--cache", default="./index_cache")
    ap_chat.add_argument("--topk", type=int, default=100)
    ap_chat.set_defaults(func=cmd_chat)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
