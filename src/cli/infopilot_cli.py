import argparse
import sys
from pathlib import Path
import pandas as pd
import os
import string
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

# 중앙 설정 및 새로운 쿼리 파서 가져오기
from src.config import (
    EXCLUDE_DIRS, SUPPORTED_EXTS,
    FOUND_FILES_CSV, CORPUS_PARQUET, CACHE_DIR,
    DEFAULT_TOP_K
)
from src.core.corpus import CorpusBuilder
from src.core.indexing import run_indexing
from src.app.chat import LNPChat
from src.core.query_parser import parse_query_and_filters # 새로운 쿼리 파서 import

def get_drives():
    """시스템에 존재하는 드라이브 목록을 반환합니다."""
    drives = []
    for letter in string.ascii_uppercase:
        drive = f"{letter}:\\"
        if os.path.exists(drive):
            drives.append(drive)
    return drives

def cmd_scan(args):
    """파일 시스템을 스캔하여 파일 목록과 메타데이터(크기, 수정시간)를 생성합니다."""
    file_rows = []
    drives = get_drives()
    print(f"🔍 Starting scan on drives: {', '.join(drives)}")
    print(f"🚫 Excluding directories containing: {', '.join(sorted(list(EXCLUDE_DIRS)))}")
    for drive in drives:
        print(f"Scanning drive {drive}...")
        try:
            for root, dirs, files in tqdm(os.walk(drive, topdown=True), desc=f"Scanning {drive}", encoding='utf-8'):
                dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
                for file in files:
                    try:
                        p = Path(root) / file
                        if p.suffix.lower() in SUPPORTED_EXTS and not any(part in EXCLUDE_DIRS for part in p.parts):
                            stat = p.stat()
                            file_rows.append({
                                'path': str(p),
                                'size': stat.st_size,
                                'mtime': stat.st_mtime
                            })
                    except (FileNotFoundError, PermissionError): continue
        except PermissionError:
            print(f"Could not access {drive}. Skipping.")
            continue
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(file_rows).to_csv(out, index=False, encoding='utf-8')
    print(f"📦 스캔 결과 저장: {out} ({len(file_rows)}개 파일)")

def cmd_train(args):
    """(전체 학습) 스캔된 파일 목록에서 텍스트를 추출하고, 인덱스를 생성합니다."""
    scan_csv_path = Path(args.scan_csv)
    corpus_path = Path(args.corpus)
    print(f"📥 스캔 목록 로드: {scan_csv_path}")
    file_rows = pd.read_csv(scan_csv_path).to_dict('records')
    print("🛠️ 문서 텍스트 추출 및 요약 생성 시작...")
    cb = CorpusBuilder(progress=True, max_workers=0)
    df_corpus = cb.build(file_rows)
    cb.save(df_corpus, corpus_path)
    print(f"💾 코퍼스 및 성공/실패 목록 저장 완료.")
    print("🚀 Starting semantic indexing...")
    run_indexing(corpus_path=corpus_path, cache_dir=Path(args.cache))

def cmd_update(args):
    """(증분 업데이트) 변경된 파일만 감지하여 코퍼스와 인덱스를 업데이트합니다."""
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print("기존 코퍼스 파일이 없습니다. 먼저 'train' 명령으로 전체 학습을 실행해주세요.", file=sys.stderr)
        return

    print("📥 기존 코퍼스 로딩 중...")
    df_old = pd.read_parquet(corpus_path)
    print(f"기존 코퍼스 로드 완료. ({len(df_old)}개 항목)")

    print("🔍 현재 파일 시스템 스캔 중...")
    # 기존 코퍼스에 있는 확장자들을 기반으로 스캔
    exts_to_scan = df_old['ext'].unique()
    # 임시 스캔 파일을 사용하지 않고 직접 스캔 결과를 받음
    current_files_list = []
    drives = get_drives()
    for drive in drives:
        for root, dirs, files in os.walk(drive, topdown=True):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for file in files:
                try:
                    p = Path(root) / file
                    if p.suffix.lower() in exts_to_scan and not any(part in EXCLUDE_DIRS for part in p.parts):
                        stat = p.stat()
                        current_files_list.append({'path': str(p), 'size': stat.st_size, 'mtime': stat.st_mtime})
                except (FileNotFoundError, PermissionError): continue
    df_current = pd.DataFrame(current_files_list)
    print(f"현재 파일 스캔 완료. ({len(df_current)}개 파일 발견)")

    print("🔄 변경된 파일 분석 중...")
    df_old.rename(columns={'mtime': 'mtime_old'}, inplace=True)
    df_merged = pd.merge(df_current, df_old, on='path', how='outer', suffixes=('', '_old'), indicator=True)

    deleted_files = df_merged[df_merged['_merge'] == 'right_only']
    new_files = df_merged[df_merged['_merge'] == 'left_only']
    both_df = df_merged[df_merged['_merge'] == 'both']
    modified_files = both_df[both_df['mtime'] > both_df['mtime_old']]

    print(f"- 신규 파일: {len(new_files)}개")
    print(f"- 수정된 파일: {len(modified_files)}개")
    print(f"- 삭제된 파일: {len(deleted_files)}개")

    files_to_process_paths = pd.concat([new_files['path'], modified_files['path']]).unique().tolist()

    if not files_to_process_paths and len(deleted_files) == 0:
        print("🎉 변경된 파일이 없습니다. 모든 데이터가 최신 상태입니다!")
        return

    if files_to_process_paths:
        print(f"🛠️ {len(files_to_process_paths)}개 파일 텍스트 추출 시작...")
        rows_to_process = df_current[df_current['path'].isin(files_to_process_paths)].to_dict('records')
        cb = CorpusBuilder(progress=True)
        df_new_corpus = cb.build(rows_to_process)
        print("텍스트 추출 완료.")
    else:
        df_new_corpus = pd.DataFrame()

    print("💾 코퍼스 업데이트 중...")
    paths_to_remove = set(deleted_files['path'].tolist() + modified_files['path'].tolist())
    df_updated = df_old[~df_old['path'].isin(paths_to_remove)]
    final_corpus = pd.concat([df_updated, df_new_corpus], ignore_index=True)
    CorpusBuilder.save(final_corpus, corpus_path)
    print("코퍼스 업데이트 완료.")

    print("🚀 벡터 인덱스 재생성 중...")
    run_indexing(corpus_path=corpus_path, cache_dir=Path(args.cache))
    print("✨ 모든 업데이트 과정 완료!")

def cmd_chat(args):
    """대화형 의미 기반 검색 모드를 시작합니다."""
    chat_session = LNPChat(corpus_path=Path(args.corpus), cache_dir=Path(args.cache), topk=args.topk)
    chat_session.ready(rebuild=False)
    print("\n💬 InfoPilot Chat (의미 기반 검색) 모드입니다. (종료: 'exit' 또는 '종료')")
    while True:
        try:
            query = input("질문> ").strip()
        except (EOFError, KeyboardInterrupt): print("\n👋 종료합니다."); break
        if not query: continue
        if query.lower() in {"exit", "quit", "종료"}: print("👋 종료합니다."); break
        cleaned_query, filters = parse_query_and_filters(query)
        print(f"[DEBUG] Cleaned Query: '{cleaned_query}', Filters: {filters}") # 디버깅 로그 추가
        result = chat_session.ask(cleaned_query, filters=filters)
        print(result["answer"])
        if result.get("suggestions"): 
            print("\n💡 이런 질문은 어떠세요?")
            for s in result["suggestions"]: print(f"   - {s}")
        print("-" * 80)

def main():
    ap = argparse.ArgumentParser(prog="infopilot", description="InfoPilot CLI - 의미 기반 문서 검색 엔진")
    sp = ap.add_subparsers(dest="cmd", required=True)

    # scan
    ap_scan = sp.add_parser("scan", help="드라이브를 스캔하여 파일 목록을 수집합니다.")
    ap_scan.add_argument("--out", default=str(FOUND_FILES_CSV), help=f"스캔 결과 CSV 파일 경로 (기본값: {FOUND_FILES_CSV})")
    ap_scan.set_defaults(func=cmd_scan)

    # train (전체 학습)
    ap_train = sp.add_parser("train", help="(전체 학습) 스캔된 파일의 텍스트를 추출하고 인덱스를 생성합니다.")
    ap_train.add_argument("--scan_csv", default=str(FOUND_FILES_CSV), help=f"입력으로 사용할 스캔 결과 CSV (기본값: {FOUND_FILES_CSV})")
    ap_train.add_argument("--corpus", default=str(CORPUS_PARQUET), help=f"생성될 코퍼스 파일 경로 (기본값: {CORPUS_PARQUET})")
    ap_train.add_argument("--cache", default=str(CACHE_DIR), help=f"인덱스 캐시 디렉토리 (기본값: {CACHE_DIR})")
    ap_train.set_defaults(func=cmd_train)

    # update (증분 업데이트)
    ap_update = sp.add_parser("update", help="(증분 업데이트) 변경된 파일만 감지하여 코퍼스와 인덱스를 업데이트합니다.")
    ap_update.add_argument("--corpus", default=str(CORPUS_PARQUET), help=f"업데이트할 코퍼스 파일 경로 (기본값: {CORPUS_PARQUET})")
    ap_update.add_argument("--cache", default=str(CACHE_DIR), help=f"인덱스 캐시 디렉토리 (기본값: {CACHE_DIR})")
    ap_update.set_defaults(func=cmd_update)

    # chat
    ap_chat = sp.add_parser("chat", help="대화형 의미 기반 검색을 시작합니다.")
    ap_chat.add_argument("--corpus", default=str(CORPUS_PARQUET), help=f"코퍼스 파일 경로 (기본값: {CORPUS_PARQUET})")
    ap_chat.add_argument("--cache", default=str(CACHE_DIR), help=f"인덱스 캐시 디렉토리 (기본값: {CACHE_DIR})")
    ap_chat.add_argument("--topk", type=int, default=DEFAULT_TOP_K, help=f"상위 K개 결과 반환 (기본값: {DEFAULT_TOP_K})")
    ap_chat.set_defaults(func=cmd_chat)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
