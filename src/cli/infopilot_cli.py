import argparse
import sys
from pathlib import Path
import pandas as pd
import re # Added for filter parsing
import os # Added for os.walk
import string # Added for get_drives
from tqdm import tqdm # Added for progress bar

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.corpus import CorpusBuilder
from src.core.indexing import run_indexing
from src.app.chat import LNPChat

def get_drives():
    """시스템에 존재하는 드라이브 목록을 반환합니다."""
    drives = []
    for letter in string.ascii_uppercase:
        drive = f"{letter}:\\"
        if os.path.exists(drive):
            drives.append(drive)
    return drives

def parse_query_and_filters(query: str) -> tuple[str, dict]:
    print(f"[DEBUG] Original query: '{query}'") # DEBUG
    filters = {}
    
    # Canonical mapping for extensions (case-insensitive matching)
    base_ext_map = {
        ".pdf": ["pdf", "피디에프"],
        ".xlsx": ["엑셀", "excel"],
        ".hwp": ["한글", "hwp"],
        ".docx": ["워드", "word"],
        ".pptx": ["파워포인트", "ppt"], # ppt can map to .pptx or .ppt, prioritize .pptx
        ".txt": ["텍스트", "txt"],
        ".csv": ["csv"],
        ".doc": ["doc"],
        ".xls": ["xls"],
        ".xlsm": ["xlsm"],
        ".ppt": ["ppt"], # Explicit .ppt for older versions
        ".py": ["py"],
        ".json": ["json"],
        ".xml": ["xml"],
        ".html": ["html"],
        ".css": ["css"],
        ".js": ["js"],
        ".md": ["md"],
    }

    ext_map = {}
    for ext, keywords in base_ext_map.items():
        ext_map[ext] = ext # Add direct extension mapping (e.g., ".pdf": ".pdf")
        for keyword in keywords:
            ext_map[keyword] = ext # Add keyword mapping (e.g., "pdf": ".pdf")

    temp_query = query
    
    # 1. Extract explicit filters (key:value) first
    explicit_filter_pattern = re.compile(r'(\w+):([^\s]+)')
    explicit_matches = list(explicit_filter_pattern.finditer(temp_query))
    for match in reversed(explicit_matches): # Process in reverse to avoid index issues with replacement
        key = match.group(1).lower()
        value = match.group(2)
        filters[key] = value
        temp_query = temp_query[:match.start()] + " " * len(match.group(0)) + temp_query[match.end():]
    print(f"[DEBUG] After explicit filters: '{temp_query}', filters: {filters}") # DEBUG

    # 2. Extract implicit filters from the remaining query
    
    # Process for implicit extension
    # Try to match direct extensions like ".pdf" first
    direct_ext_pattern = re.compile(r'\.(\w+)\b', re.IGNORECASE) # Matches ".ext"
    match = direct_ext_pattern.search(temp_query)
    if match:
        matched_ext = "." + match.group(1).lower() # e.g., ".pdf"
        if matched_ext in ext_map: # Check if it's a valid extension we support
            filters['ext'] = ext_map[matched_ext] # Use ext_map to get canonical form
            temp_query = temp_query.replace(match.group(0), " " * len(match.group(0)), 1)
    print(f"[DEBUG] After direct ext match: '{temp_query}', filters: {filters}") # DEBUG
    
    # Then try to match keywords like "pdf", "엑셀" with optional suffixes
    if 'ext' not in filters: # Only if extension filter hasn't been set by direct match
        # Generate regex parts from the keys of the generated ext_map that are not direct extensions
        sorted_ext_keywords = sorted([k for k in ext_map.keys() if not k.startswith('.')], key=len, reverse=True)
        ext_keyword_regex_parts = []
        for k in sorted_ext_keywords:
            ext_keyword_regex_parts.append(re.escape(k) + r'(?:\s*(?:파일|문서|자료))?')
        
        if ext_keyword_regex_parts: # Ensure there are parts to join
            implicit_ext_keyword_pattern = re.compile(r'\b(' + '|'.join(ext_keyword_regex_parts) + r')\b', re.IGNORECASE)
            match = implicit_ext_keyword_pattern.search(temp_query)
            if match:
                matched_text = match.group(1).lower()
                matched_keyword = re.sub(r'\s*(?:파일|문서|자료)$', '', matched_text) 
                if matched_keyword in ext_map:
                    filters['ext'] = ext_map[matched_keyword]
                    temp_query = temp_query.replace(match.group(0), " " * len(match.group(0)), 1)
    print(f"[DEBUG] After implicit keyword ext match: '{temp_query}', filters: {filters}") # DEBUG

    # Process for implicit title
    implicit_title_patterns = [
        re.compile(r'(제목이|이름이)\s*(\S+)(?:인|인문서|인파일)?', re.IGNORECASE), # "제목이 보고서인" -> "보고서"
        re.compile(r'(\S+)(?:라는|이라는)\s*(제목의|이름의)', re.IGNORECASE), # "보고서라는 제목의" -> "보고서"
    ]

    for pattern in implicit_title_patterns:
        match = pattern.search(temp_query)
        if match:
            title_value = match.group(2) if len(match.groups()) >= 2 else match.group(1)
            filters['title'] = title_value
            temp_query = temp_query.replace(match.group(0), " " * len(match.group(0)), 1)
            break # Only extract one title filter for simplicity
    print(f"[DEBUG] After implicit title match: '{temp_query}', filters: {filters}") # DEBUG

    # Clean up the query: remove placeholders and extra spaces
    cleaned_query = re.sub(r'\s+', ' ', temp_query).strip()
    
    return cleaned_query, filters

def cmd_scan(args):
    """파일 시스템을 스캔하여 파일 목록을 생성합니다."""
    EXCLUDE_DIRS = {
        ".git", ".venv", "venv", "node_modules", "__pycache__", ".idea", ".vscode",
        "Windows", "Program Files", "Program Files (x86)", "AppData", 
        "$RECYCLE.BIN", "System Volume Information", "Recovery", "PerfLogs",
        "Downloads",
        ".gradle", "plastic4", "ESTsoft", "Bitdefender", "Autodesk", "Intel", "NVIDIA", "Zoom", "Wondershare",
    }

    # Default supported extensions for CLI scan. Can be made configurable via args if needed.
    SUPPORTED_EXTS = {
        '.txt', '.csv', '.md', '.py', '.json', '.xml', '.html', '.css', '.js', 
        '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', 
        '.pdf', 
        '.hwp'
    }

    file_rows = []
    drives = get_drives()
    
    print(f"🔍 Starting scan on drives: {', '.join(drives)}")
    print(f"🚫 Excluding directories containing: {', '.join(sorted(list(EXCLUDE_DIRS)))}")

    for drive in drives:
        print(f"Scanning drive {drive}...")
        try:
            for root, dirs, files in tqdm(os.walk(drive, topdown=True), desc=f"Scanning {drive}", encoding='utf-8'): # Added encoding='utf-8'
                dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
                
                for file in files:
                    try:
                        p = Path(root) / file
                        if p.suffix.lower() in SUPPORTED_EXTS:
                            if not any(part in EXCLUDE_DIRS for part in p.parts):
                                file_rows.append({'path': str(p), 'size': p.stat().st_size})
                    except (FileNotFoundError, PermissionError):
                        continue # Skip files that can't be accessed
        except PermissionError:
            print(f"Could not access {drive}. Skipping.")
            continue

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(file_rows)
    df.to_csv(out, index=False, encoding='utf-8')
    
    print(f"📦 스캔 결과 저장: {out} ({len(file_rows)}개 파일)")

def cmd_train(args):
    """스캔된 파일 목록에서 텍스트를 추출하고, 의미 기반 인덱스를 생성합니다."""
    scan_csv_path = Path(args.scan_csv)
    corpus_path = Path(args.corpus)

    print(f"📥 스캔 목록 로드: {scan_csv_path}")
    df_scan = pd.read_csv(scan_csv_path)
    file_rows = df_scan.to_dict('records')

    print("🛠️ 문서 텍스트 추출 및 요약 생성 시작...")
    # --- CRITICAL FIX: Force sequential processing for debugging CLI crashes ---
    cb = CorpusBuilder(progress=True, max_workers=0) 
    df_corpus = cb.build(file_rows)
    
    cb.save(df_corpus, corpus_path)
    print(f"💾 코퍼스 및 성공/실패 목록 저장 완료.")

    print("🚀 Starting semantic indexing...")
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
        
        # Parse query and filters
        cleaned_query, filters = parse_query_and_filters(query)
        
        # Pass filters to the ask method
        result = chat_session.ask(cleaned_query, filters=filters)
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
    ap_train = sp.add_parser("train", help="스캔된 파일의 텍스트를 추출하고 의미 벡터 인덱스를 생성합니다.")
    ap_train.add_argument("--scan_csv", default="./data/found_files.csv")
    ap_train.add_argument("--corpus", default="./data/corpus.parquet")
    ap_train.add_argument("--cache", default="./index_cache")
    ap_train.set_defaults(func=cmd_train)

    # chat
    ap_chat = sp.add_parser("chat", help="대화형 의미 기반 검색을 시작합니다.")
    ap_chat.add_argument("--corpus", default="./data/corpus.parquet")
    ap_chat.add_argument("--cache", default="./index_cache")
    ap_chat.add_argument("--topk", type=int, default=10)
    ap_chat.set_defaults(func=cmd_chat)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
