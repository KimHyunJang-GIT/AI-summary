import sys
import os
from pathlib import Path
import pandas as pd
import time, joblib
import streamlit as st
import string
import re # For parse_query_and_filters
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

# Add project root to sys.path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.app.paths import Paths, have_all_artifacts
# from src.core.file_finder import FileFinder # Replaced with robust os.walk
from src.core.corpus import CorpusBuilder
from src.core.indexing import run_indexing
# from src.app.chat import LNPChat # REMOVED: LNPChat is now defined directly below

# --- Moved from src/app/chat.py ---
from src.core.retrieval import Retriever
from src.core.file_finder import StartupSpinner as Spinner

@dataclass
class ChatTurn:
    role: str
    text: str
    hits: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class LNPChat:
    corpus_path: Path
    cache_dir: Path = Path("./index_cache")
    topk: int = 10
    similarity_threshold: float = 0.05 # 유사도 임계값 설정

    retr: Optional[Retriever] = field(init=False, default=None)
    history: List[ChatTurn] = field(init=False, default_factory=list)
    ready_done: bool = field(init=False, default=False)

    # 초기화: Retriever 준비(인덱스 로드 or 빌드)
    def ready(self, rebuild: bool = False):
        spin = Spinner(prefix="엔진 초기화")
        spin.start()
        try:
            self.retr = Retriever(
                corpus_path=self.corpus_path,
                cache_dir=self.cache_dir,
            )
            self.retr.ready(rebuild=rebuild)
            self.ready_done = True
        finally:
            spin.stop()
        print("✅ LNP Chat 준비 완료")

    # 한 턴 처리
    def ask(self, query: str, topk: Optional[int] = None, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.ready_done:
            self.ready(rebuild=False)
        k = topk or self.topk

        spin = Spinner(prefix="검색 중")
        spin.start()
        t0 = time.time()
        try:
            # 1. Retriever는 일단 가능한 많은 후보를 가져옴 (topk * 2, 최소 20개)
            candidate_hits = self.retr.search(query, top_k=max(k * 2, 20), filters=filters)
        finally:
            spin.stop()
        dt = time.time() - t0

        # 2. 유사도 임계값(0.05)을 기준으로 필터링
        filtered_hits = [h for h in candidate_hits if h['similarity'] >= self.similarity_threshold]
        
        # 3. 최종 결과는 필터링된 것에서 topk 만큼만 잘라서 사용
        final_hits = filtered_hits[:k]

        self.history.append(ChatTurn(role="user", text=query))
        self.history.append(ChatTurn(role="assistant", text="", hits=final_hits))

        # 4. 필터링된 결과(final_hits)를 기반으로 답변 생성
        if not final_hits:
            answer_lines = [f"‘{query}’와 관련된 내용을 찾지 못했습니다."]
        else:
            answer_lines = [f"‘{query}’와(과) 의미상 유사한 문서 Top {len(final_hits)} (검색 {dt:.2f}s):"]
            for i, h in enumerate(final_hits, 1):
                sim = f"{h['similarity']:.3f}"
                answer_lines.append(f"{i}. {h['path']}  (유사도: {sim})")
                if h.get("summary"):
                    answer_lines.append(f"   요약: {h['summary']}")

        return {
            "answer": "\n".join(answer_lines),
            "hits": final_hits,
            "suggestions": self._suggest_followups(query, final_hits),
        }

    # 후속 질문 제안
    def _suggest_followups(self, query: str, hits: List[Dict[str, Any]]) -> List[str]:
        base = []
        if hits:
            base.append("이 문서의 핵심 내용을 요약해줘")
            base.append("위 문서들과 비슷한 다른 문서를 더 찾아줘")
            base.append("결과를 표 형식으로 정리해줘")
        else:
            base.append("다른 표현으로 같은 의미의 질의를 시도")
            base.append("문서 유형(엑셀/한글/PDF 등)을 지정해서 검색")
        
        seen, out = set(), []
        for s in base:
            if s not in seen:
                out.append(s); seen.add(s)
        return out[:3]
# --- End Moved from src/app/chat.py ---


# --- Helper Functions (Copied from infopilot_cli.py for self-containment) ---
def get_drives():
    """시스템에 존재하는 드라이브 목록을 반환합니다."""
    drives = []
    for letter in string.ascii_uppercase:
        drive = f"{letter}:\\"
        if os.path.exists(drive):
            drives.append(drive)
    return drives

def parse_query_and_filters(query: str) -> tuple[str, dict]:
    filters = {}
    base_ext_map = {
        ".pdf": ["pdf", "피디에프"],
        ".xlsx": ["엑셀", "excel"],
        ".hwp": ["한글", "hwp"],
        ".docx": ["워드", "word"],
        ".pptx": ["파워포인트", "ppt"],
        ".txt": ["텍스트", "txt"],
        ".csv": ["csv"],
        ".doc": ["doc"],
        ".xls": ["xls"],
        ".xlsm": ["xlsm"],
        ".ppt": ["ppt"],
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
        ext_map[ext] = ext
        for keyword in keywords:
            ext_map[keyword] = ext

    temp_query = query
    explicit_filter_pattern = re.compile(r'(\w+):([^\s]+)')
    explicit_matches = list(explicit_filter_pattern.finditer(temp_query))
    for match in reversed(explicit_matches):
        key = match.group(1).lower()
        value = match.group(2)
        filters[key] = value
        temp_query = temp_query[:match.start()] + " " * len(match.group(0)) + temp_query[match.end():]

    direct_ext_pattern = re.compile(r'\.(\w+)\b', re.IGNORECASE)
    match = direct_ext_pattern.search(temp_query)
    if match:
        matched_ext = "." + match.group(1).lower()
        if matched_ext in ext_map:
            filters['ext'] = ext_map[matched_ext]
            temp_query = temp_query.replace(match.group(0), " " * len(match.group(0)), 1)
    
    if 'ext' not in filters:
        sorted_ext_keywords = sorted([k for k in ext_map.keys() if not k.startswith('.')], key=len, reverse=True)
        ext_keyword_regex_parts = []
        for k in sorted_ext_keywords:
            ext_keyword_regex_parts.append(re.escape(k) + r'(?:\s*(?:파일|문서|자료))?')
        
        if ext_keyword_regex_parts:
            implicit_ext_keyword_pattern = re.compile(r'\b(' + '|'.join(ext_keyword_regex_parts) + r')\b', re.IGNORECASE)
            match = implicit_ext_keyword_pattern.search(temp_query)
            if match:
                matched_text = match.group(1).lower()
                matched_keyword = re.sub(r'\s*(?:파일|문서|자료)$', '', matched_text) 
                if matched_keyword in ext_map:
                    filters['ext'] = ext_map[matched_keyword]
                    temp_query = temp_query.replace(match.group(0), " " * len(match.group(0)), 1)

    implicit_title_patterns = [
        re.compile(r'(제목이|이름이)\s*(\S+)(?:인|인문서|인파일)?', re.IGNORECASE),
        re.compile(r'(\S+)(?:라는|이라는)\s*(제목의|이름의)', re.IGNORECASE),
    ]
    for pattern in implicit_title_patterns:
        match = pattern.search(temp_query)
        if match:
            title_value = match.group(2) if len(match.groups()) >= 2 else match.group(1)
            filters['title'] = title_value
            temp_query = temp_query.replace(match.group(0), " " * len(match.group(0)), 1)
            break

    cleaned_query = re.sub(r'\s+', ' ', temp_query).strip()
    return cleaned_query, filters
# --- End Helper Functions ---

st.set_page_config(page_title="InfoPilot Split Bundle", page_icon="🧭", layout="wide")
p = Paths().ensure()

st.title("InfoPilot (Split Bundle)")

def show_paths():
    st.caption("경로 및 상태")
    st.code(f"""
base:            {p.base}
data_dir:        {p.data_dir}
models_dir:      {p.models_dir}
cache_dir:       {p.cache_dir}
corpus.csv:      {p.corpus_csv.exists()}
corpus.parquet:  {p.corpus_parquet.exists()}
topic_model:     {p.topic_model.exists()}
""", language="bash")

if "mode" not in st.session_state:
    st.session_state["mode"] = "home"

def go(m): st.session_state["mode"] = m

def home():
    st.subheader("시작하기")
    if have_all_artifacts(p):
        st.success("필요 데이터가 모두 준비되었습니다.")
        c1, c2 = st.columns(2)
        if c1.button("🔁 다시 교육하기"):
            go("train")
        if c2.button("💬 채팅창으로 가기"):
            go("chat")
    else:
        st.warning("학습 데이터가 없습니다. 먼저 '교육시키기'를 실행하세요.")
        if st.button("🚀 교육시키기"):
            go("train")
    st.divider()
    show_paths()

def train():
    st.subheader("교육(스캔 → 코퍼스 → 인덱스)")
    exts_text = st.text_input("확장자 필터(쉼표 구분)", ".hwp,.doc,.docx,.xlsx,.xlsm,.xls,.pdf,.ppt,.pptx,.csv,.txt")
    do_scan = st.checkbox("드라이브 스캔 실행", value=True)
    found_csv = p.data_dir / "found_files.csv"

    if st.button("▶️ 교육 시작"):
        rows = None
        if do_scan:
            with st.status("드라이브 스캔 중...", expanded=True) as status:
                EXCLUDE_DIRS = {
                    ".git", ".venv", "venv", "node_modules", "__pycache__", ".idea", ".vscode",
                    "Windows", "Program Files", "Program Files (x86)", "AppData", 
                    "$RECYCLE.BIN", "System Volume Information", "Recovery", "PerfLogs",
                    "Downloads",
                    ".gradle", "plastic4", "ESTsoft", "Bitdefender", "Autodesk", "Intel", "NVIDIA", "Zoom", "Wondershare",
                }
                SUPPORTED_EXTS = {e.strip() for e in exts_text.split(",") if e.strip()}
                
                file_list = []
                drives = get_drives()
                status.update(label=f"Scanning drives: {', '.join(drives)}")

                for drive in drives:
                    for root, dirs, files in os.walk(drive, topdown=True):
                        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
                        for file in files:
                            try:
                                p_file = Path(root) / file
                                if p_file.suffix.lower() in SUPPORTED_EXTS:
                                    if not any(part in EXCLUDE_DIRS for part in p_file.parts):
                                        file_list.append({'path': str(p_file), 'size': p_file.stat().st_size})
                            except (FileNotFoundError, PermissionError):
                                continue
                rows = file_list
                df_scan = pd.DataFrame(rows)
                df_scan.to_csv(found_csv, index=False, encoding="utf-8")
                status.update(label=f"스캔 완료. {len(rows)}개 파일 발견.", state="complete")

        df_corpus = pd.DataFrame() 
        with st.status("텍스트 추출 및 코퍼스 생성...", expanded=True) as status:
            if (p.corpus_parquet.exists()):
                try: p.corpus_parquet.unlink()
                except: pass
            
            cb = CorpusBuilder(progress=True) 
            
            if rows is None and found_csv.exists():
                df_scan = pd.read_csv(found_csv)
                rows = df_scan.to_dict("records")
            
            if rows is not None:
                df_corpus = cb.build(rows)
                cb.save(df_corpus, p.corpus_parquet)
            else:
                st.error("스캔된 파일이 없어 코퍼스를 생성할 수 없습니다.")
            status.update(label="코퍼스 생성 완료", state="complete")

        with st.status("벡터 인덱싱 중...", expanded=True) as status:
            if p.corpus_parquet.exists():
                run_indexing(corpus_path=p.corpus_parquet, cache_dir=p.cache_dir)
                status.update(label="인덱싱 완료", state="complete")
            else:
                st.warning("코퍼스 파일이 없어 인덱싱을 건너뜁니다.")
                status.update(label="인덱싱 건너뜀", state="skipped")

        with st.status("토픽 모델 메타 저장...", expanded=True) as status:
            meta = {
                "model_name": "jhgan/ko-sroberta-multitask",
                "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "corpus_path": str(p.corpus_parquet),
                "cache_dir": str(p.cache_dir),
            }
            joblib.dump(meta, p.topic_model)
            status.update(label="완료", state="complete")

        st.success("학습이 완료되었습니다. 이제 채팅을 시작할 수 있어요!")
        st.button("💬 채팅창으로 가기", on_click=lambda: go("chat"))

    if found_csv.exists():
        st.write("최근 스캔 결과(일부):")
        try:
            st.dataframe(pd.read_csv(found_csv).head(200))
        except Exception as e:
            st.info(f"스캔 결과 미리보기 실패: {e}")

    st.button("🏠 처음으로", on_click=lambda: go("home"))
    st.divider()
    show_paths()

def chat():
    st.subheader("의미 기반 검색 채팅")
    if not have_all_artifacts(p):
        st.warning("먼저 교육을 실행해주세요.")
        if st.button("🚀 교육시키기"):
            go("train")
        return

    if "chat_engine" not in st.session_state:
        st.session_state["chat_engine"] = LNPChat(corpus_path=p.corpus_parquet, cache_dir=p.cache_dir, topk=10)
        st.session_state["chat_engine"].ready()

    query = st.text_input("질문을 입력하세요", placeholder="예: 세금 관련 파일 찾아줘")
    if st.button("검색") and query.strip():
        with st.status("검색 중...", expanded=False):
            # Parse query and filters
            cleaned_query, filters = parse_query_and_filters(query.strip())
            result = st.session_state["chat_engine"].ask(cleaned_query, filters=filters)
        st.write(result.get("answer", ""))
        hits = result.get("hits", [])
        if hits:
            df = pd.DataFrame([
                {"유사도": h.get("similarity"), "경로": h.get("path"), "요약": h.get("summary", "")}
                for h in hits
            ])
            st.dataframe(df)
        if result.get("suggestions"):
            st.caption("💡 제안")
            for s in result["suggestions"]:
                st.write("- ", s)

    st.button("🏠 처음으로", on_click=lambda: go("home"))
    st.divider()
    show_paths()

mode = st.session_state["mode"]
if mode == "home":
    home()
elif mode == "train":
    train()
elif mode == "chat":
    chat()
else:
    home()
