# -*- coding: utf-8 -*-
"""
InfoPilot 프로젝트의 모든 공통 설정 변수를 관리하는 파일입니다.
"""
from pathlib import Path

# ===================================
# 1. 모델 및 학습 관련 설정
# ===================================

# 의미 검색 및 요약에 사용할 SentenceTransformer 모델 이름
MODEL_NAME = 'sentence-transformers/LaBSE'

# AI 요약 시 사용할 주제 후보 목록
TOPIC_CANDIDATES = [
    "파이썬 프로그래밍", "자바 프로그래밍", "SQL 데이터베이스", "웹 개발", "자바스크립트",
    "머신러닝 및 인공지능", "데이터 분석 및 시각화", "클라우드 컴퓨팅", "보안",
    "업무 보고서", "회의록", "기획서", "제안서", "계약서", "법률 문서", "인사 관리", "재무 및 회계",
    "강의 교안", "연구 논문", "기술 문서", "사용자 매뉴얼",
]

# 텍스트 추출 시 최대 글자 수 제한
DEFAULT_MAX_TEXT_CHARS = 500_000

# 코퍼스 생성 시 요약 생성의 배치 크기
DEFAULT_BATCH_SIZE = 32


# ===================================
# 2. 검색 및 필터링 관련 설정
# ===================================

# 검색 결과로 보여줄 기본 문서 개수 (top-k)
DEFAULT_TOP_K = 10

# 검색 결과 필터링 시 사용할 최소 유사도 임계값
DEFAULT_SIMILARITY_THRESHOLD = 0.05

# 사용자 자연어 질의의 확장자 키워드를 실제 확장자로 변환하는 규칙
BASE_EXT_MAP = {
    ".pdf": ["pdf", "피디에프"],
    ".xlsx": ["엑셀", "excel"],
    ".hwp": ["한글", "hwp"],
    ".docx": ["워드", "word"],
    ".pptx": ["파워포인트", "ppt"], # .ppt는 .pptx 또는 .ppt에 매핑될 수 있으므로 .pptx를 우선시
    ".txt": ["텍스트", "txt"],
    ".csv": ["csv"],
    ".doc": ["doc"],
    ".xls": ["xls"],
    ".xlsm": ["xlsm"],
    ".ppt": ["ppt"], # 구형 .ppt 형식
    ".py": ["py", "파이썬"],
    ".json": ["json"],
    ".xml": ["xml"],
    ".html": ["html"],
    ".css": ["css"],
    ".js": ["js", "자바스크립트"],
    ".md": ["md", "마크다운"],
}


# ===================================
# 3. 파일 시스템 스캔 관련 설정
# ===================================

# 파일 시스템 스캔 시 제외할 디렉토리 목록
EXCLUDE_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".idea", ".vscode",
    "Windows", "Program Files", "Program Files (x86)", "AppData", 
    "$RECYCLE.BIN", "System Volume Information", "Recovery", "PerfLogs",
    # "Downloads", # 사용자의 요청으로 검색 대상에 포함
    ".gradle", "plastic4", "ESTsoft", "Bitdefender", "Autodesk", "Intel", "NVIDIA", "Zoom", "Wondershare",
}

# CLI 스캔 시 기본으로 지원할 파일 확장자 목록
SUPPORTED_EXTS = {
    '.txt', '.csv', '.md', '.py', '.json', '.xml', '.html', '.css', '.js', 
    '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', 
    '.pdf', 
    '.hwp'
}


# ===================================
# 4. 기본 경로 설정
# ===================================

# 프로젝트 루트 디렉토리
# 이 파일(config.py)의 위치를 기준으로 상위 폴더(src)의 상위 폴더(InfoPilot)를 루트로 설정
PROJECT_ROOT = Path(__file__).parent.parent

# 데이터, 모델, 캐시 등을 저장할 기본 디렉토리
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = PROJECT_ROOT / "index_cache"

# 주요 파일 경로
FOUND_FILES_CSV = DATA_DIR / "found_files.csv"
CORPUS_PARQUET = DATA_DIR / "corpus.parquet"
TOPIC_MODEL_PATH = MODELS_DIR / "topic_model.joblib"
