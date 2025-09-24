from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import sys
from pathlib import Path
import traceback # traceback 모듈 임포트

# Add project root to sys.path to allow imports from src
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.cli.infopilot_cli import cmd_train, cmd_update # cmd_scan은 이제 helpers의 perform_scan_to_csv로 대체
from src.config import FOUND_FILES_CSV, CORPUS_PARQUET, CACHE_DIR, DEFAULT_TOP_K, FASTAPI_HOST, FASTAPI_PORT # Import default paths and FastAPI config
from src.app.chat import LNPChat # Import LNPChat
from src.core.query_parser import parse_query_and_filters # Import query parser
from src.core.helpers import perform_scan_to_csv # perform_scan_to_csv 임포트

app = FastAPI()

# Initialize LNPChat globally for now. 
# For production, consider using FastAPI's dependency injection or lifespan events.
chat_session: Optional[LNPChat] = None # chat_session을 Optional로 선언
try:
    chat_session = LNPChat(corpus_path=CORPUS_PARQUET, cache_dir=CACHE_DIR, topk=DEFAULT_TOP_K)
    chat_session.ready(rebuild=False)
    print("✅ FastAPI: LNPChat session initialized successfully.")
except Exception as e:
    print(f"❌ FastAPI: Failed to initialize LNPChat session: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.stderr.flush() # 오류 즉시 출력
    chat_session = None # 초기화 실패 시 None으로 설정

# Pydantic model for scan request body
class ScanRequest(BaseModel):
    out: Optional[str] = str(FOUND_FILES_CSV)
    exts_text: str # exts_text 필드 추가

# Pydantic model for train request body
class TrainRequest(BaseModel):
    scan_csv: Optional[str] = str(FOUND_FILES_CSV)
    corpus: Optional[str] = str(CORPUS_PARQUET)
    cache: Optional[str] = str(CACHE_DIR)

# Pydantic model for update request body
class UpdateRequest(BaseModel):
    corpus: Optional[str] = str(CORPUS_PARQUET)
    cache: Optional[str] = str(CACHE_DIR)

# Pydantic model for chat request body
class ChatRequest(BaseModel):
    query: str
    topk: Optional[int] = DEFAULT_TOP_K
    filters: Optional[Dict[str, Any]] = None
    use_bm25: Optional[bool] = False
    use_reranker: Optional[bool] = False
    action: Optional[str] = None # 새로운 action 필드 추가
    target_info: Optional[Dict[str, Any]] = None # 새로운 target_info 필드 추가
    history: Optional[List[Dict[str, Any]]] = None # 대화 기록 필드 추가

@app.get("/")
async def read_root():
    return {"message": "InfoPilot FastAPI backend is running!"}

# 새로운 /status 엔드포인트 추가
@app.get("/status")
async def get_status():
    return {"status": "success", "chat_engine_ready": chat_session is not None and chat_session.ready_done}

@app.post("/scan")
async def scan_endpoint(request: ScanRequest):
    try:
        # cmd_scan 대신 perform_scan_to_csv 호출
        file_count = perform_scan_to_csv(Path(request.out), request.exts_text)
        return {"status": "success", "message": f"Scan initiated. {file_count} files found. Results saved to {request.out}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/train")
async def train_endpoint(request: TrainRequest):
    # Create a mock args object for cmd_train
    class MockArgs:
        def __init__(self, scan_csv, corpus, cache):
            self.scan_csv = scan_csv
            self.corpus = corpus
            self.cache = cache
    
    mock_args = MockArgs(request.scan_csv, request.corpus, request.cache)
    
    try:
        cmd_train(mock_args)
        return {"status": "success", "message": "Training initiated."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/update")
async def update_endpoint(request: UpdateRequest):
    # Create a mock args object for cmd_update
    class MockArgs:
        def __init__(self, corpus, cache):
            self.corpus = corpus
            self.cache = cache
    
    mock_args = MockArgs(request.corpus, request.cache)
    
    try:
        cmd_update(mock_args)
        return {"status": "success", "message": "Update initiated."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    print(f"DEBUG: chat_endpoint received request.query: '{request.query}', action: {request.action}") # 디버깅 로그 추가
    if chat_session is None or not chat_session.ready_done:
        return {"status": "error", "message": "Chat engine is not initialized or not ready. Please check backend logs.", "suggestions": []}
    try:
        # 액션이 지정된 경우 해당 액션 처리
        if request.action == "summarize":
            if not request.target_info:
                return {"status": "error", "message": "요약할 문서 정보(target_info)가 필요합니다.", "suggestions": []}
            summary_text = chat_session.summarize_document(request.target_info)
            return {"status": "success", "answer": summary_text, "hits": [], "suggestions": []}
        elif request.action == "find_similar":
            if not request.target_info:
                return {"status": "error", "message": "유사 문서를 찾을 기준 정보(target_info)가 필요합니다.", "suggestions": []}
            similar_hits = chat_session.find_similar_documents(request.target_info, request.topk)
            answer_lines = [f"'{request.target_info.get('path', request.target_info.get('query_text', '문서'))}'와(과) 유사한 문서 Top {len(similar_hits)}:"]
            for i, h in enumerate(similar_hits, 1):
                sim = f"{h['similarity']:.3f}"
                answer_lines.append(f"{i}. {h['path']}  (유사도: {sim})")
                if h.get("summary"):
                    answer_lines.append(f"   요약: {h['summary']}")
            return {"status": "success", "answer": "\n".join(answer_lines), "hits": similar_hits, "suggestions": []}
        elif request.action == "format_table":
            # 현재 세션의 마지막 검색 결과를 표 형식으로 변환
            last_hits = []
            if chat_session.history:
                # 마지막 어시스턴트 턴에서 hits를 가져옴
                for turn in reversed(chat_session.history):
                    if turn.role == "assistant" and turn.hits:
                        last_hits = turn.hits
                        break
            
            if not last_hits:
                return {"status": "error", "message": "표 형식으로 정리할 이전 검색 결과가 없습니다.", "suggestions": []}

            table_text = chat_session.format_hits_as_table(last_hits)
            return {"status": "success", "answer": table_text, "hits": last_hits, "suggestions": []}

        # 일반 질의 처리
        cleaned_query, parsed_filters = parse_query_and_filters(request.query)
        print(f"DEBUG: chat_endpoint after parse_query_and_filters: cleaned_query='{cleaned_query}', parsed_filters={parsed_filters}") # 디버깅 로그 추가
        
        # Merge filters from request body with parsed filters
        final_filters = parsed_filters
        if request.filters:
            final_filters.update(request.filters)
        print(f"DEBUG: chat_endpoint final_filters: {final_filters}") # 디버깅 로그 추가

        result = chat_session.ask(
            cleaned_query,
            topk=request.topk,
            filters=final_filters,
            use_bm25=request.use_bm25,
            use_reranker=request.use_reranker,
            history_context=request.history # 대화 기록 전달
        )
        print(f"DEBUG: chat_endpoint chat_session.ask returned: {result}") # 디버깅 로그 추가
        return {"status": "success", "answer": result["answer"], "hits": result["hits"], "suggestions": result["suggestions"]}
    except Exception as e:
        # API 엔드포인트 내에서도 오류 로깅 추가
        print(f"❌ FastAPI: Error in chat_endpoint: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush() # 오류 즉시 출력
        return {"status": "error", "message": str(e), "suggestions": []}
