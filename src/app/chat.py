from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import time
import traceback # traceback 모듈 임포트
import sys # sys 모듈 임포트

import google.generativeai as genai # Gemini API 임포트
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM # Hugging Face pipeline, AutoTokenizer, AutoModelForCausalLM 임포트
from src.config import DEFAULT_TOP_K, DEFAULT_SIMILARITY_THRESHOLD, GEMINI_API_KEY, USE_LOCAL_LLM, LOCAL_LLM_MODEL_NAME, LOCAL_LLM_MAX_NEW_TOKENS # GEMINI_API_KEY 및 로컬 LLM 설정 임포트
from src.core.retrieval import Retriever
from src.core.utils import StartupSpinner as Spinner # 경로 수정

@dataclass
class ChatTurn:
    role: str
    text: str
    hits: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class LNPChat:
    corpus_path: Path
    cache_dir: Path
    topk: int = DEFAULT_TOP_K
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    translation_cache: Optional[Dict[str, str]] = None # 번역 캐시 주입 인터페이스

    retr: Optional[Retriever] = field(init=False, default=None)
    history: List[ChatTurn] = field(init=False, default_factory=list)
    ready_done: bool = field(init=False, default=False)
    gemini_model: Any = field(init=False, default=None) # Gemini 모델 필드 추가
    local_llm_pipeline: Any = field(init=False, default=None) # 로컬 LLM 파이프라인 필드 추가
    local_llm_max_input_length: int = field(init=False, default=0) # 로컬 LLM 최대 입력 길이 필드 추가
    local_llm_tokenizer: Any = field(init=False, default=None) # 로컬 LLM 토크나이저 필드 추가

    def __post_init__(self):
        # Gemini 모델 초기화
        if not USE_LOCAL_LLM:
            if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY":
                try:
                    genai.configure(api_key=GEMINI_API_KEY)
                    self.gemini_model = genai.GenerativeModel('gemini-pro')
                    print("✅ Gemini model initialized successfully.")
                except Exception as e:
                    print(f"❌ Failed to initialize Gemini model: {e}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                    sys.stderr.flush()
            else:
                print("⚠️ GEMINI_API_KEY is not set or is default. Conversational features will be limited.", file=sys.stderr)
        else: # 로컬 LLM 사용 시
            try:
                print(f"🧠 Loading local LLM: {LOCAL_LLM_MODEL_NAME}...")
                # Explicitly load tokenizer and model to get max_position_embeddings
                self.local_llm_tokenizer = AutoTokenizer.from_pretrained(LOCAL_LLM_MODEL_NAME)
                model = AutoModelForCausalLM.from_pretrained(LOCAL_LLM_MODEL_NAME)

                self.local_llm_max_input_length = model.config.max_position_embeddings
                self.local_llm_pipeline = pipeline(
                    "text-generation", 
                    model=model,
                    tokenizer=self.local_llm_tokenizer,
                    # max_length와 truncation은 pipeline 호출 시점에 전달하여 동적으로 제어
                    # device=0 if torch.cuda.is_available() else -1 # GPU 사용 시 주석 해제 및 torch 임포트 필요
                )
                # Set pad_token_id if not already set (common for GPT-like models)
                if self.local_llm_tokenizer.pad_token is None:
                    self.local_llm_tokenizer.pad_token = self.local_llm_tokenizer.eos_token

                print(f"✅ Local LLM initialized successfully. Max input length: {self.local_llm_max_input_length}")
            except ImportError:
                print("❌ 'transformers' library not found. Please install it: pip install transformers", file=sys.stderr)
            except Exception as e:
                print(f"❌ Failed to initialize local LLM: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()

    def ready(self, rebuild: bool = False):
        spin = Spinner(prefix="엔진 초기화")
        spin.start()
        try:
            self.retr = Retriever(corpus_path=self.corpus_path, cache_dir=self.cache_dir)
            self.retr.ready(rebuild=rebuild)
            self.ready_done = True
        finally:
            spin.stop()
        print("✅ LNP Chat 준비 완료")

    def _is_conversational_query(self, query: str) -> bool:
        """
        Determines if a query is conversational rather than a search query.
        This is a simple heuristic and can be improved with NLU models.
        """
        query_lower = query.lower().strip()
        conversational_keywords = [
            "안녕", "반가워", "안녕하세요", "잘 지내", "뭐해", "고마워", "감사합니다",
            "누구세요", "무엇을 할 수 있나요", "도와줘", "알겠습니다", "네", "아니오",
            "고마워요", "천만에요", "수고했어요", "수고하셨습니다", "좋아요", "싫어요",
            "응", "아니", "그래", "아니야", "맞아", "틀려", "왜", "어떻게", "언제", "어디서", "누가", "무엇을"
        ]
        
        # Check if the query is a greeting or a very general conversational phrase
        if any(keyword in query_lower for keyword in conversational_keywords):
            return True
        
        # Check if the query is very short and not clearly a search term
        if len(query_lower.split()) <= 3 and not any(word in query_lower for word in ["문서", "파일", "검색", "찾아", "요약", "표", "정리"]):
            return True
            
        return False

    def _get_llm_response(self, query: str, history_context: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Integrates with Google Gemini API or a local Hugging Face LLM for conversational responses.
        """
        if USE_LOCAL_LLM:
            if not self.local_llm_pipeline or not self.local_llm_tokenizer:
                return "로컬 LLM이 초기화되지 않았습니다. 모델 로딩을 확인해주세요."
            try:
                # Convert history_context to a more structured prompt for local LLM
                prompt_parts = []
                
                # System message to guide the LLM
                prompt_parts.append("당신은 InfoPilot이라는 이름의 친절하고 유능한 한국어 챗봇입니다. 사용자에게 자연스럽고 공손하게 응답해주세요. 질문에 대한 답변은 간결하게 해주세요.")
                prompt_parts.append("\n--- 대화 시작 ---") # 대화 예시 시작을 명확히
                
                # Add few-shot examples
                prompt_parts.append("사용자: 안녕 반가워")
                prompt_parts.append("InfoPilot: 안녕하세요! 만나서 반갑습니다. 무엇을 도와드릴까요?")
                prompt_parts.append("사용자: 오늘 날씨는 어때?")
                prompt_parts.append("InfoPilot: 저는 날씨 정보를 알 수 없지만, 오늘 하루 즐겁게 보내시길 바랍니다!")
                
                # Add history context
                if history_context:
                    # Only include the last few turns to keep the prompt short for small models
                    for turn in history_context[-2:]: # Consider only last 2 turns for brevity
                        role_prefix = "사용자: " if turn.get("role") == "user" else "InfoPilot: "
                        prompt_parts.append(f"{role_prefix}{turn.get('text', '')}")
                
                # Add current query
                prompt_parts.append(f"사용자: {query}")
                prompt_parts.append("InfoPilot: ") # LLM이 답변을 시작하도록 유도
                
                full_prompt = "\n".join(prompt_parts)

                # Explicitly tokenize the prompt and prepare for generation
                inputs = self.local_llm_tokenizer(
                    full_prompt,
                    return_tensors="pt",
                    max_length=self.local_llm_max_input_length, # 모델의 최대 입력 길이 사용
                    truncation=True,
                    padding=True # 패딩 다시 추가
                )

                # Move inputs to the same device as the model (if using GPU, requires torch import)
                # if torch.cuda.is_available():
                #     inputs = {k: v.to(self.local_llm_pipeline.device) for k, v in inputs.items()}
                
                # Generate response using the underlying model's generate method
                output_sequences = self.local_llm_pipeline.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=LOCAL_LLM_MAX_NEW_TOKENS,
                    num_return_sequences=1,
                    pad_token_id=self.local_llm_tokenizer.pad_token_id, # Explicitly set pad_token_id
                    eos_token_id=self.local_llm_tokenizer.eos_token_id,
                    max_length=self.local_llm_max_input_length # 생성될 전체 시퀀스 길이 제한
                )

                # Decode the generated text
                generated_text = self.local_llm_tokenizer.decode(
                    output_sequences[0], skip_special_tokens=True
                )

                # Extract only the newly generated part
                # Decode the input_ids to get the exact input text that was fed to the model
                input_text_decoded = self.local_llm_tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
                
                answer = generated_text[len(input_text_decoded):].strip()

                # Clean up any remaining prompt parts if the model repeats them
                if answer.startswith("InfoPilot:"):
                    answer = answer[len("InfoPilot:"):].strip()

                # Remove any trailing "사용자:" or "InfoPilot:" if the model generates incomplete turns
                if "사용자:" in answer:
                    answer = answer.split("사용자:")[0].strip()
                if "InfoPilot:" in answer:
                    answer = answer.split("InfoPilot:")[0].strip()

                return answer
            except Exception as e:
                print(f"❌ Error calling local LLM: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
                return "로컬 LLM 호출 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
        else: # Gemini 사용
            if not self.gemini_model:
                return "Gemini 모델이 초기화되지 않았습니다. API 키를 확인해주세요."

            try:
                # Convert history_context to Gemini's format
                gemini_history = []
                if history_context:
                    for turn in history_context:
                        role = "user" if turn.get("role") == "user" else "model" # Gemini expects 'model' for assistant
                        gemini_history.append({"role": role, "parts": [turn.get("text", "")]})

                chat = self.gemini_model.start_chat(history=gemini_history)
                response = chat.send_message(query)
                return response.text
            except Exception as e:
                print(f"❌ Error calling Gemini API: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
                return "Gemini API 호출 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."

    def ask(self, query: str, topk: Optional[int] = None, filters: Optional[Dict[str, Any]] = None, use_bm25: bool = False, use_reranker: bool = False, history_context: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]: # history_context 타입 힌트 수정
        print(f"DEBUG: LNPChat.ask received query: '{query}'") # 디버깅 로그 추가
        if not self.ready_done: self.ready(rebuild=False)
        
        # 빈 쿼리 처리
        if not query.strip() and not history_context: # history_context가 없으면 빈 쿼리 처리
            return {"answer": "질의 내용이 비어있습니다.", "hits": [], "suggestions": self._suggest_followups(query, [])}

        # 1. 의도 분류: 대화형 쿼리인지 확인
        if self._is_conversational_query(query):
            llm_answer = self._get_llm_response(query, history_context)
            return {"answer": llm_answer, "hits": [], "suggestions": []} # 대화형 응답은 hits와 suggestions 없음

        k = topk or self.topk
        spin = Spinner(prefix="검색 중")
        spin.start()
        t0 = time.time()

        # history_context를 활용하여 쿼리 보강
        enriched_query = query
        if history_context:
            context_texts = [turn["text"] for turn in history_context if "text" in turn]
            if context_texts:
                enriched_query = " ".join(context_texts) + " " + query
            print(f"DEBUG: Enriched query with history: '{enriched_query}'")

        try:
            # Retriever.search에 enriched_query 전달
            candidate_hits = self.retr.search(enriched_query, top_k=max(k * 2, 20), filters=filters, use_bm25=use_bm25, use_reranker=use_reranker, history_context=history_context)
        except Exception as e:
            print(f"❌ LNPChat.ask: Error during self.retr.search: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush() # 오류 즉시 출력
            raise # 오류를 다시 발생시켜 FastAPI에서 500 에러를 반환하도록 함
        finally:
            spin.stop()
        dt = time.time() - t0
        filtered_hits = [h for h in candidate_hits if h['similarity'] >= self.similarity_threshold]
        final_hits = filtered_hits[:k]
        
        # LNPChat 내부 history는 FastAPI의 chat_endpoint에서 관리하도록 변경
        # self.history.append(ChatTurn(role="user", text=query))
        # self.history.append(ChatTurn(role="assistant", text="", hits=final_hits))
        
        if not final_hits:
            answer_lines = [f"‘{query}’와 관련된 내용을 찾지 못했습니다."]
        else:
            answer_lines = [f"‘{query}’와(과) 의미상 유사한 문서 Top {len(final_hits)} (검색 {dt:.2f}s):"]
            for i, h in enumerate(final_hits, 1):
                sim = f"{h['similarity']:.3f}"
                answer_lines.append(f"{i}. {h['path']}  (유사도: {sim})")
                if h.get("summary"):
                    answer_lines.append(f"   요약: {h['summary']}")
        return {"answer": "\n".join(answer_lines), "hits": final_hits, "suggestions": self._suggest_followups(query, final_hits)}

    def summarize_document(self, target_info: Dict[str, Any]) -> str:
        """
        Summarizes the content of a document given its path.
        target_info is expected to contain 'path'.
        """
        doc_path = target_info.get('path')
        if not doc_path:
            return "요약할 문서 경로를 찾을 수 없습니다."

        if not self.retr: # Retriever가 초기화되지 않았다면
            self.ready(rebuild=False)
            if not self.retr:
                return "문서 요약 엔진이 준비되지 않았습니다."

        full_content = self.retr.get_document_content(doc_path)
        if not full_content:
            return f"'{doc_path}' 경로의 문서 내용을 찾을 수 없습니다."

        # 간단한 요직: 처음 300자 또는 첫 3문장
        summary_length = 300
        if len(full_content) > summary_length:
            # 첫 300자 이후 가장 가까운 마침표를 찾아 자르기
            end_index = full_content.find('.', summary_length)
            if end_index == -1: # 마침표가 없으면 그냥 자르기
                end_index = summary_length
            summary = full_content[:end_index+1].strip()
            if len(full_content) > end_index + 1: # 원본 텍스트가 더 길면 ... 추가
                summary += "..."
        else:
            summary = full_content.strip()

        return f"'{doc_path}' 문서 요약:\n\n{summary}"

    def find_similar_documents(self, target_info: Dict[str, Any], topk: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Finds similar documents based on a given document path or query text.
        target_info is expected to contain 'path' or 'query_text'.
        """
        base_query_text = target_info.get('query_text')
        doc_path = target_info.get('path')
        
        search_query = ""
        if doc_path:
            if not self.retr:
                self.ready(rebuild=False)
                if not self.retr:
                    return [] # Retriever not ready
            doc_content = self.retr.get_document_content(doc_path)
            if doc_content:
                search_query = doc_content
            else:
                return [] # Document content not found
        elif base_query_text:
            search_query = base_query_text
        else:
            return [] # No valid base for search

        if not search_query.strip():
            return [] # Empty search query

        k = topk or self.topk
        
        # Perform search using the determined query
        similar_hits = self.retr.search(
            search_query,
            top_k=k + 1, # Retrieve one more to potentially exclude the original document
            use_bm25=True, # 유사 문서 검색 시 BM25와 재랭킹을 활용하는 것이 일반적
            use_reranker=True
        )

        final_similar_hits = []
        if doc_path: # If search was based on a document, exclude it from results
            for hit in similar_hits:
                if hit.get('path') != doc_path:
                    final_similar_hits.append(hit)
        else:
            final_similar_hits = similar_hits

        return final_similar_hits[:k]

    def format_hits_as_table(self, hits: List[Dict[str, Any]]) -> str:
        """
        Formats a list of hits into a simple table string.
        """
        if not hits:
            return "표 형식으로 정리할 검색 결과가 없습니다."

        table_lines = ["| 순번 | 파일 경로 | 유사도 | 요약 |", "|---|---|---|---|"]
        for i, hit in enumerate(hits, 1):
            path = hit.get('path', 'N/A')
            similarity = f"{hit.get('similarity', 0.0):.3f}"
            summary = hit.get('summary', '요약 없음')
            table_lines.append(f"| {i} | {path} | {similarity} | {summary} |")
        return "\n".join(table_lines)

    def _suggest_followups(self, query: str, hits: List[Dict[str, Any]]) -> List[str]:
        base = ["이 문서의 핵심 내용을 요약해줘", "위 문서들과 비슷한 다른 문서를 더 찾아줘", "결과를 표 형식으로 정리해줘"] if hits else ["다른 표현으로 같은 의미의 질의를 시도", "문서 유형(엑셀/한글/PDF 등)을 지정해서 검색"]
        seen, out = set(), []
        for s in base:
            if s not in seen: out.append(s); seen.add(s)
        return out[:3]
