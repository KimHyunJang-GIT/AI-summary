import customtkinter as ctk
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional
import time
import threading
import requests # requests 라이브러리 임포트

# Core logic and helpers
from src.core.helpers import parse_query_and_filters, have_all_artifacts
from src.config import CORPUS_PARQUET, CACHE_DIR, DEFAULT_TOP_K, DEFAULT_SIMILARITY_THRESHOLD, FASTAPI_URL # FASTAPI_URL 임포트

# LNPChat 클래스는 이제 백엔드에서 관리하므로 제거합니다.

class ChatScreen(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # --- Chat UI Elements ---
        self.chat_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.chat_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.chat_frame.grid_columnconfigure(0, weight=1)
        self.chat_frame.grid_rowconfigure(0, weight=1) # For results_textbox
        self.chat_frame.grid_rowconfigure(1, weight=0) # For suggestions_frame

        self.results_textbox = ctk.CTkTextbox(self.chat_frame, font=ctk.CTkFont(size=14), state="disabled")
        self.results_textbox.grid(row=0, column=0, sticky="nsew")

        self.suggestions_frame = ctk.CTkFrame(self.chat_frame, fg_color="transparent")
        self.suggestions_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        self.suggestions_frame.grid_columnconfigure(0, weight=1) # To center buttons or align them

        self.last_hits: List[Dict[str, Any]] = [] # Initialize last_hits
        self.chat_history: List[Dict[str, Any]] = [] # 대화 기록 저장용 리스트 추가

        # --- Input and Options Frame ---
        self.input_options_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.input_options_frame.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="ew")
        self.input_options_frame.grid_columnconfigure(0, weight=1)

        self.search_entry = ctk.CTkEntry(self.input_options_frame, placeholder_text="질문을 입력하세요...", height=40)
        self.search_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.search_entry.bind("<Return>", self.search_event)

        self.search_button = ctk.CTkButton(self.input_options_frame, text="검색", width=100, height=40, command=self.search_event)
        self.search_button.grid(row=0, column=1)

        # --- Search Options ---
        self.options_frame = ctk.CTkFrame(self.input_options_frame, fg_color="transparent")
        self.options_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        self.options_frame.grid_columnconfigure((0,1,2,3), weight=1)

        self.bm25_checkbox = ctk.CTkCheckBox(self.options_frame, text="BM25 사용")
        self.bm25_checkbox.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.reranker_checkbox = ctk.CTkCheckBox(self.options_frame, text="재랭킹 사용")
        self.reranker_checkbox.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        # Initialize UI elements that will be managed by refresh_state
        self.warning_label = ctk.CTkLabel(self, text="", font=ctk.CTkFont(size=16))
        self.train_button_redirect = ctk.CTkButton(self, text="🚀 전체 학습시키기",
                                                   command=lambda: master.select_frame("train"))

        self.refresh_state()  # Call refresh_state initially

    def log_message(self, message: str):
        """결과 텍스트 박스에 메시지를 로깅합니다."""
        # log_message는 이제 chat_history에 추가되지 않고, 단순히 UI에 메시지를 표시하는 용도로 사용
        self.update_results(message, [], []) 

    def refresh_state(self):
        # Clear previous state by forgetting grid layout
        self.warning_label.grid_forget()
        self.train_button_redirect.grid_forget()
        self.chat_frame.grid_forget()
        self.input_options_frame.grid_forget()

        if not have_all_artifacts():
            self.grid_rowconfigure(0, weight=1)
            self.warning_label.configure(text="⚠️ 학습된 데이터가 없습니다. 먼저 '학습시키기'를 실행하세요.")
            self.warning_label.grid(row=0, column=0, pady=(20, 10))
            self.train_button_redirect.grid(row=1, column=0, pady=10)
            self._set_search_controls_state("disabled")
        else:
            # Re-create/show chat and input frames
            self.grid_rowconfigure(0, weight=1)
            self.grid_rowconfigure(1, weight=0) # Input frame should not expand vertically
            self.chat_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
            self.input_options_frame.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="ew")

            # Check if chat engine is ready via API
            self.chat_history.clear() # 새로운 세션 시작 시 대화 기록 초기화
            self.update_results("엔진 초기화 상태 확인 중...", [], [])
            self._set_search_controls_state("disabled")
            threading.Thread(target=self._check_engine_ready_status, daemon=True).start()

    def _set_search_controls_state(self, state):
        self.search_entry.configure(state=state)
        self.search_button.configure(state=state)
        self.bm25_checkbox.configure(state=state)
        self.reranker_checkbox.configure(state=state)

    def _check_engine_ready_status(self):
        try:
            status_url = f"{FASTAPI_URL}/status"
            self.log_message(f"DEBUG: Checking engine status at {status_url}") # Added debug log
            response = requests.get(status_url, timeout=5)
            response.raise_for_status()
            data = response.json()

            if data.get("chat_engine_ready"): # chat_engine_ready 필드 확인
                self.update_results("엔진 초기화 완료. 질문을 입력하세요.", [], [])
                self._set_search_controls_state("normal")
            else:
                # 백엔드 엔진이 준비되지 않은 경우
                self.update_results(f"엔진 초기화 실패: 백엔드 엔진이 준비되지 않았습니다. 백엔드 로그를 확인해주세요.", [], [])
                self._set_search_controls_state("disabled")

        except requests.exceptions.ConnectionError as e:
            self.log_message(f"FATAL: 엔진 초기화 실패: FastAPI 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요: {FASTAPI_URL}. 오류: {e} (타입: {type(e)}) ") # 오류 타입 추가
            self._set_search_controls_state("disabled")
        except requests.exceptions.RequestException as e:
            self.log_message(f"FATAL: 엔진 초기화 실패: API 요청 중 오류 발생: {e} (타입: {type(e)}). 다시 시도해주세요.") # 오류 타입 추가
            self._set_search_controls_state("disabled")
        except Exception as e:
            self.log_message(f"FATAL: 엔진 초기화 실패: 알 수 없는 오류 발생: {e} (타입: {type(e)}).") # 오류 타입 추가
            self._set_search_controls_state("disabled")

    def on_show(self):
        # Called when the frame is brought to front
        self.refresh_state()

    def search_event(self, event=None):
        query = self.search_entry.get().strip()
        if not query or self.search_button.cget("state") == "disabled":
            return

        self.search_entry.configure(state="disabled")
        self.search_button.configure(state="disabled")
        self.bm25_checkbox.configure(state="disabled")
        self.reranker_checkbox.configure(state="disabled")
        
        # 사용자 쿼리를 대화 기록에 추가
        self.chat_history.append({"role": "user", "text": query})
        self.update_results(f"> {query}\n\n검색 중입니다...", [], []) # Initial call with empty hits/suggestions

        # Run search in a thread to keep the UI responsive
        # 대화 기록을 run_search_thread로 전달
        threading.Thread(target=self.run_search_thread, args=(query, None, None, self.chat_history), daemon=True).start() 

    def run_search_thread(self, query: str, action: Optional[str] = None, target_info: Optional[Dict[str, Any]] = None, current_history: Optional[List[Dict[str, Any]]] = None):
        try:
            cleaned_query = query # 액션 요청 시 query는 비어있을 수 있음
            filters = {} # 액션 요청 시 필터는 사용하지 않음
            if not action: # 일반 검색 요청일 경우에만 쿼리 파싱
                cleaned_query, filters = parse_query_and_filters(query)
            
            payload = {
                "query": cleaned_query,
                "topk": DEFAULT_TOP_K, # 또는 UI에서 설정 가능한 값
                "filters": filters,
                "use_bm25": self.bm25_checkbox.get() == 1,
                "use_reranker": self.reranker_checkbox.get() == 1,
                "action": action, # Pass action
                "target_info": target_info, # Pass target_info
                "history": current_history # 대화 기록 전달
            }
            
            chat_url = f"{FASTAPI_URL}/chat"
            response = requests.post(chat_url, json=payload)
            response.raise_for_status() # HTTP 오류 발생 시 예외 발생
            data = response.json()

            if data["status"] == "success":
                answer = data.get("answer", "오류가 발생했습니다.")
                hits = data.get("hits", []) # Get hits from response
                suggestions = data.get("suggestions", []) # Get suggestions from response
                
                # Store last hits for action requests
                self.last_hits = hits 

                # 어시스턴트 답변을 대화 기록에 추가
                self.chat_history.append({"role": "assistant", "text": answer, "hits": hits})

                # Pass all relevant data to update_results
                self.update_results(answer, hits, suggestions) 
            else:
                error_message = data.get("message", "알 수 없는 오류")
                # 오류 메시지도 대화 기록에 추가
                self.chat_history.append({"role": "assistant", "text": f"오류 발생: {error_message}"})
                self.update_results(f"> {query}\n\n검색 중 오류 발생: {error_message}", [], []) # Pass empty hits/suggestions on error

        except requests.exceptions.ConnectionError:
            error_message = f"FATAL: FastAPI 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요: {FASTAPI_URL}"
            self.chat_history.append({"role": "assistant", "text": error_message})
            self.update_results(f"> {query}\n\n{error_message}", [], [])
        except requests.exceptions.RequestException as e:
            error_message = f"FATAL: API 요청 중 오류 발생: {e}"
            self.chat_history.append({"role": "assistant", "text": error_message})
            self.update_results(f"> {query}\n\n{error_message}", [], [])
        except Exception as e:
            error_message = f"FATAL: 알 수 없는 오류 발생: {e}"
            self.chat_history.append({"role": "assistant", "text": error_message})
            self.update_results(f"> {query}\n\n{error_message}", [], [])
        finally:
            self.search_entry.configure(state="normal")
            self.search_button.configure(state="normal")
            self.bm25_checkbox.configure(state="normal")
            self.reranker_checkbox.configure(state="normal")

    def update_results(self, answer_text: str, hits: List[Dict[str, Any]], suggestions: List[str]): # Modified signature
        # answer_text는 이제 _do_update_results에서 chat_history를 기반으로 생성
        self.after(0, self._do_update_results, hits, suggestions) # Pass hits and suggestions

    def _do_update_results(self, hits: List[Dict[str, Any]], suggestions: List[str]): # Modified signature
        self.results_textbox.configure(state="normal")
        self.results_textbox.delete("1.0", "end")
        
        # 전체 대화 기록을 텍스트 박스에 표시
        full_chat_display = []
        for turn in self.chat_history:
            role = turn["role"]
            text = turn["text"]
            if role == "user":
                full_chat_display.append(f"\n>> 사용자: {text}")
            else:
                full_chat_display.append(f"\n<< InfoPilot: {text}")
                # 어시스턴트 턴에 hits가 있다면 함께 표시 (선택 사항)
                # if turn.get("hits"):
                #     for i, h in enumerate(turn["hits"], 1):
                #         full_chat_display.append(f"   - {h['path']} (유사도: {h['similarity']:.3f})")
        
        self.results_textbox.insert("1.0", "\n".join(full_chat_display).strip())
        self.results_textbox.configure(state="disabled")

        # Clear previous suggestions
        for widget in self.suggestions_frame.winfo_children():
            widget.destroy()

        if suggestions:
            suggestion_label = ctk.CTkLabel(self.suggestions_frame, text="💡 이런 질문은 어떠세요?", font=ctk.CTkFont(size=14, weight="bold"))
            suggestion_label.grid(row=0, column=0, columnspan=len(suggestions), sticky="w", pady=(5, 0))

            for i, s_text in enumerate(suggestions):
                # Determine action and target_info based on suggestion text
                action = None
                target_info = None
                
                if s_text == "이 문서의 핵심 내용을 요약해줘":
                    action = "summarize"
                    if self.last_hits: # Use the first hit from last search as target for summarization
                        target_info = {"path": self.last_hits[0]["path"]}
                elif s_text == "위 문서들과 비슷한 다른 문서를 더 찾아줘":
                    action = "find_similar"
                    if self.last_hits: # Use the first hit from last search as target for similar search
                        target_info = {"path": self.last_hits[0]["path"]}
                elif s_text == "결과를 표 형식으로 정리해줘":
                    action = "format_table"
                    # target_info is not needed for format_table, backend uses history

                # Create a button for each suggestion
                # Use a lambda to capture current s_text, action, and target_info
                button = ctk.CTkButton(
                    self.suggestions_frame,
                    text=s_text,
                    command=lambda a=action, ti=target_info: self._send_action_request(a, ti),
                    fg_color="gray", # Make it look like a clickable suggestion
                    hover_color="darkgray",
                    text_color="white",
                    font=ctk.CTkFont(size=12)
                )
                button.grid(row=1, column=i, padx=5, pady=5, sticky="w")

    def _send_action_request(self, action: Optional[str], target_info: Optional[Dict[str, Any]]):
        # Disable search controls while processing action
        self._set_search_controls_state("disabled")
        # 액션 요청도 대화 기록에 추가
        action_query = f"액션: {action}" + (f" (대상: {target_info.get('path', '')})" if target_info else "")
        self.chat_history.append({"role": "user", "text": action_query})
        self.update_results(action_query, [], []) # UI에 액션 처리 중 메시지 표시

        # Run the action in a separate thread
        # Query is empty for action requests
        threading.Thread(target=self.run_search_thread, args=("", action, target_info, self.chat_history), daemon=True).start()