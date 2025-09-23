import customtkinter as ctk
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import time
import threading

# Core logic and helpers
from src.core.helpers import parse_query_and_filters, have_all_artifacts
from src.config import CORPUS_PARQUET, CACHE_DIR, DEFAULT_TOP_K, DEFAULT_SIMILARITY_THRESHOLD
from src.core.retrieval import Retriever


@dataclass
class LNPChat:
    corpus_path: Path
    cache_dir: Path
    topk: int = DEFAULT_TOP_K
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    retr: Optional[Retriever] = field(init=False, default=None)
    ready_done: bool = field(init=False, default=False)

    def ready(self, rebuild: bool = False):
        print("엔진 초기화 시작...")
        self.retr = Retriever(corpus_path=self.corpus_path, cache_dir=self.cache_dir)
        self.retr.ready(rebuild=rebuild)
        self.ready_done = True
        print("✅ LNP Chat 준비 완료")

    def ask(self, query: str, topk: Optional[int] = None, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.ready_done: self.ready(rebuild=False)
        k = topk or self.topk
        t0 = time.time()
        candidate_hits = self.retr.search(query, top_k=max(k * 2, 20), filters=filters)
        dt = time.time() - t0
        filtered_hits = [h for h in candidate_hits if h['similarity'] >= self.similarity_threshold]
        final_hits = filtered_hits[:k]
        if not final_hits:
            answer_lines = [f"‘{query}’와 관련된 내용을 찾지 못했습니다."]
        else:
            answer_lines = [f"‘{query}’와(과) 의미상 유사한 문서 Top {len(final_hits)} (검색 {dt:.2f}s):"]
            for i, h in enumerate(final_hits, 1):
                sim = f"{h['similarity']:.3f}"
                answer_lines.append(f"{i}. {h['path']}  (유사도: {sim})")
                if h.get("summary"): answer_lines.append(f"   요약: {h['summary']}")
        return {"answer": "\n".join(answer_lines), "hits": final_hits}


class ChatScreen(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.chat_engine = None
        self.grid_columnconfigure(0, weight=1)

        # Initialize UI elements that will be managed by refresh_state
        self.warning_label = ctk.CTkLabel(self, text="", font=ctk.CTkFont(size=16))
        self.train_button_redirect = ctk.CTkButton(self, text="🚀 전체 학습시키기",
                                                   command=lambda: master.select_frame("train"))
        self.input_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.search_entry = ctk.CTkEntry(self.input_frame, placeholder_text="질문을 입력하세요...", height=40)
        self.search_button = ctk.CTkButton(self.input_frame, text="검색", width=100, height=40, command=self.search_event)
        self.results_textbox = ctk.CTkTextbox(self, font=ctk.CTkFont(size=14), state="disabled")

        self.refresh_state()  # Call refresh_state initially

    def setup_ui(self):
        # This method is no longer directly called, its logic is integrated into refresh_state
        pass

    def refresh_state(self):
        # Clear previous state by forgetting grid layout
        self.warning_label.grid_forget()
        self.train_button_redirect.grid_forget()  # Use grid_forget instead of pack_forget
        self.input_frame.grid_forget()
        self.results_textbox.grid_forget()

        if not have_all_artifacts():
            self.grid_rowconfigure(0, weight=1)
            self.warning_label.configure(text="⚠️ 학습된 데이터가 없습니다. 먼저 '학습시키기'를 실행하세요.")
            self.warning_label.grid(row=0, column=0, pady=(20, 10))
            self.train_button_redirect.grid(row=1, column=0, pady=10)  # Use grid instead of pack
            self.search_entry.configure(state="disabled")
            self.search_button.configure(state="disabled")
        else:
            # Re-create/show input_frame and results_textbox
            self.grid_rowconfigure(1, weight=1)
            self.input_frame.grid(row=0, column=0, padx=20, pady=20, sticky="ew")
            self.input_frame.grid_columnconfigure(0, weight=1)
            self.search_entry.grid(row=0, column=0, sticky="ew")
            self.search_entry.bind("<Return>", self.search_event)
            self.search_button.grid(row=0, column=1, padx=(10, 0))
            self.results_textbox.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")

            # Initialize chat engine if not already done
            if self.chat_engine is None or not self.chat_engine.ready_done:
                self.update_results("엔진을 초기화하는 중입니다... 잠시만 기다려주세요.")
                self.search_entry.configure(state="disabled")
                self.search_button.configure(state="disabled")
                threading.Thread(target=self.initialize_engine, daemon=True).start()
            else:
                self.update_results("질문을 입력하세요.")
                self.search_entry.configure(state="normal")
                self.search_button.configure(state="normal")

    def on_show(self):
        # Called when the frame is brought to front
        self.refresh_state()

    def initialize_engine(self):
        try:
            self.chat_engine = LNPChat(corpus_path=CORPUS_PARQUET, cache_dir=CACHE_DIR)
            self.chat_engine.ready()
            self.update_results("엔진 초기화 완료. 질문을 입력하세요.")
            self.search_entry.configure(state="normal")
            self.search_button.configure(state="normal")
        except Exception as e:
            self.update_results(f"엔진 초기화 중 오류 발생: {e}")
            self.search_entry.configure(state="disabled")
            self.search_button.configure(state="disabled")

    def search_event(self, event=None):
        query = self.search_entry.get().strip()
        if not query or self.search_button.cget("state") == "disabled":
            return

        self.search_entry.configure(state="disabled")
        self.search_button.configure(state="disabled")
        self.update_results(f"> {query}\n\n검색 중입니다...")

        # Run search in a thread to keep the UI responsive
        threading.Thread(target=self.run_search_thread, args=(query,), daemon=True).start()

    def run_search_thread(self, query):
        try:
            cleaned_query, filters = parse_query_and_filters(query)
            result = self.chat_engine.ask(cleaned_query, filters=filters)
            answer = result.get("answer", "오류가 발생했습니다.")
            self.update_results(f"> {query}\n\n{answer}")
        except Exception as e:
            self.update_results(f"> {query}\n\n검색 중 오류 발생: {e}")
        finally:
            self.search_entry.configure(state="normal")
            self.search_button.configure(state="normal")

    def update_results(self, text):
        self.after(0, self._do_update_results, text)

    def _do_update_results(self, text):
        self.results_textbox.configure(state="normal")
        self.results_textbox.delete("1.0", "end")
        self.results_textbox.insert("1.0", text)
        self.results_textbox.configure(state="disabled")
