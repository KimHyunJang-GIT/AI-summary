import customtkinter as ctk
import os
import threading
import requests # requests 라이브러리 임포트
from pathlib import Path

# Core logic and helpers (이제 직접 사용하지 않고 API를 통해 호출)
from src.core.helpers import have_all_artifacts
from src.config import (
    DATA_DIR, MODELS_DIR, CACHE_DIR,
    CORPUS_PARQUET, FOUND_FILES_CSV, TOPIC_MODEL_PATH, FASTAPI_URL # FASTAPI_URL 임포트
)

# FastAPI 백엔드 URL (이제 config.py에서 가져오므로 제거합니다)
# FASTAPI_BASE_URL = "http://127.0.0.1:8000"

# _run_update_index_logic 함수는 이제 사용하지 않으므로 제거합니다.

class UpdateScreen(ctk.CTkFrame):
    def __init__(self, master, start_task_callback, end_task_callback, **kwargs):
        super().__init__(master, **kwargs)
        self.start_task_callback = start_task_callback
        self.end_task_callback = end_task_callback

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Initialize UI elements
        self.warning_label = ctk.CTkLabel(self, text="", font=ctk.CTkFont(size=16))
        self.train_button_redirect = ctk.CTkButton(self, text="🚀 전체 학습시키기", command=lambda: master.select_frame("train"))
        self.options_frame = ctk.CTkFrame(self)
        self.start_button = ctk.CTkButton(self.options_frame, text="▶️ 업데이트 시작", command=self.start_update_thread)
        self.log_textbox = ctk.CTkTextbox(self, state="disabled", font=ctk.CTkFont(family="monospace"))

        self.refresh_state() # Call refresh_state initially

    def setup_ui(self):
        # This method is no longer directly called, its logic is integrated into refresh_state
        pass

    def refresh_state(self):
        # Clear previous state
        self.warning_label.grid_forget()
        self.train_button_redirect.grid_forget()
        self.options_frame.grid_forget()
        self.log_textbox.grid_forget()

        if not have_all_artifacts():
            self.grid_rowconfigure(0, weight=1)
            self.warning_label.configure(text="⚠️ 기존 학습 데이터가 없습니다. 전체 학습을 먼저 실행해주세요.")
            self.warning_label.grid(row=0, column=0, pady=(20, 10))
            self.train_button_redirect.grid(row=1, column=0, pady=10)
        else:
            # Re-create/show options_frame and log_textbox
            self.options_frame.grid(row=0, column=0, padx=20, pady=20, sticky="ew")
            self.options_frame.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(self.options_frame, text="새로 추가되거나 수정된 파일만 효율적으로 업데이트합니다.", justify="left").grid(row=0, column=0, padx=10, pady=10)
            self.start_button.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

            self.log_textbox.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")

    def on_show(self):
        # Called when the frame is brought to front
        self.refresh_state()

    def log_message(self, message):
        self.after(0, self._insert_log, message)

    def _insert_log(self, message):
        self.log_textbox.configure(state="normal")
        self.log_textbox.insert("end", f"{message}\n")
        self.log_textbox.see("end")
        self.log_textbox.configure(state="disabled")

    def update_done(self):
        self.after(0, self._enable_button)
        self.end_task_callback() # Notify App that task is done

    def _enable_button(self):
        self.start_button.configure(state="normal", text="▶️ 업데이트 시작")

    def start_update_thread(self):
        # UI 응답성을 위해 별도의 스레드에서 API 호출 시작
        threading.Thread(target=self._start_update_api_calls, daemon=True).start()

    def _start_update_api_calls(self):
        self.start_task_callback() # Notify App that task is starting
        self.after(0, lambda: self.start_button.configure(state="disabled", text="업데이트 진행 중..."))
        self.after(0, lambda: self.log_textbox.configure(state="normal"))
        self.after(0, lambda: self.log_textbox.delete("1.0", "end"))
        self.after(0, lambda: self.log_textbox.configure(state="disabled"))

        try:
            self.log_message("INFO: 업데이트 시작 (API 호출 중)...\n")
            update_url = f"{FASTAPI_URL}/update"
            update_payload = {
                "corpus": str(CORPUS_PARQUET),
                "cache": str(CACHE_DIR)
            }
            update_response = requests.post(update_url, json=update_payload)
            update_response.raise_for_status() # HTTP 오류 발생 시 예외 발생
            update_data = update_response.json()

            if update_data["status"] == "success":
                self.log_message(f"SUCCESS: 업데이트 완료. {update_data["message"]}\n")
            else:
                self.log_message(f"ERROR: 업데이트 실패 - {update_data["message"]}\n")

        except requests.exceptions.ConnectionError:
            self.log_message(f"FATAL: FastAPI 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요: {FASTAPI_URL}\n")
        except requests.exceptions.RequestException as e:
            self.log_message(f"FATAL: API 요청 중 오류 발생 - {e}\n")
        except Exception as e:
            self.log_message(f"FATAL: 알 수 없는 오류 발생 - {e}\n")
        finally:
            self.update_done()
