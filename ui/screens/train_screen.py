import customtkinter as ctk
import os
import threading
import requests # requests 라이브러리 임포트
from pathlib import Path

# Core logic and helpers (이제 직접 사용하지 않고 API를 통해 호출)
from src.config import (
    DATA_DIR, MODELS_DIR, CACHE_DIR,
    CORPUS_PARQUET, FOUND_FILES_CSV, TOPIC_MODEL_PATH, SUPPORTED_EXTS, FASTAPI_URL # FASTAPI_URL 임포트
)

# FastAPI 백엔드 URL (이제 config.py에서 가져오므로 제거합니다)
# FASTAPI_BASE_URL = "http://127.0.0.1:8000"

# _run_full_train_logic 함수는 이제 사용하지 않으므로 제거합니다.

class TrainScreen(ctk.CTkFrame):
    def __init__(self, master, start_task_callback, end_task_callback, **kwargs):
        super().__init__(master, **kwargs)
        self.start_task_callback = start_task_callback
        self.end_task_callback = end_task_callback

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # --- Options Frame ---
        options_frame = ctk.CTkFrame(self)
        options_frame.grid(row=0, column=0, padx=20, pady=20, sticky="ew")
        options_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(options_frame, text="파일 확장자", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=10)
        self.exts_entry = ctk.CTkEntry(options_frame)
        self.exts_entry.insert(0, ",".join(SUPPORTED_EXTS))
        self.exts_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        self.scan_checkbox = ctk.CTkCheckBox(options_frame, text="PC 전체 드라이브 스캔 실행 (시간이 오래 걸릴 수 있습니다)")
        self.scan_checkbox.select()
        self.scan_checkbox.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="w")

        self.start_button = ctk.CTkButton(options_frame, text="▶️ 전체 학습 시작", command=self.start_training_thread)
        self.start_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        # --- Log Frame ---
        self.log_textbox = ctk.CTkTextbox(self, state="disabled", font=ctk.CTkFont(family="monospace"))
        self.log_textbox.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")

    def log_message(self, message):
        self.after(0, self._insert_log, message)

    def _insert_log(self, message):
        self.log_textbox.configure(state="normal")
        self.log_textbox.insert("end", f"{message}\n")
        self.log_textbox.see("end")
        self.log_textbox.configure(state="disabled")

    def training_done(self):
        self.after(0, self._enable_button)
        self.end_task_callback() # Notify App that task is done

    def _enable_button(self):
        self.start_button.configure(state="normal", text="▶️ 전체 학습 시작")

    def start_training_thread(self):
        # UI 응답성을 위해 별도의 스레드에서 API 호출 시작
        threading.Thread(target=self._start_training_api_calls, daemon=True).start()

    def _start_training_api_calls(self):
        self.start_task_callback() # Notify App that task is starting
        self.after(0, lambda: self.start_button.configure(state="disabled", text="학습 진행 중..."))
        self.after(0, lambda: self.log_textbox.configure(state="normal"))
        self.after(0, lambda: self.log_textbox.delete("1.0", "end"))
        self.after(0, lambda: self.log_textbox.configure(state="disabled"))

        exts_text = self.exts_entry.get()
        do_scan = self.scan_checkbox.get() == 1

        try:
            if do_scan:
                self.log_message("INFO: 드라이브 스캔 시작 (API 호출 중)...\n")
                scan_url = f"{FASTAPI_URL}/scan"
                scan_payload = {"out": str(FOUND_FILES_CSV), "exts_text": exts_text}
                scan_response = requests.post(scan_url, json=scan_payload)
                scan_response.raise_for_status() # HTTP 오류 발생 시 예외 발생
                scan_data = scan_response.json()
                if scan_data["status"] == "success":
                    self.log_message(f"SUCCESS: 스캔 완료. {scan_data["message"]}\n")
                else:
                    self.log_message(f"ERROR: 스캔 실패 - {scan_data["message"]}\n")
                    self.training_done()
                    return
            else:
                self.log_message("INFO: 스캔 건너뛰기.\n")

            self.log_message("INFO: 텍스트 추출 및 코퍼스 생성 시작 (API 호출 중)...\n")
            train_url = f"{FASTAPI_URL}/train"
            train_payload = {
                "scan_csv": str(FOUND_FILES_CSV),
                "corpus": str(CORPUS_PARQUET),
                "cache": str(CACHE_DIR)
            }
            train_response = requests.post(train_url, json=train_payload)
            train_response.raise_for_status() # HTTP 오류 발생 시 예외 발생
            train_data = train_response.json()

            if train_data["status"] == "success":
                self.log_message(f"SUCCESS: 학습 완료. {train_data["message"]}\n")
            else:
                self.log_message(f"ERROR: 학습 실패 - {train_data["message"]}\n")

        except requests.exceptions.ConnectionError:
            self.log_message(f"FATAL: FastAPI 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요: {FASTAPI_URL}\n")
        except requests.exceptions.RequestException as e:
            self.log_message(f"FATAL: API 요청 중 오류 발생 - {e}\n")
        except Exception as e:
            self.log_message(f"FATAL: 알 수 없는 오류 발생 - {e}\n")
        finally:
            self.training_done()
