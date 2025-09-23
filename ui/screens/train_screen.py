
import customtkinter as ctk
import os
import pandas as pd
import time
import joblib
import threading
from pathlib import Path

# Core logic and helpers
from src.core.helpers import get_drives
from src.config import (
    EXCLUDE_DIRS, SUPPORTED_EXTS,
    DATA_DIR, MODELS_DIR, CACHE_DIR,
    CORPUS_PARQUET, FOUND_FILES_CSV, TOPIC_MODEL_PATH
)
from src.core.corpus import CorpusBuilder
from src.core.indexing import run_indexing

def _run_full_train_logic(exts_text, do_scan, log_callback, done_callback):
    try:
        log_callback("INFO: 필요 디렉토리 생성 중...")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        rows = None
        if do_scan:
            log_callback("INFO: 드라이브 스캔 시작...")
            current_supported_exts = {e.strip() for e in exts_text.split(",") if e.strip()}
            file_list = []
            for drive in get_drives():
                log_callback(f"INFO: {drive} 스캔 중...")
                for root, dirs, files in os.walk(drive, topdown=True):
                    dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
                    for file in files:
                        try:
                            p_file = Path(root) / file
                            if p_file.suffix.lower() in current_supported_exts:
                                if not any(part in EXCLUDE_DIRS for part in p_file.parts):
                                    stat = p_file.stat()
                                    file_list.append({'path': str(p_file), 'size': stat.st_size, 'mtime': stat.st_mtime})
                        except (FileNotFoundError, PermissionError): continue
            rows = file_list
            pd.DataFrame(rows).to_csv(FOUND_FILES_CSV, index=False, encoding="utf-8")
            log_callback(f"SUCCESS: 스캔 완료. {len(rows):,}개 파일 발견.")

        log_callback("INFO: 텍스트 추출 및 코퍼스 생성 시작...")
        if CORPUS_PARQUET.exists(): CORPUS_PARQUET.unlink()
        
        cb = CorpusBuilder(progress=True)
        
        if rows is None and FOUND_FILES_CSV.exists():
            rows = pd.read_csv(FOUND_FILES_CSV).to_dict("records")
        
        if rows:
            log_callback(f"INFO: {len(rows)}개 파일에서 텍스트를 추출합니다... (진행률은 콘솔 창에 표시됩니다)")
            df_corpus = cb.build(rows)
            cb.save(df_corpus, CORPUS_PARQUET)
            log_callback("SUCCESS: 코퍼스 생성 완료.")
        else:
            log_callback("ERROR: 스캔된 파일이 없어 코퍼스를 생성할 수 없습니다.")
            done_callback()
            return

        log_callback("INFO: 벡터 인덱싱 시작... (진행률은 콘솔 창에 표시됩니다)")
        if CORPUS_PARQUET.exists():
            run_indexing(corpus_path=CORPUS_PARQUET, cache_dir=CACHE_DIR)
            log_callback("SUCCESS: 인덱싱 완료.")
        else:
            log_callback("WARNING: 코퍼스 파일이 없어 인덱싱을 건너뜁니다.")

        log_callback("INFO: 학습 메타 정보 저장 중...")
        meta = {"indexed_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        joblib.dump(meta, TOPIC_MODEL_PATH)
        log_callback("🎉 SUCCESS: 모든 학습 과정 완료!")

    except Exception as e:
        log_callback(f"FATAL: 학습 중 오류 발생 - {e}")
    finally:
        done_callback()

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

        self.start_button = ctk.CTkButton(options_frame, text="▶️ 전체 학습 시작", command=self.start_training)
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

    def start_training(self):
        self.start_task_callback() # Notify App that task is starting
        self.start_button.configure(state="disabled", text="학습 진행 중...")
        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("1.0", "end")
        self.log_textbox.configure(state="disabled")

        exts_text = self.exts_entry.get()
        do_scan = self.scan_checkbox.get() == 1

        train_thread = threading.Thread(
            target=_run_full_train_logic,
            args=(exts_text, do_scan, self.log_message, self.training_done)
        )
        train_thread.daemon = True
        train_thread.start()
