import customtkinter as ctk

# config에서 필요한 경로 및 설정 가져오기
from src.config import CORPUS_PARQUET, TOPIC_MODEL_PATH, DATA_DIR, MODELS_DIR, CACHE_DIR
# helpers에서 have_all_artifacts 가져오기
from src.core.helpers import have_all_artifacts


class HomeScreen(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # --- Title ---
        self.title_label = ctk.CTkLabel(self, text="시작하기", font=ctk.CTkFont(size=24, weight="bold"))
        self.title_label.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="nw")

        # --- Main Content Area ---
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.main_frame.grid_columnconfigure((0, 1, 2), weight=1)

        # Initialize UI elements (they will be updated by refresh_state)
        self.status_label = ctk.CTkLabel(self.main_frame, text="", font=ctk.CTkFont(size=16))
        self.status_label.grid(row=0, column=0, columnspan=3, pady=(10, 20))

        self.chat_button = ctk.CTkButton(self.main_frame, text="💬 채팅 시작하기", height=40,
                                         command=lambda: master.select_frame("chat"))
        self.update_button = ctk.CTkButton(self.main_frame, text="🆕 새 파일/수정된 파일만 업데이트", height=40,
                                           command=lambda: master.select_frame("update"))
        self.train_button_full = ctk.CTkButton(self.main_frame, text="🔁 전체 다시 학습하기", height=40,
                                               command=lambda: master.select_frame("train"))
        self.train_button_initial = ctk.CTkButton(self.main_frame, text="🚀 전체 학습시키기", height=40,
                                                  command=lambda: master.select_frame("train"))

        # --- Path Info Area ---
        self.path_info_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.path_info_frame.grid(row=2, column=0, padx=20, pady=20, sticky="sew")
        self.path_info_frame.grid_columnconfigure(0, weight=1)

        path_label = ctk.CTkLabel(self.path_info_frame, text="설정된 주요 경로", font=ctk.CTkFont(weight="bold"))
        path_label.grid(row=0, column=0, sticky="w")

        path_text = f"""
데이터 디렉토리: {DATA_DIR}
모델 디렉토리:    {MODELS_DIR}
인덱스 캐시:      {CACHE_DIR}
코퍼스 파일:      {CORPUS_PARQUET} (존재: {CORPUS_PARQUET.exists()})
토픽 모델:        {TOPIC_MODEL_PATH} (존재: {TOPIC_MODEL_PATH.exists()})
"""

        self.path_textbox = ctk.CTkTextbox(self.path_info_frame, height=150, font=ctk.CTkFont(family="monospace"))
        self.path_textbox.insert("1.0", path_text)
        self.path_textbox.configure(state="disabled")  # Make it read-only
        self.path_textbox.grid(row=1, column=0, sticky="ew", pady=(5, 0))

        self.refresh_state()  # Call refresh_state initially

    def refresh_state(self):
        # Forget all buttons first to clear previous state
        self.chat_button.grid_forget()
        self.update_button.grid_forget()
        self.train_button_full.grid_forget()
        self.train_button_initial.grid_forget()

        if have_all_artifacts():
            self.status_label.configure(text="✅ 필요한 데이터가 모두 준비되었습니다.")
            self.chat_button.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
            self.update_button.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
            self.train_button_full.grid(row=1, column=2, padx=5, pady=5, sticky="ew")
        else:
            self.status_label.configure(text="⚠️ 학습된 데이터가 없습니다. 먼저 '학습시키기'를 실행하세요.")
            self.train_button_initial.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="ew")
