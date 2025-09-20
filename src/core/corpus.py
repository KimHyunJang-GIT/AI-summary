import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

from .retrieval import Retriever
from .extract import EXT_MAP, PdfExtractor

@dataclass
class ExtractRecord:
    path: str
    ext: str
    ok: bool
    text: str
    summary: str
    title: str
    columns: List[str] = field(default_factory=list)
    error_reason: str = ""

def _extract_worker(file_row: Dict[str, Any], max_text_chars: int) -> ExtractRecord:
    p = Path(file_row["path"])
    ext = p.suffix.lower()
    
    # --- DEBUG LOG --- 
    # Print the file path BEFORE processing to identify the problematic file if a crash occurs.
    print(f"[DEBUG] Processing: {str(p)}")

    record_data = {
        "path": str(p), "ext": ext, "title": p.stem, "ok": False,
        "text": "", "summary": "", "columns": [], "error_reason": ""
    }

    if ext == ".pdf":
        PdfExtractor.get_ocr_reader()

    extractor = EXT_MAP.get(ext)
    if not extractor:
        record_data["error_reason"] = f"지원되지 않는 파일 형식: {ext}"
        return ExtractRecord(**record_data)

    try:
        extract_result = extractor.extract(p)
        is_ok = bool(extract_result.get("ok", False))
        raw_text = (extract_result.get("text", "") or "")[:max_text_chars]

        if not is_ok or not raw_text.strip():
            reason = extract_result.get("text", "(내용 추출 실패 또는 비어있음)")
            record_data["error_reason"] = reason if reason.strip() else "(내용 추출 실패 또는 비어있음)"
            return ExtractRecord(**record_data)
        
        record_data["ok"] = True
        record_data["text"] = raw_text
        record_data["columns"] = extract_result.get("columns", [])
        return ExtractRecord(**record_data)

    except Exception as e:
        full_traceback = traceback.format_exc()
        error_msg = f"File Processing Error: {e}"
        sys.stderr.write(f"[오류] {p.name}: {error_msg}\nTraceback:\n{full_traceback}\n")
        record_data["error_reason"] = error_msg
        return ExtractRecord(**record_data)

class CorpusBuilder:
    TOPIC_CANDIDATES = [
        "파이썬 프로그래밍", "자바 프로그래밍", "SQL 데이터베이스", "웹 개발", "자바스크립트",
        "머신러닝 및 인공지능", "데이터 분석 및 시각화", "클라우드 컴퓨팅", "보안",
        "업무 보고서", "회의록", "기획서", "제안서", "계약서", "법률 문서", "인사 관리", "재무 및 회계",
        "강의 교안", "연구 논문", "기술 문서", "사용자 매뉴얼",
    ]

    def __init__(self, max_text_chars:int=500_000, progress:bool=True, max_workers: int = None):
        self.max_text_chars = max_text_chars
        self.progress = progress
        
        # --- FORCE SEQUENTIAL FOR DEBUGGING ---
        # This is a temporary measure to find the file causing the crash.
        self.max_workers = 0 
        sys.stderr.write("\n--- 🧠 Initializing CorpusBuilder in SEQUENTIAL DEBUG MODE ---\n")

    def build(self, file_rows: List[Dict[str, Any]]) -> pd.DataFrame:
        total_files = len(file_rows)
        records: List[ExtractRecord] = []

        # --- Text Extraction (Sequential Debug Mode) ---
        sys.stderr.write("📥 Starting sequential text extraction to find the root cause of failures...\n")
        iterator = file_rows
        if self.progress:
            iterator = tqdm(iterator, total=total_files, desc="📥 Extracting text (Sequential Debug Mode)")
        
        for row in iterator:
            records.append(_extract_worker(row, self.max_text_chars))

        # --- DataFrame Creation ---
        if not records:
            sys.stderr.write("⚠️ No records were processed. Returning an empty DataFrame.\n")
            return pd.DataFrame()
        df = pd.DataFrame([r.__dict__ for r in records])

        # --- Batch Summarization ---
        successful_records_df = df[df["ok"]].copy()
        sys.stderr.write(f"\n✍️  Starting summarization for {len(successful_records_df)} successful files...\n")
        
        if not successful_records_df.empty:
            batch_size = 32
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            sys.stderr.write(f"Using device: {device} for summarization.\n")
            semantic_model = SentenceTransformer(Retriever.MODEL_NAME, device=device)
            topic_embeddings = semantic_model.encode(self.TOPIC_CANDIDATES, convert_to_tensor=True, device=device)

            summaries = []
            iterator = range(0, len(successful_records_df), batch_size)
            if self.progress:
                iterator = tqdm(iterator, desc="✍️  Summarizing in batches")

            for i in iterator:
                batch_texts = successful_records_df.iloc[i:i + batch_size]["text"].tolist()
                if not batch_texts: continue

                try:
                    doc_embeddings = semantic_model.encode(batch_texts, convert_to_tensor=True, show_progress_bar=False)
                    cos_scores = util.cos_sim(doc_embeddings, topic_embeddings)
                    best_topic_indices = cos_scores.argmax(dim=1)

                    for j in range(len(batch_texts)):
                        best_topic = self.TOPIC_CANDIDATES[best_topic_indices[j]]
                        summaries.append(f"이 문서는 '{best_topic}' 관련 자료로 보입니다.")

                    if device == 'cuda':
                        del doc_embeddings, cos_scores
                        torch.cuda.empty_cache()

                except Exception as e:
                    sys.stderr.write(f"[오류] Batch Summary Error: {e}\n")
                    summaries.extend(["(요약 중 오류 발생)"] * len(batch_texts))
            
            successful_records_df["summary"] = summaries
            df.update(successful_records_df)
        else:
            sys.stderr.write("No successful files to summarize.\n")

        ok_count = int(df['ok'].sum())
        sys.stderr.write(f"✅ Text extraction & summarization complete: {ok_count} successful, {total_files} total.\n")
        return df

    @staticmethod
    def save(df: pd.DataFrame, out_path: Path):
        if df.empty:
            sys.stderr.write("⚠️ DataFrame is empty, nothing to save.\n")
            return
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(out_path, index=False)
            sys.stderr.write(f"💾 Corpus saved to Parquet: {out_path}\n")
        except Exception as e:
            csv_path = out_path.with_suffix(".csv")
            df.to_csv(csv_path, index=False, encoding="utf-8")
            sys.stderr.write(f"⚠️ Parquet save failed, saved to CSV instead: {csv_path}\n   Error: {e}\n")
        
        output_dir = out_path.parent
        try:
            if 'ok' in df.columns:
                df_success = df[df['ok'] == True]
                df_failure = df[df['ok'] == False]
                
                sys.stderr.write(f"[INFO] Saving success/failure CSVs to {output_dir}...\n")
                sys.stderr.write(f"[INFO] Success count: {len(df_success)}, Failure count: {len(df_failure)}\n")

                success_path = output_dir / "corpus_success.csv"
                failure_path = output_dir / "corpus_failure.csv"

                df_success.to_csv(success_path, index=False, encoding="utf-8")
                df_failure.to_csv(failure_path, index=False, encoding="utf-8")
                sys.stderr.write("[SUCCESS] Successfully saved success/failure CSVs.\n")
            else:
                sys.stderr.write("[WARNING] 'ok' column not found in DataFrame. Skipping success/failure CSV save.\n")
        except Exception as e:
            sys.stderr.write(f"[ERROR] An error occurred while saving success/failure CSV files: {e}\n")
