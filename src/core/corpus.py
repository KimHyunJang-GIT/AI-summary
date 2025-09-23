import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any
import traceback

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# 설정 파일에서 공통 변수 가져오기
from src.config import (
    MODEL_NAME,
    TOPIC_CANDIDATES,
    DEFAULT_MAX_TEXT_CHARS,
    DEFAULT_BATCH_SIZE
)
from .extract import EXT_MAP, PdfExtractor

@dataclass
class ExtractRecord:
    path: str
    ext: str
    ok: bool
    text: str
    summary: str
    title: str
    size: int
    mtime: float
    columns: List[str] = field(default_factory=list)
    error_reason: str = ""

def _extract_worker(file_row: Dict[str, Any], max_text_chars: int) -> ExtractRecord:
    p = Path(file_row["path"])
    ext = p.suffix.lower()
    
    record_data = {
        "path": str(p), "ext": ext, "title": p.stem, "ok": False,
        "text": "", "summary": "", "columns": [], "error_reason": "",
        "size": file_row.get('size', 0),
        "mtime": file_row.get('mtime', 0.0)
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
    def __init__(self, max_text_chars:int=DEFAULT_MAX_TEXT_CHARS, progress:bool=True, max_workers: int = None):
        self.max_text_chars = max_text_chars
        self.progress = progress
        self.max_workers = 0 
        sys.stderr.write("\n--- 🧠 Initializing CorpusBuilder in SEQUENTIAL DEBUG MODE ---\n")

    # MODIFIED: Accept optional *args and **kwargs to prevent crashing
    def build(self, file_rows: List[Dict[str, Any]], *args, **kwargs) -> pd.DataFrame:
        # Log if unexpected arguments are passed, for debugging.
        if args or kwargs:
            sys.stderr.write(f"[DEBUG] CorpusBuilder.build() called with extra arguments. Args: {args}, Kwargs: {kwargs}\n")

        total_files = len(file_rows)
        records: List[ExtractRecord] = []

        iterator = file_rows
        if self.progress:
            iterator = tqdm(iterator, total=total_files, desc="📥 Extracting text")
        
        for row in iterator:
            records.append(_extract_worker(row, self.max_text_chars))

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        successful_df = df[df["ok"]].copy()
        
        if not successful_df.empty:
            sys.stderr.write(f"\n✍️  Starting summarization for {len(successful_df)} successful files...\n")
            batch_size = DEFAULT_BATCH_SIZE
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            semantic_model = SentenceTransformer(MODEL_NAME, device=device)
            topic_embeddings = semantic_model.encode(TOPIC_CANDIDATES, convert_to_tensor=True, device=device)

            summaries = []
            iterator = range(0, len(successful_df), batch_size)
            if self.progress:
                iterator = tqdm(iterator, desc="✍️  Summarizing in batches")

            for i in iterator:
                batch_texts = successful_df.iloc[i:i + batch_size]["text"].tolist()
                if not batch_texts: continue
                try:
                    doc_embeddings = semantic_model.encode(batch_texts, convert_to_tensor=True, show_progress_bar=False)
                    cos_scores = util.cos_sim(doc_embeddings, topic_embeddings)
                    best_topic_indices = cos_scores.argmax(dim=1)
                    for j in range(len(batch_texts)):
                        best_topic = TOPIC_CANDIDATES[best_topic_indices[j]]
                        summaries.append(f"이 문서는 '{best_topic}' 관련 자료로 보입니다.")
                    if device == 'cuda':
                        del doc_embeddings, cos_scores
                        torch.cuda.empty_cache()
                except Exception as e:
                    sys.stderr.write(f"[오류] Batch Summary Error: {e}\n")
                    summaries.extend(["(요약 중 오류 발생)"] * len(batch_texts))
            
            successful_df["summary"] = summaries
            df.update(successful_df)
        
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
            required_cols = ['path', 'ext', 'ok', 'text', 'summary', 'title', 'size', 'mtime', 'columns', 'error_reason']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = pd.NA
            
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
                success_path = output_dir / "corpus_success.csv"
                failure_path = output_dir / "corpus_failure.csv"
                df_success.to_csv(success_path, index=False, encoding="utf-8")
                df_failure.to_csv(failure_path, index=False, encoding="utf-8")
        except Exception as e:
            sys.stderr.write(f"[ERROR] An error occurred while saving success/failure CSV files: {e}\n")
