# -*- coding: utf-8 -*-
"""
Step2: 텍스트 추출(CorpusBuilder) + 인덱싱 실행(run_indexing)
- PDF 추출 성능 향상 (PyMuPDF)
- 모든 문서에 대한 자동 요약 기능 추가
"""
import os, re, sys, time, threading
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

# --- 의존성 라이브러리 ---
try:
    import pandas as pd
except ImportError:
    pd = None
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
try:
    import docx
except ImportError:
    docx = None
try:
    import pptx
except ImportError:
    pptx = None
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from retriever import Retriever

# =========================
# 유틸리티 함수
# =========================
class TextCleaner:
    _multi = re.compile(r"\s+")
    @classmethod
    def clean(cls, s:str)->str:
        if not s: return ""
        s = "".join(ch if ch.isprintable() or ch in "\t\n\r" else " " for ch in s)
        s = s.replace("\x00"," ")
        return cls._multi.sub(" ", s).strip()

def simple_summary(text: str, max_len: int = 400) -> str:
    if not text: return ""
    sents = re.split(r'(?<=[.!?])\s+|', text.strip())
    sents = [s.strip() for s in sents if s.strip()]
    summary = " ".join(sents[:3]) if sents else text[:max_len]
    return (summary[:max_len] + "…") if len(summary) > max_len else summary

# =========================
# 파일 유형별 텍스트 추출기
# =========================
class BaseExtractor:
    exts: Tuple[str,...] = ()
    def can_handle(self, p:Path)->bool: return p.suffix.lower() in self.exts
    def extract(self, p:Path)->Dict[str,Any]: raise NotImplementedError

class DocxExtractor(BaseExtractor):
    exts=(".docx",)
    def extract(self, p:Path)->Dict[str,Any]:
        if not docx: return {"ok": False, "text": "'python-docx' not installed"}
        try:
            d = docx.Document(str(p))
            t = "\n".join(par.text for par in d.paragraphs)
            return {"ok":True,"text":TextCleaner.clean(t)}
        except Exception as e:
            return {"ok":False,"text": f"DOCX Error: {e}"}

class ExcelExtractor(BaseExtractor):
    exts=(".xlsx",".xls",".xlsm",".xlsb",".xltx",".csv")
    def extract(self, p:Path)->Dict[str,Any]:
        if not pd: return {"ok": False, "text": "'pandas' not installed"}
        try:
            df = pd.read_csv(p, nrows=200, on_bad_lines='skip') if p.suffix.lower() == ".csv" else pd.read_excel(p, sheet_name=0, nrows=200)
            header = " | ".join(map(str, df.columns))
            rows_text = "\n".join([" | ".join(map(str, row)) for _, row in df.head(20).iterrows()])
            return {"ok":True, "text":TextCleaner.clean(f"{header}\n{rows_text}"), "columns": df.columns.tolist()}
        except Exception as e:
            return {"ok":False,"text": f"Excel/CSV Error: {e}"}

class PdfExtractor(BaseExtractor):
    exts=(".pdf",)
    def extract(self, p:Path)->Dict[str,Any]:
        if not fitz: return {"ok": False, "text": "'PyMuPDF' not installed"}
        try:
            doc = fitz.open(p)
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
            return {"ok":True,"text":TextCleaner.clean(text)}
        except Exception as e:
            return {"ok":False,"text": f"PDF Error: {e}"}

class PptxExtractor(BaseExtractor):
    exts=(".pptx",)
    def extract(self, p:Path)->Dict[str,Any]:
        if not pptx: return {"ok": False, "text": "'python-pptx' not installed"}
        try:
            pres = pptx.Presentation(str(p))
            texts = [shape.text for slide in pres.slides for shape in slide.shapes if hasattr(shape, "text") and shape.text.strip()]
            return {"ok":True,"text":TextCleaner.clean("\n".join(texts))}
        except Exception as e:
            return {"ok":False,"text": f"PPTX Error: {e}"}

EXTRACTORS = [DocxExtractor(), ExcelExtractor(), PdfExtractor(), PptxExtractor()]
EXT_MAP = {e:ex for ex in EXTRACTORS for e in ex.exts}

# =========================
# 코퍼스 빌더 (데이터 처리)
# =========================
@dataclass
class ExtractRecord:
    path:str; ext:str; ok:bool; text:str; summary:str; title:str; columns:List[str]

class CorpusBuilder:
    def __init__(self, max_text_chars:int=500_000, progress:bool=True):
        self.max_text_chars=max_text_chars
        self.progress=progress

    def build(self, file_rows:List[Dict[str,Any]]) -> pd.DataFrame:
        if not pd: raise RuntimeError("pandas is required. Please run: pip install pandas")
        
        iterator = tqdm(file_rows, desc="📥 Extracting text", unit="file") if self.progress and tqdm else file_rows
        records: List[ExtractRecord] = []
        
        for row in iterator:
            p = Path(row["path"]); ext = p.suffix.lower()
            extractor = EXT_MAP.get(ext)
            if not extractor:
                records.append(ExtractRecord(str(p), ext, False, "", "", p.stem, [])); continue
            
            try:
                extract_result = extractor.extract(p)
                raw_text = (extract_result.get("text","") or "")[:self.max_text_chars]
                summary = simple_summary(raw_text) if raw_text else ""
                columns = extract_result.get("columns", [])
                records.append(ExtractRecord(str(p), ext, bool(extract_result.get("ok",False)), raw_text, summary, p.stem, columns))
            except Exception as e:
                records.append(ExtractRecord(str(p), ext, False, f"Build Error: {e}", "", p.stem, []))
        
        df = pd.DataFrame(records)
        print(f"✅ Text extraction complete: {int(df['ok'].sum())} successful, {len(df)} total.")
        return df

    @staticmethod
    def save(df: pd.DataFrame, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(out_path, index=False)
            print(f"💾 Corpus saved to Parquet: {out_path}")
        except Exception as e:
            csv_path = out_path.with_suffix(".csv")
            df.to_csv(csv_path, index=False, encoding="utf-8")
            print(f"⚠️ Parquet engine not found, saved to CSV instead: {csv_path}\n   Error: {e}")

# =========================
# 인덱싱 실행 함수
# =========================
def run_indexing(corpus_path: Path, cache_dir: Path):
    """코퍼스를 기반으로 의미 벡터 인덱스를 생성합니다."""
    print("🚀 Starting semantic indexing...")
    retriever = Retriever(corpus_path=corpus_path, cache_dir=cache_dir)
    retriever.ready(rebuild=True) # rebuild=True forces re-indexing
    print("✨ Indexing complete.")
