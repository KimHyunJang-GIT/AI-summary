# -*- coding: utf-8 -*-
"""
Step2: 텍스트 추출(CorpusBuilder) + 인덱싱 실행(run_indexing)
- AI 기반 토픽 분석을 통한 지능형 요약 생성 기능 추가
"""
import os, re, sys, time, threading
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

# --- 의존성 라이브러리 ---
try:
    import pandas as pd
except ImportError: pd = None
try:
    import fitz  # PyMuPDF
except ImportError: fitz = None
try:
    import docx
except ImportError: docx = None
try:
    import pptx
except ImportError: pptx = None
try:
    from tqdm import tqdm
except ImportError: tqdm = None
try:
    from sentence_transformers import SentenceTransformer, util
except ImportError: SentenceTransformer, util = None, None

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
            d = docx.Document(str(p)); t = "\n".join(par.text for par in d.paragraphs)
            return {"ok":True,"text":TextCleaner.clean(t)}
        except Exception as e: return {"ok":False,"text": f"DOCX Error: {e}"}

class ExcelExtractor(BaseExtractor):
    exts=(".xlsx",".xls",".xlsm",".xlsb",".xltx",".csv")
    def extract(self, p:Path)->Dict[str,Any]:
        if not pd: return {"ok": False, "text": "'pandas' not installed"}
        try:
            df = pd.read_csv(p, nrows=200, on_bad_lines='skip') if p.suffix.lower() == ".csv" else pd.read_excel(p, sheet_name=0, nrows=200)
            header = " | ".join(map(str, df.columns)); rows_text = "\n".join([" | ".join(map(str, row)) for _, row in df.head(20).iterrows()])
            return {"ok":True, "text":TextCleaner.clean(f"{header}\n{rows_text}"), "columns": df.columns.tolist()}
        except Exception as e: return {"ok":False,"text": f"Excel/CSV Error: {e}"}

class PdfExtractor(BaseExtractor):
    exts=(".pdf",)
    def extract(self, p:Path)->Dict[str,Any]:
        if not fitz: return {"ok": False, "text": "'PyMuPDF' not installed"}
        try:
            doc = fitz.open(p); text = "\n".join(page.get_text() for page in doc); doc.close()
            return {"ok":True,"text":TextCleaner.clean(text)}
        except Exception as e: return {"ok":False,"text": f"PDF Error: {e}"}

class PptxExtractor(BaseExtractor):
    exts=(".pptx",)
    def extract(self, p:Path)->Dict[str,Any]:
        if not pptx: return {"ok": False, "text": "'python-pptx' not installed"}
        try:
            pres = pptx.Presentation(str(p)); texts = [shape.text for slide in pres.slides for shape in slide.shapes if hasattr(shape, "text") and shape.text.strip()]
            return {"ok":True,"text":TextCleaner.clean("\n".join(texts))}
        except Exception as e: return {"ok":False,"text": f"PPTX Error: {e}"}

EXTRACTORS = [DocxExtractor(), ExcelExtractor(), PdfExtractor(), PptxExtractor()]
EXT_MAP = {e:ex for ex in EXTRACTORS for e in ex.exts}

# =========================
# 코퍼스 빌더 (AI 요약 기능 추가)
# =========================
@dataclass
class ExtractRecord:
    path:str; ext:str; ok:bool; text:str; summary:str; title:str; columns:List[str]

class CorpusBuilder:
    TOPIC_CANDIDATES = [
        "파이썬 프로그래밍", "자바 프로그래밍", "SQL 데이터베이스", "웹 개발", "자바스크립트",
        "머신러닝 및 인공지능", "데이터 분석 및 시각화", "클라우드 컴퓨팅", "보안",
        "업무 보고서", "회의록", "기획서", "제안서", "계약서", "법률 문서", "인사 관리", "재무 및 회계",
        "강의 교안", "연구 논문", "기술 문서", "사용자 매뉴얼",
    ]

    def __init__(self, max_text_chars:int=500_000, progress:bool=True):
        self.max_text_chars=max_text_chars
        self.progress=progress
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is not installed. Please run: pip install sentence-transformers")
        print("🧠 Loading model for summarization...")
        self.semantic_model = SentenceTransformer(Retriever.MODEL_NAME)
        self.topic_embeddings = self.semantic_model.encode(self.TOPIC_CANDIDATES, convert_to_tensor=True)

    def _generate_ai_summary(self, text: str) -> str:
        if not text or not text.strip():
            return "(내용이 없어 요약할 수 없습니다)"
        try:
            doc_embedding = self.semantic_model.encode(text, convert_to_tensor=True)
            cos_scores = util.cos_sim(doc_embedding, self.topic_embeddings)[0]
            best_topic_index = cos_scores.argmax()
            best_topic = self.TOPIC_CANDIDATES[best_topic_index]
            return f"이 문서는 '{best_topic}' 관련 자료로 보입니다."
        except Exception as e:
            return f"(요약 생성 중 오류 발생: {e})"

    def build(self, file_rows:List[Dict[str,Any]]) -> pd.DataFrame:
        if not pd: raise RuntimeError("pandas is required. Please run: pip install pandas")
        iterator = tqdm(file_rows, desc="📥 Extracting & Summarizing", unit="file") if self.progress and tqdm else file_rows
        records: List[ExtractRecord] = []
        
        for row in iterator:
            p = Path(row["path"]); ext = p.suffix.lower()
            extractor = EXT_MAP.get(ext)
            if not extractor:
                records.append(ExtractRecord(str(p), ext, False, "", "(알 수 없는 파일 형식)", p.stem, [])); continue
            
            try:
                extract_result = extractor.extract(p)
                raw_text = (extract_result.get("text","") or "")[:self.max_text_chars]
                summary = self._generate_ai_summary(raw_text)
                columns = extract_result.get("columns", [])
                records.append(ExtractRecord(str(p), ext, bool(extract_result.get("ok",False)), raw_text, summary, p.stem, columns))
            except Exception as e:
                records.append(ExtractRecord(str(p), ext, False, f"Build Error: {e}", "(처리 중 오류 발생)", p.stem, []))
        
        df = pd.DataFrame(records)
        print(f"✅ Text extraction & summarization complete: {int(df['ok'].sum())} successful, {len(df)} total.")
        return df

    @staticmethod
    def save(df: pd.DataFrame, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(out_path, index=False); print(f"💾 Corpus saved to Parquet: {out_path}")
        except Exception as e:
            csv_path = out_path.with_suffix(".csv"); df.to_csv(csv_path, index=False, encoding="utf-8")
            print(f"⚠️ Parquet engine not found, saved to CSV instead: {csv_path}\n   Error: {e}")

# =========================
# 인덱싱 실행 함수
# =========================
def run_indexing(corpus_path: Path, cache_dir: Path):
    print("🚀 Starting semantic indexing...")
    retriever = Retriever(corpus_path=corpus_path, cache_dir=cache_dir)
    retriever.ready(rebuild=True)
    print("✨ Indexing complete.")
