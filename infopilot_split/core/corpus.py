"""corpus module split from pipeline (auto-split from originals)."""
from __future__ import annotations
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

from .retrieval import Retriever
from .extract import EXT_MAP


@dataclass
class ExtractRecord:
    path: str
    ext: str
    ok: bool
    text: str
    summary: str
    title: str
    columns: List[str] = field(default_factory=list)

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
            sys.stderr.write(f"AI Summary Error: {e}\n")
            return f"[요약 오류: {e}]"

    def build(self, file_rows:List[Dict[str,Any]]) -> pd.DataFrame:
        iterator = tqdm(file_rows, desc="📥 Extracting & Summarizing", unit="file") if self.progress else file_rows
        records: List[ExtractRecord] = []

        for row in iterator:
            p = Path(row["path"]); ext = p.suffix.lower()
            
            # Streamlit UI에 현재 처리 중인 파일 이름 표시
            print(f"📄 텍스트 추출 중: {p.name}")

            extractor = EXT_MAP.get(ext)
            if not extractor:
                records.append(ExtractRecord(str(p), ext, False, "", "(알 수 없는 파일 형식)", p.stem, [])); continue

            try:
                extract_result = extractor.extract(p)
                raw_text = (extract_result.get("text","") or "")[:self.max_text_chars]

                if not raw_text.strip():
                    sys.stderr.write(f"[경고] '{p.name}'에서 추출된 텍스트가 비어있습니다.\n")

                summary = self._generate_ai_summary(raw_text)
                columns = extract_result.get("columns", [])
                records.append(ExtractRecord(str(p), ext, bool(extract_result.get("ok",False)), raw_text, summary, p.stem, columns))
            except Exception as e:
                sys.stderr.write(f"Corpus Build Error for {p.name}: {e}\n")
                records.append(ExtractRecord(str(p), ext, False, f"Build Error: {e}", "(처리 중 오류 발생)", p.stem, []))

        df = pd.DataFrame([r.__dict__ for r in records])
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
