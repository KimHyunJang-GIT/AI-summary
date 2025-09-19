"""retrieval module split from retriever (auto-split from originals)."""
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from sentence_transformers import SentenceTransformer

# VectorIndex는 index_store.py에 정의된 것으로 가정합니다.
from .index_store import VectorIndex


class Retriever:
    MODEL_NAME = 'jhgan/ko-sroberta-multitask'

    def __init__(self, corpus_path: Path, cache_dir: Path = Path("./index_cache")):
        if SentenceTransformer is None: raise ImportError("Please run: pip install sentence-transformers")
        self.corpus_path = Path(corpus_path)
        self.cache_dir = Path(cache_dir)
        print(f"🧠 Loading Semantic Model: {self.MODEL_NAME}...")
        self.model = SentenceTransformer(self.MODEL_NAME)
        self.index = VectorIndex()
        self._ready = False

    def ready(self, rebuild: bool = False):
        emb_npy = self.cache_dir / "doc_embeddings.npy"
        meta_json = self.cache_dir / "doc_meta.json"
        if not rebuild and emb_npy.exists() and meta_json.exists():
            print(f"✅ Loading index from cache: {self.cache_dir}")
            self.index.load(emb_npy, meta_json)
            self._ready = True
            return

        if pd is None: raise RuntimeError("pandas is required. Please run: pip install pandas")

        print(f"📥 Loading corpus from {self.corpus_path}...")
        df = pd.read_parquet(self.corpus_path)

        work_df = df[df["ok"] & df["text"].str.len() > 0].copy()
        if len(work_df) == 0: raise RuntimeError("No valid text documents found in the corpus.")

        print(f"🧠 Encoding documents... (total: {len(work_df):,})")
        # AI가 학습할 텍스트는 '요약'이 아닌 '전체 텍스트'를 사용해야 합니다.
        doc_embeddings = self.model.encode(work_df["text"].tolist(), convert_to_tensor=False, show_progress_bar=True)
        
        # 인덱스 빌드 시, 'summary' 컬럼을 전달합니다.
        self.index.build(doc_embeddings, work_df["path"].tolist(), work_df["summary"].tolist())
        paths = self.index.save(self.cache_dir)
        print(f"💾 Index saved: {paths.emb_npy}, {paths.meta_json}")
        self._ready = True

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self._ready: self.ready(False)
        query_embedding = self.model.encode(query, convert_to_tensor=False)
        return self.index.search(query_embedding, top_k)

    @staticmethod
    def format_results(query: str, results: List[Dict[str, Any]]) -> str:
        if not results: return f"“{query}”와 유사한 문서를 찾지 못했습니다."
        lines = [f"‘{query}’와(과) 의미상 유사한 문서 Top {len(results)}:"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['path']}  (유사도: {r['similarity']:.3f})")
            if r.get("summary"):
                lines.append(f"   요약: {r['summary']}")
        return "\n".join(lines)
