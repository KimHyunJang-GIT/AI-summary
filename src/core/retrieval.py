"""retrieval module split from retriever (auto-split from originals)."""
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from sentence_transformers import SentenceTransformer

from src.config import MODEL_NAME, MODELS_DIR # MODELS_DIR 임포트 추가
from .index_store import VectorIndex


class Retriever:
    def __init__(self, corpus_path: Path, cache_dir: Path = Path("./index_cache")):
        if SentenceTransformer is None: raise ImportError("Please run: pip install sentence-transformers")
        self.corpus_path = Path(corpus_path)
        self.cache_dir = Path(cache_dir)
        print(f"🧠 Loading Semantic Model: {MODEL_NAME}...")
        # 모델 캐시 폴더를 MODELS_DIR로 지정
        self.model = SentenceTransformer(MODEL_NAME, cache_folder=str(MODELS_DIR))
        self.index = VectorIndex()
        self._ready = False

    def ready(self, rebuild: bool = False):
        emb_npy = self.cache_dir / "doc_embeddings.npy"
        # --- UPDATED: Switched to .jsonl for metadata file ---
        meta_json = self.cache_dir / "doc_meta.jsonl"
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
        doc_embeddings = self.model.encode(work_df["text"].tolist(), convert_to_tensor=False, show_progress_bar=True)
        
        # --- UPDATED: Pass the entire metadata DataFrame to the index ---
        self.index.build(doc_embeddings, work_df)
        
        paths = self.index.save(self.cache_dir)
        print(f"💾 Index saved: {paths.emb_npy}, {paths.meta_json}")
        self._ready = True

    def search(self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not self._ready: self.ready(False)
        query_embedding = self.model.encode(query, convert_to_tensor=False)
        
        # --- UPDATED: Pass filters to the index search method ---
        return self.index.search(query_embedding, top_k, filters)

    @staticmethod
    def format_results(query: str, results: List[Dict[str, Any]]) -> str:
        if not results: return f"“{query}”와 유사한 문서를 찾지 못했습니다."
        lines = [f"‘{query}’와(과) 의미상 유사한 문서 Top {len(results)}:"]
        for i, r in enumerate(results, 1):
            # Now result `r` is a dictionary containing all metadata columns
            lines.append(f"{i}. {r.get('path', 'N/A')}  (유사도: {r.get('similarity', 0.0):.3f})")
            if r.get("summary"):
                lines.append(f"   요약: {r.get('summary')}")
        return "\n".join(lines)
