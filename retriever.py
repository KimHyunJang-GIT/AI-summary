# retriever.py  (Step3: 검색기)
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# =========================
# VectorIndex: 벡터와 메타데이터 저장/검색
# =========================
@dataclass
class IndexPaths:
    emb_npy: Path
    meta_json: Path

class VectorIndex:
    def __init__(self):
        self.embeddings: Optional[np.ndarray] = None
        self.paths: List[str] = []
        self.summaries: List[str] = [] # 'previews' -> 'summaries'로 이름 변경

    @staticmethod
    def _normalize_rows(M: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        return (M / (norms + 1e-12)).astype(np.float32, copy=False)

    def build(self, embeddings: np.ndarray, paths: List[str], summaries: List[str]):
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")
        self.embeddings = self._normalize_rows(embeddings)
        self.paths = list(paths)
        self.summaries = list(summaries)

    def save(self, out_dir: Path) -> IndexPaths:
        out_dir.mkdir(parents=True, exist_ok=True)
        emb_path = out_dir / "doc_embeddings.npy"
        meta_path = out_dir / "doc_meta.json"
        if self.embeddings is None:
            raise RuntimeError("Index is not built. Call build() before saving.")
        np.save(emb_path, self.embeddings)
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump({"paths": self.paths, "summaries": self.summaries}, f, ensure_ascii=False)
        return IndexPaths(emb_npy=emb_path, meta_json=meta_path)

    def load(self, emb_npy: Path, meta_json: Path):
        self.embeddings = np.load(emb_npy).astype(np.float32, copy=False)
        with meta_json.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        self.paths = meta["paths"]
        self.summaries = meta["summaries"]
        if self.embeddings.shape[0] != len(self.paths):
            raise RuntimeError("Embedding rows do not match meta entries.")

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.embeddings is None: raise RuntimeError("Index is not loaded.")
        qv = self._normalize_rows(query_vector.reshape(1, -1))
        sims = (self.embeddings @ qv.T).ravel()
        
        k = min(top_k, len(sims) - 1)
        if k <= 0: return []
        
        top_k_indices = np.argpartition(-sims, k)[:k]
        sorted_indices = top_k_indices[np.argsort(-sims[top_k_indices])]
        
        return [
            {
                "path": self.paths[i],
                "summary": self.summaries[i], # 'preview' -> 'summary'로 변경
                "similarity": float(sims[i]),
            }
            for i in sorted_indices
        ]

# =========================
# Retriever: 의미 기반 검색 엔진
# =========================
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
