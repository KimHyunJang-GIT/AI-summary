"""index_store module split from retriever (auto-split from originals)."""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np


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
