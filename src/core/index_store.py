"""index_store module split from retriever (auto-split from originals)."""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd

@dataclass
class IndexPaths:
    emb_npy: Path
    meta_json: Path

class VectorIndex:
    def __init__(self):
        self.embeddings: Optional[np.ndarray] = None
        # --- NEW: Store all metadata in a pandas DataFrame for powerful filtering ---
        self.metadata: pd.DataFrame = pd.DataFrame()

    @staticmethod
    def _normalize_rows(M: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        return (M / (norms + 1e-12)).astype(np.float32, copy=False)

    def build(self, embeddings: np.ndarray, metadata: pd.DataFrame):
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")
        if len(embeddings) != len(metadata):
            raise ValueError("Length of embeddings and metadata must be the same.")
        
        self.embeddings = self._normalize_rows(embeddings)
        # Reset index to ensure it's a simple 0, 1, 2, ... range, which is crucial for mapping.
        self.metadata = metadata.reset_index(drop=True)

    def save(self, out_dir: Path) -> IndexPaths:
        out_dir.mkdir(parents=True, exist_ok=True)
        emb_path = out_dir / "doc_embeddings.npy"
        meta_path = out_dir / "doc_meta.jsonl" # Use .jsonl for line-delimited JSON
        if self.embeddings is None:
            raise RuntimeError("Index is not built. Call build() before saving.")
        
        np.save(emb_path, self.embeddings)
        # Save metadata as line-delimited JSON for efficient reading
        self.metadata.to_json(meta_path, orient="records", lines=True, force_ascii=False)
        
        return IndexPaths(emb_npy=emb_path, meta_json=meta_path)

    def load(self, emb_npy: Path, meta_json: Path):
        self.embeddings = np.load(emb_npy).astype(np.float32, copy=False)
        self.metadata = pd.read_json(meta_json, orient="records", lines=True)
        if len(self.embeddings) != len(self.metadata):
            raise RuntimeError("Embedding rows do not match meta entries.")

    def search(self, query_vector: np.ndarray, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if self.embeddings is None or self.metadata.empty:
            raise RuntimeError("Index is not loaded or is empty.")

        # --- HYBRID SEARCH: METADATA FILTERING ---
        candidate_indices = np.arange(len(self.metadata))
        if filters:
            mask = pd.Series(True, index=self.metadata.index)
            for key, value in filters.items():
                if key in self.metadata.columns:
                    # For now, we support simple equality check. Can be extended.
                    mask &= (self.metadata[key] == value)
            
            candidate_indices = self.metadata[mask].index.to_numpy()

        if len(candidate_indices) == 0:
            return [] # No documents match the filter criteria

        # Perform search only on the filtered candidates
        candidate_embeddings = self.embeddings[candidate_indices]
        # -----------------------------------------

        qv = self._normalize_rows(query_vector.reshape(1, -1))
        sims = (candidate_embeddings @ qv.T).ravel()
        
        k = min(top_k, len(sims))
        if k <= 0: return []
        
        # Get top_k indices *within the candidate set*
        top_k_candidate_indices = np.argpartition(-sims, k-1)[:k]
        
        # Sort these candidate indices by similarity
        sorted_candidate_indices = top_k_candidate_indices[np.argsort(-sims[top_k_candidate_indices])]

        # Map back to original indices to fetch metadata
        original_indices = candidate_indices[sorted_candidate_indices]
        
        return [
            {
                # Retrieve all metadata for the result
                **self.metadata.iloc[i].to_dict(),
                "similarity": float(sims[sorted_candidate_indices[j]])
            }
            for j, i in enumerate(original_indices)
        ]
