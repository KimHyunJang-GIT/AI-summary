# -*- coding: utf-8 -*-
"""
retriever.py (Step3): 요약 우선 의미 기반 검색기
- sentence-transformers로 문서/질의 임베딩
- corpus의 'summary'가 있으면 우선 사용, 없으면 'text' 폴백
- 인덱스(doc_embeddings.npy + doc_meta.json) 캐시
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# =========================
# VectorIndex
# =========================
@dataclass
class IndexPaths:
    emb_npy: Path
    meta_json: Path

class VectorIndex:
    def __init__(self):
        self.embeddings: Optional[np.ndarray] = None
        self.paths: List[str] = []
        self.exts: List[str] = []
        self.previews: List[str] = []

    @staticmethod
    def _normalize_rows(M: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        return (M / (norms + 1e-12)).astype(np.float32, copy=False)

    def build(self, embeddings: np.ndarray, paths: List[str], exts: List[str], previews: List[str]):
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")
        self.embeddings = self._normalize_rows(embeddings)
        self.paths = list(paths)
        self.exts = list(exts)
        self.previews = [(t[:180] + "…") if isinstance(t, str) and len(t) > 180 else (t or "") for t in previews]

    def save(self, out_dir: Path) -> IndexPaths:
        out_dir.mkdir(parents=True, exist_ok=True)
        emb_path = out_dir / "doc_embeddings.npy"
        meta_path = out_dir / "doc_meta.json"
        if self.embeddings is None:
            raise RuntimeError("Index is not built. Call build() before saving.")
        np.save(emb_path, self.embeddings)
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump({"paths": self.paths, "exts": self.exts, "previews": self.previews}, f, ensure_ascii=False)
        return IndexPaths(emb_npy=emb_path, meta_json=meta_path)

    def load(self, emb_npy: Path, meta_json: Path):
        self.embeddings = np.load(emb_npy).astype(np.float32, copy=False)
        with meta_json.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        self.paths = meta["paths"]
        self.exts = meta["exts"]
        self.previews = meta["previews"]
        if self.embeddings.shape[0] != len(self.paths):
            raise RuntimeError("Embedding rows do not match meta entries.")

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.embeddings is None:
            raise RuntimeError("Index is not loaded.")
        qv = self._normalize_rows(query_vector.reshape(1, -1))
        sims = (self.embeddings @ qv.T).ravel()
        if len(sims) == 0:
            return []
        k = min(top_k, len(sims))
        idx = np.argpartition(-sims, k-1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [
            {
                "path": self.paths[i],
                "ext": self.exts[i],
                "similarity": float(sims[i]),
                "preview": self.previews[i],
            } for i in idx
        ]

# =========================
# Retriever
# =========================
class Retriever:
    MODEL_NAME = "jhgan/ko-sroberta-multitask"

    def __init__(self, corpus_path: Path, cache_dir: Path = Path("./index_cache")):
        if SentenceTransformer is None:
            raise ImportError("SentenceTransformer 미설치. pip install sentence-transformers")
        self.corpus_path = Path(corpus_path)
        self.cache_dir = Path(cache_dir)
        print(f"🧠 Loading Semantic Model: {self.MODEL_NAME} ...")
        self.model = SentenceTransformer(self.MODEL_NAME)
        self.index = VectorIndex()
        self._ready = False

    def _load_corpus(self) -> "pd.DataFrame":
        if pd is None:
            raise RuntimeError("pandas 미설치. pip install pandas")
        if self.corpus_path.suffix.lower() == ".parquet":
            try:
                return pd.read_parquet(self.corpus_path)
            except Exception:
                # Parquet 엔진 없을 때 CSV 폴백
                return pd.read_csv(self.corpus_path.with_suffix(".csv"))
        return pd.read_csv(self.corpus_path)

    def ready(self, rebuild: bool = False):
        emb_npy = self.cache_dir / "doc_embeddings.npy"
        meta_json = self.cache_dir / "doc_meta.json"
        if not rebuild and emb_npy.exists() and meta_json.exists():
            print(f"✅ Loading index from cache: {self.cache_dir}")
            self.index.load(emb_npy, meta_json)
            self._ready = True
            return

        print("📥 Loading corpus...")
        df = self._load_corpus()

        # 요약이 있으면 우선, 없으면 텍스트 폴백
        text_col = "summary" if "summary" in df.columns else "text"
        if text_col not in df.columns:
            raise RuntimeError("corpus에 'text' 컬럼이 없습니다. Step2를 먼저 실행하세요.")

        mask = df[text_col].astype(str).str.len() > 0
        work = df[mask].copy()
        if len(work) == 0:
            raise RuntimeError("유효 텍스트/요약이 없습니다.")

        # 프리뷰: title > summary/text
        if "title" in work.columns:
            previews = work["title"].fillna("").astype(str).tolist()
            if "summary" in work.columns:
                for i, p in enumerate(previews):
                    if len(p.strip()) < 4:
                        previews[i] = work["summary"].iloc[i]
        else:
            previews = work[text_col].astype(str).tolist()

        print(f"🧠 Encoding documents (column='{text_col}') ... (total: {len(work):,})")
        doc_embeddings = self.model.encode(
            work[text_col].astype(str).tolist(),
            convert_to_tensor=False,
            show_progress_bar=True
        )

        self.index.build(
            embeddings=np.asarray(doc_embeddings, dtype=np.float32),
            paths=work["path"].astype(str).tolist(),
            exts=work["ext"].astype(str).tolist(),
            previews=previews
        )
        paths = self.index.save(self.cache_dir)
        print(f"💾 Index saved: {paths.emb_npy.name}, {paths.meta_json.name} → {self.cache_dir}")
        self._ready = True

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self._ready:
            self.ready(rebuild=False)
        q_emb = self.model.encode([query], convert_to_tensor=False)
        q_vec = np.asarray(q_emb[0], dtype=np.float32)
        return self.index.search(q_vec, top_k=top_k)

    @staticmethod
    def format_results(query: str, results: List[Dict[str, Any]]) -> str:
        if not results:
            return f"“{query}”와 유사한 문서를 찾지 못했습니다."
        lines = [f"‘{query}’와(과) 의미상 유사한 문서 Top {len(results)}:"]
        for i, r in enumerate(results, 1):
            sim = f"{r['similarity']:.3f}"
            lines.append(f"{i}. {r['path']} [{r['ext']}]  유사도={sim}")
            if r.get("preview"):
                lines.append(f"   미리보기: {r['preview']}")
        return "\n".join(lines)
