# retriever.py  (Step3: 검색기)
from __future__ import annotations
import os, sys, json, time, importlib, types
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import joblib
except Exception:
    joblib = None


# =========================
# 레거시 모듈 별칭 주입
# =========================
def _alias_legacy_modules():
    """
    과거 joblib이 'TextCleaner' 모듈 경로를 기억하고 있을 때
    현재 Step2 모듈명으로 연결해 준다.
    """
    candidates = [
        "pipeline",                # 현재 Step2 파일명 (여기에 맞춤)
        "step2_module_progress",   # 이전 이름들
        "step2_pipeline",
        "Step2_module_progress",
    ]
    for name in candidates:
        try:
            mod = importlib.import_module(name)
            sys.modules["TextCleaner"] = mod
            return True
        except Exception:
            continue
    shim = types.ModuleType("TextCleaner")
    sys.modules["TextCleaner"] = shim
    return False


# =========================
# QueryEncoder
# =========================
class QueryEncoder:
    """Step2 topic_model.joblib에서 파이프라인을 꺼내 질의/문서 임베딩 변환"""
    def __init__(self, model_path: Path):
        if joblib is None:
            raise RuntimeError("joblib이 필요합니다. pip install joblib")

        try:
            obj = joblib.load(model_path)
        except ModuleNotFoundError as e:
            if "TextCleaner" in str(e):
                print("⚠ 레거시 모델 감지: TextCleaner → pipeline 별칭 주입 후 재시도")
                _alias_legacy_modules()
                obj = joblib.load(model_path)
            else:
                raise

        self.pipeline = obj["pipeline"]
        self.tfidf = self.pipeline.named_steps["tfidf"]
        self.svd = self.pipeline.named_steps["svd"]

    def encode_docs(self, texts: List[str]) -> np.ndarray:
        X = self.tfidf.transform(texts)
        Z = self.svd.transform(X)
        return Z.astype(np.float32, copy=False)

    def encode_query(self, query: str) -> np.ndarray:
        Xq = self.tfidf.transform([query])
        Zq = self.svd.transform(Xq)
        return Zq.astype(np.float32, copy=False)


# =========================
# VectorIndex
# =========================
@dataclass
class IndexPaths:
    emb_npy: Path
    meta_json: Path

class VectorIndex:
    def __init__(self):
        self.Z: Optional[np.ndarray] = None
        self.paths: List[str] = []
        self.exts: List[str] = []
        self.preview: List[str] = []

    @staticmethod
    def _normalize_rows(M: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
        return (M / norms).astype(np.float32, copy=False)

    def build(self, embeddings: np.ndarray, paths: List[str], exts: List[str], texts: List[str]):
        if embeddings.ndim != 2:
            raise ValueError("embeddings는 2차원이어야 합니다.")
        self.Z = self._normalize_rows(embeddings)
        self.paths = list(paths)
        self.exts = list(exts)
        self.preview = [(t[:180] + "…") if len(t) > 180 else t for t in texts]

    def save(self, out_dir: Path) -> IndexPaths:
        out_dir.mkdir(parents=True, exist_ok=True)
        emb_path = out_dir / "doc_embeddings.npy"
        meta_path = out_dir / "doc_meta.json"
        if self.Z is None:
            raise RuntimeError("인덱스가 비어있습니다. build() 후 저장하세요.")
        np.save(emb_path, self.Z)
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump({"paths": self.paths, "exts": self.exts, "preview": self.preview}, f, ensure_ascii=False)
        return IndexPaths(emb_npy=emb_path, meta_json=meta_path)

    def load(self, emb_npy: Path, meta_json: Path):
        self.Z = np.load(emb_npy).astype(np.float32, copy=False)
        with meta_json.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        self.paths = meta["paths"]
        self.exts = meta["exts"]
        self.preview = meta["preview"]
        if self.Z.shape[0] != len(self.paths):
            raise RuntimeError("임베딩 행 수와 메타 항목 수가 다릅니다.")

    def search(self, qvec: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.Z is None:
            raise RuntimeError("인덱스가 로드되지 않았습니다.")
        qv = qvec.reshape(1, -1)
        qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-12)
        sims = (self.Z @ qv.T).ravel()
        idx = np.argpartition(-sims, kth=min(top_k, len(sims)-1))[:top_k]
        idx = idx[np.argsort(-sims[idx])]
        return [
            {"path": self.paths[i], "ext": self.exts[i], "similarity": float(sims[i]), "preview": self.preview[i]}
            for i in idx
        ]


# =========================
# Retriever
# =========================
class Retriever:
    def __init__(self, model_path: Path, corpus_path: Path, cache_dir: Path = Path("./index_cache")):
        self.model_path = Path(model_path)
        self.corpus_path = Path(corpus_path)
        self.cache_dir = Path(cache_dir)
        self.encoder = QueryEncoder(self.model_path)
        self.index = VectorIndex()
        self._ready = False

    def ready(self, rebuild: bool = False):
        emb_npy = self.cache_dir / "doc_embeddings.npy"
        meta_json = self.cache_dir / "doc_meta.json"
        if not rebuild and emb_npy.exists() and meta_json.exists():
            self.index.load(emb_npy, meta_json)
            self._ready = True
            print(f"✅ 인덱스 로드: {self.cache_dir}")
            return

        if pd is None:
            raise RuntimeError("pandas 필요. pip install pandas")

        print("📥 코퍼스 로드…")
        if self.corpus_path.suffix.lower() == ".parquet":
            try:
                df = pd.read_parquet(self.corpus_path)
            except Exception:
                df = pd.read_csv(self.corpus_path.with_suffix(".csv"))
        else:
            df = pd.read_csv(self.corpus_path)

        mask = df["text"].astype(str).str.len() > 0
        work = df[mask].copy()
        if len(work) == 0:
            raise RuntimeError("유효 텍스트 문서가 없습니다.")

        print(f"🧠 문서 임베딩 생성… (docs={len(work):,})")
        Z = self.encoder.encode_docs(work["text"].tolist())
        self.index.build(Z, work["path"].tolist(), work["ext"].tolist(), work["text"].tolist())
        paths = self.index.save(self.cache_dir)
        print(f"💾 인덱스 저장: {paths.emb_npy}, {paths.meta_json}")
        self._ready = True

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self._ready:
            self.ready(False)
        q = self.encoder.encode_query(query)
        return self.index.search(q, top_k)

    @staticmethod
    def format_results(query: str, results: List[Dict[str, Any]]) -> str:
        if not results:
            return f"“{query}”와 유사한 문서를 찾지 못했습니다."
        lines = [f"‘{query}’와 유사한 문서 Top {len(results)}:"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['path']} [{r['ext']}]  유사도={r['similarity']:.2f}")
            if r.get("preview"):
                lines.append(f"   미리보기: {r['preview']}")
        return "\n".join(lines)
