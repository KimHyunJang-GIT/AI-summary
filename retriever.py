# retriever.py  (Step3: 검색기)
from __future__ import annotations
import os, sys, json, time, importlib, types, re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set

import numpy as np

MODEL_TEXT_COLUMN = "text_model"
_META_SPLIT_RE = re.compile(r"[^0-9A-Za-z가-힣]+")


def _normalize_ext(ext: str) -> str:
    if not ext:
        return ""
    ext = ext.strip().lower()
    if not ext:
        return ""
    if not ext.startswith('.'):
        ext = f".{ext}"
    return ext


_DOC_SYNONYMS: Set[str] = {
    "워드",
    "ms word",
    "microsoft word",
    "ms-word",
    "msword",
    "word file",
    "word document",
    "word doc",
    "워드 문서",
    "워드파일",
}

_PPT_SYNONYMS: Set[str] = {
    "ppt",
    "파워포인트",
    "파워 포인트",
    "powerpoint",
    "power point",
    "power-point",
    "presentation deck",
    "presentation file",
    "slide deck",
    "슬라이드",
}

_EXCEL_SYNONYMS: Set[str] = {
    "excel",
    "엑셀",
    "excel file",
    "excel sheet",
    "spreadsheet",
    "스프레드시트",
    "엑셀 시트",
}

_CSV_SYNONYMS: Set[str] = {
    "csv",
    "comma separated",
    "comma-separated",
    "comma separated values",
    "씨에스브이",
    "쉼표 구분",
}

_DOMAIN_EXT_HINTS: Dict[str, Set[str]] = {
    "보고서": {".pdf", ".docx", ".hwp"},
    "회의록": {".docx", ".hwp", ".pdf"},
    "계약서": {".pdf", ".hwp"},
    "예산": {".xlsx", ".xls", ".xlsm"},
    "세금": {".xlsx", ".xls", ".pdf"},
    "레퍼런스": {".pdf", ".xlsx"},
    "참고": {".pdf", ".xlsx"},
    "초안": {".doc", ".docx", ".hwp"},
    "참고문헌": {".docx", ".xlsx", ".csv"},
}

_EXT_SYNONYMS: Dict[str, Set[str]] = {
    ".pdf": {"pdf", "피디에프", "acrobat", "portable document", "포터블 문서"},
    ".hwp": {"hwp", "한글", "한컴", "한컴오피스", "hanword", "han word", "hangul", "hangeul"},
    ".doc": set(_DOC_SYNONYMS),
    ".docx": set(_DOC_SYNONYMS),
    ".ppt": set(_PPT_SYNONYMS),
    ".pptx": set(_PPT_SYNONYMS) | {"pptx"},
    ".xlsx": set(_EXCEL_SYNONYMS),
    ".xls": set(_EXCEL_SYNONYMS),
    ".xlsm": set(_EXCEL_SYNONYMS),
    ".xlsb": set(_EXCEL_SYNONYMS),
    ".xltx": set(_EXCEL_SYNONYMS),
    ".csv": set(_CSV_SYNONYMS),
}


def _iter_query_units(lowered: str) -> Set[str]:
    tokens = _split_tokens(lowered)
    units: Set[str] = set(tokens)
    units.add(lowered)
    length = len(tokens)
    for n in (2, 3):
        if length < n:
            continue
        for i in range(length - n + 1):
            segment = tokens[i:i + n]
            units.add(" ".join(segment))
            units.add("".join(segment))
    return units


def _extract_query_exts(query: str, *, available_exts: Set[str]) -> Set[str]:
    if not query or not available_exts:
        return set()
    lowered = query.lower()
    units = _iter_query_units(lowered)

    requested: Set[str] = set()

    for unit in units:
        normalized = _normalize_ext(unit)
        if normalized in available_exts:
            requested.add(normalized)

    for unit in units:
        for ext, keywords in _EXT_SYNONYMS.items():
            if ext not in available_exts:
                continue
            for keyword in keywords:
                if keyword in unit:
                    requested.add(ext)
                    break

    if requested:
        return requested

    for keyword, hinted_exts in _DOMAIN_EXT_HINTS.items():
        if keyword in units or keyword in lowered:
            for ext in hinted_exts:
                if ext in available_exts:
                    requested.add(ext)

    return requested


def _prioritize_ext_hits(
    hits: List[Dict[str, Any]], *, desired_exts: Set[str], top_k: int
) -> List[Dict[str, Any]]:
    if not desired_exts:
        return hits[:top_k]
    matched = [h for h in hits if _normalize_ext(h.get("ext")) in desired_exts]
    if len(matched) >= top_k:
        return matched[:top_k]
    remaining = [h for h in hits if _normalize_ext(h.get("ext")) not in desired_exts]
    combined = matched + remaining
    return combined[:top_k]


def _split_tokens(source: str) -> List[str]:
    if not source:
        return []
    return [tok for tok in _META_SPLIT_RE.split(source) if tok]


def _metadata_text(path: str, ext: str, drive: str) -> str:
    tokens: List[str] = []
    if path:
        try:
            p = Path(path)
        except Exception:
            p = None
        if p:
            name = p.name
            if name:
                tokens.append(name)
            stem = p.stem
            if stem and stem != name:
                tokens.append(stem)
            tokens.extend(_split_tokens(stem))
            parent_name = p.parent.name if p.parent else ""
            if parent_name:
                tokens.append(parent_name)
                tokens.extend(_split_tokens(parent_name))
        else:
            tokens.append(str(path))
    if ext:
        ext_clean = str(ext).strip()
        if ext_clean:
            tokens.append(ext_clean)
            ext_no_dot = ext_clean.lstrip(".")
            if ext_no_dot:
                tokens.append(ext_no_dot)
    if drive:
        drive_str = str(drive)
        tokens.append(drive_str)
        tokens.extend(_split_tokens(drive_str))

    seen = set()
    normalized: List[str] = []
    cleaner = None
    if "TextCleaner" in sys.modules:
        cleaner = getattr(sys.modules.get("TextCleaner"), "TextCleaner", None)

    for token in tokens:
        cleaned = str(token)
        if cleaner and hasattr(cleaner, "clean"):
            try:
                cleaned = cleaner.clean(cleaned)
            except Exception:
                pass
        cleaned = cleaned.strip().lower()
        if not cleaned:
            continue
        if cleaned not in seen:
            seen.add(cleaned)
            normalized.append(cleaned)
    return " ".join(normalized)


def _compose_model_text(base_text: str, metadata: str) -> str:
    base_text = base_text or ""
    metadata = metadata or ""
    if metadata and base_text:
        return f"{base_text}\n\n{metadata}"
    if metadata:
        return metadata
    return base_text


def _prepare_text_frame(df: "pd.DataFrame") -> "pd.DataFrame":
    if pd is None or df is None:
        return df
    if df.empty:
        if MODEL_TEXT_COLUMN not in df.columns:
            df[MODEL_TEXT_COLUMN] = pd.Series(dtype=str)
        return df

    for column in ("text", "text_original"):
        if column in df.columns:
            df[column] = df[column].fillna("").astype(str)

    if "text" not in df.columns:
        df["text"] = ""

    paths = df.get("path")
    if paths is None:
        paths = pd.Series([""] * len(df))
    else:
        paths = paths.fillna("").astype(str)

    exts = df.get("ext")
    if exts is None:
        exts = pd.Series([""] * len(df))
    else:
        exts = exts.fillna("").astype(str)

    drives = df.get("drive")
    if drives is None:
        drives = pd.Series([""] * len(df))
    else:
        drives = drives.fillna("").astype(str)

    base_texts = df["text"].tolist()
    metadata_list = [
        _metadata_text(paths.iat[idx], exts.iat[idx], drives.iat[idx])
        for idx in range(len(df))
    ]
    df[MODEL_TEXT_COLUMN] = [
        _compose_model_text(base_texts[idx], metadata_list[idx])
        for idx in range(len(df))
    ]
    return df

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

    @staticmethod
    def _sanitize_texts(texts: List[Any]) -> List[str]:
        cleaned: List[str] = []
        for raw in texts:
            if raw is None:
                cleaned.append("")
                continue
            if pd is not None:
                try:
                    if pd.isna(raw):
                        cleaned.append("")
                        continue
                except Exception:
                    pass
            if isinstance(raw, (float, np.floating)):
                if np.isnan(raw):
                    cleaned.append("")
                    continue
            try:
                if raw != raw:  # NaN check (handles e.g. float('nan'))
                    cleaned.append("")
                    continue
            except Exception:
                cleaned.append("")
                continue
            cleaned.append(str(raw))
        return cleaned

    def encode_docs(self, texts: List[str]) -> np.ndarray:
        clean_texts = self._sanitize_texts(texts)
        X = self.tfidf.transform(clean_texts)
        Z = self.svd.transform(X)
        return Z.astype(np.float32, copy=False)

    def encode_query(self, query: str) -> np.ndarray:
        clean_query = self._sanitize_texts([query])
        Xq = self.tfidf.transform(clean_query)
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

    def build(self, embeddings: np.ndarray, paths: List[str], exts: List[str], preview_texts: List[str]):
        if embeddings.ndim != 2:
            raise ValueError("embeddings는 2차원이어야 합니다.")
        self.Z = self._normalize_rows(embeddings)
        self.paths = list(paths)
        self.exts = list(exts)
        self.preview = []
        for text in preview_texts:
            src = (text or "").strip()
            if len(src) > 180:
                self.preview.append(src[:180] + "…")
            else:
                self.preview.append(src)

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

    def search(self, qvec: np.ndarray, top_k: int = 5, *, oversample: int = 1) -> List[Dict[str, Any]]:
        if self.Z is None:
            raise RuntimeError("인덱스가 로드되지 않았습니다.")
        qv = qvec.reshape(1, -1)
        qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-12)
        sims = (self.Z @ qv.T).ravel()
        fetch = max(1, min(len(sims), top_k if oversample <= 1 else top_k * oversample))
        idx = np.argpartition(-sims, kth=fetch - 1)[:fetch]
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
            cached_dim = self.index.Z.shape[1] if self.index.Z is not None else 0
            model_dim = getattr(self.encoder.svd, "n_components", None)
            if model_dim is None:
                components = getattr(self.encoder.svd, "components_", None)
                if components is not None:
                    model_dim = components.shape[0]
            if cached_dim and model_dim and cached_dim != model_dim:
                print(
                    f"⚠️ 인덱스 차원({cached_dim})과 모델 차원({model_dim})이 달라 재생성합니다."
                )
                rebuild = True
            else:
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

        df = df.copy()
        _prepare_text_frame(df)

        if MODEL_TEXT_COLUMN not in df.columns:
            raise RuntimeError("코퍼스에 학습 텍스트 컬럼이 없습니다.")

        mask = df[MODEL_TEXT_COLUMN].str.len() > 0
        work = df.loc[mask].copy()
        if len(work) == 0:
            raise RuntimeError("유효 텍스트 문서가 없습니다.")

        print(f"🧠 문서 임베딩 생성… (docs={len(work):,})")
        Z = self.encoder.encode_docs(work[MODEL_TEXT_COLUMN].tolist())
        preview_series = work["text_original"] if "text_original" in work.columns else work["text"]
        preview_list = preview_series.fillna("").astype(str).tolist()
        self.index.build(Z, work["path"].tolist(), work["ext"].tolist(), preview_list)
        paths = self.index.save(self.cache_dir)
        print(f"💾 인덱스 저장: {paths.emb_npy}, {paths.meta_json}")
        self._ready = True

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self._ready:
            self.ready(False)
        available_exts: Set[str] = set()
        for ext in self.index.exts:
            normalized = _normalize_ext(ext)
            if normalized:
                available_exts.add(normalized)
        requested_exts = _extract_query_exts(query, available_exts=available_exts)
        oversample = 4 if requested_exts else 1
        q = self.encoder.encode_query(query)
        raw_hits = self.index.search(q, top_k=max(top_k, 1), oversample=oversample)
        return _prioritize_ext_hits(raw_hits, desired_exts=requested_exts, top_k=top_k)

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
