# retriever.py  (Step3: 검색기)
from __future__ import annotations
import os, sys, json, time, importlib, types, re, math, calendar
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple

from datetime import datetime, timedelta

import numpy as np
from index_manager import IndexManager

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except Exception:
    SentenceTransformer = None
    CrossEncoder = None

try:
    import torch
except Exception:
    torch = None

MODEL_TEXT_COLUMN = "text_model"
_META_SPLIT_RE = re.compile(r"[^0-9A-Za-z가-힣]+")
MODEL_TYPE_SENTENCE_TRANSFORMER = "sentence-transformer"
DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _normalize_ext(ext: str) -> str:
    if not ext:
        return ""
    ext = ext.strip().lower()
    if not ext:
        return ""
    if not ext.startswith('.'):
        ext = f".{ext}"
    return ext


def _looks_like_extension(token: str) -> bool:
    if not token:
        return False
    if token.startswith('.'):
        return True
    if len(token) > 5:
        return False
    ascii_token = token.replace('-', '')
    return ascii_token.isascii() and ascii_token.isalnum()


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

_DOMAIN_KEYWORDS_BY_EXT: Dict[str, Set[str]] = {}
for _keyword, _exts in _DOMAIN_EXT_HINTS.items():
    for _ext in _exts:
        normalized_ext = _normalize_ext(_ext)
        if not normalized_ext:
            continue
        bucket = _DOMAIN_KEYWORDS_BY_EXT.setdefault(normalized_ext, set())
        bucket.add(_keyword)

_SEMANTIC_SYNONYMS: Dict[str, Set[str]] = {
    "보고서": {"report", "document", "summary"},
    "자료": {"material", "resource", "document"},
    "계약서": {"contract", "agreement"},
    "회의록": {"meeting", "minutes", "meeting minutes"},
    "예산": {"budget", "financial plan"},
    "발표": {"presentation", "slide", "deck"},
    "제안서": {"proposal", "pitch", "offer"},
    "계획": {"plan", "planning"},
    "정리": {"summary", "overview"},
}

_LEXICAL_WEIGHT = 0.25
_EXTENSION_MATCH_BONUS = 0.05


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
    if not hits:
        return []

    def _sort_key(entry: Dict[str, Any]) -> tuple:
        ext = _normalize_ext(entry.get("ext"))
        in_desired = 1 if ext in desired_exts else 0
        score = entry.get("score", entry.get("similarity", 0.0))
        return in_desired, score

    ordered = sorted(hits, key=_sort_key, reverse=True)
    return ordered[:top_k]


def _split_tokens(source: str) -> List[str]:
    if not source:
        return []
    return [tok for tok in _META_SPLIT_RE.split(source) if tok]


def _extension_related_tokens(ext: str) -> Set[str]:
    normalized = _normalize_ext(ext)
    if not normalized:
        return set()

    related: Set[str] = set()
    base = normalized.lstrip(".")
    if base:
        related.add(base)

    for keyword in _EXT_SYNONYMS.get(normalized, set()):
        keyword_norm = keyword.strip().lower()
        if not keyword_norm:
            continue
        related.add(keyword_norm)
        related.update(_split_tokens(keyword_norm))

    for domain_keyword in _DOMAIN_KEYWORDS_BY_EXT.get(normalized, set()):
        keyword_norm = domain_keyword.strip().lower()
        if not keyword_norm:
            continue
        related.add(keyword_norm)
        related.update(_split_tokens(keyword_norm))

    return {tok for tok in related if tok}


def _expand_query_text(query: str) -> str:
    lowered = query.lower()
    tokens = {tok for tok in _split_tokens(lowered) if tok}
    expansions: Set[str] = set()

    for token in tokens:
        ext_token = None
        if _looks_like_extension(token):
            ext_token = _normalize_ext(token)
        if ext_token:
            expansions.update(_extension_related_tokens(ext_token))
            continue

        for ext, synonyms in _EXT_SYNONYMS.items():
            if token in synonyms:
                expansions.add(ext)
                expansions.update(_extension_related_tokens(ext))

        if token in _SEMANTIC_SYNONYMS:
            expansions.update(_SEMANTIC_SYNONYMS[token])

    for keyword, hinted_exts in _DOMAIN_EXT_HINTS.items():
        if keyword in lowered or keyword in tokens:
            expansions.add(keyword)
            for ext in hinted_exts:
                expansions.update(_extension_related_tokens(ext))

    cleaned_expansions = [word for word in expansions if word and word not in tokens]
    if not cleaned_expansions:
        return query
    return f"{query} {' '.join(sorted(set(cleaned_expansions)))}"


def _time_tokens(epoch: Optional[float]) -> List[str]:
    if not epoch:
        return []
    try:
        dt = datetime.fromtimestamp(float(epoch))
    except Exception:
        return []
    return [
        dt.strftime("%Y"),
        dt.strftime("%Y-%m"),
        dt.strftime("%Y-%m-%d"),
        dt.strftime("%B"),
        dt.strftime("%m"),
    ]


def _size_bucket(size: Optional[int]) -> Optional[str]:
    if size is None:
        return None
    try:
        size = int(size)
    except (TypeError, ValueError):
        return None
    if size <= 0:
        return None
    if size < 10 * 1024:
        return "size:tiny"
    if size < 1 * 1024 * 1024:
        return "size:small"
    if size < 10 * 1024 * 1024:
        return "size:medium"
    if size < 50 * 1024 * 1024:
        return "size:large"
    return "size:huge"


def _metadata_text(
    path: str,
    ext: str,
    drive: str,
    *,
    size: Optional[int] = None,
    mtime: Optional[float] = None,
    ctime: Optional[float] = None,
    owner: Optional[str] = None,
) -> str:
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
            tokens.extend(_extension_related_tokens(ext_clean))
    if drive:
        drive_str = str(drive)
        tokens.append(drive_str)
        tokens.extend(_split_tokens(drive_str))
    for epoch in (mtime, ctime):
        tokens.extend(_time_tokens(epoch))
    bucket = _size_bucket(size)
    if bucket:
        tokens.append(bucket)
    if owner:
        tokens.append(str(owner))
        tokens.extend(_split_tokens(str(owner)))

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


def _collect_hit_tokens(hit: Dict[str, Any]) -> Set[str]:
    tokens: Set[str] = set()

    path = hit.get("path")
    if path:
        path_text = str(path).lower()
        tokens.add(path_text)
        tokens.update(_split_tokens(path_text))

    preview = hit.get("preview")
    if preview:
        preview_text = str(preview).lower()
        tokens.add(preview_text)
        tokens.update(_split_tokens(preview_text))

    tokens.update(_extension_related_tokens(hit.get("ext", "")))
    return {tok for tok in tokens if tok}


def _lexical_overlap_score(query_tokens: Set[str], hit_tokens: Set[str]) -> float:
    if not query_tokens or not hit_tokens:
        return 0.0

    total = 0.0
    for token in query_tokens:
        if token in hit_tokens:
            total += 1.0
            continue

        for candidate in hit_tokens:
            if token in candidate or candidate in token:
                total += 0.5
                break

    return total / max(len(query_tokens), 1)


def _compose_rerank_document(hit: Dict[str, Any]) -> str:
    sections: List[str] = []
    path = str(hit.get("path") or "").strip()
    if path:
        sections.append(f"파일 경로: {path}")

    ext = str(hit.get("ext") or "").strip()
    if ext:
        sections.append(f"확장자: {ext}")

    drive = str(hit.get("drive") or "").strip()
    if drive:
        sections.append(f"드라이브: {drive}")

    owner = str(hit.get("owner") or "").strip()
    if owner:
        sections.append(f"작성자: {owner}")

    mtime_label = _format_human_time(hit.get("mtime"))
    if mtime_label:
        sections.append(f"수정일: {mtime_label}")

    size_label = _format_size(hit.get("size"))
    if size_label:
        sections.append(f"파일 크기: {size_label}")

    preview = str(hit.get("preview") or "").strip()
    if preview:
        sections.append(preview)

    return "\n".join(section for section in sections if section)


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        *,
        device: Optional[str] = None,
        batch_size: int = 16,
    ) -> None:
        if CrossEncoder is None:
            raise RuntimeError("sentence-transformers의 CrossEncoder를 사용할 수 없습니다.")

        self.model_name = model_name
        self.device = device or None
        self.batch_size = max(1, int(batch_size) if batch_size else 1)

        load_kwargs: Dict[str, Any] = {}
        if self.device:
            load_kwargs["device"] = self.device

        t0 = time.time()
        self.model = CrossEncoder(model_name, **load_kwargs)
        dt = time.time() - t0
        device_label = self.device or getattr(self.model, "device", "cpu")
        print(f"✨ 재랭킹 모델 로드 완료: {model_name} (device={device_label}, {dt:.1f}s)")

    def rerank(
        self,
        query: str,
        hits: List[Dict[str, Any]],
        *,
        desired_exts: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        if not hits:
            return []

        pairs: List[List[str]] = []
        prepared_hits: List[Dict[str, Any]] = []

        for hit in hits:
            doc_text = _compose_rerank_document(hit)
            pairs.append([query, doc_text])
            prepared_hits.append(dict(hit))

        try:
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            print(f"⚠️ 재랭킹 예측 중 오류가 발생해 기본 점수를 사용합니다: {exc}")
            return hits

        ext_preferences = desired_exts or set()

        scored: List[Dict[str, Any]] = []
        for hit, score in zip(prepared_hits, scores.tolist() if hasattr(scores, "tolist") else scores):
            base_similarity = float(hit.get("similarity", 0.0))
            if "vector_similarity" not in hit:
                hit["vector_similarity"] = base_similarity
            final_score = float(score)
            ext_bonus = 0.0
            ext = _normalize_ext(hit.get("ext"))
            if ext and ext in ext_preferences:
                ext_bonus = _EXTENSION_MATCH_BONUS
                final_score += ext_bonus
            hit["rerank_score"] = float(score)
            hit["similarity"] = final_score
            hit["score"] = final_score
            if ext_bonus:
                hit["rerank_ext_bonus"] = ext_bonus
            scored.append(hit)

        scored.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return scored


def _rerank_hits(
    raw_query: str,
    expanded_query: str,
    hits: List[Dict[str, Any]],
    *,
    desired_exts: Set[str],
    top_k: int,
) -> List[Dict[str, Any]]:
    if not hits:
        return []

    base_tokens = {tok for tok in _split_tokens(raw_query.lower()) if tok}
    expanded_tokens = {tok for tok in _split_tokens(expanded_query.lower()) if tok}
    query_tokens = base_tokens or expanded_tokens
    synonym_tokens = expanded_tokens - base_tokens
    scored_hits: List[Dict[str, Any]] = []

    for raw_hit in hits:
        hit = dict(raw_hit)
        hit_tokens = _collect_hit_tokens(hit)
        lexical_score = _lexical_overlap_score(query_tokens, hit_tokens)
        if synonym_tokens:
            synonym_score = _lexical_overlap_score(synonym_tokens, hit_tokens)
            lexical_score = max(lexical_score, synonym_score)
        base_similarity = float(hit.get("similarity", 0.0))
        ext = _normalize_ext(hit.get("ext"))
        ext_bonus = _EXTENSION_MATCH_BONUS if ext in desired_exts else 0.0
        hit["lexical_score"] = lexical_score
        hit["vector_similarity"] = base_similarity
        hit["score"] = base_similarity + (lexical_score * _LEXICAL_WEIGHT) + ext_bonus
        scored_hits.append(hit)

    scored_hits.sort(key=lambda item: item.get("score", item.get("similarity", 0.0)), reverse=True)
    return _prioritize_ext_hits(scored_hits, desired_exts=desired_exts, top_k=top_k)


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

    sizes = df.get("size")
    if sizes is None:
        sizes = pd.Series([0] * len(df))
    else:
        sizes = sizes.fillna(0)

    mtimes = df.get("mtime")
    if mtimes is None:
        mtimes = pd.Series([0.0] * len(df))
    else:
        mtimes = mtimes.fillna(0.0)

    ctimes = df.get("ctime")
    if ctimes is None:
        ctimes = pd.Series([0.0] * len(df))
    else:
        ctimes = ctimes.fillna(0.0)

    owners = df.get("owner")
    if owners is None:
        owners = pd.Series([""] * len(df))
    else:
        owners = owners.fillna("").astype(str)

    base_texts = df["text"].tolist()
    metadata_list = [
        _metadata_text(
            paths.iat[idx],
            exts.iat[idx],
            drives.iat[idx],
            size=sizes.iat[idx],
            mtime=mtimes.iat[idx],
            ctime=ctimes.iat[idx],
            owner=owners.iat[idx],
        )
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
PARQUET_ENGINE: Optional[str] = None
if pd is not None:
    for candidate in ("fastparquet", "pyarrow"):
        try:
            importlib.import_module(candidate)
            PARQUET_ENGINE = candidate
            break
        except ImportError:
            continue

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

        self.model_type = MODEL_TYPE_SENTENCE_TRANSFORMER
        self.embedding_dim: Optional[int] = None
        self.embedder: Optional[SentenceTransformer] = None
        self.pipeline = None
        self.tfidf = None
        self.svd = None

        if isinstance(obj, dict) and obj.get("model_type") == MODEL_TYPE_SENTENCE_TRANSFORMER:
            if SentenceTransformer is None:
                raise RuntimeError(
                    "sentence-transformers 라이브러리가 필요합니다. pip install sentence-transformers"
                )
            model_name = obj.get("model_name") or DEFAULT_EMBED_MODEL
            print(f"🔌 Sentence-BERT 로드: {model_name}", flush=True)
            self.embedder = SentenceTransformer(model_name)
            detected_dim = obj.get("embedding_dim")
            if detected_dim:
                try:
                    self.embedding_dim = int(detected_dim)
                except (TypeError, ValueError):
                    self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
            else:
                self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
            self.cluster_model = obj.get("cluster_model")
            self.train_config = obj.get("train_config")
        else:
            self.model_type = "tfidf"
            self.pipeline = obj["pipeline"]
            self.tfidf = self.pipeline.named_steps["tfidf"]
            self.svd = self.pipeline.named_steps["svd"]
            self.embedding_dim = getattr(self.svd, "n_components", None)
            self.cluster_model = None
            self.train_config = obj.get("cfg") if isinstance(obj, dict) else None

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
        if self.model_type == MODEL_TYPE_SENTENCE_TRANSFORMER and self.embedder is not None:
            embeddings = self.embedder.encode(
                clean_texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            if isinstance(embeddings, list):
                embeddings = np.asarray(embeddings, dtype=np.float32)
            return np.asarray(embeddings, dtype=np.float32)

        if self.tfidf is None or self.svd is None:
            raise RuntimeError("TF-IDF 파이프라인이 초기화되지 않았습니다.")
        X = self.tfidf.transform(clean_texts)
        Z = self.svd.transform(X)
        return Z.astype(np.float32, copy=False)

    def encode_query(self, query: str) -> np.ndarray:
        clean_query = self._sanitize_texts([query])
        if self.model_type == MODEL_TYPE_SENTENCE_TRANSFORMER and self.embedder is not None:
            Zq = self.embedder.encode(
                clean_query,
                batch_size=8,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            if isinstance(Zq, list):
                Zq = np.asarray(Zq, dtype=np.float32)
            return np.asarray(Zq, dtype=np.float32)

        if self.tfidf is None or self.svd is None:
            raise RuntimeError("TF-IDF 파이프라인이 초기화되지 않았습니다.")
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
        self.sizes: List[Optional[int]] = []
        self.mtimes: List[Optional[float]] = []
        self.ctimes: List[Optional[float]] = []
        self.owners: List[Optional[str]] = []

    @staticmethod
    def _normalize_rows(M: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
        return (M / norms).astype(np.float32, copy=False)

    def build(
        self,
        embeddings: np.ndarray,
        paths: List[str],
        exts: List[str],
        preview_texts: List[str],
        *,
        sizes: Optional[List[Optional[int]]] = None,
        mtimes: Optional[List[Optional[float]]] = None,
        ctimes: Optional[List[Optional[float]]] = None,
        owners: Optional[List[Optional[str]]] = None,
    ):
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
        total = len(self.paths)
        def _safe_list(values, fallback=None):
            if values is None:
                return [fallback] * total
            if len(values) != total:
                raise ValueError("메타데이터 길이가 문서 수와 다릅니다.")
            return list(values)

        self.sizes = _safe_list(sizes, 0)
        self.mtimes = _safe_list(mtimes, 0.0)
        self.ctimes = _safe_list(ctimes, 0.0)
        self.owners = _safe_list(owners, "")

    def save(self, out_dir: Path) -> IndexPaths:
        out_dir.mkdir(parents=True, exist_ok=True)
        emb_path = out_dir / "doc_embeddings.npy"
        meta_path = out_dir / "doc_meta.json"
        if self.Z is None:
            raise RuntimeError("인덱스가 비어있습니다. build() 후 저장하세요.")
        np.save(emb_path, self.Z.astype(np.float32, copy=False))
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "paths": self.paths,
                    "exts": self.exts,
                    "preview": self.preview,
                    "sizes": self.sizes,
                    "mtimes": self.mtimes,
                    "ctimes": self.ctimes,
                    "owners": self.owners,
                },
                f,
                ensure_ascii=False,
            )
        return IndexPaths(emb_npy=emb_path, meta_json=meta_path)

    def load(self, emb_npy: Path, meta_json: Path, *, use_mmap: bool = True):
        mmap_mode = "r" if use_mmap else None
        self.Z = np.load(emb_npy, mmap_mode=mmap_mode)
        if self.Z.dtype != np.float32:
            self.Z = self.Z.astype(np.float32, copy=False)
        with meta_json.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        self.paths = meta["paths"]
        self.exts = meta["exts"]
        self.preview = meta["preview"]
        total = len(self.paths)
        self.sizes = meta.get("sizes", [0] * total)
        self.mtimes = meta.get("mtimes", [0.0] * total)
        self.ctimes = meta.get("ctimes", [0.0] * total)
        self.owners = meta.get("owners", [""] * total)
        if len(self.sizes) != total:
            self.sizes = [0] * total
        if len(self.mtimes) != total:
            self.mtimes = [0.0] * total
        if len(self.ctimes) != total:
            self.ctimes = [0.0] * total
        if len(self.owners) != total:
            self.owners = [""] * total
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
        results: List[Dict[str, Any]] = []
        for i in idx:
            results.append(
                {
                    "path": self.paths[i],
                    "ext": self.exts[i],
                    "similarity": float(sims[i]),
                    "preview": self.preview[i],
                    "size": self.sizes[i] if i < len(self.sizes) else None,
                    "mtime": self.mtimes[i] if i < len(self.mtimes) else None,
                    "ctime": self.ctimes[i] if i < len(self.ctimes) else None,
                    "owner": self.owners[i] if i < len(self.owners) else None,
                }
            )
        return results


# =========================
# Metadata filters
# =========================


@dataclass
class MetadataFilters:
    mtime_from: Optional[float] = None
    mtime_to: Optional[float] = None
    ctime_from: Optional[float] = None
    ctime_to: Optional[float] = None
    size_min: Optional[int] = None
    size_max: Optional[int] = None
    owners: Set[str] = field(default_factory=set)

    def is_active(self) -> bool:
        return any(
            value is not None
            for value in [
                self.mtime_from,
                self.mtime_to,
                self.ctime_from,
                self.ctime_to,
                self.size_min,
                self.size_max,
            ]
        ) or bool(self.owners)

    def matches(self, hit: Dict[str, Any]) -> bool:
        if not self.is_active():
            return True
        mtime = _to_float(hit.get("mtime"))
        ctime = _to_float(hit.get("ctime"))
        size = _to_int(hit.get("size"))
        owner = str(hit.get("owner") or "").lower()

        if self.mtime_from is not None and (mtime is None or mtime < self.mtime_from):
            return False
        if self.mtime_to is not None and (mtime is None or mtime > self.mtime_to):
            return False
        if self.ctime_from is not None and (ctime is None or ctime < self.ctime_from):
            return False
        if self.ctime_to is not None and (ctime is None or ctime > self.ctime_to):
            return False
        if self.size_min is not None and (size is None or size < self.size_min):
            return False
        if self.size_max is not None and (size is None or size > self.size_max):
            return False
        if self.owners:
            if not owner:
                return False
            if owner not in self.owners:
                return False
        return True


def _to_float(value: Any) -> Optional[float]:
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _year_bounds(year: int) -> Tuple[float, float]:
    start = datetime(year, 1, 1, 0, 0, 0)
    end = datetime(year, 12, 31, 23, 59, 59)
    return start.timestamp(), end.timestamp()


def _month_bounds(dt: datetime, months_ago: int = 0) -> Tuple[float, float]:
    base = _shift_months(dt, months_ago)
    start = base.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    last_day = calendar.monthrange(start.year, start.month)[1]
    end = start.replace(day=last_day, hour=23, minute=59, second=59)
    return start.timestamp(), end.timestamp()


def _shift_months(dt: datetime, months: int) -> datetime:
    year = dt.year
    month = dt.month - months
    while month <= 0:
        month += 12
        year -= 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


def _approx_range(center: float, tolerance_ratio: float = 0.2) -> Tuple[float, float]:
    delta = center * tolerance_ratio
    return max(0.0, center - delta), center + delta


def _parse_size_expression(value: str, unit: str) -> int:
    unit = unit.lower()
    base = float(value)
    multiplier = {
        "kb": 1024,
        "mb": 1024 ** 2,
        "gb": 1024 ** 3,
        "tb": 1024 ** 4,
    }.get(unit, 1)
    return int(base * multiplier)


def _normalize_owner(owner: str) -> str:
    return owner.strip().lower()


def _extract_metadata_filters(query: str) -> MetadataFilters:
    filters = MetadataFilters()
    lowered = query.lower()
    now = datetime.now()

    # Year-based filters (e.g., 2021년)
    year_match = re.search(r"(20\d{2}|19\d{2})\s*년", query)
    if year_match:
        year = int(year_match.group(1))
        filters.mtime_from, filters.mtime_to = _year_bounds(year)

    # Relative year (e.g., 3년 전)
    rel_year = re.search(r"(\d+)\s*년\s*전", query)
    if rel_year:
        years = int(rel_year.group(1))
        target_year = now.year - years
        filters.mtime_from, filters.mtime_to = _year_bounds(target_year)

    if "작년" in lowered:
        filters.mtime_from, filters.mtime_to = _year_bounds(now.year - 1)
    if "재작년" in lowered:
        filters.mtime_from, filters.mtime_to = _year_bounds(now.year - 2)
    if "올해" in lowered or "올 해" in lowered or "금년" in lowered:
        filters.mtime_from, filters.mtime_to = _year_bounds(now.year)

    # Month-based filters
    rel_month = re.search(r"(\d+)\s*개월\s*전", query)
    if rel_month:
        months = int(rel_month.group(1))
        filters.mtime_from, filters.mtime_to = _month_bounds(now, months)
    if "지난달" in lowered:
        filters.mtime_from, filters.mtime_to = _month_bounds(now, 1)
    if "이번달" in lowered or "이 달" in lowered:
        filters.mtime_from, filters.mtime_to = _month_bounds(now, 0)

    if any(keyword in lowered for keyword in ["최근", "요즘", "요근래", "최근에"]):
        horizon = now - timedelta(days=180)
        filters.mtime_from = horizon.timestamp()

    # Size expressions (e.g., 10MB 이상)
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(kb|mb|gb|tb)\s*(이상|이하|초과|미만|보다 큰|보다 작은)?", lowered):
        value = match.group(1)
        unit = match.group(2)
        qualifier = match.group(3) or ""
        size_bytes = _parse_size_expression(value, unit)
        if any(token in qualifier for token in ["이상", "초과", "보다 큰", "at least", "over"]):
            filters.size_min = max(filters.size_min or 0, size_bytes)
        elif any(token in qualifier for token in ["이하", "미만", "보다 작은", "at most", "under"]):
            filters.size_max = min(filters.size_max or size_bytes, size_bytes)
        else:
            approx_min, approx_max = _approx_range(size_bytes)
            filters.size_min = max(filters.size_min or 0, int(approx_min))
            filters.size_max = min(filters.size_max or int(approx_max), int(approx_max))

    # Owner expressions (작성자 홍길동 / author:alice / @username)
    for match in re.finditer(r"(?:작성자|author|owner)[:\s]+([\w가-힣@.]+)", query, re.IGNORECASE):
        filters.owners.add(_normalize_owner(match.group(1)))
    for mention in re.findall(r"@([\w가-힣._-]+)", query):
        filters.owners.add(_normalize_owner(mention))

    return filters


def _apply_metadata_filters(hits: List[Dict[str, Any]], filters: MetadataFilters) -> List[Dict[str, Any]]:
    if not filters.is_active():
        return hits
    filtered = [hit for hit in hits if filters.matches(hit)]
    return filtered


def _format_human_time(epoch: Any) -> str:
    value = _to_float(epoch)
    if value is None or value <= 0:
        return ""
    try:
        return datetime.fromtimestamp(value).strftime("%Y-%m-%d")
    except Exception:
        return ""


def _format_size(size: Any) -> str:
    num = _to_int(size)
    if num is None or num <= 0:
        return ""
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)}{unit}"
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{num}B"


def _similarity_to_percent(value: Any, *, decimals: int = 1) -> str:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return "-"
    score = max(0.0, min(score, 1.0))
    pct = score * 100.0
    return f"{pct:.{decimals}f}%"


def _pick_rerank_device(requested: Optional[str]) -> str:
    if requested:
        return str(requested)
    if torch is not None:
        try:
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
    return "cpu"


# =========================
# Retriever
# =========================
class Retriever:
    def __init__(
        self,
        model_path: Path,
        corpus_path: Path,
        cache_dir: Path = Path("./index_cache"),
        *,
        search_wait_timeout: float = 0.5,
        use_rerank: bool = False,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rerank_depth: int = 80,
        rerank_batch_size: int = 16,
        rerank_device: Optional[str] = None,
    ):
        self.model_path = Path(model_path)
        self.corpus_path = Path(corpus_path)
        self.cache_dir = Path(cache_dir)
        self.encoder = QueryEncoder(self.model_path)
        self.search_wait_timeout = search_wait_timeout
        depth = max(0, int(rerank_depth))
        self.rerank_depth = depth
        self.rerank_model = rerank_model
        self.rerank_batch_size = max(1, int(rerank_batch_size) if rerank_batch_size else 1)
        self.rerank_device = rerank_device or None
        self.use_rerank = bool(use_rerank and depth > 0)
        self._reranker: Optional[CrossEncoderReranker] = None
        self.index_manager = IndexManager(
            loader=self._load_cached_index,
            builder=self._rebuild_index,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def ready(self, rebuild: bool = False, *, wait: bool = True) -> bool:
        if rebuild:
            self.index_manager.schedule_rebuild(priority=True)
        else:
            self.index_manager.ensure_loaded()
        if wait:
            self.index_manager.wait_until_ready()
        return self.index_manager.get_index(wait=False) is not None

    def wait_until_ready(self, timeout: Optional[float] = None) -> bool:
        return self.index_manager.wait_until_ready(timeout=timeout)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        index = self.index_manager.get_index(wait=False)
        if index is None:
            self.index_manager.ensure_loaded()
            if not self.index_manager.wait_until_ready(timeout=self.search_wait_timeout):
                return []
            index = self.index_manager.get_index(wait=False)
            if index is None:
                return []

        available_exts: Set[str] = set()
        for ext in index.exts:
            normalized = _normalize_ext(ext)
            if normalized:
                available_exts.add(normalized)
        requested_exts = _extract_query_exts(query, available_exts=available_exts)
        metadata_filters = _extract_metadata_filters(query)
        oversample = 4 if requested_exts else 2
        if metadata_filters.is_active():
            oversample = max(oversample, 8)
        vector_query = _expand_query_text(query)
        q = self.encoder.encode_query(vector_query)
        use_rerank = bool(getattr(self, "use_rerank", False))
        rerank_depth = int(getattr(self, "rerank_depth", 0) or 0)
        search_top_k = max(top_k, 1)
        search_oversample = oversample
        if use_rerank and rerank_depth:
            search_top_k = max(search_top_k, rerank_depth)
            search_oversample = max(1, min(oversample, 2))
        raw_hits = index.search(q, top_k=search_top_k, oversample=search_oversample)
        filtered_hits = _apply_metadata_filters(raw_hits, metadata_filters)
        if not filtered_hits:
            return []

        lexical_limit = max(top_k, 1)
        if use_rerank and rerank_depth:
            lexical_limit = max(lexical_limit, min(rerank_depth, len(filtered_hits)))

        lexical_ranking = _rerank_hits(
            query,
            vector_query,
            filtered_hits,
            desired_exts=requested_exts,
            top_k=lexical_limit,
        )

        if not use_rerank:
            return lexical_ranking[:top_k]

        reranker = self._ensure_reranker()
        if reranker is None:
            return lexical_ranking[:top_k]

        reranked = reranker.rerank(query, lexical_ranking, desired_exts=requested_exts)
        return reranked[:top_k]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_reranker(self) -> Optional[CrossEncoderReranker]:
        if not getattr(self, "use_rerank", False):
            return None
        existing = getattr(self, "_reranker", None)
        if existing is not None:
            return existing

        if CrossEncoder is None:
            print("⚠️ CrossEncoder 클래스를 불러올 수 없어 재랭킹을 비활성화합니다.")
            self.use_rerank = False
            return None

        try:
            reranker = CrossEncoderReranker(
                getattr(self, "rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
                device=_pick_rerank_device(getattr(self, "rerank_device", None)),
                batch_size=int(getattr(self, "rerank_batch_size", 16) or 16),
            )
            self._reranker = reranker
        except Exception as exc:
            print(f"⚠️ 재랭킹 모델 로드 실패로 벡터 점수만 사용합니다: {exc}")
            self.use_rerank = False
            self._reranker = None
        return getattr(self, "_reranker", None)

    def _load_cached_index(self) -> Optional[VectorIndex]:
        emb_npy = self.cache_dir / "doc_embeddings.npy"
        meta_json = self.cache_dir / "doc_meta.json"
        if not emb_npy.exists() or not meta_json.exists():
            return None

        index = VectorIndex()
        try:
            index.load(emb_npy, meta_json, use_mmap=True)
        except Exception as exc:
            print(f"⚠️ 인덱스 로드 실패로 재생성을 시도합니다: {exc}")
            return None

        if not self._index_matches_model(index):
            print("⚠️ 모델 차수와 인덱스 차수가 달라 재생성을 진행합니다.")
            return None

        print(f"✅ 인덱스 로드: {self.cache_dir}")
        return index

    def _rebuild_index(self) -> VectorIndex:
        if pd is None:
            raise RuntimeError("pandas 필요. pip install pandas")

        df = self._load_corpus().copy()
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

        size_list = work["size"].fillna(0).astype(int).tolist() if "size" in work.columns else [0] * len(work)
        mtime_list = work["mtime"].fillna(0.0).astype(float).tolist() if "mtime" in work.columns else [0.0] * len(work)
        ctime_list = work["ctime"].fillna(0.0).astype(float).tolist() if "ctime" in work.columns else [0.0] * len(work)
        owner_list = work["owner"].fillna("").astype(str).tolist() if "owner" in work.columns else [""] * len(work)

        index = VectorIndex()
        index.build(
            Z,
            work["path"].tolist(),
            work["ext"].tolist(),
            preview_list,
            sizes=size_list,
            mtimes=mtime_list,
            ctimes=ctime_list,
            owners=owner_list,
        )
        paths = index.save(self.cache_dir)
        print(f"💾 인덱스 저장: {paths.emb_npy}, {paths.meta_json}")

        fresh = VectorIndex()
        fresh.load(paths.emb_npy, paths.meta_json, use_mmap=True)
        return fresh

    def _load_corpus(self):
        print("📥 코퍼스 로드…")
        if self.corpus_path.suffix.lower() == ".parquet":
            engine_kwargs = {}
            engine_label = PARQUET_ENGINE or "auto"
            if PARQUET_ENGINE:
                engine_kwargs["engine"] = PARQUET_ENGINE
            try:
                return pd.read_parquet(self.corpus_path, **engine_kwargs)
            except Exception as exc:
                print(
                    f"⚠️ Parquet 로드 실패(engine={engine_label}): {exc} → CSV로 재시도",
                    flush=True,
                )
                return pd.read_csv(self.corpus_path.with_suffix(".csv"))
        return pd.read_csv(self.corpus_path)

    def _index_matches_model(self, index: VectorIndex) -> bool:
        if index.Z is None:
            return False
        cached_dim = index.Z.shape[1]
        model_dim = getattr(self.encoder, "embedding_dim", None)
        if model_dim is None and getattr(self.encoder, "svd", None) is not None:
            model_dim = getattr(self.encoder.svd, "n_components", None)
            if model_dim is None:
                components = getattr(self.encoder.svd, "components_", None)
                if components is not None:
                    model_dim = components.shape[0]
        if cached_dim and model_dim and cached_dim != model_dim:
            return False
        return True

    @staticmethod
    def format_results(query: str, results: List[Dict[str, Any]]) -> str:
        if not results:
            return f"“{query}”와 유사한 문서를 찾지 못했습니다."
        lines = [f"‘{query}’와 유사한 문서 Top {len(results)}:"]
        for i, r in enumerate(results, 1):
            similarity_label = _similarity_to_percent(r.get("similarity", r.get("vector_similarity")))
            lines.append(f"{i}. {r['path']} [{r['ext']}]  유사도={similarity_label}")
            meta_bits: List[str] = []
            mod_date = _format_human_time(r.get("mtime") or r.get("ctime"))
            if mod_date:
                meta_bits.append(f"수정일 {mod_date}")
            size_label = _format_size(r.get("size"))
            if size_label:
                meta_bits.append(size_label)
            owner = str(r.get("owner") or "").strip()
            if owner:
                meta_bits.append(f"작성자 {owner}")
            if meta_bits:
                lines.append("   메타: " + ", ".join(meta_bits))
            if r.get("preview"):
                lines.append(f"   미리보기: {r['preview']}")
        return "\n".join(lines)
