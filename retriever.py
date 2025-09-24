# retriever.py  (Step3: 검색기)
from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
import threading
import time
import types
import weakref
import importlib
import calendar
import hashlib
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover - defensive fallback for minimal envs
    class _NumpyStub:
        ndarray = list
        float32 = float
        int64 = int
        _is_stub = True

        def __getattr__(self, name: str) -> Any:
            raise ModuleNotFoundError(
                "numpy 모듈이 필요합니다. pip install numpy 로 설치 후 다시 시도해 주세요."
            )

    np = _NumpyStub()  # type: ignore[assignment]

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import joblib
except Exception:
    joblib = None

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
except Exception:
    CrossEncoder = None
    SentenceTransformer = None

try:
    import torch
except Exception:
    torch = None

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

from index_manager import IndexManager

MODEL_TEXT_COLUMN = "text_model"
MODEL_TYPE_SENTENCE_TRANSFORMER = "sentence-transformer"
DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MAX_BM25_TOKENS = 8000
MAX_PREVIEW_CHARS = 180
DEFAULT_MMR_LAMBDA = 0.7
RRF_DEFAULT_K = 60
_SESSION_HISTORY_LIMIT = 50
_SESSION_CHAT_HISTORY_LIMIT = 20
_SESSION_PREF_DECAY = 0.85
_SESSION_CLICK_WEIGHT = 0.35
_SESSION_PIN_WEIGHT = 0.6
_SESSION_LIKE_WEIGHT = 0.45
_SESSION_DISLIKE_WEIGHT = -0.5
_SESSION_EXT_PREF_SCALE = 0.05
_SESSION_OWNER_PREF_SCALE = 0.04
_META_SPLIT_RE = re.compile(r"[^0-9A-Za-z가-힣]+")


logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@lru_cache(maxsize=4096)
def _split_tokens_cached(source: str) -> Tuple[str, ...]:
    if not source:
        return ()
    return tuple(tok for tok in _META_SPLIT_RE.split(source) if tok)


def _split_tokens(source: Any) -> List[str]:
    if not source:
        return []
    return list(_split_tokens_cached(str(source)))


def _mask_path(path: str) -> str:
    if not path:
        return ""
    try:
        return Path(path).name
    except Exception:
        return "<invalid-path>"


def _normalize_ext(ext: str) -> str:
    if not ext:
        return ""
    ext = str(ext).strip().lower()
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


def _rescale_inner_product(value: float) -> float:
    if value is None:
        return 0.0
    try:
        if math.isnan(value):
            return 0.0
    except TypeError:
        return 0.0
    scaled = 0.5 * (float(value) + 1.0)
    if scaled < 0.0:
        return 0.0
    if scaled > 1.0:
        return 1.0
    return scaled


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

_DOMAIN_KEYWORDS_BY_EXT: Dict[str, Set[str]] = {}
for keyword, exts in _DOMAIN_EXT_HINTS.items():
    for ext in exts:
        norm = _normalize_ext(ext)
        if not norm:
            continue
        _DOMAIN_KEYWORDS_BY_EXT.setdefault(norm, set()).add(keyword)

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


def _keyword_forms(keyword: str) -> Set[str]:
    base = str(keyword).strip().lower()
    if not base:
        return set()
    forms: Set[str] = {base}
    tokens = _split_tokens(base)
    if tokens:
        forms.add("".join(tokens))
        forms.add(" ".join(tokens))
        forms.update(tokens)
    return {form for form in forms if form}


_EXTENSION_KEYWORD_MAP: Dict[str, Set[str]] = {}
for ext, synonyms in _EXT_SYNONYMS.items():
    normalized_ext = _normalize_ext(ext)
    if not normalized_ext:
        continue
    keyword_pool = set(synonyms)
    keyword_pool.add(normalized_ext)
    keyword_pool.add(normalized_ext.lstrip('.'))
    for keyword in keyword_pool:
        for form in _keyword_forms(keyword):
            bucket = _EXTENSION_KEYWORD_MAP.setdefault(form, set())
            bucket.add(normalized_ext)

_DOMAIN_KEYWORD_MAP: Dict[str, Set[str]] = {}
for keyword, exts in _DOMAIN_EXT_HINTS.items():
    for form in _keyword_forms(keyword):
        mapped_exts = {_normalize_ext(ext) for ext in exts if _normalize_ext(ext)}
        if mapped_exts:
            _DOMAIN_KEYWORD_MAP.setdefault(form, set()).update(mapped_exts)

# 의미 검색만 사용하도록 BM25 가중치를 비활성화
_LEXICAL_WEIGHT = 0.0
_EXTENSION_MATCH_BONUS = 0.05


def _iter_query_units(lowered: str) -> Set[str]:
    tokens = _split_tokens(lowered)
    units: Set[str] = set(tokens)
    units.add(lowered)
    for token in tokens:
        for segment in re.findall(r"[a-z0-9]{1,5}", token):
            if segment:
                units.add(segment)
    length = len(tokens)
    for n in (2, 3):
        if length < n:
            continue
        for i in range(length - n + 1):
            segment = tokens[i:i + n]
            units.add(" ".join(segment))
            units.add("".join(segment))
    return units


def _token_contains(unit: str, keyword: str) -> bool:
    if not unit or not keyword:
        return False
    unit_clean = str(unit).strip().lower()
    keyword_clean = str(keyword).strip().lower()
    if not unit_clean or not keyword_clean:
        return False
    if unit_clean == keyword_clean:
        return True
    unit_tokens = set(_split_tokens(unit_clean))
    if keyword_clean in unit_tokens:
        return True
    keyword_tokens = _split_tokens(keyword_clean)
    if keyword_tokens and all(tok in unit_tokens for tok in keyword_tokens):
        return True
    return False


@dataclass
class SessionState:
    recent_queries: Deque[str] = field(default_factory=lambda: deque(maxlen=_SESSION_HISTORY_LIMIT))
    clicked_doc_ids: Set[int] = field(default_factory=set)
    preferred_exts: Dict[str, float] = field(default_factory=dict)
    owner_prior: Dict[str, float] = field(default_factory=dict)
    chat_history: Deque[Tuple[str, str]] = field(
        default_factory=lambda: deque(maxlen=_SESSION_CHAT_HISTORY_LIMIT)
    )

    def add_query(self, query: str) -> None:
        if not query:
            return
        self.recent_queries.append(query)

    def record_user_message(self, message: str) -> None:
        self._append_chat_turn("user", message)

    def record_assistant_message(self, message: str) -> None:
        self._append_chat_turn("assistant", message)

    def get_chat_history(self) -> List[Tuple[str, str]]:
        return list(self.chat_history)

    def record_click(
        self,
        *,
        doc_id: Optional[int] = None,
        ext: Optional[str] = None,
        owner: Optional[str] = None,
    ) -> None:
        if doc_id is not None:
            self.clicked_doc_ids.add(int(doc_id))
        self._apply_preference(ext=ext, owner=owner, delta=_SESSION_CLICK_WEIGHT)

    def record_pin(
        self,
        *,
        doc_id: Optional[int] = None,
        ext: Optional[str] = None,
        owner: Optional[str] = None,
    ) -> None:
        if doc_id is not None:
            self.clicked_doc_ids.add(int(doc_id))
        self._apply_preference(ext=ext, owner=owner, delta=_SESSION_PIN_WEIGHT)

    def record_like(
        self,
        *,
        ext: Optional[str] = None,
        owner: Optional[str] = None,
    ) -> None:
        self._apply_preference(ext=ext, owner=owner, delta=_SESSION_LIKE_WEIGHT)

    def record_dislike(
        self,
        *,
        ext: Optional[str] = None,
        owner: Optional[str] = None,
    ) -> None:
        self._apply_preference(ext=ext, owner=owner, delta=_SESSION_DISLIKE_WEIGHT)

    def _apply_preference(
        self,
        *,
        ext: Optional[str],
        owner: Optional[str],
        delta: float,
    ) -> None:
        if ext:
            normalized_ext = _normalize_ext(ext)
            if normalized_ext:
                self._update_pref(self.preferred_exts, normalized_ext, delta)
        if owner:
            normalized_owner = _normalize_owner(owner)
            if normalized_owner:
                self._update_pref(self.owner_prior, normalized_owner, delta)

    def _update_pref(self, store: Dict[str, float], key: str, delta: float) -> None:
        current = store.get(key, 0.0) * _SESSION_PREF_DECAY
        updated = _clamp(current + delta, -1.0, 1.0)
        if abs(updated) < 1e-4:
            store.pop(key, None)
        else:
            store[key] = updated

    def _append_chat_turn(self, role: str, message: str) -> None:
        text = (message or "").strip()
        if not text:
            return
        normalized_role = role if role in {"user", "assistant"} else "assistant"
        self.chat_history.append((normalized_role, text))

@dataclass
class EarlyStopConfig:
    score_threshold: float = 0.05
    window_size: int = 0
    patience: int = 2

    def create_state(self, batch_size: int) -> "EarlyStopState":
        window = self.window_size or batch_size
        return EarlyStopState(
            threshold=max(0.0, float(self.score_threshold)),
            window=max(1, int(window)),
            patience=max(1, int(self.patience)),
        )


@dataclass
class EarlyStopState:
    threshold: float
    window: int
    patience: int
    scores: Deque[float] = field(default_factory=deque)
    patience_hits: int = 0
    last_average: float = 0.0

    def observe(self, new_scores: Iterable[float]) -> bool:
        for score in new_scores:
            self.scores.append(float(score))
            if len(self.scores) > self.window:
                self.scores.popleft()
        if len(self.scores) < self.window:
            self.patience_hits = 0
            self.last_average = 0.0
            return False
        self.last_average = sum(self.scores) / len(self.scores)
        if self.last_average < self.threshold:
            self.patience_hits += 1
            if self.patience_hits >= self.patience:
                return True
        else:
            self.patience_hits = 0
        return False



def _extract_query_exts(query: str, *, available_exts: Set[str]) -> Set[str]:
    if not query or not available_exts:
        return set()
    lowered = query.lower()
    units = _iter_query_units(lowered)
    requested: Set[str] = set()

    for unit in units:
        normalized = _normalize_ext(unit)
        if normalized and normalized in available_exts:
            requested.add(normalized)

    if requested:
        return requested

    for unit in units:
        mapped = _EXTENSION_KEYWORD_MAP.get(unit)
        if mapped:
            requested.update(mapped & available_exts)
    if requested:
        return requested

    for unit in units:
        for ext, keywords in _EXT_SYNONYMS.items():
            norm_ext = _normalize_ext(ext)
            if norm_ext not in available_exts:
                continue
            if any(_token_contains(unit, keyword) for keyword in keywords):
                requested.add(norm_ext)
    if requested:
        return requested

    for unit in units:
        mapped = _DOMAIN_KEYWORD_MAP.get(unit)
        if mapped:
            requested.update(mapped & available_exts)

    if requested:
        return requested

    for keyword, hinted_exts in _DOMAIN_EXT_HINTS.items():
        if any(_token_contains(unit, keyword) for unit in units):
            for ext in hinted_exts:
                norm = _normalize_ext(ext)
                if norm in available_exts:
                    requested.add(norm)
    return requested


def _prioritize_ext_hits(
    hits: List[Dict[str, Any]], *, desired_exts: Set[str], top_k: int
) -> List[Dict[str, Any]]:
    if not hits:
        return []
    if not desired_exts:
        return hits[:top_k]

    desired_hits: List[Dict[str, Any]] = []
    other_hits: List[Dict[str, Any]] = []

    for hit in hits:
        ext = _normalize_ext(hit.get("ext", ""))
        if ext in desired_exts:
            desired_hits.append(hit)
        else:
            other_hits.append(hit)

    if not desired_hits:
        return hits[:top_k]

    required_matches = max(1, min(top_k, int(math.ceil(top_k * 0.95))))
    take_from_desired = min(len(desired_hits), required_matches)

    ordered: List[Dict[str, Any]] = desired_hits[:take_from_desired]

    remaining_slots = top_k - len(ordered)
    if remaining_slots > 0 and take_from_desired < len(desired_hits):
        additional = desired_hits[take_from_desired : take_from_desired + remaining_slots]
        ordered.extend(additional)
        remaining_slots = top_k - len(ordered)

    if remaining_slots > 0:
        ordered.extend(other_hits[:remaining_slots])

    return ordered[:top_k]


def _minmax_scale(values: Sequence[float]) -> List[float]:
    data = [float(v) for v in values if v is not None]
    if not data:
        return []
    vmin = min(data)
    vmax = max(data)
    if math.isclose(vmax, vmin, abs_tol=1e-12):
        return [0.5] * len(values)
    span = vmax - vmin
    return [((float(v) - vmin) / span) if v is not None else 0.0 for v in values]


def _dynamic_oversample(
    top_k: int,
    *,
    has_ext_pref: bool,
    filters_active: bool,
    corpus_size: int,
) -> int:
    capped_top_k = max(1, int(top_k))
    base = 2
    if has_ext_pref:
        base += 2
    if filters_active:
        base = max(base, 6)
    if capped_top_k <= 3:
        base = max(base, 4)

    # hard upper bound so that oversample * top_k stays reasonably small
    upper_by_limit = max(1, min(10, 200 // capped_top_k))

    if corpus_size > 0:
        max_batches = max(1, (corpus_size + capped_top_k - 1) // capped_top_k)
        upper_by_limit = min(upper_by_limit, max_batches)

    oversample = min(base, upper_by_limit)
    return max(1, oversample)


def _classify_query(
    query: str,
    *,
    metadata_filters: "MetadataFilters",
    requested_exts: Set[str],
) -> str:
    tokens = _split_tokens((query or "").lower())
    if metadata_filters.is_active() or requested_exts:
        return "narrow"
    if len(tokens) <= 3:
        return "narrow"
    if len(tokens) >= 10:
        return "broad"
    return "broad"


def _dynamic_search_params(
    query: str,
    top_k: int,
    *,
    metadata_filters: "MetadataFilters",
    requested_exts: Set[str],
) -> Dict[str, int]:
    classification = _classify_query(
        query,
        metadata_filters=metadata_filters,
        requested_exts=requested_exts,
    )
    base_top_k = max(1, int(top_k))
    if classification == "narrow":
        oversample = min(6, max(2, base_top_k))
        rerank_depth = max(base_top_k * 2, 40)
        fusion_depth = max(base_top_k * 2, base_top_k + 10)
    else:
        oversample = min(10, max(3, base_top_k * 2))
        rerank_depth = max(base_top_k * 3, 120)
        fusion_depth = max(base_top_k * 3, base_top_k + 20)
    return {
        "oversample": max(1, oversample),
        "rerank_depth": max(base_top_k, rerank_depth),
        "fusion_depth": max(base_top_k, fusion_depth),
    }


def _compute_extension_bonus(
    ext: Optional[str],
    desired_exts: Set[str],
    session: Optional[SessionState],
) -> Tuple[float, float, float]:
    normalized = _normalize_ext(ext)
    desired_bonus = _EXTENSION_MATCH_BONUS if normalized and normalized in desired_exts else 0.0
    session_bonus = 0.0
    if session is not None and normalized:
        preference = session.preferred_exts.get(normalized, 0.0)
        if preference:
            session_bonus = preference * _SESSION_EXT_PREF_SCALE
    return desired_bonus + session_bonus, desired_bonus, session_bonus


def _compute_owner_bonus(owner: Optional[str], session: Optional[SessionState]) -> float:
    if session is None or not owner:
        return 0.0
    normalized = _normalize_owner(str(owner))
    if not normalized:
        return 0.0
    preference = session.owner_prior.get(normalized, 0.0)
    if not preference:
        return 0.0
    return preference * _SESSION_OWNER_PREF_SCALE


def _should_expand_query(
    query: str,
    *,
    metadata_filters: "MetadataFilters",
    requested_exts: Set[str],
) -> bool:
    tokens = _split_tokens((query or "").lower())
    if len(tokens) > 6:
        return False
    if metadata_filters.is_active():
        return False
    if any(_looks_like_extension(tok) for tok in tokens):
        return False
    lowered = (query or "").lower()
    if any(keyword in lowered for keyword in ("확장자", "extension", "file type")):
        return False
    return True


def _rrf(rank_lists: Sequence[List[Dict[str, Any]]], *, k: int = RRF_DEFAULT_K) -> List[Dict[str, Any]]:
    if not rank_lists:
        return []
    cumulative: Dict[Tuple[Any, Any], float] = {}
    keeper: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
    for rlist in rank_lists:
        for rank, hit in enumerate(rlist, start=1):
            key = (hit.get("doc_id"), hit.get("path"))
            if key not in keeper:
                keeper[key] = hit
            cumulative[key] = cumulative.get(key, 0.0) + 1.0 / (k + rank)
    ordered = sorted(cumulative.items(), key=lambda item: item[1], reverse=True)
    return [keeper[key] for key, _ in ordered]


def _mmr(
    index: "VectorIndex",
    candidates: List[Dict[str, Any]],
    qvec: Optional[np.ndarray],
    top_k: int,
    *,
    lambda_: float = DEFAULT_MMR_LAMBDA,
) -> List[Dict[str, Any]]:
    if not candidates or top_k <= 0:
        return []

    if qvec is None:
        return candidates[:top_k]

    valid_hits: List[Dict[str, Any]] = []
    doc_vectors: List[np.ndarray] = []
    for hit in candidates:
        doc_id = hit.get("doc_id")
        if doc_id is None:
            continue
        vec = index.embeddings.get(int(doc_id)) if hasattr(index, "embeddings") else None
        if vec is None:
            continue
        valid_hits.append(hit)
        doc_vectors.append(vec)

    if not valid_hits:
        return candidates[:top_k]

    D = np.vstack(doc_vectors).astype(np.float32, copy=False)
    q = VectorIndex._normalize_vector(np.asarray(qvec, dtype=np.float32))
    sim_to_query = (D @ q.reshape(-1, 1)).ravel()

    selected_indices: List[int] = []
    chosen_hits: List[Dict[str, Any]] = []
    remaining = set(range(len(valid_hits)))

    while remaining and len(chosen_hits) < min(top_k, len(valid_hits)):
        if not selected_indices:
            best = int(max(remaining, key=lambda idx: sim_to_query[idx]))
            selected_indices.append(best)
            remaining.discard(best)
            chosen_hits.append(valid_hits[best])
            continue

        selected_matrix = D[selected_indices]
        inter = D @ selected_matrix.T
        max_inter = inter.max(axis=1)
        mmr_scores = lambda_ * sim_to_query - (1.0 - lambda_) * max_inter
        for idx in selected_indices:
            mmr_scores[idx] = -1e9
        best = int(max(remaining, key=lambda idx: mmr_scores[idx]))
        selected_indices.append(best)
        remaining.discard(best)
        chosen_hits.append(valid_hits[best])

    if len(chosen_hits) < top_k:
        # fill with remaining candidates preserving order
        seen = {id(hit) for hit in chosen_hits}
        for hit in valid_hits:
            if len(chosen_hits) >= top_k:
                break
            if id(hit) in seen:
                continue
            chosen_hits.append(hit)

    return chosen_hits[:top_k]


def _summarize_metadata_filters(filters: MetadataFilters) -> List[str]:
    if not filters.is_active():
        return []
    summary: List[str] = []
    if filters.mtime_from is not None or filters.mtime_to is not None:
        summary.append("수정일 조건 일치")
    if filters.ctime_from is not None or filters.ctime_to is not None:
        summary.append("생성일 조건 일치")
    if filters.size_min is not None or filters.size_max is not None:
        summary.append("파일 크기 조건 일치")
    if filters.owners:
        owners = ", ".join(sorted(filters.owners))
        summary.append(f"작성자 조건: {owners}")
    return summary


def _annotate_hits(
    hits: List[Dict[str, Any]],
    *,
    desired_exts: Set[str],
    raw_query_tokens: Set[str],
    expanded_query_tokens: Set[str],
    metadata_filters: MetadataFilters,
    lexical_weight: float,
) -> List[Dict[str, Any]]:
    synonym_tokens = {tok for tok in expanded_query_tokens if tok not in raw_query_tokens}
    metadata_summary = _summarize_metadata_filters(metadata_filters)

    def _safe_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _unique(items: Iterable[str]) -> List[str]:
        seen: Set[str] = set()
        ordered: List[str] = []
        for item in items:
            if item and item not in seen:
                seen.add(item)
                ordered.append(item)
        return ordered

    for hit in hits:
        hit_tokens = _collect_hit_tokens(hit)

        matched_terms = sorted(tok for tok in raw_query_tokens if tok in hit_tokens)
        synonym_matches = sorted(tok for tok in synonym_tokens if tok in hit_tokens)

        breakdown: Dict[str, float] = {}
        reasons: List[str] = []

        vector_score = _safe_float(hit.get("vector_similarity"))
        if vector_score is not None:
            breakdown["vector"] = round(vector_score, 4)
            reasons.append(f"임베딩 유사도 {breakdown['vector']:.2f}")

        lexical_score = _safe_float(hit.get("lexical_score"))
        if lexical_score is not None:
            breakdown["lexical"] = round(lexical_score, 4)
            if lexical_weight > 0.0 and lexical_score > 0.0:
                reasons.append(
                    f"키워드 일치 점수 {breakdown['lexical']:.2f} (가중치 {lexical_weight:.2f})"
                )
            elif lexical_score > 0.0:
                reasons.append(f"키워드 일치 점수 {breakdown['lexical']:.2f}")

        rerank_score = _safe_float(hit.get("rerank_score"))
        if rerank_score is not None:
            breakdown["rerank"] = round(rerank_score, 4)
            reasons.append(f"Cross-Encoder 점수 {breakdown['rerank']:.2f}")

        total_score = _safe_float(hit.get("combined_score", hit.get("score")))
        if total_score is not None:
            breakdown["final"] = round(total_score, 4)

        ext = _normalize_ext(hit.get("ext"))
        desired_ext_bonus = _safe_float(hit.get("desired_extension_bonus")) or 0.0
        session_ext_bonus = _safe_float(hit.get("session_ext_bonus")) or 0.0
        session_owner_bonus = _safe_float(hit.get("session_owner_bonus")) or 0.0

        breakdown["extension_bonus"] = round(desired_ext_bonus, 4) if desired_ext_bonus else 0.0
        if desired_ext_bonus > 0 and ext:
            reasons.append(f"요청 확장자 {ext} 우선")
        if session_ext_bonus:
            breakdown["session_ext"] = round(session_ext_bonus, 4)
            if session_ext_bonus > 0:
                reasons.append("세션 선호 확장자 가중치")
            else:
                reasons.append("세션 비선호 확장자 페널티")
        if session_owner_bonus:
            breakdown["session_owner"] = round(session_owner_bonus, 4)
            if session_owner_bonus > 0:
                reasons.append("세션 선호 작성자 가중치")
            else:
                reasons.append("세션 비선호 작성자 페널티")

        metadata_reasons = metadata_summary if metadata_summary else []

        if matched_terms:
            snippet = ", ".join(matched_terms[:4])
            reasons.append(f"질문 키워드 일치: {snippet}")
        if synonym_matches:
            snippet = ", ".join(synonym_matches[:4])
            reasons.append(f"확장/동의어 매칭: {snippet}")
        reasons.extend(metadata_reasons)

        hit["score_breakdown"] = breakdown
        hit["matched_terms"] = matched_terms
        hit["matched_synonyms"] = synonym_matches
        hit["metadata_matches"] = list(metadata_reasons)
        hit["match_reasons"] = _unique(reasons)

    return hits


@lru_cache(maxsize=256)
def _extension_related_tokens_cached(ext: str) -> Tuple[str, ...]:
    normalized = _normalize_ext(ext)
    if not normalized:
        return ()
    related: Set[str] = set()
    base = normalized.lstrip('.')
    if base:
        related.add(base)
    related.add(normalized)
    for keyword in _EXT_SYNONYMS.get(normalized, set()):
        related.update(_keyword_forms(keyword))
    for domain_keyword in _DOMAIN_KEYWORDS_BY_EXT.get(normalized, set()):
        related.update(_keyword_forms(domain_keyword))
    return tuple(sorted({tok for tok in related if tok}))


def _extension_related_tokens(ext: str) -> Set[str]:
    return set(_extension_related_tokens_cached(ext))


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

        mapped_exts = _EXTENSION_KEYWORD_MAP.get(token)
        if mapped_exts:
            for ext in mapped_exts:
                expansions.update(_extension_related_tokens(ext))
            continue

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


def _clean_token(token: str) -> str:
    if not token:
        return ""
    cleaned = re.sub(r"\s+", " ", str(token)).strip().lower()
    return cleaned


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

    seen: Set[str] = set()
    normalized: List[str] = []
    for token in tokens:
        cleaned = _clean_token(token)
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
            for value in (
                self.mtime_from,
                self.mtime_to,
                self.ctime_from,
                self.ctime_to,
                self.size_min,
                self.size_max,
            )
        ) or bool(self.owners)

    def matches(self, hit: Dict[str, Any]) -> bool:
        if not self.is_active():
            return True
        mtime = _to_float(hit.get("mtime"))
        ctime = _to_float(hit.get("ctime"))
        size = _to_int(hit.get("size"))
        owner = _clean_token(hit.get("owner", ""))

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
        if self.owners and owner and owner not in self.owners:
            return False
        if self.owners and not owner:
            return False
        return True


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _year_bounds(year: int) -> Tuple[float, float]:
    start = datetime(year, 1, 1)
    end = datetime(year, 12, 31, 23, 59, 59)
    return start.timestamp(), end.timestamp()


def _month_bounds(now: datetime, months_ago: int) -> Tuple[float, float]:
    year = now.year
    month = now.month - months_ago
    while month <= 0:
        month += 12
        year -= 1
    start = datetime(year, month, 1)
    last_day = calendar.monthrange(year, month)[1]
    end = datetime(year, month, last_day, 23, 59, 59)
    return start.timestamp(), end.timestamp()


def _approx_range(value: int, *, tolerance: float = 0.15) -> Tuple[float, float]:
    delta = max(1.0, value * tolerance)
    return value - delta, value + delta


def _normalize_owner(owner: str) -> str:
    return _clean_token(owner)


def _parse_size_expression(value: str, unit: str) -> int:
    base = float(value)
    unit = unit.lower()
    multiplier = {
        "kb": 1024,
        "mb": 1024 ** 2,
        "gb": 1024 ** 3,
        "tb": 1024 ** 4,
    }.get(unit, 1)
    return int(base * multiplier)


def _extract_metadata_filters(query: str) -> MetadataFilters:
    filters = MetadataFilters()
    lowered = query.lower()
    now = datetime.now()

    year_match = re.search(r"(20\d{2}|19\d{2})\s*년", lowered)
    if year_match:
        year = int(year_match.group(1))
        filters.mtime_from, filters.mtime_to = _year_bounds(year)

    rel_year = re.search(r"(\d+)\s*년\s*전", lowered)
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

    rel_month = re.search(r"(\d+)\s*개월\s*전", lowered)
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

    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(kb|mb|gb|tb)\s*(이상|이하|초과|미만|보다 큰|보다 작은|at least|over|under|at most)?", lowered):
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

    for match in re.finditer(r"(?:작성자|author|owner)[:\s]+([\w가-힣@.]+)", query, re.IGNORECASE):
        filters.owners.add(_normalize_owner(match.group(1)))
    for mention in re.findall(r"@([\w가-힣._-]+)", query):
        filters.owners.add(_normalize_owner(mention))

    return filters


def _apply_metadata_filters(hits: List[Dict[str, Any]], filters: MetadataFilters) -> List[Dict[str, Any]]:
    if not filters.is_active():
        return hits
    return [hit for hit in hits if filters.matches(hit)]


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


def _rerank_hits(
    raw_query: str,
    expanded_query: str,
    hits: List[Dict[str, Any]],
    *,
    desired_exts: Set[str],
    top_k: int,
    session: Optional[SessionState] = None,
) -> List[Dict[str, Any]]:
    if not hits:
        return []

    base_tokens = {tok for tok in _split_tokens(raw_query.lower()) if tok}
    expanded_tokens = {tok for tok in _split_tokens(expanded_query.lower()) if tok}
    query_tokens = base_tokens or expanded_tokens
    synonym_tokens = expanded_tokens - base_tokens
    scored_hits: List[Dict[str, Any]] = []
    use_lexical_overlap = _LEXICAL_WEIGHT > 0.0

    for raw_hit in hits:
        hit = dict(raw_hit)
        hit_tokens = _collect_hit_tokens(hit)
        lexical_score = 0.0
        if use_lexical_overlap:
            existing_lexical = hit.get("lexical_score")
            if existing_lexical is None:
                lexical_score = _lexical_overlap_score(query_tokens, hit_tokens)
                if synonym_tokens:
                    synonym_score = _lexical_overlap_score(synonym_tokens, hit_tokens)
                    lexical_score = max(lexical_score, synonym_score)
                hit["lexical_score"] = lexical_score
            else:
                try:
                    lexical_score = float(existing_lexical)
                except (TypeError, ValueError):
                    lexical_score = 0.0
        else:
            hit.setdefault("lexical_score", 0.0)

        base_similarity = float(hit.get("vector_similarity", hit.get("similarity", 0.0)))
        if "vector_similarity" not in hit:
            hit["vector_similarity"] = base_similarity

        base_score = hit.get("score")
        if base_score is None:
            base_score = base_similarity
            if use_lexical_overlap:
                base_score += float(lexical_score) * _LEXICAL_WEIGHT

        ext_raw = hit.get("ext")
        ext_norm = _normalize_ext(ext_raw)
        total_ext_bonus, desired_ext_bonus, session_ext_bonus = _compute_extension_bonus(
            ext_norm,
            desired_exts,
            session,
        )
        owner_bonus = _compute_owner_bonus(hit.get("owner"), session)
        final_score = float(base_score) + total_ext_bonus + owner_bonus
        hit["score"] = final_score
        if "vector_similarity" in hit:
            hit["similarity"] = float(hit.get("vector_similarity", 0.0))
        else:
            hit["similarity"] = final_score
        hit["desired_extension_bonus"] = float(desired_ext_bonus)
        hit["session_ext_bonus"] = float(session_ext_bonus)
        hit["session_owner_bonus"] = float(owner_bonus)
        scored_hits.append(hit)

    scored_hits.sort(key=lambda item: item.get("score", item.get("similarity", 0.0)), reverse=True)
    return _prioritize_ext_hits(scored_hits, desired_exts=desired_exts, top_k=top_k)


def _compose_rerank_document(hit: Dict[str, Any]) -> str:
    sections: List[str] = []
    path = str(hit.get("path") or "").strip()
    if path:
        sections.append(f"파일 경로: {_mask_path(path)}")
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
        early_stop: Optional[EarlyStopConfig] = None,
    ) -> None:
        if CrossEncoder is None:
            raise RuntimeError("sentence-transformers의 CrossEncoder를 사용할 수 없습니다.")
        self.model_name = model_name
        self.device = device or None
        self.batch_size = max(1, int(batch_size) if batch_size else 1)
        self.early_stop_config = early_stop or EarlyStopConfig(window_size=self.batch_size)
        if self.early_stop_config.window_size <= 0:
            self.early_stop_config.window_size = self.batch_size

        load_kwargs: Dict[str, Any] = {}
        if self.device:
            load_kwargs["device"] = self.device

        t0 = time.time()
        self.model = CrossEncoder(model_name, **load_kwargs)
        dt = time.time() - t0
        device_label = self.device or getattr(self.model, "device", "cpu")
        logger.info("reranker loaded: model=%s device=%s dt=%.1fs", model_name, device_label, dt)

    def rerank(
        self,
        query: str,
        hits: List[Dict[str, Any]],
        *,
        desired_exts: Optional[Set[str]] = None,
        session: Optional[SessionState] = None,
    ) -> List[Dict[str, Any]]:
        if not hits:
            return []

        pairs: List[List[str]] = []
        prepared_hits: List[Dict[str, Any]] = []
        for hit in hits:
            doc_text = _compose_rerank_document(hit)
            pairs.append([query, doc_text])
            prepared_hits.append(dict(hit))

        collected: List[float] = []
        try:
            for batch_scores in self._predict_iter(pairs):
                collected.extend(batch_scores)
        except Exception as exc:  # pragma: no cover - defensive path
            logger.warning("rerank inference failed, fallback to previous scores: %s", exc)
            return hits

        if not collected:
            return hits

        ext_preferences = desired_exts or set()
        rerank_raw = [float(s) for s in collected[: len(prepared_hits)]]
        vector_components = [float(h.get("vector_similarity", 0.0)) for h in prepared_hits]
        lexical_components = [float(h.get("lexical_score", 0.0)) for h in prepared_hits]

        rerank_scaled = _minmax_scale(rerank_raw)
        vector_scaled = _minmax_scale(vector_components)
        lexical_scaled = _minmax_scale(lexical_components)

        alpha, beta, gamma = 0.60, 0.25, 0.15
        combined_hits: List[Dict[str, Any]] = []

        for idx, hit in enumerate(prepared_hits):
            rerank_score = rerank_raw[idx] if idx < len(rerank_raw) else 0.0
            rerank_component = rerank_scaled[idx] if idx < len(rerank_scaled) else 0.0
            vector_component = vector_scaled[idx] if idx < len(vector_scaled) else 0.0
            lexical_component = lexical_scaled[idx] if idx < len(lexical_scaled) else 0.0

            ext = _normalize_ext(hit.get("ext"))
            total_ext_bonus, desired_ext_bonus, session_ext_bonus = _compute_extension_bonus(
                ext,
                ext_preferences,
                session,
            )
            owner_bonus = _compute_owner_bonus(hit.get("owner"), session)

            combined = (
                (alpha * rerank_component)
                + (beta * vector_component)
                + (gamma * lexical_component)
                + total_ext_bonus
                + owner_bonus
            )

            original_vector = float(hit.get("vector_similarity", hit.get("similarity", 0.0)))
            hit["vector_similarity"] = original_vector
            hit.setdefault("pre_rerank_score", float(hit.get("score", 0.0)))
            hit["rerank_score"] = float(rerank_score)
            hit["combined_score"] = float(combined)
            hit["score"] = float(combined)
            hit["similarity"] = original_vector
            hit["desired_extension_bonus"] = float(desired_ext_bonus)
            hit["session_ext_bonus"] = float(session_ext_bonus)
            hit["session_owner_bonus"] = float(owner_bonus)
            if total_ext_bonus:
                hit["rerank_ext_bonus"] = total_ext_bonus
            match_reasons = hit.get("match_reasons")
            if match_reasons is not None and session_ext_bonus:
                label = "세션 선호 확장자 가중치" if session_ext_bonus > 0 else "세션 비선호 확장자 페널티"
                if label not in match_reasons:
                    match_reasons.append(label)
            if match_reasons is not None and owner_bonus:
                owner_label = "세션 선호 작성자 가중치" if owner_bonus > 0 else "세션 비선호 작성자 페널티"
                if owner_label not in match_reasons:
                    match_reasons.append(owner_label)
            combined_hits.append(hit)

        combined_hits.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return combined_hits

    def _predict_iter(self, pairs: List[List[str]]) -> Iterable[np.ndarray]:
        total = len(pairs)
        if total <= self.batch_size:
            yield self.model.predict(
                pairs,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return

        start = 0
        stop_state = self.early_stop_config.create_state(self.batch_size)
        while start < total:
            end = min(total, start + self.batch_size)
            batch = pairs[start:end]
            batch_scores = self.model.predict(
                batch,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            yield batch_scores
            if stop_state.observe(batch_scores):
                logger.info(
                    "reranker early stop: avg=%.4f window=%d processed=%d/%d",
                    stop_state.last_average,
                    stop_state.window,
                    end,
                    total,
                )
                break
            start = end


try:
    import pandas as pd  # noqa: F811 - re-import for static analyzers
except Exception:  # pragma: no cover - already handled
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
    import joblib  # noqa: F811 - re-import for static analyzers
except Exception:  # pragma: no cover
    joblib = None


def _alias_legacy_modules() -> bool:
    candidates = ["pipeline", "step2_module_progress", "step2_pipeline", "Step2_module_progress"]
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


class QueryEncoder:
    def __init__(self, model_path: Path):
        if joblib is None:
            raise RuntimeError("joblib이 필요합니다. pip install joblib")

        try:
            obj = joblib.load(model_path)
        except ModuleNotFoundError as e:
            if "TextCleaner" in str(e):
                logger.warning("legacy model detected; injecting TextCleaner alias and retrying")
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
            logger.info("Sentence-BERT loaded: %s", model_name)
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
            cleaned.append(str(raw))
        return cleaned

    def encode_docs(self, texts: List[str]) -> np.ndarray:
        texts = self._sanitize_texts(texts)
        if self.model_type == MODEL_TYPE_SENTENCE_TRANSFORMER and self.embedder is not None:
            Z = self.embedder.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return np.asarray(Z, dtype=np.float32)

        if self.tfidf is None or self.svd is None:
            raise RuntimeError("TF-IDF 파이프라인이 초기화되지 않았습니다.")
        X = self.tfidf.transform(texts)
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
                normalize_embeddings=True,
            )
            return np.asarray(Zq, dtype=np.float32)

        if self.tfidf is None or self.svd is None:
            raise RuntimeError("TF-IDF 파이프라인이 초기화되지 않았습니다.")
        Xq = self.tfidf.transform(clean_query)
        Zq = self.svd.transform(Xq)
        return Zq.astype(np.float32, copy=False)


@dataclass
class IndexPaths:
    emb_npy: Optional[Path]
    meta_json: Path
    faiss_index: Optional[Path] = None


class VectorIndex:
    def __init__(self) -> None:
        self.dimension: Optional[int] = None
        self.doc_ids: List[int] = []
        self.entries: Dict[int, Dict[str, Any]] = {}
        self.lexical_tokens: Dict[int, List[str]] = {}
        self.embeddings: Dict[int, np.ndarray] = {}
        self._path_to_id: Dict[str, int] = {}

        self.paths: List[str] = []
        self.exts: List[str] = []
        self.preview: List[str] = []
        self.sizes: List[Optional[int]] = []
        self.mtimes: List[Optional[float]] = []
        self.ctimes: List[Optional[float]] = []
        self.owners: List[Optional[str]] = []
        self.drives: List[Optional[str]] = []
        self.chunk_ids: List[Optional[int]] = []
        self.chunk_counts: List[Optional[int]] = []
        self.chunk_tokens: List[Optional[int]] = []

        self.Z: Optional[np.ndarray] = None
        self.faiss_index = None
        self.lexical_index: Optional[BM25Okapi] = None
        self.ann_index = None

        self._matrix_dirty = True
        self._faiss_dirty = True
        self._lexical_dirty = True
        self._ann_dirty = True

        self.lexical_weight = 0.0
        self.ann_threshold = 50000
        self.ann_m = 32
        self.ann_ef_construction = 80
        self.ann_ef_search = 64

    @staticmethod
    def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
        return (matrix / norms).astype(np.float32, copy=False)

    @staticmethod
    def _normalize_vector(vec: np.ndarray) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(arr)) + 1e-12
        return (arr / norm).astype(np.float32, copy=False)

    @staticmethod
    def _normalize_path(path: str) -> str:
        try:
            return os.path.normcase(str(Path(path).resolve()))
        except Exception:
            return os.path.normcase(str(path))

    @staticmethod
    def _truncate_preview(text: str, limit: int = MAX_PREVIEW_CHARS) -> str:
        src = (text or "").strip()
        return src if len(src) <= limit else f"{src[:limit]}…"

    @staticmethod
    def _generate_doc_ids(paths: Iterable[str]) -> List[int]:
        assigned: Set[int] = set()
        ids: List[int] = []
        mask = (1 << 63) - 1
        for raw_path in paths:
            norm = VectorIndex._normalize_path(raw_path)
            digest = hashlib.sha1(norm.encode("utf-8")).digest()
            candidate = int.from_bytes(digest[:8], byteorder="big") & mask
            while candidate in assigned:
                candidate = (candidate + 1) & mask
            assigned.add(candidate)
            ids.append(candidate)
        return ids

    def _allocate_doc_id(self, path: str) -> int:
        normalized = self._normalize_path(path)
        existing = self._path_to_id.get(normalized)
        if existing is not None:
            return existing
        mask = (1 << 63) - 1
        digest = hashlib.sha1(normalized.encode("utf-8")).digest()
        candidate = int.from_bytes(digest[:8], byteorder="big") & mask
        while candidate in self.entries and self.entries[candidate].get("path") != path:
            candidate = (candidate + 1) & mask
        return candidate

    def _rebuild_lists(self) -> None:
        self.paths = []
        self.exts = []
        self.preview = []
        self.sizes = []
        self.mtimes = []
        self.ctimes = []
        self.owners = []
        self.drives = []
        self.chunk_ids = []
        self.chunk_counts = []
        self.chunk_tokens = []
        for doc_id in self.doc_ids:
            entry = self.entries.get(doc_id, {})
            self.paths.append(entry.get("path", ""))
            self.exts.append(entry.get("ext", ""))
            self.preview.append(entry.get("preview", ""))
            self.sizes.append(entry.get("size"))
            self.mtimes.append(entry.get("mtime"))
            self.ctimes.append(entry.get("ctime"))
            self.owners.append(entry.get("owner"))
            self.drives.append(entry.get("drive"))
            self.chunk_ids.append(entry.get("chunk_id"))
            self.chunk_counts.append(entry.get("chunk_count"))
            self.chunk_tokens.append(entry.get("chunk_tokens"))

    def _mark_faiss_dirty(self) -> None:
        self._faiss_dirty = True

    def _mark_lexical_dirty(self) -> None:
        self._lexical_dirty = True

    def _mark_ann_dirty(self) -> None:
        self._ann_dirty = True

    def _ensure_matrix(self) -> None:
        if not self._matrix_dirty:
            return
        if not self.embeddings:
            self.Z = None
            self._matrix_dirty = False
            return
        ordered = [self.embeddings[doc_id] for doc_id in self.doc_ids if doc_id in self.embeddings]
        if not ordered:
            self.Z = None
            self._matrix_dirty = False
            return
        self.Z = np.vstack(ordered).astype(np.float32, copy=False)
        self.dimension = self.Z.shape[1]
        self._matrix_dirty = False

    def _ensure_faiss_index(self) -> None:
        if not self._faiss_dirty:
            return
        if faiss is None:
            self.faiss_index = None
            self._faiss_dirty = False
            return
        if not self.doc_ids:
            self.faiss_index = None
            self._faiss_dirty = False
            return
        if self.Z is None:
            self._ensure_matrix()
        if self.Z is None or self.Z.size == 0:
            self.faiss_index = None
            self._faiss_dirty = False
            return
        dim = self.Z.shape[1]
        base = faiss.IndexFlatIP(dim)
        index = faiss.IndexIDMap(base)
        ids = np.asarray(self.doc_ids, dtype=np.int64)
        index.add_with_ids(self.Z, ids)
        self.faiss_index = index
        self.dimension = dim
        self._faiss_dirty = False

    def _ensure_lexical_index(self) -> None:
        if not self._lexical_dirty:
            return
        if BM25Okapi is None or not self.doc_ids:
            self.lexical_index = None
            self._lexical_dirty = False
            return
        corpus = [self.lexical_tokens.get(doc_id, []) for doc_id in self.doc_ids]
        try:
            self.lexical_index = BM25Okapi(corpus) if corpus else None
        except Exception:
            self.lexical_index = None
        self._lexical_dirty = False

    def _tokenize_entry(self, entry: Dict[str, Any]) -> List[str]:
        corpus = " ".join(
            str(entry.get(field, "")) for field in ("preview", "path", "ext", "owner", "drive")
        )
        return [tok for tok in _split_tokens(corpus.lower()) if tok]

    def build(
        self,
        embeddings: np.ndarray,
        paths: List[str],
        exts: List[str],
        preview_texts: List[str],
        *,
        tokens: Optional[List[List[str]]] = None,
        sizes: Optional[List[Optional[int]]] = None,
        mtimes: Optional[List[Optional[float]]] = None,
        ctimes: Optional[List[Optional[float]]] = None,
        owners: Optional[List[Optional[str]]] = None,
        drives: Optional[List[Optional[str]]] = None,
        doc_ids: Optional[List[int]] = None,
        extra_meta: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if embeddings.ndim != 2:
            raise ValueError("embeddings는 2차원이어야 합니다.")

        normalized_embeddings = self._normalize_rows(embeddings)
        count = normalized_embeddings.shape[0]
        if doc_ids is None:
            doc_ids = self._generate_doc_ids(paths)
        if len(doc_ids) != count:
            raise ValueError("doc_id 수와 임베딩 수가 다릅니다.")

        self.dimension = normalized_embeddings.shape[1]
        self.doc_ids = [int(doc_id) for doc_id in doc_ids]
        self.entries.clear()
        self.lexical_tokens.clear()
        self.embeddings = {
            doc_id: normalized_embeddings[idx]
            for idx, doc_id in enumerate(self.doc_ids)
        }
        self._path_to_id.clear()

        def _meta_list(values: Optional[List[Any]], fallback: Any) -> List[Any]:
            if values is None:
                return [fallback] * count
            if len(values) != count:
                raise ValueError("메타데이터 길이가 문서 수와 다릅니다.")
            return list(values)

        size_list = _meta_list(sizes, 0)
        mtime_list = _meta_list(mtimes, 0.0)
        ctime_list = _meta_list(ctimes, 0.0)
        owner_list = _meta_list(owners, "")
        drive_list = _meta_list(drives, "")

        for idx, doc_id in enumerate(self.doc_ids):
            entry = {
                "path": paths[idx],
                "ext": exts[idx],
                "preview": preview_texts[idx],
                "size": size_list[idx],
                "mtime": mtime_list[idx],
                "ctime": ctime_list[idx],
                "owner": owner_list[idx],
                "drive": drive_list[idx],
            }
            if extra_meta and idx < len(extra_meta):
                for key, value in extra_meta[idx].items():
                    entry[key] = value
            entry["preview"] = self._truncate_preview(entry.get("preview", ""))
            self.entries[doc_id] = entry
            provided_tokens = tokens[idx] if tokens and idx < len(tokens) else None
            self.lexical_tokens[doc_id] = list(provided_tokens) if provided_tokens else self._tokenize_entry(entry)
            self._path_to_id[self._normalize_path(entry["path"])] = doc_id

        self._rebuild_lists()
        self._matrix_dirty = True
        self._mark_faiss_dirty()
        self._mark_lexical_dirty()
        self._mark_ann_dirty()
        self._ensure_matrix()
        self._ensure_faiss_index()
        self._ensure_lexical_index()

    def save(self, out_dir: Path) -> IndexPaths:
        out_dir.mkdir(parents=True, exist_ok=True)
        emb_path = out_dir / "doc_embeddings.npy"
        meta_path = out_dir / "doc_meta.json"
        faiss_path: Optional[Path] = None

        self._ensure_matrix()
        if self.Z is not None:
            np.save(emb_path, self.Z.astype(np.float32, copy=False), allow_pickle=False)
        else:
            emb_path = None

        tokens_payload = [self.lexical_tokens.get(doc_id, []) for doc_id in self.doc_ids]
        chunk_id_payload = [self.entries.get(doc_id, {}).get("chunk_id") for doc_id in self.doc_ids]
        chunk_count_payload = [self.entries.get(doc_id, {}).get("chunk_count") for doc_id in self.doc_ids]
        chunk_tokens_payload = [self.entries.get(doc_id, {}).get("chunk_tokens") for doc_id in self.doc_ids]
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "doc_ids": self.doc_ids,
                    "paths": self.paths,
                    "exts": self.exts,
                    "preview": self.preview,
                    "sizes": self.sizes,
                    "mtimes": self.mtimes,
                    "ctimes": self.ctimes,
                    "owners": self.owners,
                    "drives": self.drives,
                    "tokens": tokens_payload,
                    "chunk_id": chunk_id_payload,
                    "chunk_count": chunk_count_payload,
                    "chunk_tokens": chunk_tokens_payload,
                },
                f,
                ensure_ascii=False,
            )

        if faiss is not None:
            self._ensure_faiss_index()
            if self.faiss_index is not None:
                faiss_path = out_dir / "doc_index.faiss"
                faiss.write_index(self.faiss_index, str(faiss_path))

        return IndexPaths(emb_npy=emb_path, meta_json=meta_path, faiss_index=faiss_path)

    def load(
        self,
        emb_npy: Optional[Path],
        meta_json: Path,
        *,
        faiss_path: Optional[Path] = None,
        use_mmap: bool = True,
    ) -> None:
        if not meta_json.exists():
            raise FileNotFoundError(f"메타데이터 파일을 찾을 수 없습니다: {meta_json}")

        with meta_json.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        paths = meta.get("paths", [])
        exts = meta.get("exts", [])
        previews = meta.get("preview", [])
        doc_ids = [int(x) for x in meta.get("doc_ids", list(range(len(paths))))]
        sizes = meta.get("sizes", [0] * len(paths))
        mtimes = meta.get("mtimes", [0.0] * len(paths))
        ctimes = meta.get("ctimes", [0.0] * len(paths))
        owners = meta.get("owners", [""] * len(paths))
        drives = meta.get("drives", [""] * len(paths))
        tokens_payload = meta.get("tokens", [[] for _ in doc_ids])
        chunk_id_payload = meta.get("chunk_id", [None for _ in doc_ids])
        chunk_count_payload = meta.get("chunk_count", [None for _ in doc_ids])
        chunk_tokens_payload = meta.get("chunk_tokens", [None for _ in doc_ids])

        self.doc_ids = doc_ids
        self.entries.clear()
        self.lexical_tokens.clear()
        self.embeddings.clear()
        self._path_to_id.clear()

        for idx, doc_id in enumerate(self.doc_ids):
            entry = {
                "path": paths[idx] if idx < len(paths) else "",
                "ext": exts[idx] if idx < len(exts) else "",
                "preview": previews[idx] if idx < len(previews) else "",
                "size": sizes[idx] if idx < len(sizes) else 0,
                "mtime": mtimes[idx] if idx < len(mtimes) else 0.0,
                "ctime": ctimes[idx] if idx < len(ctimes) else 0.0,
                "owner": owners[idx] if idx < len(owners) else "",
                "drive": drives[idx] if idx < len(drives) else "",
            }
            def _coerce_optional_int(raw: Any) -> Optional[int]:
                if raw in (None, "", "null"):
                    return None
                if isinstance(raw, int):
                    return raw
                try:
                    as_float = float(raw)
                except (TypeError, ValueError):
                    return None
                if math.isnan(as_float):
                    return None
                return int(as_float)

            if idx < len(chunk_id_payload):
                entry["chunk_id"] = _coerce_optional_int(chunk_id_payload[idx])
            if idx < len(chunk_count_payload):
                entry["chunk_count"] = _coerce_optional_int(chunk_count_payload[idx])
            if idx < len(chunk_tokens_payload):
                entry["chunk_tokens"] = _coerce_optional_int(chunk_tokens_payload[idx])
            entry["preview"] = self._truncate_preview(entry.get("preview", ""))
            self.entries[doc_id] = entry
            provided_tokens = tokens_payload[idx] if idx < len(tokens_payload) else []
            self.lexical_tokens[doc_id] = list(provided_tokens)
            self._path_to_id[self._normalize_path(entry["path"])] = doc_id

        self._rebuild_lists()

        if emb_npy and emb_npy.exists():
            mmap_mode = "r" if use_mmap else None
            matrix = np.load(emb_npy, mmap_mode=mmap_mode)
            if matrix.dtype != np.float32:
                matrix = matrix.astype(np.float32, copy=False)
            self.Z = matrix
            self.dimension = matrix.shape[1] if matrix.ndim == 2 and matrix.size else None
            for idx, doc_id in enumerate(self.doc_ids):
                if idx < matrix.shape[0]:
                    self.embeddings[doc_id] = matrix[idx]
            self._matrix_dirty = False
        else:
            self.Z = None
            self._matrix_dirty = True

        if faiss is not None and faiss_path and faiss_path.exists():
            try:
                self.faiss_index = faiss.read_index(str(faiss_path))
                self.dimension = getattr(self.faiss_index, "d", self.dimension)
                self._faiss_dirty = False
            except Exception:
                self.faiss_index = None
                self._faiss_dirty = True
        else:
            self.faiss_index = None
            self._faiss_dirty = True

        if self.dimension is None and self.embeddings:
            any_vec = next(iter(self.embeddings.values()))
            self.dimension = len(any_vec)

        self._mark_lexical_dirty()
        self._ensure_lexical_index()
        self._mark_ann_dirty()

    def configure_ann(
        self,
        *,
        threshold: Optional[int] = None,
        ef_search: Optional[int] = None,
        ef_construction: Optional[int] = None,
        m: Optional[int] = None,
    ) -> None:
        rebuild = False
        if threshold is not None:
            new_threshold = max(0, int(threshold))
            if new_threshold != self.ann_threshold:
                self.ann_threshold = new_threshold
                rebuild = True
        if ef_construction is not None:
            new_ef_construction = max(16, int(ef_construction))
            if new_ef_construction != self.ann_ef_construction:
                self.ann_ef_construction = new_ef_construction
                rebuild = True
        if m is not None:
            new_m = max(8, int(m))
            if new_m != self.ann_m:
                self.ann_m = new_m
                rebuild = True
        if rebuild:
            self._mark_ann_dirty()
        if ef_search is not None:
            new_ef_search = max(8, int(ef_search))
            if new_ef_search != self.ann_ef_search:
                self.ann_ef_search = new_ef_search
                if self.ann_index is not None:
                    try:
                        self.ann_index.hnsw.efSearch = self.ann_ef_search
                    except Exception:
                        pass

    def search(
        self,
        qvec: np.ndarray,
        top_k: int = 5,
        *,
        oversample: int = 1,
        lexical_weight: float = 0.0,
        query_tokens: Optional[List[str]] = None,
        min_similarity: float = 0.0,
        use_ann: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        if not self.doc_ids:
            return []
        q = self._normalize_vector(qvec)
        fetch = max(1, min(len(self.doc_ids), top_k * max(1, oversample)))

        vector_scores, vector_order = self._vector_scores(q, fetch, use_ann=use_ann)
        lex_weight = max(0.0, min(1.0, lexical_weight))
        use_lexical = lex_weight > 0.0 and bool(query_tokens)

        lexical_scores: Dict[int, float] = {}
        if use_lexical:
            lexical_fetch = min(len(self.doc_ids), max(fetch, top_k * 8))
            lexical_scores = self._lexical_scores(query_tokens, lexical_fetch)

        candidate_ids: Set[int] = set(vector_scores.keys())
        if use_lexical:
            candidate_ids |= set(lexical_scores.keys())
        if not candidate_ids:
            candidate_ids = set(self.doc_ids[:fetch])
        lexical_max = max(lexical_scores.values(), default=0.0)

        def _lexical_component(raw: float) -> float:
            if lexical_max <= 0.0:
                return 0.0
            return float(raw) / float(lexical_max) if lexical_max else 0.0

        threshold = max(0.0, min(1.0, float(min_similarity)))

        results: List[Dict[str, Any]] = []
        for doc_id in candidate_ids:
            entry = self.entries.get(doc_id)
            if not entry:
                continue
            vector_raw = float(vector_scores.get(doc_id, 0.0))
            vector_component = _rescale_inner_product(vector_raw)
            lexical_raw = float(lexical_scores.get(doc_id, 0.0))
            lexical_component = _lexical_component(lexical_raw)
            hybrid_score = ((1.0 - lex_weight) * vector_component) + (lex_weight * lexical_component)

            if threshold > 0.0:
                passes_vector = vector_component >= threshold
                passes_hybrid = hybrid_score >= threshold if use_lexical else False
                if not (passes_vector or passes_hybrid):
                    continue

            hit = {
                "doc_id": doc_id,
                "path": entry.get("path"),
                "ext": entry.get("ext"),
                "preview": entry.get("preview"),
                "size": entry.get("size"),
                "mtime": entry.get("mtime"),
                "ctime": entry.get("ctime"),
                "owner": entry.get("owner"),
                "drive": entry.get("drive"),
                "chunk_id": entry.get("chunk_id"),
                "chunk_count": entry.get("chunk_count"),
                "chunk_tokens": entry.get("chunk_tokens"),
                "vector_similarity": vector_component,
                "vector_raw": vector_raw,
                "lexical_score": lexical_component,
                "lexical_raw": lexical_raw,
                "score": hybrid_score,
                "hybrid_score": hybrid_score,
                "similarity": vector_component,
            }
            results.append(hit)

        if not results:
            return []

        vector_rank = {doc_id: idx for idx, doc_id in enumerate(vector_order)}
        results.sort(
            key=lambda item: (
                item.get("score", 0.0),
                item.get("vector_similarity", 0.0),
                -vector_rank.get(item.get("doc_id"), len(vector_rank)),
            ),
            reverse=True,
        )

        limit = min(len(results), max(top_k, fetch))
        return results[:limit]

    def _vector_scores(
        self,
        qvec: np.ndarray,
        fetch: int,
        *,
        use_ann: Optional[bool] = None,
    ) -> Tuple[Dict[int, float], List[int]]:
        scores: Dict[int, float] = {}
        order: List[int] = []
        ann_choice = use_ann
        if ann_choice is None and self._should_use_ann():
            ann_choice = True
        if ann_choice:
            scores, order = self._ann_scores(qvec, fetch)
            if scores:
                return scores, order
        elif ann_choice is False:
            pass
        self._ensure_faiss_index()
        if self.faiss_index is not None and faiss is not None:
            query = qvec.reshape(1, -1).astype(np.float32, copy=False)
            k = min(fetch, len(self.doc_ids))
            if k <= 0:
                return scores, order
            distances, ids = self.faiss_index.search(query, k)
            for score, doc_id in zip(distances[0], ids[0]):
                if doc_id < 0:
                    continue
                doc_id_int = int(doc_id)
                scores[doc_id_int] = float(score)
                order.append(doc_id_int)
            return scores, order

        self._ensure_matrix()
        if self.Z is None or self.Z.size == 0:
            return scores, order

        sims = (self.Z @ qvec.reshape(-1, 1)).ravel()
        limit = min(fetch, sims.shape[0])
        idx = np.argpartition(-sims, limit - 1)[:limit]
        idx = idx[np.argsort(-sims[idx])]
        for pos in idx:
            doc_id = self.doc_ids[pos]
            scores[doc_id] = float(sims[pos])
            order.append(doc_id)
        return scores, order

    def _should_use_ann(self) -> bool:
        if faiss is None:
            return False
        if len(self.doc_ids) < max(1, self.ann_threshold):
            return False
        self._ensure_ann_index()
        return self.ann_index is not None

    def _ensure_ann_index(self) -> None:
        if not self._ann_dirty:
            return
        if faiss is None or not self.doc_ids:
            self.ann_index = None
            self._ann_dirty = False
            return
        if len(self.doc_ids) < max(1, self.ann_threshold):
            self.ann_index = None
            self._ann_dirty = False
            return
        self._ensure_matrix()
        if self.Z is None or self.Z.size == 0:
            self.ann_index = None
            self._ann_dirty = False
            return
        dim = self.Z.shape[1]
        try:
            hnsw_index = faiss.IndexHNSWFlat(dim, max(8, int(self.ann_m)))
        except Exception:
            self.ann_index = None
            self._ann_dirty = False
            return
        hnsw_index.hnsw.efConstruction = max(16, int(self.ann_ef_construction))
        hnsw_index.hnsw.efSearch = max(8, int(self.ann_ef_search))
        ids = np.asarray(self.doc_ids, dtype=np.int64)
        target_index = hnsw_index if hasattr(hnsw_index, "add_with_ids") else faiss.IndexIDMap(hnsw_index)
        try:
            target_index.add_with_ids(self.Z, ids)
        except Exception:
            target_index = faiss.IndexIDMap(hnsw_index)
            target_index.add_with_ids(self.Z, ids)
        self.ann_index = target_index
        self._ann_hnsw = hnsw_index
        self._ann_dirty = False

    def _ann_scores(self, qvec: np.ndarray, fetch: int) -> Tuple[Dict[int, float], List[int]]:
        self._ensure_ann_index()
        if self.ann_index is None:
            return {}, []
        k = min(len(self.doc_ids), max(fetch, self.ann_ef_search))
        if k <= 0:
            return {}, []
        query = qvec.reshape(1, -1).astype(np.float32, copy=False)
        try:
            hnsw_struct = None
            if hasattr(self.ann_index, "hnsw"):
                hnsw_struct = self.ann_index.hnsw
            elif hasattr(self, "_ann_hnsw") and hasattr(self._ann_hnsw, "hnsw"):
                hnsw_struct = self._ann_hnsw.hnsw
            if hnsw_struct is not None:
                hnsw_struct.efSearch = max(self.ann_ef_search, fetch)
            distances, ids = self.ann_index.search(query, k)
        except Exception:
            return {}, []
        scores: Dict[int, float] = {}
        order: List[int] = []
        if ids.size == 0:
            return scores, order
        for doc_id in ids[0]:
            if doc_id < 0:
                continue
            doc_id_int = int(doc_id)
            vec = self.embeddings.get(doc_id_int)
            if vec is None:
                continue
            raw = float(np.dot(vec, qvec))
            scores[doc_id_int] = raw
            order.append(doc_id_int)
            if len(order) >= fetch:
                break
        return scores, order

    def _lexical_scores(self, query_tokens: Optional[List[str]], fetch: int) -> Dict[int, float]:
        if not query_tokens:
            return {}
        self._ensure_lexical_index()
        if self.lexical_index is None:
            return {}
        try:
            scores = self.lexical_index.get_scores(query_tokens)
        except Exception:
            return {}
        scores_arr = np.asarray(scores, dtype=np.float32)
        if scores_arr.size == 0:
            return {}
        limit = min(fetch, scores_arr.shape[0])
        idx = np.argpartition(-scores_arr, limit - 1)[:limit]
        idx = idx[np.argsort(-scores_arr[idx])]
        result: Dict[int, float] = {}
        for pos in idx:
            score = float(scores_arr[pos])
            if score <= 0:
                continue
            doc_id = self.doc_ids[pos]
            result[doc_id] = score
        return result

    def remove_paths(self, paths: Iterable[str]) -> int:
        to_remove: List[int] = []
        for raw in paths:
            doc_id = self._path_to_id.pop(self._normalize_path(raw), None)
            if doc_id is not None:
                to_remove.append(doc_id)

        removed = 0
        for doc_id in to_remove:
            if self._remove_doc_id(doc_id):
                removed += 1

        if removed:
            self._rebuild_lists()
            self._matrix_dirty = True
            self._mark_faiss_dirty()
            self._mark_lexical_dirty()
        return removed

    def _remove_doc_id(self, doc_id: int) -> bool:
        if doc_id not in self.entries:
            return False
        self.entries.pop(doc_id, None)
        self.lexical_tokens.pop(doc_id, None)
        self.embeddings.pop(doc_id, None)
        if doc_id in self.doc_ids:
            self.doc_ids.remove(doc_id)
        return True

    def upsert(
        self,
        *,
        path: str,
        ext: str,
        embedding: np.ndarray,
        preview: str,
        size: Optional[int] = None,
        mtime: Optional[float] = None,
        ctime: Optional[float] = None,
        owner: Optional[str] = None,
        tokens: Optional[List[str]] = None,
    ) -> int:
        doc_id = self._allocate_doc_id(path)
        normalized_path = self._normalize_path(path)
        self._path_to_id[normalized_path] = doc_id

        self.embeddings[doc_id] = self._normalize_vector(embedding)
        entry = {
            "path": path,
            "ext": ext,
            "preview": preview,
            "size": size or 0,
            "mtime": mtime or 0.0,
            "ctime": ctime or 0.0,
            "owner": owner or "",
        }
        entry["preview"] = self._truncate_preview(entry.get("preview", ""))
        self.entries[doc_id] = entry
        self.lexical_tokens[doc_id] = list(tokens) if tokens is not None else self._tokenize_entry(entry)

        if doc_id not in self.doc_ids:
            self.doc_ids.append(doc_id)

        self._rebuild_lists()
        self._matrix_dirty = True
        self._mark_faiss_dirty()
        self._mark_lexical_dirty()
        return doc_id


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
            if torch.cuda.is_available():  # type: ignore[attr-defined]
                return "cuda"
        except Exception:
            pass
    return "cpu"


class CacheSignatureMonitor:
    """Polls cache signature changes and invokes a callback when stable."""

    def __init__(
        self,
        compute_signature: Callable[[], Tuple[float, float, float]],
        on_change: Callable[[Tuple[float, float, float], Tuple[float, float, float]], None],
        *,
        interval: float = 1.5,
        stability_checks: int = 2,
        thread_name: str = "retriever-cache-monitor",
    ) -> None:
        self._compute_signature = compute_signature
        self._on_change = on_change
        self._interval = max(0.1, float(interval))
        self._stability_checks = max(1, int(stability_checks))
        self._thread_name = thread_name

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._last_signature: Optional[Tuple[float, float, float]] = None
        self._pending_signature: Optional[Tuple[float, float, float]] = None
        self._pending_hits = 0
        self._last_error: Optional[BaseException] = None

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, name=self._thread_name, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            thread = self._thread
            if not thread:
                self._stop_event.set()
                return
        self._stop_event.set()
        thread.join(timeout=max(0.1, self._interval * 2))
        with self._lock:
            self._thread = None

    close = stop

    def is_running(self) -> bool:
        with self._lock:
            return bool(self._thread and self._thread.is_alive())

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def prime(self, signature: Tuple[float, float, float]) -> None:
        with self._lock:
            self._last_signature = signature
            self._pending_signature = None
            self._pending_hits = 0

    def check_once(self) -> None:
        try:
            current = self._compute_signature()
        except BaseException as exc:  # pragma: no cover - defensive guard
            self._last_error = exc
            return

        with self._lock:
            previous = self._last_signature
            if previous is None:
                self._last_signature = current
                self._pending_signature = None
                self._pending_hits = 0
                return

            if current == previous:
                self._pending_signature = None
                self._pending_hits = 0
                return

            if self._pending_signature != current:
                self._pending_signature = current
                self._pending_hits = 1
                return

            self._pending_hits += 1
            if self._pending_hits < self._stability_checks:
                return

            self._pending_signature = None
            self._pending_hits = 0
            self._last_signature = current
            before = previous

        try:
            self._on_change(before, current)
        except Exception as exc:  # pragma: no cover - defensive guard
            self._last_error = exc

    @property
    def last_signature(self) -> Optional[Tuple[float, float, float]]:
        with self._lock:
            return self._last_signature

    @property
    def last_error(self) -> Optional[BaseException]:
        return self._last_error

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.check_once()
            self._stop_event.wait(self._interval)


class Retriever:
    def __init__(
        self,
        model_path: Path,
        corpus_path: Path,
        cache_dir: Path = Path("./index_cache"),
        *,
        search_wait_timeout: float = 0.5,
        use_rerank: bool = True,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rerank_depth: int = 80,
        rerank_batch_size: int = 16,
        rerank_device: Optional[str] = None,
        rerank_min_score: Optional[float] = 0.35,
        lexical_weight: float = 0.0,
        min_similarity: float = 0.35,
        auto_refresh: bool = True,
        refresh_interval: float = 1.5,
        refresh_stability_checks: int = 2,
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
        base_weight = max(0.0, min(1.0, float(lexical_weight)))
        self.base_lexical_weight = base_weight
        self.lexical_weight = base_weight
        self.min_similarity = max(0.0, min(1.0, float(min_similarity)))
        try:
            self.rerank_min_score = float(rerank_min_score) if rerank_min_score is not None else None
        except (TypeError, ValueError):
            self.rerank_min_score = None
        self._reranker: Optional[CrossEncoderReranker] = None
        self.index_manager = IndexManager(
            loader=self._load_cached_index,
            builder=self._rebuild_index,
        )

        self._cache_signature: Optional[Tuple[float, float, float]] = self._compute_cache_signature()
        self._cache_monitor: Optional[CacheSignatureMonitor] = None
        self._auto_refresh = bool(auto_refresh)
        self._refresh_interval = max(0.1, float(refresh_interval)) if refresh_interval else 0.0
        self._refresh_stability_checks = max(1, int(refresh_stability_checks))

        if self._auto_refresh and self._refresh_interval > 0.0:
            self._cache_monitor = CacheSignatureMonitor(
                self._compute_cache_signature,
                self._on_cache_signature_change,
                interval=self._refresh_interval,
                stability_checks=self._refresh_stability_checks,
            )
            if self._cache_signature is not None:
                self._cache_monitor.prime(self._cache_signature)
            self._cache_monitor.start()

        self._finalizer = weakref.finalize(self, self._shutdown_background_tasks)

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

    def shutdown(self) -> None:
        finalizer = getattr(self, "_finalizer", None)
        if finalizer is not None and getattr(finalizer, "alive", False):
            finalizer()

    def _shutdown_background_tasks(self) -> None:
        monitor = getattr(self, "_cache_monitor", None)
        if monitor is not None:
            try:
                monitor.stop()
            except Exception:
                pass
        if hasattr(self, "index_manager") and self.index_manager is not None:
            try:
                self.index_manager.shutdown()
            except Exception:
                pass

    def _ensure_index(self) -> Optional[VectorIndex]:
        self._refresh_if_cache_changed()
        index = self.index_manager.get_index(wait=False)
        if index is not None:
            return index
        self.index_manager.ensure_loaded()
        if not self.index_manager.wait_until_ready(timeout=self.search_wait_timeout):
            return None
        return self.index_manager.get_index(wait=False)

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        session: Optional[SessionState] = None,
        use_ann: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        index = self._ensure_index()
        if index is None:
            return []

        if session is not None:
            session.add_query(query)

        available_exts: Set[str] = set()
        for ext in getattr(index, "exts", []):
            normalized = _normalize_ext(ext)
            if normalized:
                available_exts.add(normalized)
        requested_exts = _extract_query_exts(query, available_exts=available_exts)
        metadata_filters = _extract_metadata_filters(query)
        corpus_size = len(getattr(index, "doc_ids", [])) if hasattr(index, "doc_ids") else 0

        search_params = _dynamic_search_params(
            query,
            top_k,
            metadata_filters=metadata_filters,
            requested_exts=requested_exts,
        )

        oversample = max(
            search_params["oversample"],
            _dynamic_oversample(
                top_k,
                has_ext_pref=bool(requested_exts),
                filters_active=metadata_filters.is_active(),
                corpus_size=corpus_size,
            ),
        )

        should_expand = _should_expand_query(
            query,
            metadata_filters=metadata_filters,
            requested_exts=requested_exts,
        )
        vector_query = _expand_query_text(query) if should_expand else query
        query_lower = (query or "").lower()
        raw_query_tokens_set: Set[str] = {tok for tok in _split_tokens(query_lower) if tok}
        expanded_query_tokens_set: Set[str] = {
            tok for tok in _split_tokens(vector_query.lower()) if tok
        }
        q = self.encoder.encode_query(vector_query)
        q_vector: Optional[np.ndarray]
        try:
            q_array = np.asarray(q, dtype=np.float32)
            if q_array.ndim == 0:
                q_vector = q_array.reshape(1)
            else:
                q_vector = q_array.reshape(-1)
        except Exception:
            q_vector = None
        query_tokens: Optional[List[str]] = None
        if getattr(self, "base_lexical_weight", 0.0) > 0.0:
            query_tokens = list(expanded_query_tokens_set)
        configured_rerank_depth = int(getattr(self, "rerank_depth", 0) or 0)
        effective_rerank_depth = max(search_params["rerank_depth"], configured_rerank_depth)
        use_rerank = bool(getattr(self, "use_rerank", False) and effective_rerank_depth > 0)
        fusion_depth = search_params["fusion_depth"]
        search_top_k = max(top_k, 1, effective_rerank_depth if use_rerank else top_k)
        search_oversample = oversample
        if use_rerank:
            search_top_k = max(search_top_k, effective_rerank_depth)
            search_oversample = max(1, min(oversample, 2))
        adaptive_lex_weight = self._dynamic_lexical_weight(query_tokens)
        self._last_lexical_weight = adaptive_lex_weight
        if hasattr(index, "configure_ann"):
            ann_ef = max(32, search_top_k * max(1, search_oversample))
            try:
                index.configure_ann(ef_search=ann_ef)
            except Exception:
                pass
        raw_hits = index.search(
            q,
            top_k=search_top_k,
            oversample=search_oversample,
            lexical_weight=adaptive_lex_weight,
            query_tokens=query_tokens,
            min_similarity=self.min_similarity,
            use_ann=use_ann,
        )
        filtered_hits = _apply_metadata_filters(raw_hits, metadata_filters)
        if not filtered_hits:
            return []

        lexical_limit = max(top_k, 1)
        if use_rerank and effective_rerank_depth:
            lexical_limit = max(lexical_limit, min(effective_rerank_depth, len(filtered_hits)))

        lexical_ranking = _rerank_hits(
            query,
            vector_query,
            filtered_hits,
            desired_exts=requested_exts,
            top_k=lexical_limit,
            session=session,
        )

        if not use_rerank:
            mmr_limit = max(top_k, min(len(lexical_ranking), fusion_depth))
            mmr_candidates = lexical_ranking[:mmr_limit]
            final_hits = _mmr(index, mmr_candidates, q_vector, top_k)
            return _annotate_hits(
                final_hits,
                desired_exts=requested_exts,
                raw_query_tokens=raw_query_tokens_set,
                expanded_query_tokens=expanded_query_tokens_set,
                metadata_filters=metadata_filters,
                lexical_weight=adaptive_lex_weight,
            )

        reranker = self._ensure_reranker()
        if reranker is None:
            mmr_limit = max(top_k, min(len(lexical_ranking), fusion_depth))
            mmr_candidates = lexical_ranking[:mmr_limit]
            final_hits = _mmr(index, mmr_candidates, q_vector, top_k)
            return _annotate_hits(
                final_hits,
                desired_exts=requested_exts,
                raw_query_tokens=raw_query_tokens_set,
                expanded_query_tokens=expanded_query_tokens_set,
                metadata_filters=metadata_filters,
                lexical_weight=adaptive_lex_weight,
            )

        reranked = reranker.rerank(
            query,
            lexical_ranking,
            desired_exts=requested_exts,
            session=session,
        )
        if self.rerank_min_score is not None:
            filtered: List[Dict[str, Any]] = []
            threshold = self.rerank_min_score

            for hit in reranked:
                raw_score = hit.get("rerank_score", hit.get("score", hit.get("similarity", 0.0)))
                try:
                    value = float(raw_score)
                except (TypeError, ValueError):
                    continue
                if value >= threshold:
                    filtered.append(hit)

            if filtered:
                reranked = filtered
            else:
                return []

        rank_sources: List[List[Dict[str, Any]]] = []
        if lexical_ranking:
            rank_sources.append(lexical_ranking[:fusion_depth])
        if reranked:
            rank_sources.append(reranked[:fusion_depth])
        fused_candidates = _rrf(rank_sources) if rank_sources else []

        if not fused_candidates:
            fused_candidates = reranked[:fusion_depth]
        mmr_pool_size = max(top_k * 2, fusion_depth)
        mmr_candidates = fused_candidates[:mmr_pool_size]
        final_hits = _mmr(index, mmr_candidates, q_vector, top_k)
        return _annotate_hits(
            final_hits,
            desired_exts=requested_exts,
            raw_query_tokens=raw_query_tokens_set,
            expanded_query_tokens=expanded_query_tokens_set,
            metadata_filters=metadata_filters,
            lexical_weight=adaptive_lex_weight,
        )

    def _ensure_reranker(self) -> Optional[CrossEncoderReranker]:
        if not getattr(self, "use_rerank", False):
            return None
        if getattr(self, "_reranker", None) is not None:
            return self._reranker
        try:
            device = _pick_rerank_device(self.rerank_device)
            self._reranker = CrossEncoderReranker(
                self.rerank_model,
                device=device,
                batch_size=self.rerank_batch_size,
            )
        except Exception as exc:
            logger.warning("reranker load failed; disabling rerank: %s", exc)
            self.use_rerank = False
            self._reranker = None
        return getattr(self, "_reranker", None)

    def _dynamic_lexical_weight(self, query_tokens: Optional[List[str]]) -> float:
        base = max(0.0, float(getattr(self, "base_lexical_weight", 0.0)))
        if base <= 0.0:
            return 0.0
        if not query_tokens:
            return base
        distinct = len({tok for tok in query_tokens if tok})
        if distinct <= 2:
            return min(0.75, max(base, 0.45))
        if distinct >= 8:
            return max(0.15, min(base, 0.30))
        return base

    def _load_cached_index(self) -> Optional[VectorIndex]:
        emb_npy = self.cache_dir / "doc_embeddings.npy"
        meta_json = self.cache_dir / "doc_meta.json"
        faiss_path = self.cache_dir / "doc_index.faiss"
        if not meta_json.exists():
            return None

        index = VectorIndex()
        try:
            index.load(
                emb_npy if emb_npy.exists() else None,
                meta_json,
                faiss_path=faiss_path if faiss_path.exists() else None,
                use_mmap=True,
            )
        except Exception as exc:
            logger.warning("index load failed; rebuild scheduled: %s", exc)
            return None

        if not self._index_matches_model(index):
            logger.warning("index dimension mismatch detected; triggering rebuild")
            return None

        logger.info("index loaded: cache=%s", _mask_path(str(self.cache_dir)))
        self._cache_signature = self._compute_cache_signature()
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
        if work.empty:
            raise RuntimeError("유효 텍스트 문서가 없습니다.")

        logger.info("encoding documents for index build: docs=%d", len(work))
        Z = self.encoder.encode_docs(work[MODEL_TEXT_COLUMN].tolist())

        preview_source = work.get("preview")
        if preview_source is None:
            preview_source = work.get("text_original")
        if preview_source is None:
            preview_source = work.get("text")
        if preview_source is None:
            preview_source = work[MODEL_TEXT_COLUMN]
        preview_list = preview_source.fillna("").astype(str).tolist()

        token_lists: Optional[List[List[str]]] = None
        if BM25Okapi is not None:
            tokens_raw = work.get("tokens")
            if tokens_raw is not None:
                token_lists = [tokens_raw.iloc[i] if isinstance(tokens_raw.iloc[i], list) else _split_tokens(tokens_raw.iloc[i]) for i in range(len(work))]
            else:
                token_lists = [
                    [tok for tok in _split_tokens(preview_list[idx].lower()) if tok]
                    for idx in range(len(work))
                ]
                total_tokens = sum(len(tokens) for tokens in token_lists)
                if total_tokens > MAX_BM25_TOKENS:
                    factor = MAX_BM25_TOKENS / total_tokens
                    limited_tokens: List[List[str]] = []
                    truncated = 0
                    for tokens in token_lists:
                        keep = max(1, int(len(tokens) * factor))
                        if keep < len(tokens):
                            truncated += len(tokens) - keep
                        limited_tokens.append(tokens[:keep])
                    token_lists = limited_tokens
                    if truncated:
                        logger.info(
                            "bm25 tokens truncated: removed=%d limit=%d",
                            truncated,
                            MAX_BM25_TOKENS,
                        )

        size_list = work["size"].fillna(0).astype(int).tolist() if "size" in work.columns else [0] * len(work)
        mtime_list = work["mtime"].fillna(0.0).astype(float).tolist() if "mtime" in work.columns else [0.0] * len(work)
        ctime_list = work["ctime"].fillna(0.0).astype(float).tolist() if "ctime" in work.columns else [0.0] * len(work)
        owner_list = work["owner"].fillna("").astype(str).tolist() if "owner" in work.columns else [""] * len(work)
        drive_list = work["drive"].fillna("").astype(str).tolist() if "drive" in work.columns else [""] * len(work)

        extra_meta: Optional[List[Dict[str, Any]]] = None
        chunk_columns = [col for col in ("chunk_id", "chunk_count", "chunk_tokens") if col in work.columns]
        if chunk_columns:
            extra_meta = []
            for i in range(len(work)):
                meta: Dict[str, Any] = {}
                if "chunk_id" in work.columns:
                    value = work["chunk_id"].iloc[i]
                    if pd.notna(value):
                        meta["chunk_id"] = int(value)
                if "chunk_count" in work.columns:
                    value = work["chunk_count"].iloc[i]
                    if pd.notna(value):
                        meta["chunk_count"] = int(value)
                if "chunk_tokens" in work.columns:
                    value = work["chunk_tokens"].iloc[i]
                    if pd.notna(value):
                        meta["chunk_tokens"] = int(value)
                extra_meta.append(meta)

        index = VectorIndex()
        index.build(
            Z,
            work["path"].tolist(),
            work["ext"].tolist(),
            preview_list,
            tokens=token_lists,
            sizes=size_list,
            mtimes=mtime_list,
            ctimes=ctime_list,
            owners=owner_list,
            drives=drive_list,
            extra_meta=extra_meta,
        )
        paths = index.save(self.cache_dir)
        saved_files = [str(paths.meta_json)]
        if paths.emb_npy:
            saved_files.append(str(paths.emb_npy))
        if paths.faiss_index:
            saved_files.append(str(paths.faiss_index))
        logger.info(
            "index saved: %s",
            ", ".join(_mask_path(path) for path in saved_files),
        )

        fresh = VectorIndex()
        fresh.load(
            paths.emb_npy,
            paths.meta_json,
            faiss_path=paths.faiss_index,
            use_mmap=True,
        )
        self._cache_signature = self._compute_cache_signature()
        return fresh

    def _load_corpus(self):
        logger.info("loading corpus: path=%s", _mask_path(str(self.corpus_path)))
        if self.corpus_path.suffix.lower() == ".parquet":
            engine_kwargs = {}
            engine_label = PARQUET_ENGINE or "auto"
            if PARQUET_ENGINE:
                engine_kwargs["engine"] = PARQUET_ENGINE
            try:
                return pd.read_parquet(self.corpus_path, **engine_kwargs)
            except Exception as exc:
                logger.warning(
                    "parquet load failed (engine=%s); retrying via CSV: %s",
                    engine_label,
                    exc,
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

    def _compute_cache_signature(self) -> Tuple[float, float, float]:
        def _mtime(path: Path) -> float:
            try:
                return path.stat().st_mtime
            except OSError:
                return 0.0

        emb = self.cache_dir / "doc_embeddings.npy"
        meta = self.cache_dir / "doc_meta.json"
        faiss_path = self.cache_dir / "doc_index.faiss"
        return (_mtime(meta), _mtime(emb), _mtime(faiss_path))

    def _refresh_if_cache_changed(self) -> None:
        current = self._compute_cache_signature()
        previous = getattr(self, "_cache_signature", None)
        if previous is None:
            self._cache_signature = current
            monitor = getattr(self, "_cache_monitor", None)
            if monitor is not None:
                monitor.prime(current)
            return
        if current != previous:
            self._on_cache_signature_change(previous, current)

    def _on_cache_signature_change(
        self,
        previous: Tuple[float, float, float],
        current: Tuple[float, float, float],
    ) -> None:
        self._cache_signature = current
        try:
            self.index_manager.clear()
            loaded = self.index_manager.ensure_loaded()
            if loaded is None:
                self.index_manager.schedule_rebuild(priority=True)
        except Exception as exc:
            logger.warning("index refresh after cache change failed: %s", exc)

    @staticmethod
    def format_results(query: str, results: List[Dict[str, Any]]) -> str:
        if not results:
            return f"“{query}”와 유사한 문서를 찾지 못했습니다."
        lines = [f"‘{query}’와 유사한 문서 Top {len(results)}:"]
        for i, r in enumerate(results, 1):
            vector_similarity = r.get("vector_similarity", r.get("similarity"))
            similarity_label = _similarity_to_percent(vector_similarity)
            final_score = r.get("combined_score", r.get("score", 0.0))
            try:
                final_score = float(final_score)
            except (TypeError, ValueError):
                final_score = 0.0
            path_raw = str(r.get("path", "") or "")
            name = _mask_path(path_raw) or path_raw or "<unknown>"
            ext_label = str(r.get("ext", "") or "")
            ext_display = f" [{ext_label}]" if ext_label else ""
            lines.append(
                f"{i}. {name}{ext_display}  유사도={similarity_label}  종합점수={final_score:.3f}"
            )
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
            drive_label = str(r.get("drive") or "").strip()
            if drive_label:
                meta_bits.append(f"드라이브 {drive_label}")
            if meta_bits:
                lines.append("   메타: " + ", ".join(meta_bits))
            if r.get("preview"):
                lines.append(f"   미리보기: {r['preview']}")
        return "\n".join(lines)


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
        sizes = sizes.fillna(0).astype(int)

    mtimes = df.get("mtime")
    if mtimes is None:
        mtimes = pd.Series([0.0] * len(df))
    else:
        mtimes = mtimes.fillna(0.0).astype(float)

    ctimes = df.get("ctime")
    if ctimes is None:
        ctimes = pd.Series([0.0] * len(df))
    else:
        ctimes = ctimes.fillna(0.0).astype(float)

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
