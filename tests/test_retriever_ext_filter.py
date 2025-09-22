import sys
import time
from pathlib import Path
from types import SimpleNamespace
import types

import retriever

if "numpy" not in sys.modules:
    np_stub = types.ModuleType("numpy")
    np_stub.ndarray = list
    np_stub.float32 = float

    def _unsupported(*_args, **_kwargs):  # pragma: no cover - 테스트 환경 안전장치
        raise NotImplementedError("numpy 기능은 테스트 더블로 대체됩니다.")

    np_stub.array = _unsupported
    np_stub.stack = _unsupported
    np_stub.linalg = types.SimpleNamespace(norm=_unsupported)
    np_stub.argpartition = _unsupported
    np_stub.argsort = _unsupported
    np_stub.save = _unsupported
    np_stub.load = _unsupported
    np_stub.dot = _unsupported
    sys.modules["numpy"] = np_stub

try:
    import pytest
except ModuleNotFoundError:  # pytest 미설치 환경 대비
    class _Mark:
        def smoke(self, func):
            return func

        def parametrize(self, _argnames, argvalues):
            def decorator(func):
                def wrapper():
                    for params in argvalues:
                        if not isinstance(params, tuple):
                            params = (params,)
                        func(*params)
                return wrapper
            return decorator

    pytest = types.SimpleNamespace(mark=_Mark())

from retriever import (
    Retriever,
    _metadata_text,
    _similarity_to_percent,
    _pick_rerank_device,
)


def _make_stub_retriever() -> Retriever:
    retr = Retriever.__new__(Retriever)  # type: ignore
    retr.model_path = Path("dummy_model.joblib")
    retr.corpus_path = Path("dummy_corpus.csv")
    retr.cache_dir = Path("dummy_cache")
    def _encode_query(query: str):
        retr._last_query = query
        return query

    retr.encoder = SimpleNamespace(
        encode_query=_encode_query,
        encode_docs=lambda docs: docs,
    )

    class SimpleIndex:
        def __init__(self):
            self.exts = [
                ".xlsx",
                ".pdf",
                ".pdf",
                ".xlsm",
                ".docx",
                ".pptx",
                ".doc",
                ".hwp",
            ]
            now = time.time()
            year = 365 * 24 * 3600
            self._hits = [
                {"path": "xlsx_doc", "ext": ".xlsx", "similarity": 0.99, "preview": "", "mtime": now - (0.1 * year), "size": 5_000},
                {"path": "pdf_doc_one", "ext": ".pdf", "similarity": 0.95, "preview": "", "mtime": now - (0.2 * year), "size": 20_000},
                {"path": "pdf_doc_two", "ext": ".pdf", "similarity": 0.94, "preview": "", "mtime": now - (1.1 * year), "size": 18_000},
                {"path": "xlsm_budget", "ext": ".xlsm", "similarity": 0.93, "preview": "", "mtime": now - (2.0 * year), "size": 35_000},
                {"path": "docx_doc", "ext": ".docx", "similarity": 0.92, "preview": "", "mtime": now - (3.1 * year), "size": 12_000},
                {"path": "pptx_deck", "ext": ".pptx", "similarity": 0.91, "preview": "", "mtime": now - (4.5 * year), "size": 8_000},
                {"path": "doc_report", "ext": ".doc", "similarity": 0.90, "preview": "", "mtime": now - (5.0 * year), "size": 16_000},
                {"path": "hwp_contract", "ext": ".hwp", "similarity": 0.89, "preview": "", "mtime": now - (6.0 * year), "size": 14_000},
            ]

        def search(self, _qvec, top_k: int, oversample: int = 1):
            fetch = max(1, min(len(self._hits), top_k * max(1, oversample)))
            return self._hits[:fetch]

    simple_index = SimpleIndex()

    class StubIndexManager:
        def __init__(self, index):
            self._index = index

        def ensure_loaded(self):
            return self._index

        def schedule_rebuild(self, priority: bool = False):
            return None

        def wait_until_ready(self, timeout=None):
            return True

        def get_index(self, wait: bool = False, timeout=None):
            return self._index

    retr.index_manager = StubIndexManager(simple_index)
    retr.index = simple_index  # 편의상 노출 (기존 테스트 호환)
    retr.search_wait_timeout = 0.0
    return retr


@pytest.mark.smoke
def test_pdf_extension_query_prioritises_pdf_hits():
    retr = _make_stub_retriever()
    hits = retr.search("확장자가 pdf인 파일 어떤것있어", top_k=3)
    assert len(hits) == 3
    assert [h["ext"] for h in hits[:2]] == [".pdf", ".pdf"]
    assert hits[0]["path"] == "pdf_doc_one"


@pytest.mark.smoke
def test_query_without_extension_keeps_original_order():
    retr = _make_stub_retriever()
    hits = retr.search("파일 어떤것있어", top_k=3)
    assert len(hits) == 3
    assert hits[0]["ext"] == ".xlsx"


@pytest.mark.smoke
def test_semantic_query_expansion_hits_expected_extensions():
    retr = _make_stub_retriever()
    hits = retr.search("계약서 보여줘", top_k=2)
    expanded_query = getattr(retr, "_last_query", "")
    assert "contract" in expanded_query or "agreement" in expanded_query
    assert hits[0]["ext"] in {".pdf", ".hwp"}


@pytest.mark.parametrize(
    "query, expected_exts",
    [
        ("파워 포인트 자료 있어?", {".ppt", ".pptx"}),
        ("ms word 보고서 찾고싶어", {".doc", ".docx"}),
        ("spreadsheet 공유해줘", {".xls", ".xlsx", ".xlsm", ".xlsb", ".xltx"}),
        ("계약서 파일 보여줘", {".pdf", ".hwp"}),
        ("예산 자료 어디 있어?", {".xlsx", ".xls", ".xlsm"}),
    ],
)
def test_extension_synonym_queries_prioritise_expected_type(query, expected_exts):
    retr = _make_stub_retriever()
    hits = retr.search(query, top_k=2)
    assert hits, "검색 결과가 비어있습니다."
    assert hits[0]["ext"] in expected_exts


@pytest.mark.smoke
def test_metadata_text_includes_extension_context_keywords():
    metadata = _metadata_text("팀/분기_계획.pptx", ".pptx", "driveA")
    assert "pptx" in metadata
    assert "파워포인트" in metadata
    assert "presentation" in metadata


@pytest.mark.smoke
def test_filename_overlap_boosts_results_without_explicit_extension():
    retr = _make_stub_retriever()
    retr.index._hits = [
        {"path": "random_notes.txt", "ext": ".txt", "similarity": 0.99, "preview": ""},
        {"path": "연간 계획.docx", "ext": ".docx", "similarity": 0.88, "preview": ""},
        {"path": "yearly_budget.xlsx", "ext": ".xlsx", "similarity": 0.85, "preview": ""},
    ]
    retr.index.exts = [hit["ext"] for hit in retr.index._hits]

    hits = retr.search("연간 계획 자료", top_k=2)
    assert hits[0]["path"] == "연간 계획.docx"
    assert hits[0]["score"] >= hits[1]["score"]


@pytest.mark.smoke
def test_metadata_filters_recognise_relative_year():
    retr = _make_stub_retriever()
    now = time.time()
    # Make only one document fall within 3 years ago window
    retr.index._hits[4]["mtime"] = now - (3 * 365 * 24 * 3600)
    retr.index._hits[5]["mtime"] = now - (6 * 365 * 24 * 3600)
    hits = retr.search("3년 전 작성한 자료", top_k=3)
    assert hits and hits[0]["path"] == "docx_doc"


def test_similarity_to_percent_clamps_range():
    assert _similarity_to_percent(0.873) == "87.3%"
    assert _similarity_to_percent(1.4) == "100.0%"
    assert _similarity_to_percent(-0.2) == "0.0%"
    assert _similarity_to_percent("oops") == "-"


def test_pick_rerank_device_prefers_explicit_value():
    assert _pick_rerank_device("cuda:1") == "cuda:1"


def test_pick_rerank_device_uses_cuda_when_available():
    original_torch = retriever.torch

    class _CudaTorch:
        class cuda:  # type: ignore
            @staticmethod
            def is_available() -> bool:
                return True

    retriever.torch = _CudaTorch()
    try:
        assert _pick_rerank_device(None) == "cuda"
    finally:
        retriever.torch = original_torch


def test_pick_rerank_device_falls_back_to_cpu():
    original_torch = retriever.torch

    class _CpuTorch:
        class cuda:  # type: ignore
            @staticmethod
            def is_available() -> bool:
                return False

    retriever.torch = _CpuTorch()
    try:
        assert _pick_rerank_device(None) == "cpu"
    finally:
        retriever.torch = original_torch
