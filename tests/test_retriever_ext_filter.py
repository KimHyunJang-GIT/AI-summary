import sys
from pathlib import Path
from types import SimpleNamespace
import types

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

from retriever import Retriever


def _make_stub_retriever() -> Retriever:
    retr = Retriever.__new__(Retriever)  # type: ignore
    retr.model_path = Path("dummy_model.joblib")
    retr.corpus_path = Path("dummy_corpus.csv")
    retr.cache_dir = Path("dummy_cache")
    retr.encoder = SimpleNamespace(
        encode_query=lambda query: query,
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
            self._hits = [
                {"path": "xlsx_doc", "ext": ".xlsx", "similarity": 0.99, "preview": ""},
                {"path": "pdf_doc_one", "ext": ".pdf", "similarity": 0.95, "preview": ""},
                {"path": "pdf_doc_two", "ext": ".pdf", "similarity": 0.94, "preview": ""},
                {"path": "xlsm_budget", "ext": ".xlsm", "similarity": 0.93, "preview": ""},
                {"path": "docx_doc", "ext": ".docx", "similarity": 0.92, "preview": ""},
                {"path": "pptx_deck", "ext": ".pptx", "similarity": 0.91, "preview": ""},
                {"path": "doc_report", "ext": ".doc", "similarity": 0.90, "preview": ""},
                {"path": "hwp_contract", "ext": ".hwp", "similarity": 0.89, "preview": ""},
            ]

        def search(self, _qvec, top_k: int, oversample: int = 1):
            fetch = max(1, min(len(self._hits), top_k * max(1, oversample)))
            return self._hits[:fetch]

    retr.index = SimpleIndex()
    retr._ready = True
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
