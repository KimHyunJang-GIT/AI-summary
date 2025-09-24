"""Pytest configuration helpers."""
from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
root_str = str(PROJECT_ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")

if "JOBLIB_TEMP_FOLDER" not in os.environ:
    joblib_tmp = Path(tempfile.mkdtemp(prefix="joblib-", dir=root_str))
    os.environ["JOBLIB_TEMP_FOLDER"] = str(joblib_tmp)
    atexit.register(shutil.rmtree, joblib_tmp, True)

from backend.api.app_factory import create_app
from backend.api.settings import Settings


class StubRetriever:
    def __init__(self) -> None:
        self.search_calls = []

    def search(self, query: str, top_k: int = 5, session=None):
        self.search_calls.append((query, top_k))
        if session is not None:
            session.add_query(query)
        return [
            {
                "path": "stub.txt",
                "ext": ".txt",
                "combined_score": 1.0,
                "match_reasons": ["stub"],
            }
        ]

    def ready(self, *args, **kwargs):  # pragma: no cover - stub
        return True


@pytest.fixture(autouse=True)
def set_testing_env(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    monkeypatch.setenv("TESTING", "1")
    yield
    os.environ.pop("TESTING", None)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    stub = StubRetriever()
    settings = Settings(TESTING=True, STARTUP_LOAD=False)

    app = create_app(settings=settings, retriever_provider=lambda: stub)

    with TestClient(app) as test_client:
        test_client.app.state.retriever = stub
        yield test_client
