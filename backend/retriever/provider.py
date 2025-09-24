from __future__ import annotations

from pathlib import Path

from retriever import Retriever
from backend.api.settings import Settings


def real_retriever_factory(settings: Settings) -> Retriever:
    retr = Retriever(
        model_path=Path("data/topic_model.joblib"),
        corpus_path=Path("data/corpus.parquet"),
        cache_dir=Path("index_cache"),
    )
    retr.ready(wait=False)
    return retr
