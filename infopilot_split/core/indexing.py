"""indexing module split from pipeline (auto-split from originals)."""
from __future__ import annotations
from pathlib import Path

from .retrieval import Retriever

def run_indexing(corpus_path: Path, cache_dir: Path):
    print("🚀 Starting semantic indexing...")
    retriever = Retriever(corpus_path=corpus_path, cache_dir=cache_dir)
    retriever.ready(rebuild=True)
    print("✨ Indexing complete.")
