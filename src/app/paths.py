"""Common paths & artifact checks (auto-split from originals)."""
from __future__ import annotations


from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class Paths:
    base: Path = Path.cwd()
    data_dir: Path = base / "data"
    models_dir: Path = base / "models"
    cache_dir: Path = base / "index_cache"
    corpus_csv: Path = data_dir / "corpus.csv"
    corpus_parquet: Path = data_dir / "corpus.parquet"
    topic_model: Path = models_dir / "topic_model.joblib"

    def ensure(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self

def have_all_artifacts(p: Paths) -> bool:
    return (p.corpus_csv.exists() or p.corpus_parquet.exists()) and p.topic_model.exists()
