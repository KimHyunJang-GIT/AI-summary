"""Vector index storage and search (auto-split from originals)."""
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass # dataclass 임포트 추가

import numpy as np
import pandas as pd
from sentence_transformers import util

class VectorIndex:
    def __init__(self):
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: Optional[pd.DataFrame] = None

    def build(self, embeddings: np.ndarray, metadata: pd.DataFrame):
        self.embeddings = embeddings
        self.metadata = metadata

    def save(self, cache_dir: Path) -> 'Paths':
        cache_dir.mkdir(parents=True, exist_ok=True)
        emb_path = cache_dir / "doc_embeddings.npy"
        meta_path = cache_dir / "doc_meta.jsonl"
        np.save(emb_path, self.embeddings)
        self.metadata.to_json(meta_path, orient='records', lines=True, force_ascii=False)
        return self.Paths(emb_npy=emb_path, meta_json=meta_path)

    def load(self, emb_path: Path, meta_path: Path):
        self.embeddings = np.load(emb_path)
        self.metadata = pd.read_json(meta_path, lines=True)

    def search(self, query_embedding: np.ndarray, top_k: int, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if self.embeddings is None or self.metadata is None:
            return []

        # 필터링 조건이 있으면 데이터프레임 필터링
        filtered_df = self.metadata
        if filters:
            for key, value in filters.items():
                if key in filtered_df.columns:
                    # 'path' 필터는 경로 문자열에 특정 단어가 포함되어 있는지 확인
                    if key == 'path':
                        filtered_df = filtered_df[filtered_df[key].str.contains(value, case=False, na=False)]
                    else:
                        # 다른 필터는 완전 일치(case-insensitive)로 처리
                        filtered_df = filtered_df[filtered_df[key].str.lower() == str(value).lower()]

        if filtered_df.empty:
            return []

        # 필터링된 인덱스를 사용하여 검색
        filtered_indices = filtered_df.index.to_numpy()
        filtered_embeddings = self.embeddings[filtered_indices]

        if len(filtered_embeddings) == 0:
            return []

        # 코사인 유사도 계산
        cos_scores = util.pytorch_cos_sim(query_embedding, filtered_embeddings)[0].cpu().numpy()
        
        # 상위 K개 결과 선정
        top_k_filtered = min(top_k, len(cos_scores))
        top_results_indices_filtered = np.argpartition(-cos_scores, range(top_k_filtered))[:top_k_filtered]

        results = []
        for i in top_results_indices_filtered:
            original_index = filtered_indices[i]
            hit = self.metadata.iloc[original_index].to_dict()
            hit['similarity'] = cos_scores[i]
            results.append(hit)
        
        return sorted(results, key=lambda x: x['similarity'], reverse=True)

    @dataclass(frozen=True)
    class Paths:
        emb_npy: Path
        meta_json: Path
