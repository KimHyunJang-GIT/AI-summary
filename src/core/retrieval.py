from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import json # For saving/loading tokenized corpus
import numpy as np # numpy 임포트 추가
import sys # sys 모듈 임포트
import traceback # traceback 모듈 임포트

from src.config import MODEL_NAME, MODELS_DIR
from .index_store import VectorIndex
from src.core.helpers import _mask_path # Import _mask_path

# Helper function to convert numpy types to Python native types for JSON serialization
def _convert_numpy_types_to_python_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_numpy_types_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types_to_python_types(elem) for elem in obj]
    else:
        return obj


class Retriever:
    def __init__(self, corpus_path: Path, cache_dir: Path = Path("./index_cache")):
        if SentenceTransformer is None: raise ImportError("Please run: pip install sentence-transformers")
        self.corpus_path = Path(corpus_path)
        self.cache_dir = Path(cache_dir)
        print(f"🧠 Loading Semantic Model: {MODEL_NAME}...")
        self.model = SentenceTransformer(MODEL_NAME, cache_folder=str(MODELS_DIR))
        self.index = VectorIndex()
        self._ready = False

        self.bm25_index: Optional[BM25Okapi] = None
        self.corpus_metadata: Optional[pd.DataFrame] = None # Store corpus metadata for BM25
        
        self.cross_encoder: Optional[CrossEncoder] = None
        self.cross_encoder_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        print(f"🧠 Loading Cross-Encoder Model: {self.cross_encoder_model_name}...")
        self.cross_encoder = CrossEncoder(self.cross_encoder_model_name, max_length=512)


    def ready(self, rebuild: bool = False):
        emb_npy = self.cache_dir / "doc_embeddings.npy"
        meta_json = self.cache_dir / "doc_meta.jsonl"
        bm25_tokens_path = self.cache_dir / "bm25_tokens.json"

        if not rebuild and emb_npy.exists() and meta_json.exists() and bm25_tokens_path.exists():
            print(f"✅ Loading index from cache: {self.cache_dir}")
            try:
                self.index.load(emb_npy, meta_json)
                self.corpus_metadata = self.index.metadata # Use metadata from VectorIndex

                with open(bm25_tokens_path, 'r', encoding='utf-8') as f:
                    tokenized_corpus = json.load(f)
                self.bm25_index = BM25Okapi(tokenized_corpus)
                self._ready = True
                print("DEBUG: Retriever.ready - Loaded from cache successfully.")
                return
            except Exception as e:
                print(f"❌ Retriever.ready: Error loading from cache: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
                # 캐시 로드 실패 시, 재빌드 로직으로 넘어감

        if pd is None: raise RuntimeError("pandas is required. Please run: pip install pandas")

        print(f"📥 Loading corpus from {self.corpus_path}...")
        try:
            df = pd.read_parquet(self.corpus_path)
        except Exception as e:
            print(f"❌ Retriever.ready: Error loading corpus parquet: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            raise RuntimeError(f"Failed to load corpus from {self.corpus_path}. Error: {e}")

        work_df = df[df["ok"] & df["text"].str.len() > 0].copy()
        if len(work_df) == 0: 
            print("❌ Retriever.ready: No valid text documents found in the corpus.", file=sys.stderr)
            raise RuntimeError("No valid text documents found in the corpus.")
        self.corpus_metadata = work_df # Store the metadata

        print(f"🧠 Encoding documents... (total: {len(work_df):,})")
        try:
            doc_embeddings = self.model.encode(work_df["text"].tolist(), convert_to_tensor=False, show_progress_bar=True)
        except Exception as e:
            print(f"❌ Retriever.ready: Error encoding documents: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            raise RuntimeError(f"Failed to encode documents. Error: {e}")
        
        self.index.build(doc_embeddings, work_df)
        
        try:
            paths = self.index.save(self.cache_dir)
            print(f"💾 Index saved: {paths.emb_npy}, {paths.meta_json}")
        except Exception as e:
            print(f"❌ Retriever.ready: Error saving index: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            raise RuntimeError(f"Failed to save index. Error: {e}")

        # Build BM25 index
        print("Building BM25 index...")
        try:
            # Tokenize corpus for BM25. A more advanced tokenizer could be used.
            tokenized_corpus = [doc.split(" ") for doc in work_df["text"].tolist()]
            self.bm25_index = BM25Okapi(tokenized_corpus)
            
            with open(bm25_tokens_path, 'w', encoding='utf-8') as f:
                json.dump(tokenized_corpus, f, ensure_ascii=False)
            print(f"💾 BM25 tokens saved: {bm25_tokens_path}")
        except Exception as e:
            print(f"❌ Retriever.ready: Error building or saving BM25 index: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            raise RuntimeError(f"Failed to build/save BM25 index. Error: {e}")

        self._ready = True
        print("DEBUG: Retriever.ready - Built successfully.")

    def get_document_content(self, path: str) -> Optional[str]:
        """
        Retrieves the full text content of a document given its path.
        """
        if not self._ready: self.ready(False)
        if self.corpus_metadata is None: return None

        # Find the row where 'path' matches
        # Use .iloc[0] to get the first matching row as a Series, then .get('text')
        # Using .eq() for exact match and .any() to check if any match exists
        matching_rows = self.corpus_metadata[self.corpus_metadata['path'] == path]
        if not matching_rows.empty:
            return matching_rows.iloc[0]['text']
        return None

    def search(self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None, use_bm25: bool = False, use_reranker: bool = False, history_context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        print(f"DEBUG: Retriever.search received query: '{query}'") # 디버깅 로그 추가
        if not self._ready: self.ready(False)
        
        # 빈 쿼리 처리: 빈 쿼리가 들어오면 검색을 수행하지 않고 빈 리스트 반환
        if not query.strip():
            print("DEBUG: Retriever.search - Empty query, returning empty list.")
            return []

        candidate_hits = []
        
        # 1. Vector Search (always performed as a base)
        try:
            query_embedding = self.model.encode(query, convert_to_tensor=False)
            print(f"DEBUG: Retriever.search - query_embedding created. Shape: {query_embedding.shape}")
            vector_results = self.index.search(query_embedding, top_k=top_k * 5, filters=filters) # Retrieve more for re-ranking
            candidate_hits.extend(vector_results)
        except Exception as e:
            print(f"❌ Retriever.search: Error during vector encoding or search: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            raise # 오류를 다시 발생시켜 상위 호출자에게 전달

        # 2. BM25 Search (if enabled)
        if use_bm25 and self.bm25_index and self.corpus_metadata is not None:
            print("DEBUG: BM25 검색 활성화")
            tokenized_query = query.split(" ") # Simple split for now
            if not tokenized_query: # 빈 토큰화된 쿼리 방지
                print("DEBUG: BM25 search - Tokenized query is empty, skipping BM25.")
            else:
                try:
                    bm25_scores = self.bm25_index.get_scores(tokenized_query)
                    print(f"DEBUG: BM25 scores calculated. Max score: {np.max(bm25_scores) if len(bm25_scores) > 0 else 'N/A'}")
                    
                    # Get top BM25 documents
                    bm25_top_indices = bm25_scores.argsort()[-top_k*5:][::-1] # Get top N indices
                    print(f"DEBUG: BM25 top indices: {bm25_top_indices}")
                    
                    for idx in bm25_top_indices:
                        doc_meta = self.corpus_metadata.iloc[idx].to_dict()
                        doc_meta['bm25_score'] = bm25_scores[idx] 
                        if 'similarity' not in doc_meta: doc_meta['similarity'] = 0.0 # Ensure similarity is float
                        candidate_hits.append(doc_meta)
                except Exception as e:
                    print(f"❌ Retriever.search: Error during BM25 search: {e}", file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)
                    sys.stderr.flush()
                    raise # 오류를 다시 발생시켜 상위 호출자에게 전달

        # Remove duplicates based on 'path'
        unique_paths = set()
        combined_hits = []
        for hit in candidate_hits:
            if hit['path'] not in unique_paths:
                combined_hits.append(hit)
                unique_paths.add(hit['path'])

        # 3. Re-ranking (if enabled)
        if use_reranker and self.cross_encoder:
            print("DEBUG: 재랭킹 활성화")
            if not combined_hits:
                print("DEBUG: Re-ranking - No combined hits, returning empty list.")
                return []

            # Prepare pairs for cross-encoder
            sentence_pairs = [[query, hit['text']] for hit in combined_hits]
            
            try:
                rerank_scores = self.cross_encoder.predict(sentence_pairs)
                print(f"DEBUG: Re-rank scores calculated. Max score: {np.max(rerank_scores) if len(rerank_scores) > 0 else 'N/A'}")

                # Add rerank scores to results and sort
                for i, hit in enumerate(combined_hits):
                    hit['rerank_score'] = rerank_scores[i]
                
                final_results = sorted(combined_hits, key=lambda x: x['rerank_score'], reverse=True)
            except Exception as e:
                print(f"❌ Retriever.search: Error during re-ranking: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
                raise # 오류를 다시 발생시켜 상위 호출자에게 전달
        else:
            # If no re-ranker, sort by similarity (from vector search) or BM25 score if only BM25 was used
            final_results = sorted(combined_hits, key=lambda x: x.get('similarity', x.get('bm25_score', 0.0)), reverse=True)

        # Apply generic numpy type conversion before returning
        final_results_converted = [_convert_numpy_types_to_python_types(hit) for hit in final_results]
        print(f"DEBUG: Retriever.search returning {len(final_results_converted)} results.")
        return final_results_converted[:top_k]

    @staticmethod
    def format_results(query: str, results: List[Dict[str, Any]]) -> str:
        if not results: return f"“{query}”와 유사한 문서를 찾지 못했습니다."
        lines = [f"‘{query}’와(과) 의미상 유사한 문서 Top {len(results)}:"]
        for i, r in enumerate(results, 1):
            # Now result `r` is a dictionary containing all metadata columns
            masked_path = _mask_path(r.get('path', 'N/A'))
            lines.append(f"{i}. {masked_path}  (유사도: {r.get('similarity', 0.0):.3f})")
            if r.get("summary"):
                lines.append(f"   요약: {r.get('summary')}")
            if 'rerank_score' in r:
                lines.append(f"   재랭크 점수: {r['rerank_score']:.3f}")
            if 'bm25_score' in r:
                lines.append(f"   BM25 점수: {r['bm25_score']:.3f}")
        return "\n".join(lines)
