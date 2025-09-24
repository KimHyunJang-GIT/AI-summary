#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from retriever import VectorIndex, faiss


def build_synthetic_index(
    *,
    doc_count: int,
    dim: int,
    seed: int,
    ext_pool: Sequence[str] | None = None,
    owner_pool: Sequence[str] | None = None,
) -> VectorIndex:
    if doc_count <= 0:
        raise ValueError("doc_count must be positive")
    if dim <= 0:
        raise ValueError("dim must be positive")

    rng = np.random.default_rng(seed)
    embeddings = rng.normal(size=(doc_count, dim)).astype(np.float32)
    paths = [f"/synthetic/doc_{idx}.txt" for idx in range(doc_count)]
    ext_choices = ext_pool or (".txt", ".md", ".pdf", ".pptx")
    owner_choices = owner_pool or ("alice", "bob", "carol", "dave")
    exts = [ext_choices[idx % len(ext_choices)] for idx in range(doc_count)]
    owners = [owner_choices[idx % len(owner_choices)] for idx in range(doc_count)]
    previews = [""] * doc_count

    index = VectorIndex()
    index.build(
        embeddings,
        paths,
        exts,
        previews,
        owners=owners,
        drives=["synthetic"] * doc_count,
    )
    return index


def measure_latency(
    index: VectorIndex,
    *,
    query_vectors: Sequence[np.ndarray],
    top_k: int,
    use_ann: bool,
) -> tuple[List[List[dict]], List[float]]:
    latencies: List[float] = []
    results: List[List[dict]] = []
    for query_vec in query_vectors:
        start = time.perf_counter()
        hits = index.search(
            query_vec,
            top_k=top_k,
            oversample=1,
            lexical_weight=0.0,
            query_tokens=None,
            min_similarity=0.0,
            use_ann=use_ann,
        )
        end = time.perf_counter()
        latencies.append(end - start)
        results.append(hits)
    return results, latencies


def overlap_fraction(baseline: Iterable[dict], candidate: Iterable[dict], k: int) -> float:
    base_ids = {hit.get("doc_id") for hit in list(baseline)[:k] if hit.get("doc_id") is not None}
    if not base_ids:
        return 1.0
    cand_ids = {hit.get("doc_id") for hit in list(candidate)[:k] if hit.get("doc_id") is not None}
    return len(base_ids & cand_ids) / len(base_ids)


def summarise(latencies: Sequence[float]) -> dict:
    if not latencies:
        return {"avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    ms_values = [value * 1000.0 for value in latencies]
    sorted_ms = sorted(ms_values)
    p50 = sorted_ms[len(sorted_ms) // 2]
    p95 = sorted_ms[int(len(sorted_ms) * 0.95) - 1]
    return {
        "avg_ms": statistics.fmean(ms_values),
        "p50_ms": p50,
        "p95_ms": p95,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ANN vs exact search benchmark")
    parser.add_argument("--doc-count", type=int, default=20000, help="Number of synthetic documents")
    parser.add_argument("--dim", type=int, default=384, help="Embedding dimensionality")
    parser.add_argument("--queries", type=int, default=50, help="Number of random queries to evaluate")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K cutoff for overlap computation")
    parser.add_argument(
        "--ann-threshold",
        type=int,
        default=5000,
        help="Minimum index size to trigger ANN path",
    )
    parser.add_argument("--ef-search", type=int, default=128, help="HNSW efSearch value")
    parser.add_argument("--ef-construction", type=int, default=200, help="HNSW efConstruction value")
    parser.add_argument("--ann-m", type=int, default=32, help="HNSW graph degree (M)")
    parser.add_argument("--target-overlap", type=float, default=None, help="Required overlap@k to pass (e.g. 0.95)")
    parser.add_argument("--target-p95", type=float, default=None, help="Maximum allowed ANN p95 latency in milliseconds")
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write JSON summary",
    )
    args = parser.parse_args(argv)

    if faiss is None:
        print("FAISS/HNSW backend is unavailable. Install faiss-cpu to run this benchmark.", file=sys.stderr)
        return 2

    index = build_synthetic_index(doc_count=args.doc_count, dim=args.dim, seed=args.seed)
    index.configure_ann(
        threshold=args.ann_threshold,
        ef_search=args.ef_search,
        ef_construction=args.ef_construction,
        m=args.ann_m,
    )
    # Warm-up to build ANN structures once
    _ = index.search(
        np.random.default_rng(args.seed).normal(size=args.dim).astype(np.float32),
        top_k=min(args.top_k, 5),
        use_ann=True,
    )

    rng = np.random.default_rng(args.seed + 7)
    query_vectors = [rng.normal(size=args.dim).astype(np.float32) for _ in range(args.queries)]

    baseline_res, baseline_lat = measure_latency(
        index,
        query_vectors=query_vectors,
        top_k=args.top_k,
        use_ann=False,
    )
    ann_res, ann_lat = measure_latency(
        index,
        query_vectors=query_vectors,
        top_k=args.top_k,
        use_ann=True,
    )

    overlaps = [
        overlap_fraction(base, cand, args.top_k)
        for base, cand in zip(baseline_res, ann_res)
    ]
    overlap_avg = statistics.fmean(overlaps) if overlaps else 1.0

    baseline_stats = summarise(baseline_lat)
    ann_stats = summarise(ann_lat)
    speedup = (
        baseline_stats["avg_ms"] / ann_stats["avg_ms"]
        if ann_stats["avg_ms"] > 0
        else float("inf")
    )
    meets_overlap = True if args.target_overlap is None else overlap_avg >= args.target_overlap
    ann_p95 = ann_stats.get("p95_ms", 0.0)
    meets_latency = True if args.target_p95 is None else ann_p95 <= args.target_p95
    report = {
        "doc_count": args.doc_count,
        "dim": args.dim,
        "queries": args.queries,
        "top_k": args.top_k,
        "ann_threshold": args.ann_threshold,
        "overlap@k": overlap_avg,
        "baseline_ms": baseline_stats,
        "ann_ms": ann_stats,
        "speedup": speedup,
        "target_overlap": args.target_overlap,
        "target_p95": args.target_p95,
        "meets_overlap": meets_overlap,
        "meets_latency": meets_latency,
    }

    if args.output is not None:
        try:
            import json

            with args.output.open("w", encoding="utf-8") as handle:
                json.dump(report, handle, ensure_ascii=False, indent=2)
        except Exception as exc:  # pragma: no cover - best effort
            print(f"Failed to write benchmark output: {exc}", file=sys.stderr)

    print("=== ANN Benchmark Summary ===")
    for key, value in report.items():
        print(f"{key}: {value}")

    if not (report["meets_overlap"] and report["meets_latency"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
