#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def load_labels(path: Path) -> Dict[str, Dict[str, float]]:
    mapping: Dict[str, Dict[str, float]] = defaultdict(dict)
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"query", "doc_id", "relevance"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"labels file must contain columns {sorted(required)}")
        for row in reader:
            query = row["query"].strip()
            doc_id = row["doc_id"].strip()
            if not query or not doc_id:
                continue
            try:
                relevance = float(row["relevance"])
            except (TypeError, ValueError):
                relevance = 0.0
            mapping[query][doc_id] = relevance
    return mapping


def load_predictions(path: Path) -> Dict[str, List[Tuple[int, str, float]]]:
    ranking: Dict[str, List[Tuple[int, str, float]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"query", "doc_id", "rank"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"predictions file must contain columns {sorted(required)}")
        for row in reader:
            query = row["query"].strip()
            doc_id = row["doc_id"].strip()
            if not query or not doc_id:
                continue
            try:
                rank = int(row["rank"])
            except (TypeError, ValueError):
                rank = len(ranking[query]) + 1
            try:
                score = float(row.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            ranking[query].append((rank, doc_id, score))
    for query in ranking:
        ranking[query].sort(key=lambda item: item[0])
    return ranking


def dcg(relevances: Iterable[float]) -> float:
    total = 0.0
    for idx, rel in enumerate(relevances, start=1):
        if rel <= 0:
            continue
        total += (2 ** rel - 1) / math.log2(idx + 1)
    return total


def compute_metrics(
    labels: Dict[str, Dict[str, float]],
    ranking: Dict[str, List[Tuple[int, str, float]]],
    k_values: Sequence[int],
) -> Dict[int, Dict[str, float]]:
    results: Dict[int, Dict[str, float]] = {}
    queries = sorted(set(labels.keys()) & set(ranking.keys()))
    if not queries:
        return {k: {"p": 0.0, "ndcg": 0.0, "queries": 0} for k in k_values}

    for k in k_values:
        precisions: List[float] = []
        ndcgs: List[float] = []
        for query in queries:
            gold = labels[query]
            ranked = ranking[query][:k]
            hits = sum(1 for _, doc_id, _ in ranked if gold.get(doc_id, 0.0) > 0)
            precisions.append(hits / max(1, k))

            gains = [gold.get(doc_id, 0.0) for _, doc_id, _ in ranked]
            ideal = sorted(gold.values(), reverse=True)[:k]
            ideal_dcg = dcg(ideal)
            ndcg_value = dcg(gains) / ideal_dcg if ideal_dcg > 0 else 0.0
            ndcgs.append(ndcg_value)
        results[k] = {
            "p": statistics.fmean(precisions),
            "ndcg": statistics.fmean(ndcgs),
            "queries": len(queries),
        }
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute P@K and nDCG@K from label/prediction CSVs")
    parser.add_argument("--labels", type=Path, required=True, help="CSV with query,doc_id,relevance")
    parser.add_argument("--predictions", type=Path, required=True, help="CSV with query,doc_id,rank[,score]")
    parser.add_argument("--k", type=int, nargs="+", default=[5], help="List of cutoff values (default: 5)")
    parser.add_argument("--target-p", type=float, default=None, help="Minimum acceptable P@K (applies to first K)")
    parser.add_argument("--target-ndcg", type=float, default=None, help="Minimum acceptable nDCG@K (applies to first K)")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to write JSON summary")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        label_map = load_labels(args.labels)
        prediction_map = load_predictions(args.predictions)
    except Exception as exc:  # pragma: no cover - argument errors
        print(f"Failed to load inputs: {exc}", file=sys.stderr)
        return 2

    k_values = sorted({max(1, int(k)) for k in args.k})
    metrics = compute_metrics(label_map, prediction_map, k_values)

    if args.output is not None:
        try:
            import json

            with args.output.open("w", encoding="utf-8") as handle:
                json.dump({str(k): metrics[k] for k in k_values}, handle, ensure_ascii=False, indent=2)
        except Exception as exc:  # pragma: no cover - i/o issue
            print(f"Failed to write results: {exc}", file=sys.stderr)

    for k in k_values:
        line = metrics.get(k, {"p": 0.0, "ndcg": 0.0, "queries": 0})
        print(f"K={k}: P@K={line['p']:.3f} nDCG@K={line['ndcg']:.3f} (queries={line['queries']})")

    first_k = k_values[0] if k_values else 5
    primary = metrics.get(first_k, {"p": 0.0, "ndcg": 0.0, "queries": 0})
    meets_p = True if args.target_p is None else primary["p"] >= args.target_p
    meets_ndcg = True if args.target_ndcg is None else primary["ndcg"] >= args.target_ndcg

    if not (meets_p and meets_ndcg):
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
