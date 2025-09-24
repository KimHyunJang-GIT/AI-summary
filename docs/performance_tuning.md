# Performance & Tuning Guide

This note captures the current knobs for personalization, retrieval speed, and benchmark workflows introduced in the latest cycles.

## Benchmarks

Use the synthetic benchmark script to compare ANN (HNSW) retrieval with the exact FAISS search path:

```bash
source .venv/bin/activate
python benchmarks/ann_benchmark.py --doc-count 20000 --queries 100 --top-k 10 --ann-threshold 2000 --ef-search 128 --ef-construction 200 --ann-m 32 --target-overlap 0.95 --target-p95 500 --output /tmp/ann.json
```

The script prints average/p50/p95 latency (ms), overlap@k against the exact ranking, and the relative speed-up factor. Use `--target-overlap` / `--target-p95` to fail fast when accuracy or latency budgets are missed; the command exits with status 1 if either target is violated. Adjust `doc-count` and `dim` to approximate production embeddings, and save the JSON for regression tracking.

## Early Stop (CrossEncoder Reranker)

- `EarlyStopConfig.score_threshold`: average batch score below this value counts toward patience.
- `EarlyStopConfig.window_size`: number of recent batches to average; defaults to the CrossEncoder batch size.
- `EarlyStopConfig.patience`: consecutive low-score windows allowed before halting.<br>
  Increase `window_size`/`patience` if recall drops; lower the threshold to avoid premature exits on noisy domains.

## Session Personalization Weights

`retriever.py` exposes the session scaling constants:

| Constant | Purpose | Default |
| --- | --- | --- |
| `_SESSION_EXT_PREF_SCALE` | Ext preference → score delta | `0.05` |
| `_SESSION_OWNER_PREF_SCALE` | Owner prior → score delta | `0.04` |
| `_SESSION_CLICK_WEIGHT` | Click feedback increment | `0.35` |
| `_SESSION_PIN_WEIGHT` | Pin feedback increment | `0.6` |
| `_SESSION_LIKE_WEIGHT` | Like feedback increment | `0.45` |
| `_SESSION_DISLIKE_WEIGHT` | Dislike feedback decrement | `-0.5` |
| `_SESSION_PREF_DECAY` | Exponential decay on every update | `0.85` |

Modify these constants cautiously and re-run `pytest -m "smoke"` plus the ANN benchmark to confirm recall/latency targets.

## Test Coverage

- `tests/test_ann_and_reranker.py`: early-stop logic, ANN parity, API session summary, retriever hook propagation.
- `tests/test_retriever_ext_filter.py`: extension/owner priors and metadata filters.

Run `pytest -m "smoke"` for quick feedback and `pytest -m "full"` before shipping.


### Accuracy/Latency Sweep

Run multiple configurations (e.g. varying `--ef-search`, `--ann-m`, or reranker `score_threshold`) and store the resulting JSON summaries under `benchmarks/results/`. A simple pattern is:

```bash
mkdir -p benchmarks/results
for ef in 64 96 128 160; do
  python benchmarks/ann_benchmark.py     --doc-count 20000 --queries 100 --top-k 10 --ann-threshold 2000     --ef-search "$ef" --ef-construction 200 --ann-m 32     --target-overlap 0.95 --target-p95 500     --output benchmarks/results/ann_ef${ef}.json || true
done
```

Compare overlap/latency across the saved snapshots and pick the lowest efSearch that still clears both thresholds. Capture the chosen configuration in version control (JSON artifact + PR note) so regressions are easy to spot.


## Accuracy Evaluation

Use the CSV-based evaluator to check P@K/nDCG against a labelled query set:

```bash
python benchmarks/accuracy_eval.py   --labels benchmarks/fixtures/sample_labels.csv   --predictions benchmarks/fixtures/sample_predictions.csv   --k 1 5 --target-p 0.8 --target-ndcg 0.7 || true
```

Provide your own label/prediction exports when available. The command exits with status 1 if the primary K (first value passed to `--k`) misses either threshold, making it suitable for CI regression checks. Store canonical label/prediction snapshots under `benchmarks/fixtures/` to keep evaluations reproducible.
