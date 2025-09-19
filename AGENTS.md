# Repository Guidelines

## Project Structure & Module Organization
- `infopilot.py` orchestrates the CLI scan/train/chat flow and wires the supporting modules.
- `filefinder.py` performs cross-platform scans and writes candidate metadata to CSV during Step 1.
- `pipeline.py` builds the cleaned corpus and topic model artifacts stored in `data/`.
- `retriever.py` and `lnp_chat.py` load the trained model, manage `index_cache/`, and drive the interactive chat.
- `data/` hosts generated corpora and models; `models/` is reserved for packaged exports.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` (`.\.venv\Scripts\activate` on Windows) isolates dependencies.
- `pip install -r requirements.txt` installs the baseline extractors (번역 기본 활성화를 위한 `deep-translator` 포함).
- `python infopilot.py scan --out data/found_files.csv` scans mounted drives and writes the discovery CSV.
- `python infopilot.py train --scan_csv data/found_files.csv --corpus data/corpus.parquet` builds the corpus and updates `data/topic_model.joblib`.
- `python infopilot.py chat --model data/topic_model.joblib --corpus data/corpus.parquet --cache index_cache` launches the retrieval chat; add `--no-translate` if `deep-translator` is unavailable.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation, type hints, and concise docstrings on public entry points.
- Use `snake_case` for functions, `UPPER_SNAKE_CASE` for constants, and `CamelCase` for classes such as `FileFinder`.
- Keep console output succinct and reuse the existing emoji/logging tone for user feedback.

## Testing Guidelines
- Adopt `pytest` for new coverage and place specs under `tests/` mirroring module names.
- Mock filesystem access or use temporary directories when exercising `FileFinder` or `Retriever`.
- Run `pytest -q` before pushing and rely on synthetic fixture CSVs instead of real customer data.

## Commit & Pull Request Guidelines
- Write imperative, present-tense commit subjects (e.g., `Improve PDF extractor handling`) with short context in the body.
- Reference issues via `Refs #123` or `Fixes #123`, and exclude bulky artifacts from `data/` or `index_cache/`.
- PRs should cover scope, manual verification (`scan`, `train`, `chat`), dependency changes, and attach console screenshots when UX shifts.

## Security & Data Handling
- Scrub or anonymize sensitive file paths in logs, tests, and shared troubleshooting assets.
- Clear or regenerate `index_cache/` before distributing debug bundles to prevent leaking previews.
- Add local-only artifacts to `.gitignore` entries before committing scanned data or corpora.
