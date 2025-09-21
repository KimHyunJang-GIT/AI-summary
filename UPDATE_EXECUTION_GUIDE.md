# Update Execution Guide

This guide describes what to run after modifying project files, broken down by environment.

## Prerequisites
- Ensure Python 3.9+ is installed.
- Run everything from the repository root (`/mnt/.../AI-summary-1`).

## 1. Create or Activate a Virtual Environment

### Cross-platform (recommended)
```bash
python -m venv .venv
```

### macOS / Linux
```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (Command Prompt)
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> If you prefer to reuse an existing virtual environment, skip the `python -m venv .venv` line and just activate it.

## 2. Validate the Update

### Run Tests (all platforms)
```bash
pytest -q
```
Use `py -m pytest -q` on Windows if `pytest` is not on PATH.

### Rebuild Pipelines When Data Logic Changes
1. Rescan inputs (optional after file discovery changes):
   ```bash
   python infopilot.py scan --out data/found_files.csv
   ```
2. Rebuild the corpus and topic model:
   ```bash
   python infopilot.py train --scan_csv data/found_files.csv --corpus data/corpus.parquet
   ```

### Refresh Chat Cache (only if model or corpus changed)
```bash
python infopilot.py chat --model data/topic_model.joblib --corpus data/corpus.parquet --cache index_cache --no-translate
```
Add `--no-translate` on systems without `deep-translator`.

## 3. OS-specific Notes

### macOS
- Use `python3` instead of `python` if the latter points to Python 2.
- Install Xcode Command Line Tools if `fastparquet` or other C-extensions complain about missing compilers: `xcode-select --install`.

### Windows
- Prefer the 64-bit Python installer to satisfy `fastparquet` wheels.
- If compilation of dependencies fails, ensure Microsoft C++ Build Tools are installed.

### WSL / Linux
- Install build essentials if compiling wheels: `sudo apt-get install build-essential libsnappy-dev` (snappy speeds up `fastparquet`).
- Use `python3` instead of `python` depending on your setup.

## 4. After Verification
- Deactivate the virtual environment (`deactivate`).
- Stage and commit changes following the repository guidelines.

Keep this guide alongside the codebase; update it when new workflows are introduced.
