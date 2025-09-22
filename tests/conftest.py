"""Pytest configuration helpers."""
from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
root_str = str(PROJECT_ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")

if "JOBLIB_TEMP_FOLDER" not in os.environ:
    joblib_tmp = Path(tempfile.mkdtemp(prefix="joblib-", dir=root_str))
    os.environ["JOBLIB_TEMP_FOLDER"] = str(joblib_tmp)
    atexit.register(shutil.rmtree, joblib_tmp, True)
