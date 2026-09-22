"""Run new_penn step1 with Basser/BRCA paths."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
NEW_PENN_DIR = SCRIPT_DIR.parent / "new_penn"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(NEW_PENN_DIR))

import step1_npz2nifti as step1  # noqa: E402
from config import MAPPING_CSV, OUT_ROOT  # noqa: E402

step1.OUT_ROOT = OUT_ROOT
step1.OUT_DATA_DIR = OUT_ROOT / "preprocessed" / "data"
step1.MAPPING_CSV = MAPPING_CSV

if __name__ == "__main__":
    step1.main()
