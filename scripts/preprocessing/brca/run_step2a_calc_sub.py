"""Run new_penn step2a with Basser/BRCA paths."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
NEW_PENN_DIR = SCRIPT_DIR.parent / "new_penn"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(NEW_PENN_DIR))

import step2a_calc_sub as step2a  # noqa: E402
from config import OUT_ROOT  # noqa: E402

step2a.OUT_ROOT = OUT_ROOT
step2a.DATA_ROOT = OUT_ROOT / "preprocessed" / "data"

if __name__ == "__main__":
    step2a.main()
