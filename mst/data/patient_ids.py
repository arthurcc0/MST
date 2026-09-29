"""Canonical patient-ID strings for matching MRNs across Penn tables.

The same MRN shows up as ``001144138`` (Excel/mapping), ``1144138.0`` (pandas
float after NaNs) and ``1144138`` (old-Penn EHR). Numeric IDs are reduced to
their integer digits; anything else is only stripped.
"""

from __future__ import annotations

import re

import pandas as pd

_BLANK = frozenset({"", "nan", "none", "null", "<na>"})
_NUMERIC = re.compile(r"^\d+(?:\.0+)?$")


def normalize_patient_id(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    text = str(val).strip()
    if text.lower() in _BLANK:
        return ""
    if _NUMERIC.fullmatch(text):
        return str(int(text.split(".", 1)[0]))
    return text


def patient_ids(values) -> set[str]:
    return {i for i in map(normalize_patient_id, values) if i}
