"""Build ``brca_mapping.csv`` for the Basser BRCA cohort.

One row per accession (for step1/2a/2b). Laterality expansion to left/right
UIDs happens in ``step3_create_split.py``.

Laterality rules (``breasts_to_consider``):
    - empty / NaN  -> both breasts (left + right samples in step3)
    - 1            -> right only
    - 2            -> left only

Label ``bc_overall`` (0/1) is mapped to canonical ``benign`` / ``malignant``.

When ``ENABLE_MRI_TIMING_LABEL_RULE`` is on (see ``config.py``), ``bc_overall=1``
rows are malignant only if ``mridate`` is on or before ``bc_overall_firstdate``
and at most ``MAX_YEARS_MRI_BEFORE_CANCER`` years earlier; otherwise they are
downgraded to benign.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import (
    COL_ACCESSION,
    COL_BC_FIRST_DATE,
    COL_BREASTS,
    COL_LABEL,
    COL_MRI_DATE,
    COL_PATIENT,
    COL_SPLIT,
    ENABLE_MRI_TIMING_LABEL_RULE,
    FILTERED_LABEL_TABLE,
    LABEL_TABLE,
    MAPPING_CSV,
    MAX_YEARS_MRI_BEFORE_CANCER,
    OUT_ROOT,
    STANDARD_LABEL,
    VOL_DIR,
)

DEBUG_SINGLE = False


def _read_label_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=str, keep_default_na=False)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _side_flags(breasts_to_consider) -> tuple[bool, bool]:
    """Return (include_left, include_right)."""
    if breasts_to_consider is None or (
        isinstance(breasts_to_consider, float) and np.isnan(breasts_to_consider)
    ):
        return True, True
    s = str(breasts_to_consider).strip().lower()
    if s in {"", "nan", "none", "null"}:
        return True, True
    try:
        code = int(float(s))
    except ValueError:
        return True, True
    if code == 1:
        return False, True
    if code == 2:
        return True, False
    return True, True


def _label_from_bc_overall(val) -> str | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip().lower()
    if s in {"0", "0.0", "false", "benign", "negative", "no"}:
        return "benign"
    if s in {"1", "1.0", "true", "malignant", "positive", "yes", "cancer"}:
        return "malignant"
    return None


def _parse_date(val) -> pd.Timestamp | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    if s in {"", "nan", "none", "null"}:
        return None
    ts = pd.to_datetime(s, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.normalize()


def _days_mri_before_cancer(mridate, bc_firstdate) -> int | None:
    mri_ts = _parse_date(mridate)
    bc_ts = _parse_date(bc_firstdate)
    if mri_ts is None or bc_ts is None:
        return None
    return int((bc_ts - mri_ts).days)


def _label_with_mri_timing(
    bc_overall_val,
    mridate,
    bc_firstdate,
    *,
    enable_timing_rule: bool,
    max_years_before_cancer: float,
) -> tuple[str | None, int | None, bool]:
    """Return (canonical label, days MRI before cancer, timing_downgraded)."""
    label = _label_from_bc_overall(bc_overall_val)
    if label != "malignant" or not enable_timing_rule:
        return label, _days_mri_before_cancer(mridate, bc_firstdate), False

    days_before = _days_mri_before_cancer(mridate, bc_firstdate)
    if days_before is None:
        return label, None, False

    max_days = int(round(max_years_before_cancer * 365.25))
    if days_before < 0 or days_before > max_days:
        return "benign", days_before, True
    return "malignant", days_before, False


def create_mapping(
    label_table: Path,
    mapping_csv: Path,
    vol_dir: Path,
    *,
    enable_timing_rule: bool = ENABLE_MRI_TIMING_LABEL_RULE,
    max_years_before_cancer: float = MAX_YEARS_MRI_BEFORE_CANCER,
) -> None:
    if not label_table.is_file():
        raise FileNotFoundError(f"Label table not found: {label_table}")
    if not vol_dir.is_dir():
        raise FileNotFoundError(f"Volume directory not found: {vol_dir}")

    df = _read_label_table(label_table)
    for col in (COL_ACCESSION, COL_LABEL, COL_PATIENT):
        if col not in df.columns:
            raise KeyError(f"Label table missing required column {col!r}")

    df = df.copy()
    df[COL_ACCESSION] = df[COL_ACCESSION].astype(str).str.strip()
    df[COL_PATIENT] = df[COL_PATIENT].astype(str).str.strip()
    if COL_BREASTS not in df.columns:
        df[COL_BREASTS] = np.nan

    df = df.drop_duplicates(subset=[COL_ACCESSION], keep="first")

    has_split = COL_SPLIT in df.columns
    has_mri_date = COL_MRI_DATE in df.columns
    has_bc_first_date = COL_BC_FIRST_DATE in df.columns
    if enable_timing_rule and (not has_mri_date or not has_bc_first_date):
        missing = [
            c for c in (COL_MRI_DATE, COL_BC_FIRST_DATE) if c not in df.columns
        ]
        raise KeyError(
            f"MRI timing label rule enabled but label table missing columns: {missing}"
        )

    records = []
    n_missing_npz = 0
    n_bad_label = 0
    n_timing_downgraded = 0
    n_timing_missing_dates = 0
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Building mapping"):
        acc = getattr(row, COL_ACCESSION)
        npz_path = vol_dir / f"{acc}.npz"
        if not npz_path.is_file():
            n_missing_npz += 1
            continue

        mridate = getattr(row, COL_MRI_DATE, None) if has_mri_date else None
        bc_firstdate = getattr(row, COL_BC_FIRST_DATE, None) if has_bc_first_date else None
        label, days_before, timing_downgraded = _label_with_mri_timing(
            getattr(row, COL_LABEL),
            mridate,
            bc_firstdate,
            enable_timing_rule=enable_timing_rule,
            max_years_before_cancer=max_years_before_cancer,
        )
        if label is None:
            n_bad_label += 1
            continue
        if (
            enable_timing_rule
            and _label_from_bc_overall(getattr(row, COL_LABEL)) == "malignant"
            and days_before is None
        ):
            n_timing_missing_dates += 1
        if timing_downgraded:
            n_timing_downgraded += 1

        breasts_val = getattr(row, COL_BREASTS, None)
        include_left, include_right = _side_flags(breasts_val)

        record = {
            "PatientID": acc,
            "FullPath": str(npz_path).replace("\\", "/"),
            COL_PATIENT: getattr(row, COL_PATIENT),
            COL_LABEL: getattr(row, COL_LABEL),
            STANDARD_LABEL: label,
            COL_BREASTS: "" if pd.isna(breasts_val) else str(breasts_val).strip(),
            "include_left": int(include_left),
            "include_right": int(include_right),
        }
        if has_split:
            split_val = getattr(row, COL_SPLIT, None)
            record[COL_SPLIT] = "" if split_val is None else str(split_val).strip()
        if enable_timing_rule:
            record[COL_MRI_DATE] = "" if mridate is None else str(mridate).strip()
            record[COL_BC_FIRST_DATE] = "" if bc_firstdate is None else str(bc_firstdate).strip()
            record["days_mri_before_cancer"] = "" if days_before is None else days_before
            record["label_timing_downgraded"] = int(timing_downgraded)
        records.append(record)

    if not records:
        print("No rows with both label and .npz on disk.")
        return

    out = pd.DataFrame(records)
    if DEBUG_SINGLE:
        out = out.head(1)
        print(f"[DEBUG_SINGLE] keeping only: {out['PatientID'].tolist()}")

    mapping_csv = Path(mapping_csv)
    mapping_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(mapping_csv, index=False)

    n_both = int((out["include_left"] & out["include_right"]).sum())
    n_left = int((out["include_left"] & ~out["include_right"]).sum())
    n_right = int((~out["include_left"] & out["include_right"]).sum())
    print(f"Mapping written to {mapping_csv}  (n={len(out)})")
    print(f"  missing .npz skipped:     {n_missing_npz}")
    print(f"  invalid bc_overall skipped: {n_bad_label}")
    if enable_timing_rule:
        print(
            f"  MRI timing rule:          max {max_years_before_cancer:g} year(s) "
            f"before {COL_BC_FIRST_DATE}"
        )
        print(f"  timing downgraded to benign: {n_timing_downgraded}")
        print(f"  malignant w/ missing dates: {n_timing_missing_dates}")
    print(f"  both breasts (L+R UIDs):  {n_both}")
    print(f"  left only:                {n_left}")
    print(f"  right only:               {n_right}")
    print(f"  expected step3 UID rows:  {int(out['include_left'].sum() + out['include_right'].sum())}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build brca_mapping.csv for Basser MRI cohort.")
    parser.add_argument(
        "--label-table",
        type=Path,
        default=FILTERED_LABEL_TABLE if FILTERED_LABEL_TABLE.is_file() else LABEL_TABLE,
        help="Filtered table from step0, or raw label table if step0 not run.",
    )
    parser.add_argument("--mapping-csv", type=Path, default=MAPPING_CSV)
    parser.add_argument("--vol-dir", type=Path, default=VOL_DIR)
    parser.add_argument(
        "--max-years-before-cancer",
        type=float,
        default=MAX_YEARS_MRI_BEFORE_CANCER,
        help=(
            "When bc_overall=1, keep malignant only if mridate is on/before "
            "bc_overall_firstdate and at most this many years earlier."
        ),
    )
    timing = parser.add_mutually_exclusive_group()
    timing.add_argument(
        "--enable-mri-timing-label",
        dest="enable_timing_rule",
        action="store_true",
        default=ENABLE_MRI_TIMING_LABEL_RULE,
        help="Apply MRI/cancer-date timing rule (default from config.py).",
    )
    timing.add_argument(
        "--disable-mri-timing-label",
        dest="enable_timing_rule",
        action="store_false",
        help="Use raw bc_overall only.",
    )
    args = parser.parse_args()
    create_mapping(
        args.label_table,
        args.mapping_csv,
        args.vol_dir,
        enable_timing_rule=args.enable_timing_rule,
        max_years_before_cancer=args.max_years_before_cancer,
    )


if __name__ == "__main__":
    main()
