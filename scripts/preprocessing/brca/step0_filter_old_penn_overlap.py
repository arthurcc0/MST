"""Remove Basser cohort accessions that appear in the old-Penn pretrain datasplit.

Ensures no study used for MST old-Penn pretraining is reused in BRCA
fine-tuning. Writes a filtered label table plus an audit CSV of excluded rows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import (
    COL_ACCESSION,
    EXCLUDED_OVERLAP_CSV,
    FILTERED_LABEL_TABLE,
    LABEL_TABLE,
    OLD_PENN_DATASPLIT,
    OUT_ROOT,
)


def _read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=str, keep_default_na=False)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _old_penn_accessions(split_csv: Path) -> set[str]:
    df = pd.read_csv(split_csv, dtype=str)
    if "dummy_acc" not in df.columns:
        raise KeyError(f"Old Penn split missing dummy_acc column: {split_csv}")
    return set(df["dummy_acc"].astype(str).str.strip().dropna())


def filter_old_penn_overlap(
    label_table: Path,
    old_penn_split: Path,
    filtered_out: Path,
    excluded_audit: Path,
) -> None:
    if not label_table.is_file():
        raise FileNotFoundError(f"Label table not found: {label_table}")
    if not old_penn_split.is_file():
        raise FileNotFoundError(f"Old Penn datasplit not found: {old_penn_split}")

    df = _read_table(label_table)
    if COL_ACCESSION not in df.columns:
        raise KeyError(f"Label table missing {COL_ACCESSION!r}: {label_table}")

    df = df.copy()
    df[COL_ACCESSION] = df[COL_ACCESSION].astype(str).str.strip()
    old_accs = _old_penn_accessions(old_penn_split)

    overlap_mask = df[COL_ACCESSION].isin(old_accs)
    excluded = df.loc[overlap_mask].copy()
    kept = df.loc[~overlap_mask].copy()

    filtered_out = Path(filtered_out)
    excluded_audit = Path(excluded_audit)
    filtered_out.parent.mkdir(parents=True, exist_ok=True)
    excluded_audit.parent.mkdir(parents=True, exist_ok=True)

    if filtered_out.suffix.lower() in {".xlsx", ".xls"}:
        kept.to_excel(filtered_out, index=False)
    else:
        kept.to_csv(filtered_out, index=False)

    if len(excluded):
        excluded["exclude_reason"] = "accession_in_old_penn_datasplit"
        excluded.to_csv(excluded_audit, index=False)
    else:
        pd.DataFrame(columns=list(df.columns) + ["exclude_reason"]).to_csv(
            excluded_audit, index=False
        )

    print(f"Label table:           {label_table}  (n={len(df)})")
    print(f"Old Penn accessions:   {len(old_accs)}  ({old_penn_split})")
    print(f"Overlap excluded:      {int(overlap_mask.sum())}")
    print(f"Kept:                  {len(kept)}  -> {filtered_out}")
    print(f"Excluded audit:        {excluded_audit}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Drop Basser accessions present in old_penn_datasplit."
    )
    parser.add_argument("--label-table", type=Path, default=LABEL_TABLE)
    parser.add_argument("--old-penn-split", type=Path, default=OLD_PENN_DATASPLIT)
    parser.add_argument("--filtered-out", type=Path, default=FILTERED_LABEL_TABLE)
    parser.add_argument("--excluded-audit", type=Path, default=EXCLUDED_OVERLAP_CSV)
    args = parser.parse_args()
    filter_old_penn_overlap(
        args.label_table,
        args.old_penn_split,
        args.filtered_out,
        args.excluded_audit,
    )


if __name__ == "__main__":
    main()
