"""Old-Penn train/val only (no test): more data for stage 1 and the joint graft.

Uses unique exams from ``old_penn_datasplit_noNewPenn.csv`` (new-Penn patients
already dropped). One patient-grouped 80/20 train/val (inner group 0 = val).
``Fold=0`` on every row so stage 1 and ``make_joint_datasplit --old-source-fold 0``
see a single split.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step3_create_split import (
    COL_LABEL,
    COL_PATIENT,
    N_FOLDS,
    RANDOM_STATE_INNER,
    _cv_stratum,
    _greedy_group_folds,
)

OUT_ROOT = Path(r"D:\Users\arthur\Data\MST_birads4")
SOURCE_CSV = OUT_ROOT / "old_penn_datasplit_noNewPenn.csv"
OUTPUT_CSV = OUT_ROOT / "old_penn_datasplit_trainval.csv"


def make_old_penn_trainval(
    source_csv: Path = SOURCE_CSV,
    output_csv: Path = OUTPUT_CSV,
) -> pd.DataFrame:
    src = pd.read_csv(source_csv, dtype=str)
    df = src.drop_duplicates("UID").copy()
    df["Malignant"] = pd.to_numeric(df["Malignant"])
    lab = COL_LABEL if COL_LABEL in df.columns else "label"
    df["_stratum"] = [
        _cv_stratum(lab_i, mal) for lab_i, mal in zip(df[lab], df["Malignant"])
    ]
    inner = _greedy_group_folds(
        df,
        COL_PATIENT,
        "_stratum",
        n_splits=N_FOLDS,
        random_state=RANDOM_STATE_INNER,
    )
    df["_inner"] = df[COL_PATIENT].astype(str).str.strip().map(inner)
    df["Split"] = "train"
    df.loc[df["_inner"] == 0, "Split"] = "val"
    df["Fold"] = 0
    df = df.drop(columns=["_stratum", "_inner"])
    df["Malignant"] = pd.to_numeric(df["Malignant"]).astype("Int64")

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"Source:  {source_csv}")
    print(f"Wrote:   {output_csv}  ({len(df)} unique exams, Fold=0 only)")
    print(df.groupby("Split").size().to_string())
    print("\nLabels by split:")
    print(pd.crosstab(df["Split"], df[lab], margins=True).to_string())
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, default=SOURCE_CSV)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    args = parser.parse_args()
    make_old_penn_trainval(args.source_csv, args.output_csv)


if __name__ == "__main__":
    main()
