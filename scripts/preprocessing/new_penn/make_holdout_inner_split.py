"""Build a locked-holdout + inner 5-fold split from the existing new-Penn CSV.

Outer test = original Fold 0 ``Split=test`` (same 205 UIDs). The remaining
exams keep their inner patient groups (recovered with the same greedy seed as
step3). Inner fold k uses group k as val, the other four groups as train, and
always the same holdout as test.

Does not reshuffle patients. Inner fold 0 val matches original Fold 0 val.
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
SOURCE_CSV = OUT_ROOT / "new_penn_datasplit_v5_noBenHR.csv"
OUTPUT_CSV = OUT_ROOT / "new_penn_datasplit_v5_holdoutF0.csv"
HOLDOUT_FOLD = 0


def make_holdout_inner_split(
    source_csv: Path = SOURCE_CSV,
    output_csv: Path = OUTPUT_CSV,
    holdout_fold: int = HOLDOUT_FOLD,
) -> pd.DataFrame:
    src = pd.read_csv(source_csv, dtype=str)
    src["Fold"] = pd.to_numeric(src["Fold"])
    src["Malignant"] = pd.to_numeric(src["Malignant"])
    one = src[src["Fold"] == holdout_fold].copy()
    if one.empty:
        raise ValueError(f"{source_csv} has no Fold={holdout_fold}")

    holdout = one[one["Split"].str.lower() == "test"].drop_duplicates("UID")
    trainval = one[one["Split"].str.lower() != "test"].drop_duplicates("UID")
    if holdout.empty or trainval.empty:
        raise ValueError("Need both test and train/val rows on the holdout fold.")

    orig_val = set(one.loc[one["Split"].str.lower() == "val", "UID"])
    lab = COL_LABEL if COL_LABEL in trainval.columns else "label"
    trainval = trainval.copy()
    trainval["_stratum"] = [
        _cv_stratum(lab_i, mal) for lab_i, mal in zip(trainval[lab], trainval["Malignant"])
    ]
    inner = _greedy_group_folds(
        trainval,
        COL_PATIENT,
        "_stratum",
        n_splits=N_FOLDS,
        random_state=RANDOM_STATE_INNER + holdout_fold,
    )
    trainval["_inner"] = trainval[COL_PATIENT].astype(str).str.strip().map(inner)
    rec_val = set(trainval.loc[trainval["_inner"] == 0, "UID"])
    if rec_val != orig_val:
        raise RuntimeError(
            "Recovered inner group 0 does not match original val; "
            "refusing to write a different split."
        )

    parts = []
    holdout_rows = holdout.drop(columns=["_stratum", "_inner"], errors="ignore")
    for k in range(N_FOLDS):
        block = trainval.copy()
        block["Split"] = "train"
        block.loc[block["_inner"] == k, "Split"] = "val"
        block["Fold"] = k
        test_k = holdout_rows.copy()
        test_k["Split"] = "test"
        test_k["Fold"] = k
        parts.append(pd.concat([block, test_k], ignore_index=True))

    out = pd.concat(parts, ignore_index=True)
    out = out.drop(columns=["_stratum", "_inner"], errors="ignore")
    out["Malignant"] = pd.to_numeric(out["Malignant"]).astype("Int64")
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    print(f"Source:  {source_csv}")
    print(f"Wrote:   {output_csv}  ({len(out)} rows)")
    print(f"Holdout test UIDs: {holdout['UID'].nunique()} (locked, every inner fold)")
    print(out.groupby(["Fold", "Split"]).size().unstack(fill_value=0).to_string())
    tests = [set(out.loc[(out.Fold == k) & (out.Split == "test"), "UID"]) for k in range(N_FOLDS)]
    assert all(t == tests[0] for t in tests)
    print("Test UIDs identical across inner folds.")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, default=SOURCE_CSV)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--holdout-fold", type=int, default=HOLDOUT_FOLD)
    args = parser.parse_args()
    make_holdout_inner_split(args.source_csv, args.output_csv, args.holdout_fold)


if __name__ == "__main__":
    main()
