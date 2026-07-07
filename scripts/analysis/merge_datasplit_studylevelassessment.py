"""Attach ``studylevelassessment`` to Penn datasplit CSVs from origin label tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(r"D:\Users\arthur\Data\MST_birads4")

DEFAULT_SPLITS = {
    "new": {
        "split": DATA_ROOT / "new_penn_datasplit_v2.csv",
        "origin": REPO / "tables" / "matches_birads4_all_v2.xlsx",
        "origin_acc": "newaccession",
        "out": DATA_ROOT / "new_penn_datasplit_v2_wAssessment.csv",
    },
    "old": {
        "split": DATA_ROOT / "old_penn_datasplit_v2.csv",
        "origin": REPO / "table_utils" / "lat_added_dummy_ehr_chat_no_birads4_v2.csv",
        "origin_acc": "dummy_acc",
        "out": DATA_ROOT / "old_penn_datasplit_v2_wAssessment.csv",
    },
}

STUDY_ASSESSMENT = "studylevelassessment"
SPLIT_ACC = "dummy_acc"


def _pick_assessment_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if str(col).lower() == STUDY_ASSESSMENT:
            return col
    raise KeyError(f"No {STUDY_ASSESSMENT!r} column in origin table; got {list(df.columns)}")


def _read_origin(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=str, keep_default_na=False)
    return pd.read_csv(path, dtype=str)


def merge_assessment(
    split_csv: Path,
    origin_table: Path,
    origin_acc_col: str,
    out_csv: Path,
) -> None:
    split_df = pd.read_csv(split_csv, dtype=str)
    origin_df = _read_origin(origin_table)

    if SPLIT_ACC not in split_df.columns:
        raise KeyError(f"Split CSV missing {SPLIT_ACC!r}: {split_csv}")
    if origin_acc_col not in origin_df.columns:
        raise KeyError(f"Origin table missing {origin_acc_col!r}: {origin_table}")

    sla_col = _pick_assessment_column(origin_df)
    lookup = (
        origin_df[[origin_acc_col, sla_col]]
        .drop_duplicates(subset=[origin_acc_col])
        .rename(columns={origin_acc_col: SPLIT_ACC, sla_col: STUDY_ASSESSMENT})
    )
    lookup[SPLIT_ACC] = lookup[SPLIT_ACC].astype(str).str.strip()

    split_df[SPLIT_ACC] = split_df[SPLIT_ACC].astype(str).str.strip()
    merged = split_df.merge(lookup, on=SPLIT_ACC, how="left")

    missing = int(merged[STUDY_ASSESSMENT].isna().sum()) + int(
        (merged[STUDY_ASSESSMENT].astype(str).str.strip() == "").sum()
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)

    print(f"Split:   {split_csv} ({len(split_df)} rows)")
    print(f"Origin:  {origin_table} ({len(origin_df)} rows, acc={origin_acc_col!r})")
    print(f"Wrote:   {out_csv} ({len(merged)} rows)")
    print(f"  missing/empty {STUDY_ASSESSMENT}: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge studylevelassessment into Penn datasplit CSVs.")
    parser.add_argument("--cohort", choices=["new", "old", "both"], default="both")
    parser.add_argument("--split-csv", type=Path, default=None)
    parser.add_argument("--origin-table", type=Path, default=None)
    parser.add_argument("--origin-acc", type=str, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    cohorts = ("new", "old") if args.cohort == "both" else (args.cohort,)
    for name in cohorts:
        cfg = DEFAULT_SPLITS[name]
        merge_assessment(
            args.split_csv or cfg["split"],
            args.origin_table or cfg["origin"],
            args.origin_acc or cfg["origin_acc"],
            args.out_csv or cfg["out"],
        )


if __name__ == "__main__":
    main()
