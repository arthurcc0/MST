"""Step 3b (old Penn only): drop old-Penn exams of patients who are in new Penn.

Pipeline:  new_penn_mapping.py -> step3_create_split.py -> this -> make_joint_datasplit.py

New Penn (BI-RADS-4) is the evaluation cohort, so shared patients stay there
and their old-Penn exams are removed. Every new-Penn patient is in the test
set of some fold, so this is what lets a single stage-1 model (or one joint
graft) be leak-free for all new-Penn folds.

Rows are only removed: the remaining exams keep the Fold/Split assigned by
step3, so reruns are deterministic. The patient set is the union of all
``NEW_PENN_SOURCES`` (label table, split, holdout), which also covers new-Penn
exams without laterality or missing on disk.

Outputs:
    - ``OUTPUT_CSV``: filtered old-Penn split (same columns as the input).
    - ``<OUTPUT_CSV stem>_excluded.csv``: removed exams, one row per UID.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
from mst.data.patient_ids import normalize_patient_id, patient_ids  # noqa: E402

OUT_ROOT = Path(r"D:\Users\arthur\Data\MST_birads4")
OLD_SPLIT_CSV = OUT_ROOT / "old_penn_datasplit.csv"
OUTPUT_CSV = OUT_ROOT / "old_penn_datasplit_noNewPenn.csv"
# The holdout (null laterality) is never evaluated, so its patients may stay in old Penn.
NEW_PENN_SOURCES = (OUT_ROOT / "new_penn_datasplit_v5_noBenHR.csv",)
# If ever needed, the holdout can be added back in.
# NEW_PENN_SOURCES = (
#     PROJECT_ROOT / "tables" / "matches_birads4_all_v5_noBenHR.xlsx",
#     OUT_ROOT / "new_penn_datasplit_v5_noBenHR.csv",
#     OUT_ROOT / "new_penn_holdout_v5_noBenHR.csv",
# )
COL_PATIENT = "MRN"

def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=str, keep_default_na=False)
    return pd.read_csv(path, dtype=str)


def load_new_penn_patients(sources) -> set[str]:
    ids: set[str] = set()
    for path in sources:
        df = _read_table(Path(path))
        if COL_PATIENT not in df.columns:
            raise KeyError(f"{path} has no {COL_PATIENT!r} column.")
        found = patient_ids(df[COL_PATIENT])
        print(f"  {Path(path).name}: {len(found)} patients")
        ids |= found
    return ids


def _counts(df: pd.DataFrame) -> pd.DataFrame:
    g = df.drop_duplicates(["UID", "Fold"]).groupby(["Fold", "Split"])["Malignant"]
    return g.agg(n="count", pos=lambda s: int(pd.to_numeric(s).sum()))


def exclude_new_penn_patients(
    old_split_csv: Path = OLD_SPLIT_CSV,
    output_csv: Path = OUTPUT_CSV,
    new_penn_sources=NEW_PENN_SOURCES,
) -> pd.DataFrame:
    old = pd.read_csv(old_split_csv, dtype=str)
    if COL_PATIENT not in old.columns:
        raise KeyError(f"{old_split_csv} has no {COL_PATIENT!r} column; rerun step3 with MRN grouping.")
    blank = old[COL_PATIENT].map(normalize_patient_id) == ""
    if blank.any():
        raise ValueError(f"{int(blank.sum())} rows in {old_split_csv} have no {COL_PATIENT}.")

    print("New-Penn patient sources:")
    new_ids = load_new_penn_patients(new_penn_sources)
    print(f"  union: {len(new_ids)} patients")

    shared = old[COL_PATIENT].map(normalize_patient_id).isin(new_ids)
    kept = old.loc[~shared].copy()
    removed = old.loc[shared].drop_duplicates("UID").drop(columns=["Fold", "Split"], errors="ignore")

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(output_csv, index=False)
    excluded_csv = output_csv.with_name(f"{output_csv.stem}_excluded.csv")
    removed.to_csv(excluded_csv, index=False)

    n_old = old["UID"].nunique()
    print(f"\nOld split:  {old_split_csv}  ({n_old} exams, {old[COL_PATIENT].nunique()} patients)")
    print(f"Removed:    {len(removed)} exams from {removed[COL_PATIENT].nunique()} shared patients "
          f"({len(removed) / max(n_old, 1):.1%})  -> {excluded_csv}")
    if "label" in removed.columns:
        print(f"            labels: {removed['label'].value_counts().to_dict()}")
    print(f"Wrote:      {output_csv}  ({kept['UID'].nunique()} exams)")

    before, after = _counts(old), _counts(kept)
    table = before.join(after, lsuffix="_before", rsuffix="_after")
    print("\nPer fold/split (unique exams, positives):")
    print(table.to_string())

    leftover = kept[COL_PATIENT].map(normalize_patient_id).isin(new_ids).sum()
    if leftover:
        raise RuntimeError(f"{leftover} rows still share a patient with new Penn.")
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--old-split-csv", type=Path, default=OLD_SPLIT_CSV)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--new-penn-sources", type=Path, nargs="+", default=list(NEW_PENN_SOURCES),
                        help="Tables with an MRN column listing every new-Penn patient.")
    args = parser.parse_args()
    exclude_new_penn_patients(args.old_split_csv, args.output_csv, args.new_penn_sources)


if __name__ == "__main__":
    main()
