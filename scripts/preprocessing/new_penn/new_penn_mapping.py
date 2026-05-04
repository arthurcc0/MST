"""Build a manifest CSV for the new PENN data layout.

Pipeline order (mirrors scripts/preprocessing/penn but adapted for .npz inputs):

    1. new_penn_mapping.py        <- this file
    2. step1_npz2nifti.py
    3. step2a_calc_sub.py
    4. step2b_apply_mask_split.py
    5. step3_create_split.py

Inputs:
    - VOL_DIR: a flat folder of .npz files, each named ``<newaccession>.npz``,
      containing keys ``'pre'``, ``'post'`` (volumes) and ``'dic_pre'`` /
      ``'dic_post'`` (metadata dicts; pixel spacing lives in there).
    - LABEL_TABLE: tables/matches_birads4_all.xlsx with columns
      ``newaccession``, ``laterality`` (1=R, 2=L, 9=undef), ``label``
      (``malignant`` / ``high risk`` / ``benign``), and
      ``PennChart_EpicPatientId`` (used for grouping in step3 to avoid
      patient leakage between train/val/test).

Output:
    - new_penn_mapping.csv with one row per case found on disk, joined against
      the label table.
"""

from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ---- Configurable paths -----------------------------------------------------
VOL_DIR = Path(r"\\10.156.155.77\mccarthy_lab\shared\mri_preproc\n4bc")
LABEL_TABLE = Path(r"D:\Users\arthur\Projects\MST\tables\matches_birads4_all.xlsx")
OUT_ROOT = Path(r"D:\Users\arthur\Data\MST_birads4")
OUTPUT_CSV = OUT_ROOT / "new_penn_mapping.csv"
HOLDOUT_CSV = OUT_ROOT / "new_penn_holdout.csv"  # cases with null laterality (reserved for test)

# Column names inside LABEL_TABLE
COL_NEWACC = "newaccession"
COL_LATERALITY = "lat"
COL_LABEL = "label"
COL_PATIENT = "PennChart_EpicPatientId"

# Debug: only keep the first case (set to False for full runs)
DEBUG_SINGLE = False


def _read_label_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, dtype={COL_NEWACC: str, COL_PATIENT: str})
    else:
        df = pd.read_csv(path, dtype={COL_NEWACC: str, COL_PATIENT: str})
    df[COL_NEWACC] = df[COL_NEWACC].astype(str).str.strip()
    if COL_PATIENT in df.columns:
        df[COL_PATIENT] = df[COL_PATIENT].astype(str).str.strip()
    return df


def create_mapping() -> None:
    if not VOL_DIR.is_dir():
        raise FileNotFoundError(f"Volume directory not found: {VOL_DIR}")
    if not LABEL_TABLE.is_file():
        raise FileNotFoundError(f"Label table not found: {LABEL_TABLE}")

    df_labels = _read_label_table(LABEL_TABLE)
    keep_cols = [c for c in (COL_NEWACC, COL_LATERALITY, COL_LABEL, COL_PATIENT) if c in df_labels.columns]
    df_labels = df_labels[keep_cols].drop_duplicates(subset=[COL_NEWACC])

    # Only consider .npz files whose stem appears in the label table; the
    # VOL_DIR contains extra files that are not part of this study.
    records = []
    n_missing_on_disk = 0
    for newacc in tqdm(df_labels[COL_NEWACC].tolist(), desc="Resolving volumes"):
        npz_path = VOL_DIR / f"{newacc}.npz"
        if not npz_path.is_file():
            n_missing_on_disk += 1
            continue
        records.append(
            {
                "PatientID": newacc,
                "FullPath": str(npz_path).replace("\\", "/"),
            }
        )

    if not records:
        print(f"No labelled .npz files found under {VOL_DIR}.")
        return

    df_files = pd.DataFrame(records)
    df_merged = df_files.merge(
        df_labels, how="left", left_on="PatientID", right_on=COL_NEWACC
    )
    if COL_NEWACC in df_merged.columns and COL_NEWACC != "PatientID":
        df_merged = df_merged.drop(columns=[COL_NEWACC])

    print(f"Label table rows: {len(df_labels)}; .npz on disk for those: {len(records)}; missing on disk: {n_missing_on_disk}")

    # Reserve cases with null laterality for the held-out test set: they are
    # written to HOLDOUT_CSV and excluded from the main mapping consumed by
    # step1+ scripts. The ``lat`` column contains strings ('left', 'right',
    # 'null'); we treat both the literal string 'null' and actual NaN as
    # missing.
    if COL_LATERALITY in df_merged.columns:
        lat_str = df_merged[COL_LATERALITY].astype(str).str.strip().str.lower()
        null_lat_mask = df_merged[COL_LATERALITY].isna() | lat_str.isin({"null", "nan", ""})
    else:
        null_lat_mask = pd.Series(False, index=df_merged.index)

    df_holdout = df_merged[null_lat_mask].copy()
    df_main = df_merged[~null_lat_mask].copy()

    if DEBUG_SINGLE:
        df_main = df_main.head(1)
        print(f"[DEBUG_SINGLE] keeping only the first case: {df_main['PatientID'].tolist()}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_main.to_csv(OUTPUT_CSV, index=False)
    if len(df_holdout):
        df_holdout.to_csv(HOLDOUT_CSV, index=False)

    n_total = len(df_merged)
    n_main = len(df_main)
    n_holdout = len(df_holdout)
    n_with_label = df_main[COL_LABEL].notna().sum() if COL_LABEL in df_main.columns else 0
    print(f"Mapping written to {OUTPUT_CSV}")
    print(f"  total .npz cases:                  {n_total}")
    print(f"  kept (laterality present):         {n_main}")
    print(f"  reserved for test (null laterality): {n_holdout}  -> {HOLDOUT_CSV}")
    print(f"  in main mapping & matched to label table: {n_with_label}")
    print(f"  in main mapping & unmatched (NaN label):  {n_main - n_with_label}")


if __name__ == "__main__":
    create_mapping()
