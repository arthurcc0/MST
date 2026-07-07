"""Step 3: build the train/val/test split CSV consumed by ``PENN_Dataset3D``.

Logic:
    - Prefer ``new_penn_mapping.csv`` (PatientID renamed to newaccession) so
      ``lat`` matches the finalized mapping script (codes 0/3 randomized there).
      If that file is missing, fall back to ``LABEL_TABLE`` (newaccession, lat,
      label, PennChart_EpicPatientId).
    - Map ``label`` -> ``Malignant`` flag:
        * ``malignant``   -> 1
        * ``benign``      -> 0
        * ``high risk``   -> controlled by ``HIGH_RISK_POLICY``:
            ``malignant`` (count as 1), ``benign`` (count as 0), or ``exclude``
            (drop from the split entirely).
    - Drop rows where ``lat`` is missing (NaN or the literal string 'null');
      those are reserved for the held-out test set by new_penn_mapping.py.
    - Keep only the side that has a finding: each newaccession becomes a
      single UID ``<newaccession>_<lat>`` (e.g. ``12345_left``). The
      contralateral side is discarded.
    - StratifiedGroupKFold uses a **group id** column (``COL_PATIENT``). If the
      table has no real patient id, that column is filled from ``COL_NEWACC`` so
      each accession is its own group (no cross-accession leakage when there is
      only one study per accession).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# ---- Configurable paths and parameters --------------------------------------
OUT_ROOT = Path(r"D:\Users\arthur\Data\MST_birads4")
DEFAULT_DEPTH = 32

# For new Penn data (BIRADS-4):
LABEL_TABLE = Path(r"D:\Users\arthur\Projects\MST\tables\matches_birads4_all_v3.xlsx")
MAPPING_CSV = OUT_ROOT / "new_penn_mapping_v3.csv"
OUTPUT_CSV = OUT_ROOT / "new_penn_datasplit_v3.csv"
FINAL_DATA_DIR = OUT_ROOT / "final_cropped_and_masked_slabs_n32_s3_o0"

COL_NEWACC = "newaccession"
COL_LATERALITY = "lat"
COL_LABEL = "label"
COL_PATIENT = "PennChart_EpicPatientId"
HIGH_RISK_POLICY = "malignant"  # or "benign" / "exclude"

# For old Penn data (all BI-RADS):
# LABEL_TABLE = Path(r"D:\Users\arthur\Projects\MST\table_utils\lat_added_dummy_ehr_chat_no_birads4_v2.csv")
# MAPPING_CSV = OUT_ROOT / "old_penn_mapping_v2.csv"
# OUTPUT_CSV = OUT_ROOT / "old_penn_datasplit_v2.csv"
# FINAL_DATA_DIR = OUT_ROOT / "final_cropped_and_masked_slabs_n32_s3_o0"

# COL_NEWACC = "dummy_acc"
# COL_LATERALITY = "lat"  # canonical name from new_penn_mapping.py / old_penn_mapping.csv
# COL_LABEL = "label"
# COL_PATIENT = "PennChart_EpicPatientId"
# # "malignant" | "benign" | "exclude" — how to handle canonical label "high risk"
# HIGH_RISK_POLICY = "exclude"

N_FOLDS = 5
RANDOM_STATE_OUTER = 0
RANDOM_STATE_INNER = 42

# Debug: only keep the first patient (the resulting split will be tiny / invalid,
# but useful for end-to-end smoke testing).
DEBUG_SINGLE = False


def _ensure_group_column(df: pd.DataFrame, *, source_had_patient_id: bool) -> pd.DataFrame:
    """Ensure COL_PATIENT exists for StratifiedGroupKFold; fall back to accession."""
    if COL_PATIENT not in df.columns or not source_had_patient_id:
        df[COL_PATIENT] = df[COL_NEWACC]
    else:
        df[COL_PATIENT] = df[COL_PATIENT].fillna(df[COL_NEWACC]).astype(str).str.strip()
    return df


def _read_label_table(path: Path) -> tuple[pd.DataFrame, bool]:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, dtype={COL_NEWACC: str})
    else:
        df = pd.read_csv(path, dtype={COL_NEWACC: str})
    df[COL_NEWACC] = df[COL_NEWACC].astype(str).str.strip()
    had_patient = COL_PATIENT in df.columns
    if had_patient:
        df[COL_PATIENT] = df[COL_PATIENT].astype(str).str.strip()
    return _ensure_group_column(df, source_had_patient_id=had_patient), had_patient


def _read_mapping_or_label_table(mapping_csv: Path, label_table: Path) -> tuple[pd.DataFrame, bool]:
    if mapping_csv.is_file():
        df = pd.read_csv(mapping_csv, dtype={"PatientID": str})
        if "PatientID" not in df.columns:
            raise KeyError(f"Mapping CSV missing PatientID column: {mapping_csv}")
        had_patient = COL_PATIENT in df.columns
        df = df.rename(columns={"PatientID": COL_NEWACC})
        return _ensure_group_column(df, source_had_patient_id=had_patient), had_patient
    return _read_label_table(label_table)


def _normalize_label(label) -> str:
    if not isinstance(label, str):
        return ""
    return label.strip().lower()


def _malignant_from_label(label, high_risk_policy: str) -> int | None:
    """Map label to Malignant (0/1). Returns None if row should be excluded."""
    norm = _normalize_label(label)
    if norm == "malignant":
        return 1
    if norm == "high risk":
        policy = high_risk_policy.strip().lower()
        if policy == "malignant":
            return 1
        if policy == "benign":
            return 0
        if policy == "exclude":
            return None
        raise ValueError(
            f"Invalid high_risk_policy={high_risk_policy!r}; "
            "use 'malignant', 'benign', or 'exclude'."
        )
    return 0


def _list_existing_uids(final_dir: Path) -> set:
    if not final_dir.is_dir():
        return set()
    return {p.name for p in final_dir.iterdir() if p.is_dir()}


def main(
    final_data_dir: Path = FINAL_DATA_DIR,
    output_csv: Path = OUTPUT_CSV,
    mapping_csv: Path = MAPPING_CSV,
    label_table: Path = LABEL_TABLE,
    high_risk_policy: str = HIGH_RISK_POLICY,
) -> None:
    final_data_dir = Path(final_data_dir)
    output_csv = Path(output_csv)
    mapping_csv = Path(mapping_csv)
    label_table = Path(label_table)
    print(f"Reading existing UIDs from: {final_data_dir}")
    print(f"Writing split CSV to:        {output_csv}")
    if mapping_csv.is_file():
        print(f"Using finalized mapping:   {mapping_csv}")
    else:
        print(f"Mapping CSV not found; using label table: {label_table}")

    df, had_patient_id = _read_mapping_or_label_table(mapping_csv, label_table)

    needed = {COL_NEWACC, COL_LATERALITY, COL_LABEL}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"Input is missing required columns: {missing}")

    if not had_patient_id or df[COL_PATIENT].astype(str).equals(df[COL_NEWACC].astype(str)):
        print(
            f"No distinct {COL_PATIENT} in input; StratifiedGroupKFold will group by "
            f"{COL_NEWACC} (one accession per group)."
        )
    else:
        print(f"StratifiedGroupKFold groups by {COL_PATIENT}.")

    df = df[[COL_NEWACC, COL_LATERALITY, COL_LABEL, COL_PATIENT]].drop_duplicates(subset=[COL_NEWACC])

    # Normalize the ``lat`` string column ('left' / 'right' / 'null'). Drop
    # cases with missing laterality (NaN or 'null'): they are reserved for
    # the held-out test set in new_penn_mapping.py.
    df[COL_LATERALITY] = df[COL_LATERALITY].astype(str).str.strip().str.lower()
    n_before = len(df)
    df = df[~df[COL_LATERALITY].isin({"null", "nan", ""})]
    if len(df) < n_before:
        print(f"Dropped {n_before - len(df)} rows with missing laterality.")

    print(f"High-risk label policy: {high_risk_policy!r}")
    malignant_vals = df[COL_LABEL].apply(lambda x: _malignant_from_label(x, high_risk_policy))
    exclude_mask = malignant_vals.isna()
    if exclude_mask.any():
        df = df.loc[~exclude_mask].copy()
        malignant_vals = malignant_vals.loc[~exclude_mask]
        print(f"Excluded {int(exclude_mask.sum())} rows with label 'high risk' (policy=exclude).")
    df["Malignant"] = malignant_vals.astype(int).values

    # ---- Keep only the side with a finding ----------------------------------
    # Each case contributes a single UID ``<newaccession>_<lat>``; the
    # contralateral side is dropped.
    df_expanded = pd.DataFrame(
        {
            "dummy_acc": df[COL_NEWACC].values,
            "UID": df[COL_NEWACC].astype(str) + "_" + df[COL_LATERALITY].astype(str),
            "side": df[COL_LATERALITY].values,
            COL_LATERALITY: df[COL_LATERALITY].values,
            "Malignant": df["Malignant"].values,
            COL_LABEL: df[COL_LABEL].values,
            COL_PATIENT: df[COL_PATIENT].values,
        }
    )

    # ---- Optional: keep only UIDs that exist on disk ------------------------
    existing = _list_existing_uids(final_data_dir)
    if existing:
        n_before = len(df_expanded)
        df_expanded = df_expanded[df_expanded["UID"].isin(existing)].reset_index(drop=True)
        print(
            f"Filtered to UIDs present in {final_data_dir}: kept {len(df_expanded)}/{n_before}."
        )

    if DEBUG_SINGLE and not df_expanded.empty:
        first_patient = df_expanded[COL_PATIENT].iloc[0]
        df_expanded = df_expanded[df_expanded[COL_PATIENT] == first_patient].reset_index(drop=True)
        print(f"[DEBUG_SINGLE] keeping only patient {first_patient} ({len(df_expanded)} rows).")

    if df_expanded.empty:
        raise RuntimeError("No UIDs left after filtering; nothing to split.")

    # ---- Split: outer test fold + inner train/val ---------------------------
    sgkf_outer = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE_OUTER)
    sgkf_inner = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE_INNER)

    splits = []
    for fold_i, (trainval_idx, test_idx) in enumerate(
        sgkf_outer.split(
            df_expanded["UID"], df_expanded["Malignant"], groups=df_expanded[COL_PATIENT]
        )
    ):
        df_split = df_expanded.copy()
        df_split["Fold"] = fold_i
        df_split["Split"] = np.nan

        df_trainval = df_split.iloc[trainval_idx]
        if len(df_trainval) > 1 and df_trainval["Malignant"].nunique() > 1:
            train_local, val_local = next(
                sgkf_inner.split(
                    df_trainval["UID"], df_trainval["Malignant"], groups=df_trainval[COL_PATIENT]
                )
            )
            train_global = df_trainval.iloc[train_local].index
            val_global = df_trainval.iloc[val_local].index
            df_split.loc[train_global, "Split"] = "train"
            df_split.loc[val_global, "Split"] = "val"
        else:
            df_split.loc[trainval_idx, "Split"] = "train"

        df_split.loc[test_idx, "Split"] = "test"
        splits.append(df_split)

    df_splits = pd.concat(splits, ignore_index=True)
    df_splits.drop_duplicates(subset=["UID", "Fold"], inplace=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_splits.to_csv(output_csv, index=False)
    print(f"Split written to {output_csv}: {len(df_splits)} rows across {N_FOLDS} folds.")
    print(df_splits.groupby(["Fold", "Split"])["Malignant"].agg(["count", "sum"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build the new_penn train/val/test split CSV consumed by PENN_Dataset3D.')
    parser.add_argument('--depth', type=int, default=DEFAULT_DEPTH,
                        help='D used by step2b_apply_mask_split.py. Drives the default --data_dir when != 32.')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Override the final_cropped_and_masked_data directory. Default: OUT_ROOT/final_cropped_and_masked_data[_d{D}] when D != 32.')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Override the output CSV path. Default: OUT_ROOT/new_penn_datasplit.csv (the split is depth-independent).')
    parser.add_argument('--mapping_csv', type=str, default=None,
                        help='Override new_penn_mapping.csv path; if missing file, LABEL_TABLE is used instead.')
    parser.add_argument(
        '--high-risk-policy',
        type=str,
        default=HIGH_RISK_POLICY,
        choices=("malignant", "benign", "exclude"),
        help="How to handle canonical label 'high risk': count as malignant, benign, or drop from split.",
    )
    cli_args = parser.parse_args()

    if cli_args.data_dir is not None:
        cli_data_dir = Path(cli_args.data_dir)
    elif cli_args.depth != DEFAULT_DEPTH:
        cli_data_dir = OUT_ROOT / f"final_cropped_and_masked_data_d{cli_args.depth}"
    else:
        cli_data_dir = FINAL_DATA_DIR

    cli_output_csv = Path(cli_args.output_csv) if cli_args.output_csv is not None else OUTPUT_CSV
    cli_mapping = Path(cli_args.mapping_csv) if cli_args.mapping_csv is not None else MAPPING_CSV

    main(
        final_data_dir=cli_data_dir,
        output_csv=cli_output_csv,
        mapping_csv=cli_mapping,
        high_risk_policy=cli_args.high_risk_policy,
    )
