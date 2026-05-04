"""Step 3: build the train/val/test split CSV consumed by ``PENN_Dataset3D``.

Logic:
    - Read ``tables/matches_birads4_all.xlsx`` (newaccession, lat, label,
      PennChart_EpicPatientId).
    - Map ``label`` -> ``Malignant`` flag:
        * ``malignant``   -> 1
        * ``high risk``   -> 1   (configurable via ``HIGH_RISK_AS_MALIGNANT``)
        * ``benign``      -> 0
    - Drop rows where ``lat`` is missing (NaN or the literal string 'null');
      those are reserved for the held-out test set by new_penn_mapping.py.
    - Keep only the side that has a finding: each newaccession becomes a
      single UID ``<newaccession>_<lat>`` (e.g. ``12345_left``). The
      contralateral side is discarded.
    - StratifiedGroupKFold grouped by ``PennChart_EpicPatientId`` (so the same
      patient never appears in train and val/test, even across multiple
      accessions).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# ---- Configurable paths and parameters --------------------------------------
LABEL_TABLE = Path(r"D:\Users\arthur\Projects\MST\tables\matches_birads4_all.xlsx")
OUT_ROOT = Path(r"D:\Users\arthur\Data\MST_birads4")
FINAL_DATA_DIR = OUT_ROOT / "final_cropped_and_masked_data"
OUTPUT_CSV = OUT_ROOT / "new_penn_datasplit.csv"

COL_NEWACC = "newaccession"
COL_LATERALITY = "lat"
COL_LABEL = "label"
COL_PATIENT = "PennChart_EpicPatientId"

HIGH_RISK_AS_MALIGNANT = True
N_FOLDS = 5
RANDOM_STATE_OUTER = 0
RANDOM_STATE_INNER = 42

# Debug: only keep the first patient (the resulting split will be tiny / invalid,
# but useful for end-to-end smoke testing).
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


def _patient_is_malignant(label: str) -> int:
    if not isinstance(label, str):
        return 0
    norm = label.strip().lower()
    if norm == "malignant":
        return 1
    if norm == "high risk" and HIGH_RISK_AS_MALIGNANT:
        return 1
    return 0


def _list_existing_uids(final_dir: Path) -> set:
    if not final_dir.is_dir():
        return set()
    return {p.name for p in final_dir.iterdir() if p.is_dir()}


def main() -> None:
    df = _read_label_table(LABEL_TABLE)

    needed = {COL_NEWACC, COL_LATERALITY, COL_LABEL, COL_PATIENT}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"Label table is missing required columns: {missing}")

    df = df[[COL_NEWACC, COL_LATERALITY, COL_LABEL, COL_PATIENT]].drop_duplicates(subset=[COL_NEWACC])

    # Normalize the ``lat`` string column ('left' / 'right' / 'null'). Drop
    # cases with missing laterality (NaN or 'null'): they are reserved for
    # the held-out test set in new_penn_mapping.py.
    df[COL_LATERALITY] = df[COL_LATERALITY].astype(str).str.strip().str.lower()
    n_before = len(df)
    df = df[~df[COL_LATERALITY].isin({"null", "nan", ""})]
    if len(df) < n_before:
        print(f"Dropped {n_before - len(df)} rows with missing laterality.")

    df["Malignant"] = df[COL_LABEL].apply(_patient_is_malignant)

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
    existing = _list_existing_uids(FINAL_DATA_DIR)
    if existing:
        n_before = len(df_expanded)
        df_expanded = df_expanded[df_expanded["UID"].isin(existing)].reset_index(drop=True)
        print(
            f"Filtered to UIDs present in {FINAL_DATA_DIR}: kept {len(df_expanded)}/{n_before}."
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

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_splits.to_csv(OUTPUT_CSV, index=False)
    print(f"Split written to {OUTPUT_CSV}: {len(df_splits)} rows across {N_FOLDS} folds.")
    print(df_splits.groupby(["Fold", "Split"])["Malignant"].agg(["count", "sum"]))


if __name__ == "__main__":
    main()
