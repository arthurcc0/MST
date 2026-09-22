"""Step 3: build bilateral BRCA datasplit for ``PENN_Dataset3D``.

Unlike BI-RADS-4 new Penn, each accession may contribute **two** UIDs
(``<accession>_left`` and ``<accession>_right``) when both breasts are included.
The same accession-level label applies to every side row (from ``label`` in
``brca_mapping.csv``, which may differ from raw ``bc_overall`` when the MRI
timing rule is enabled).

Split modes (``config.py``):
    - ``USE_PROVIDED_SPLIT=True``: keep train/val/test from the label-table
      ``split`` column (``Fold=0`` for all rows).
    - ``USE_PROVIDED_SPLIT=False``: ``StratifiedGroupKFold`` by ``uniqueid``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import (  # noqa: E402
    COL_ACCESSION,
    COL_LABEL,
    COL_PATIENT,
    COL_SPLIT,
    DATASPLIT_CSV,
    DEFAULT_DEPTH,
    FILTERED_LABEL_TABLE,
    FINAL_SLAB_DIR,
    MAPPING_CSV,
    N_FOLDS,
    OUT_ROOT,
    PROVIDED_SPLIT_FOLD,
    RANDOM_STATE_INNER,
    RANDOM_STATE_OUTER,
    SPLIT_EXCLUDE_VALUES,
    STANDARD_LABEL,
)
import config as brca_config  # noqa: E402

DEBUG_SINGLE = False
VALID_SPLITS = frozenset({"train", "val", "test"})


def _malignant_from_bc_overall(val) -> int | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip().lower()
    if s in {"0", "0.0", "false", "benign", "negative", "no"}:
        return 0
    if s in {"1", "1.0", "true", "malignant", "positive", "yes", "cancer"}:
        return 1
    return None


def _list_existing_uids(final_dir: Path) -> set[str]:
    if not final_dir.is_dir():
        return set()
    return {p.name for p in final_dir.iterdir() if p.is_dir()}


def _normalize_split(val) -> str | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip().lower()
    if s in {"", "nan", "none", "null"}:
        return None
    if s in SPLIT_EXCLUDE_VALUES:
        return None
    if s in VALID_SPLITS:
        return s
    raise ValueError(f"Unknown split value {val!r}; expected one of {sorted(VALID_SPLITS)}")


def _label_from_mapping_row(row) -> tuple[str, int] | None:
    label_text = row.get(STANDARD_LABEL)
    if label_text is not None and not (isinstance(label_text, float) and np.isnan(label_text)):
        s = str(label_text).strip().lower()
        if s in {"benign", "malignant"}:
            return s, int(s == "malignant")
    mal = _malignant_from_bc_overall(row.get(COL_LABEL))
    if mal is None:
        return None
    return ("malignant" if mal else "benign", mal)


def _expand_bilateral_rows(df: pd.DataFrame, *, use_provided_split: bool) -> pd.DataFrame:
    """One row per (accession, side) UID included in mapping."""
    rows = []
    for _, row in df.iterrows():
        acc = str(row["PatientID"]).strip()
        patient = str(row[COL_PATIENT]).strip()
        parsed = _label_from_mapping_row(row)
        if parsed is None:
            continue
        label_text, mal = parsed

        split_name = None
        if use_provided_split:
            split_name = _normalize_split(row.get(COL_SPLIT))
            if split_name is None:
                continue

        sides: list[tuple[str, str]] = []
        if int(row.get("include_left", 0)):
            sides.append(("left", f"{acc}_left"))
        if int(row.get("include_right", 0)):
            sides.append(("right", f"{acc}_right"))
        for side_name, uid in sides:
            item = {
                "dummy_acc": acc,
                "UID": uid,
                "side": side_name,
                "lat": side_name,
                STANDARD_LABEL: label_text,
                COL_LABEL: row.get(COL_LABEL, mal),
                "Malignant": int(mal),
                COL_PATIENT: patient,
            }
            if use_provided_split:
                item[COL_SPLIT] = split_name
            rows.append(item)
    return pd.DataFrame(rows)


def _build_kfold_splits(df_expanded: pd.DataFrame) -> pd.DataFrame:
    from sklearn.model_selection import StratifiedGroupKFold

    sgkf_outer = StratifiedGroupKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE_OUTER
    )
    sgkf_inner = StratifiedGroupKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE_INNER
    )

    splits = []
    for fold_i, (trainval_idx, test_idx) in enumerate(
        sgkf_outer.split(
            df_expanded["UID"],
            df_expanded["Malignant"],
            groups=df_expanded[COL_PATIENT],
        )
    ):
        df_split = df_expanded.copy()
        df_split["Fold"] = fold_i
        df_split["Split"] = np.nan

        df_trainval = df_split.iloc[trainval_idx]
        if len(df_trainval) > 1 and df_trainval["Malignant"].nunique() > 1:
            train_local, val_local = next(
                sgkf_inner.split(
                    df_trainval["UID"],
                    df_trainval["Malignant"],
                    groups=df_trainval[COL_PATIENT],
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
    return df_splits


def _build_provided_split(df_expanded: pd.DataFrame) -> pd.DataFrame:
    df_splits = df_expanded.copy()
    df_splits["Fold"] = PROVIDED_SPLIT_FOLD
    df_splits["Split"] = df_splits[COL_SPLIT]
    df_splits = df_splits.drop(columns=[COL_SPLIT])
    df_splits.drop_duplicates(subset=["UID", "Fold"], inplace=True)
    return df_splits


def _attach_split_from_label_table(df: pd.DataFrame, label_table: Path) -> pd.DataFrame:
    """Merge ``split`` from the filtered label table when mapping lacks it."""
    label_table = Path(label_table)
    if not label_table.is_file():
        raise FileNotFoundError(
            f"Mapping missing {COL_SPLIT!r} and label table not found: {label_table}"
        )
    if label_table.suffix.lower() in {".xlsx", ".xls"}:
        label_df = pd.read_excel(label_table, dtype=str, keep_default_na=False)
    else:
        label_df = pd.read_csv(label_table, dtype=str, keep_default_na=False)
    if COL_SPLIT not in label_df.columns or COL_ACCESSION not in label_df.columns:
        raise KeyError(
            f"Label table {label_table} missing {COL_SPLIT!r} or {COL_ACCESSION!r}"
        )
    split_df = (
        label_df[[COL_ACCESSION, COL_SPLIT]]
        .drop_duplicates(subset=[COL_ACCESSION], keep="first")
        .rename(columns={COL_ACCESSION: "PatientID"})
    )
    split_df["PatientID"] = split_df["PatientID"].astype(str).str.strip()
    out = df.copy()
    out["PatientID"] = out["PatientID"].astype(str).str.strip()
    out = out.drop(columns=[COL_SPLIT], errors="ignore").merge(
        split_df, on="PatientID", how="left", validate="m:1"
    )
    missing = out[COL_SPLIT].isna() | (out[COL_SPLIT].astype(str).str.strip() == "")
    if missing.any():
        n = int(missing.sum())
        raise KeyError(
            f"{n} mapping accessions have no {COL_SPLIT!r} in {label_table.name}"
        )
    return out


def _audit_provided_split(df_splits: pd.DataFrame, df_mapping: pd.DataFrame) -> None:
    acc_split = (
        df_mapping[["PatientID", COL_SPLIT]]
        .drop_duplicates("PatientID")
        .rename(columns={"PatientID": "dummy_acc", COL_SPLIT: "expected_split"})
    )
    acc_split["expected_split"] = acc_split["expected_split"].astype(str).str.strip().str.lower()
    got = df_splits.drop_duplicates("dummy_acc")[["dummy_acc", "Split"]]
    merged = acc_split.merge(got, on="dummy_acc", how="inner")
    merged = merged[~merged["expected_split"].isin(SPLIT_EXCLUDE_VALUES)]
    mismatch = merged[merged["Split"] != merged["expected_split"]]
    if not mismatch.empty:
        raise RuntimeError(
            f"Provided split audit failed: {len(mismatch)} accessions disagree with mapping."
        )
    print(
        f"Provided split audit: {len(merged)} accessions match mapping "
        f"({df_splits['Fold'].nunique()} fold)."
    )


def main(
    mapping_csv: Path = MAPPING_CSV,
    output_csv: Path = DATASPLIT_CSV,
    final_data_dir: Path = FINAL_SLAB_DIR,
    *,
    use_provided_split: bool | None = None,
    label_table: Path = FILTERED_LABEL_TABLE,
) -> None:
    if use_provided_split is None:
        use_provided_split = bool(brca_config.USE_PROVIDED_SPLIT)
    mapping_csv = Path(mapping_csv)
    output_csv = Path(output_csv)
    final_data_dir = Path(final_data_dir)

    if not mapping_csv.is_file():
        raise FileNotFoundError(f"Mapping CSV not found: {mapping_csv}")

    print(f"Reading mapping:     {mapping_csv}")
    print(f"Writing datasplit:   {output_csv}")
    print(f"Slab directory:    {final_data_dir}")
    print(
        f"Split mode:          "
        f"{'provided (' + COL_SPLIT + ' column, Fold=' + str(PROVIDED_SPLIT_FOLD) + ')' if use_provided_split else 'k-fold (StratifiedGroupKFold, N_FOLDS=' + str(N_FOLDS) + ')'}"
    )

    df = pd.read_csv(mapping_csv, dtype=str)
    needed = {"PatientID", COL_PATIENT, COL_LABEL, "include_left", "include_right"}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"Mapping missing columns: {sorted(missing)}")
    if use_provided_split and COL_SPLIT not in df.columns:
        print(
            f"Mapping missing {COL_SPLIT!r}; merging from label table {label_table}"
        )
        df = _attach_split_from_label_table(df, label_table)
    elif use_provided_split:
        print(f"Using {COL_SPLIT!r} from mapping CSV.")

    n_accessions = len(df)
    if use_provided_split:
        excluded = df[COL_SPLIT].astype(str).str.strip().str.lower().isin(SPLIT_EXCLUDE_VALUES)
        n_excluded = int(excluded.sum())
        if n_excluded:
            print(f"Skipping {n_excluded} accessions with split in {sorted(SPLIT_EXCLUDE_VALUES)}.")
    else:
        n_excluded = 0
        if brca_config.USE_PROVIDED_SPLIT:
            print(
                "Note: config.USE_PROVIDED_SPLIT=True but --kfold-split was passed; "
                "building new StratifiedGroupKFold splits."
            )
        print(f"StratifiedGroupKFold groups by {COL_PATIENT!r} ({N_FOLDS} folds).")

    df_expanded = _expand_bilateral_rows(df, use_provided_split=use_provided_split)
    if df_expanded.empty:
        raise RuntimeError("No UID rows after bilateral expansion.")

    print(
        f"Expanded {n_accessions} accessions -> {len(df_expanded)} side-level UIDs "
        f"({df_expanded['dummy_acc'].nunique()} accessions)."
    )
    if use_provided_split and n_excluded:
        print(f"  ({n_excluded} accessions dropped before expansion due to excluded split)")

    existing = _list_existing_uids(final_data_dir)
    if existing:
        n_before = len(df_expanded)
        df_expanded = df_expanded[df_expanded["UID"].isin(existing)].reset_index(drop=True)
        print(
            f"Filtered to UIDs on disk: kept {len(df_expanded)}/{n_before} "
            f"({df_expanded['dummy_acc'].nunique()} accessions)."
        )

    if DEBUG_SINGLE and not df_expanded.empty:
        first_patient = df_expanded[COL_PATIENT].iloc[0]
        df_expanded = df_expanded[df_expanded[COL_PATIENT] == first_patient].reset_index(drop=True)
        print(f"[DEBUG_SINGLE] patient {first_patient} only ({len(df_expanded)} UIDs).")

    if df_expanded.empty:
        raise RuntimeError("No UIDs left after disk filter.")

    if use_provided_split:
        df_splits = _build_provided_split(df_expanded)
        n_folds = 1
        _audit_provided_split(df_splits, df)
    else:
        df_splits = _build_kfold_splits(df_expanded)
        n_folds = N_FOLDS

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_splits.to_csv(output_csv, index=False)
    print(f"Split written: {len(df_splits)} rows across {n_folds} fold(s).")
    print(df_splits.groupby(["Fold", "Split"])["Malignant"].agg(["count", "sum"]))
    print(
        f"Unique patients: {df_splits[COL_PATIENT].nunique()} | "
        f"Unique accessions: {df_splits['dummy_acc'].nunique()} | "
        f"Unique UIDs: {df_splits['UID'].nunique()}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build bilateral BRCA datasplit CSV for PENN_Dataset3D."
    )
    parser.add_argument("--mapping-csv", type=Path, default=MAPPING_CSV)
    parser.add_argument("--output-csv", type=Path, default=DATASPLIT_CSV)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--label-table",
        type=Path,
        default=FILTERED_LABEL_TABLE,
        help="Label table with split column (fallback if mapping lacks split).",
    )
    parser.add_argument(
        "--kfold-split",
        action="store_true",
        default=False,
        help=(
            "Ignore config.USE_PROVIDED_SPLIT and build StratifiedGroupKFold splits "
            f"by {COL_PATIENT} instead."
        ),
    )
    args = parser.parse_args()
    use_provided_split = bool(brca_config.USE_PROVIDED_SPLIT) and not args.kfold_split

    if args.data_dir is not None:
        data_dir = args.data_dir
    elif args.depth != DEFAULT_DEPTH:
        data_dir = OUT_ROOT / f"final_cropped_and_masked_slabs_n{args.depth}_s3_o0"
    else:
        data_dir = FINAL_SLAB_DIR

    main(
        args.mapping_csv,
        args.output_csv,
        data_dir,
        use_provided_split=use_provided_split,
        label_table=args.label_table,
    )
