"""Audit why exams left the old/new Penn tables before training.

Writes one row per source-table accession (the label workbook / EHR export),
with a ``discard_reason`` field plus flags for each pipeline CSV. Original
source columns are kept so you can tabulate by any existing field.

    python scripts/preprocessing/new_penn/audit_cohort_lineage.py

Outputs under ``D:\\Users\\arthur\\Data\\MST_birads4``:
    - ``new_penn_source_with_discard.csv``  (xlsx columns + audit)
    - ``old_penn_source_with_discard.csv``
    - ``cohort_lineage.csv``  (slim, both cohorts, for counting)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import new_penn_mapping as npm  # noqa: E402
OUT_ROOT = Path(r"D:\Users\arthur\Data\MST_birads4")

NEW_SOURCE = PROJECT_ROOT / "tables" / "matches_birads4_all_v5_noBenHR.xlsx"
OLD_SOURCE = PROJECT_ROOT / "table_utils" / "lat_added_dummy_ehr_chat_no_birads4_v2.csv"

NEW_MAPPING = OUT_ROOT / "new_penn_mapping_v5_noBenHR.csv"
NEW_HOLDOUT = OUT_ROOT / "new_penn_holdout_v5_noBenHR.csv"
NEW_SPLIT = OUT_ROOT / "new_penn_datasplit_v5_noBenHR.csv"
NEW_HOLDOUT_F0 = OUT_ROOT / "new_penn_datasplit_v5_holdoutF0.csv"

OLD_MAPPING = OUT_ROOT / "old_penn_mapping.csv"
OLD_HOLDOUT = OUT_ROOT / "old_penn_holdout.csv"
OLD_SPLIT = OUT_ROOT / "old_penn_datasplit.csv"
OLD_NONEW = OUT_ROOT / "old_penn_datasplit_noNewPenn.csv"
OLD_NONEW_EXCL = OUT_ROOT / "old_penn_datasplit_noNewPenn_excluded.csv"
OLD_TRAINVAL = OUT_ROOT / "old_penn_datasplit_trainval.csv"

USED = "used"
HOLDOUT = "holdout"
DROPPED = "dropped"

REASON_DUP = "duplicate_accession"
REASON_BAD_LABEL = "invalid_or_missing_label"
REASON_BC_PRIOR = "benign_birads23_with_bc_prior"
REASON_LAT_EXCL = "laterality_exclude_4_5"
REASON_LAT_HOLD = "laterality_holdout_or_blank"
REASON_MISSING_NPZ = "missing_npz_on_disk"
REASON_HR_EXCL = "high_risk_excluded_at_split"
REASON_NO_CROP = "missing_from_cropped_data"
REASON_SHARED_NEW = "old_exam_of_new_penn_patient"


def _read_any(path: Path, acc_col: str | None = None) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, dtype=str, keep_default_na=False)
    else:
        df = pd.read_csv(path, dtype=str)
    if acc_col and acc_col in df.columns:
        df[acc_col] = df[acc_col].astype(str).str.strip()
    return df


def _acc_set(path: Path, col: str) -> set[str]:
    df = _read_any(path, col)
    if df.empty or col not in df.columns:
        return set()
    return set(df[col].astype(str).str.strip())


def _uid_set(path: Path) -> set[str]:
    df = _read_any(path)
    if df.empty or "UID" not in df.columns:
        return set()
    return set(df["UID"].astype(str).str.strip())


def _classify_lat(val, blank_mode: str) -> str:
    prev = npm.BLANK_LATERALITY_MODE
    npm.BLANK_LATERALITY_MODE = blank_mode
    try:
        return npm._classify_laterality_cell(val)
    finally:
        npm.BLANK_LATERALITY_MODE = prev


def _mapping_drop_reason(
    row: pd.Series,
    *,
    acc_col: str,
    lat_col: str,
    label_col: str,
    blank_mode: str,
    discard_benign_prior: bool,
    in_mapping: bool,
    in_holdout: bool,
) -> str | None:
    """First mapping-stage reason, or None if the exam reached mapping/holdout."""
    if in_mapping:
        return None
    if in_holdout:
        return REASON_LAT_HOLD

    lab = npm._normalize_precomputed_label(row.get(label_col))
    if pd.isna(lab):
        return REASON_BAD_LABEL

    if discard_benign_prior:
        tmp = pd.DataFrame([row])
        prev = npm.DISCARD_BENIGN_STUDY_WITH_BC_PRIOR
        npm.DISCARD_BENIGN_STUDY_WITH_BC_PRIOR = True
        try:
            if bool(npm._mask_discard_benign_study_with_bc_prior(tmp).iloc[0]):
                return REASON_BC_PRIOR
        finally:
            npm.DISCARD_BENIGN_STUDY_WITH_BC_PRIOR = prev

    lat_cls = _classify_lat(row.get(lat_col), blank_mode)
    if lat_cls == "exclude":
        return REASON_LAT_EXCL
    return REASON_MISSING_NPZ


def _audit_cohort(
    source: pd.DataFrame,
    *,
    cohort: str,
    acc_col: str,
    lat_col: str,
    label_col: str,
    blank_mode: str,
    discard_benign_prior: bool,
    mapping_accs: set[str],
    holdout_accs: set[str],
    split_uids: set[str],
    final_uids: set[str],
    step3b_removed_uids: set[str],
    hr_exclude_at_split: bool,
) -> pd.DataFrame:
    df = source.copy()
    df["_accession"] = df[acc_col].astype(str).str.strip()
    df["_dup"] = df["_accession"].duplicated(keep="first")

    reasons: list[str] = []
    statuses: list[str] = []
    uids: list[str] = []

    for _, row in df.iterrows():
        acc = row["_accession"]
        if row["_dup"]:
            reasons.append(REASON_DUP)
            statuses.append(DROPPED)
            uids.append("")
            continue

        in_map = acc in mapping_accs
        in_hold = acc in holdout_accs
        map_reason = _mapping_drop_reason(
            row,
            acc_col=acc_col,
            lat_col=lat_col,
            label_col=label_col,
            blank_mode=blank_mode,
            discard_benign_prior=discard_benign_prior,
            in_mapping=in_map,
            in_holdout=in_hold,
        )
        if map_reason == REASON_LAT_HOLD:
            reasons.append(map_reason)
            statuses.append(HOLDOUT)
            uids.append("")
            continue
        if map_reason:
            reasons.append(map_reason)
            statuses.append(DROPPED)
            uids.append("")
            continue

        lat_cls = _classify_lat(row.get(lat_col), blank_mode)
        if lat_cls == "random":
            # Mapping already assigned left/right; recover from mapping CSV via UID later.
            side = ""
        elif lat_cls in ("left", "right"):
            side = lat_cls
        else:
            side = ""

        uid = f"{acc}_{side}" if side else ""
        # Prefer the UID that actually appears in a split (handles randomized lat).
        left_u, right_u = f"{acc}_left", f"{acc}_right"
        if left_u in split_uids:
            uid = left_u
        elif right_u in split_uids:
            uid = right_u
        elif left_u in step3b_removed_uids:
            uid = left_u
        elif right_u in step3b_removed_uids:
            uid = right_u
        elif left_u in final_uids:
            uid = left_u
        elif right_u in final_uids:
            uid = right_u

        if hr_exclude_at_split:
            lab = str(row.get(label_col, "")).strip().lower().replace("-", " ")
            if lab == "high risk":
                reasons.append(REASON_HR_EXCL)
                statuses.append(DROPPED)
                uids.append(uid)
                continue

        if uid and uid in step3b_removed_uids:
            reasons.append(REASON_SHARED_NEW)
            statuses.append(DROPPED)
            uids.append(uid)
            continue

        if uid in final_uids:
            reasons.append("")
            statuses.append(USED)
            uids.append(uid)
            continue

        if uid in split_uids:
            # In an intermediate split but not the trainval/holdoutF0 file.
            reasons.append("")
            statuses.append(USED)
            uids.append(uid)
            continue

        reasons.append(REASON_NO_CROP)
        statuses.append(DROPPED)
        uids.append(uid)

    df["cohort"] = cohort
    df["accession"] = df["_accession"]
    df["UID"] = uids
    df["status"] = statuses
    df["discard_reason"] = reasons
    df["in_mapping"] = df["_accession"].isin(mapping_accs)
    df["in_holdout_csv"] = df["_accession"].isin(holdout_accs)
    df["in_split"] = [u in split_uids for u in uids]
    df["in_final_used"] = [u in final_uids for u in uids]
    return df.drop(columns=["_accession", "_dup"])


def _front_cols(df: pd.DataFrame, extra: tuple[str, ...]) -> pd.DataFrame:
    first = [
        "cohort",
        "accession",
        "UID",
        "status",
        "discard_reason",
        "in_mapping",
        "in_holdout_csv",
        "in_split",
        "in_final_used",
        *extra,
    ]
    first = [c for c in first if c in df.columns]
    rest = [c for c in df.columns if c not in first]
    return df[first + rest]


def _print_summary(df: pd.DataFrame, title: str) -> None:
    print(f"\n=== {title}  n={len(df)} ===")
    print(df["status"].value_counts().to_string())
    dropped = df.loc[df["discard_reason"] != "", "discard_reason"]
    if len(dropped):
        print("discard_reason:")
        print(dropped.value_counts().to_string())
    else:
        print("discard_reason: (none)")


def main() -> None:
    new_src = _read_any(NEW_SOURCE, "newaccession")
    old_src = _read_any(OLD_SOURCE, "dummy_acc")
    if new_src.empty:
        raise FileNotFoundError(NEW_SOURCE)
    if old_src.empty:
        raise FileNotFoundError(OLD_SOURCE)

    new_map = _acc_set(NEW_MAPPING, "PatientID")
    new_hold = _acc_set(NEW_HOLDOUT, "PatientID")
    new_split = _uid_set(NEW_SPLIT)
    new_final = _uid_set(NEW_HOLDOUT_F0)

    old_map = _acc_set(OLD_MAPPING, "PatientID")
    old_hold = _acc_set(OLD_HOLDOUT, "PatientID")
    old_split = _uid_set(OLD_SPLIT)
    old_final = _uid_set(OLD_TRAINVAL)
    old_excl = _uid_set(OLD_NONEW_EXCL)

    new_df = _audit_cohort(
        new_src,
        cohort="new_penn",
        acc_col="newaccession",
        lat_col="lat",
        label_col="label",
        blank_mode="holdout",
        discard_benign_prior=False,
        mapping_accs=new_map,
        holdout_accs=new_hold,
        split_uids=new_split,
        final_uids=new_final,
        step3b_removed_uids=set(),
        hr_exclude_at_split=False,
    )
    old_df = _audit_cohort(
        old_src,
        cohort="old_penn",
        acc_col="dummy_acc",
        lat_col="laterality",
        label_col="label",
        blank_mode="random",
        discard_benign_prior=True,
        mapping_accs=old_map,
        holdout_accs=old_hold,
        split_uids=old_split,
        final_uids=old_final,
        step3b_removed_uids=old_excl,
        hr_exclude_at_split=True,
    )

    new_out = _front_cols(
        new_df,
        ("MRN", "label", "lat", "newaccession", "oldaccession", "bc_prior", "studylevelassessment"),
    )
    old_out = _front_cols(
        old_df,
        ("MRN", "label", "laterality", "dummy_acc", "real_acc", "bc_prior", "studylevelassessment"),
    )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    new_path = OUT_ROOT / "new_penn_source_with_discard.csv"
    old_path = OUT_ROOT / "old_penn_source_with_discard.csv"
    slim_path = OUT_ROOT / "cohort_lineage.csv"
    new_out.to_csv(new_path, index=False)
    old_out.to_csv(old_path, index=False)

    slim_cols = [
        "cohort", "accession", "UID", "status", "discard_reason",
        "in_mapping", "in_holdout_csv", "in_split", "in_final_used",
        "MRN", "label",
    ]
    slim = pd.concat(
        [
            new_out[[c for c in slim_cols if c in new_out.columns]].assign(
                lat=new_out["lat"] if "lat" in new_out.columns else "",
            ),
            old_out[[c for c in slim_cols if c in old_out.columns]].assign(
                lat=old_out["laterality"] if "laterality" in old_out.columns else "",
            ),
        ],
        ignore_index=True,
    )
    slim.to_csv(slim_path, index=False)

    _print_summary(new_out, "new Penn  (matches_birads4_all_v5_noBenHR.xlsx)")
    print(f"  mapping {len(new_map)}  holdout {len(new_hold)}  "
          f"split {len(new_split)}  holdoutF0 unique {len(new_final)}")
    _print_summary(old_out, "old Penn  (lat_added_dummy_ehr_chat_no_birads4_v2.csv)")
    print(f"  mapping {len(old_map)}  holdout {len(old_hold)}  "
          f"split {len(old_split)}  noNewPenn-excl {len(old_excl)}  trainval {len(old_final)}")
    print("\nWrote:")
    print(f"  {new_path}")
    print(f"  {old_path}")
    print(f"  {slim_path}")


if __name__ == "__main__":
    main()
