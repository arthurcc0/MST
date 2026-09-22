"""Graft old-Penn train/val onto existing new-Penn folds (no re-split).

Keeps ``new_penn`` Fold/Split so the new-Penn test set stays identical to the
staged experiment. By default, new-Penn fold F gets old-Penn fold F train+val
as extra ``Split=train`` (that fold's old test stays out). Pass
``--old-source-fold N`` to reuse one old fold's train+val on every new fold.

Patient leak guard: an old exam whose group id (``MRN``, or mapping-joined
``MRN``, else ``PennChart_EpicPatientId``) appears in that fold's new-Penn
test or val is dropped from the graft. Empty / missing ids are not treated
as a shared patient. IDs are compared after stripping a pandas float suffix
(``1144138.0`` == ``1144138``). Until old Penn has real MRNs this is a no-op.

Writes a ``cohort`` column (``new_penn`` | ``old_penn``) and a unified ``MRN``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

OUT_ROOT = Path(r"D:\Users\arthur\Data\MST_birads4")
NEW_SPLIT_CSV = OUT_ROOT / "new_penn_datasplit_v5_noBenHR.csv"
OLD_SPLIT_CSV = OUT_ROOT / "old_penn_datasplit_v2.csv"
OLD_MAPPING_CSV = OUT_ROOT / "old_penn_mapping_v2.csv"
OUTPUT_CSV = OUT_ROOT / "joint_penn_datasplit_v5.csv"

REQUIRED_SPLIT_COLS = ("UID", "Fold", "Split", "Malignant")
_BLANK_IDS = frozenset({"", "nan", "none", "null", "<na>"})
# pandas often writes integer MRNs as "123.0"; keep the integer digits.
_FLOAT_INT_ID = re.compile(r"^([+-]?)(\d+)\.0+$")


def _norm_id(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    text = str(val).strip()
    if text.lower() in _BLANK_IDS:
        return ""
    match = _FLOAT_INT_ID.fullmatch(text)
    if match:
        return match.group(1) + match.group(2)
    return text


def _require_cols(df: pd.DataFrame, cols: tuple[str, ...], path: Path) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{path} missing columns {missing}; have {list(df.columns)}")


def _attach_old_mrn(old: pd.DataFrame, mapping_csv: Path | None) -> pd.DataFrame:
    """Prefer MRN already on the split; else join mapping on accession."""
    out = old.copy()
    if "MRN" in out.columns and out["MRN"].map(_norm_id).ne("").any():
        return out
    if mapping_csv is None or not mapping_csv.is_file():
        return out
    mp = pd.read_csv(mapping_csv, dtype=str)
    if "MRN" not in mp.columns:
        print(f"Old mapping has no MRN column ({mapping_csv.name}); leak guard uses Epic/accession ids.")
        return out
    acc_col = "PatientID" if "PatientID" in mp.columns else None
    if acc_col is None:
        return out
    lookup = (
        mp[[acc_col, "MRN"]]
        .assign(**{acc_col: mp[acc_col].astype(str).str.strip(), "MRN": mp["MRN"].map(_norm_id)})
        .drop_duplicates(acc_col)
    )
    if "dummy_acc" not in out.columns:
        return out
    out = out.merge(
        lookup.rename(columns={acc_col: "dummy_acc", "MRN": "_map_mrn"}),
        on="dummy_acc",
        how="left",
    )
    if "MRN" in out.columns:
        out["MRN"] = out["MRN"].map(_norm_id)
        out["MRN"] = out["MRN"].where(out["MRN"] != "", out["_map_mrn"].map(_norm_id))
    else:
        out["MRN"] = out["_map_mrn"].map(_norm_id)
    return out.drop(columns=["_map_mrn"], errors="ignore")


def _group_id(row: pd.Series, *, prefer_mrn: bool) -> str:
    if prefer_mrn:
        mrn = _norm_id(row.get("MRN"))
        if mrn:
            return mrn
    epic = _norm_id(row.get("PennChart_EpicPatientId"))
    if epic:
        return epic
    return _norm_id(row.get("dummy_acc")) or _norm_id(row.get("UID"))


def _unique_exams(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates("UID").copy()


def make_joint_datasplit(
    new_split_csv: Path = NEW_SPLIT_CSV,
    old_split_csv: Path = OLD_SPLIT_CSV,
    output_csv: Path = OUTPUT_CSV,
    old_mapping_csv: Path | None = OLD_MAPPING_CSV,
    old_source_fold: int | None = None,
) -> pd.DataFrame:
    new = pd.read_csv(new_split_csv, dtype=str)
    old = pd.read_csv(old_split_csv, dtype=str)
    _require_cols(new, REQUIRED_SPLIT_COLS, new_split_csv)
    _require_cols(old, REQUIRED_SPLIT_COLS, old_split_csv)

    new = new.copy()
    new["cohort"] = "new_penn"
    if "MRN" not in new.columns:
        new["MRN"] = new.apply(lambda r: _group_id(r, prefer_mrn=False), axis=1)
    else:
        new["MRN"] = new["MRN"].map(_norm_id)
        empty = new["MRN"] == ""
        if empty.any():
            new.loc[empty, "MRN"] = new.loc[empty].apply(
                lambda r: _group_id(r, prefer_mrn=False), axis=1
            )

    old = _attach_old_mrn(old, old_mapping_csv)
    old["cohort"] = "old_penn"
    old["MRN"] = old.apply(lambda r: _group_id(r, prefer_mrn=True), axis=1)
    old_fold = old["Fold"].astype(str).str.strip()
    old_split = old["Split"].astype(str).str.strip().str.lower()

    new_uids = set(new["UID"].astype(str).str.strip())
    old_trainval = _unique_exams(old.loc[old_split.isin({"train", "val"})])
    uid_collision = new_uids & set(old_trainval["UID"].astype(str).str.strip())
    if uid_collision:
        raise ValueError(
            f"UID collision between cohorts ({len(uid_collision)}): "
            f"{sorted(uid_collision)[:8]}"
        )

    new_folds = sorted(new["Fold"].astype(str).str.strip().unique(), key=lambda x: int(x) if x.isdigit() else x)
    old_folds = set(old_fold.unique())
    grafted_parts = []
    leak_by_fold: dict[str, int] = {}
    pool_n_by_fold: dict[str, int] = {}
    for fold in new_folds:
        src_fold = str(old_source_fold) if old_source_fold is not None else fold
        if src_fold not in old_folds:
            raise KeyError(
                f"Old-Penn split has no Fold={src_fold!r} to graft onto new-Penn fold {fold}."
            )
        old_pool = _unique_exams(
            old.loc[(old_fold == src_fold) & (old_split.isin({"train", "val"}))]
        )
        pool_n_by_fold[fold] = len(old_pool)
        fold_new = new[new["Fold"].astype(str).str.strip() == fold]
        held_out = fold_new[fold_new["Split"].astype(str).str.strip().str.lower().isin({"test", "val"})]
        blocked = {gid for gid in held_out["MRN"].map(_norm_id) if gid}
        leak_mask = old_pool["MRN"].map(_norm_id).isin(blocked) & old_pool["MRN"].map(_norm_id).ne("")
        n_leak = int(leak_mask.sum())
        leak_by_fold[fold] = n_leak
        grafted = old_pool.loc[~leak_mask].copy()
        grafted["Fold"] = fold
        grafted["Split"] = "train"
        grafted_parts.append(grafted)

    joint = pd.concat([new] + grafted_parts, ignore_index=True)

    keep = [
        c
        for c in (
            "dummy_acc",
            "UID",
            "side",
            "lat",
            "Malignant",
            "label",
            "MRN",
            "PennChart_EpicPatientId",
            "cohort",
            "Fold",
            "Split",
        )
        if c in joint.columns
    ]
    extra = [c for c in joint.columns if c not in keep]
    joint = joint[keep + extra]
    joint["Malignant"] = pd.to_numeric(joint["Malignant"], errors="coerce").astype("Int64")

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    joint.to_csv(output_csv, index=False)

    print(f"New split:  {new_split_csv}")
    print(f"Old split:  {old_split_csv}")
    print(f"Wrote:      {output_csv}  ({len(joint)} rows)")
    print(f"UID collision: {len(uid_collision)}")
    if old_source_fold is None:
        print("Old graft: matching fold (new fold F gets old fold F train+val).")
    else:
        print(f"Old graft: reuse old fold-{old_source_fold} train+val on every new fold.")
    print("Old train+val unique exams available to graft:")
    for fold, n in pool_n_by_fold.items():
        print(f"  new fold {fold}: {n}")
    print("Leak-guard drops (old exams sharing MRN with new-Penn val/test):")
    for fold, n in leak_by_fold.items():
        print(f"  fold {fold}: {n}")
    if sum(leak_by_fold.values()) == 0:
        print("  (no-op until old Penn has MRNs that match new-Penn patients)")

    print("\nPer-fold counts (unique UID):")
    for fold in new_folds:
        sub = joint[joint["Fold"].astype(str).str.strip() == fold]
        print(f"  Fold {fold}")
        for split in ("train", "val", "test"):
            rows = sub[sub["Split"].astype(str).str.strip().str.lower() == split]
            n_new = int((rows["cohort"] == "new_penn").sum())
            n_old = int((rows["cohort"] == "old_penn").sum())
            print(f"    {split:5s}  new={n_new:5d}  old={n_old:5d}  total={len(rows):5d}")

    new_test = _unique_exams(new[new["Split"].astype(str).str.strip().str.lower() == "test"])
    joint_test = _unique_exams(joint[joint["Split"].astype(str).str.strip().str.lower() == "test"])
    if set(new_test["UID"]) != set(joint_test["UID"]):
        raise RuntimeError("Joint test UIDs do not match the new-Penn test set.")
    if (joint_test["cohort"] != "new_penn").any():
        raise RuntimeError("Joint test contains non-new-Penn rows.")
    print("\nNew-Penn test UIDs preserved.")
    return joint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Graft old-Penn train/val onto each new-Penn fold as extra train."
    )
    parser.add_argument("--new-split-csv", type=Path, default=NEW_SPLIT_CSV)
    parser.add_argument("--old-split-csv", type=Path, default=OLD_SPLIT_CSV)
    parser.add_argument(
        "--old-mapping-csv",
        type=Path,
        default=OLD_MAPPING_CSV,
        help="Optional old mapping used only to attach MRN for the leak guard.",
    )
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument(
        "--old-source-fold",
        type=int,
        default=None,
        help="Reuse this old-Penn fold's train+val on every new fold. Default: match fold indices.",
    )
    args = parser.parse_args()
    make_joint_datasplit(
        new_split_csv=args.new_split_csv,
        old_split_csv=args.old_split_csv,
        output_csv=args.output_csv,
        old_mapping_csv=args.old_mapping_csv,
        old_source_fold=args.old_source_fold,
    )


if __name__ == "__main__":
    main()
