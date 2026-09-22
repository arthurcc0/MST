"""Merge pooled main_predict outputs with new-Penn mapping and datasplit metadata.

Use this to build a single table for stratified analysis (high-risk label, bc_prior,
studylevelassessment, etc.) after running predict on val and/or test splits.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = ANALYSIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from aggregate_auc import aggregate, dedupe_uid_predictions, find_fold_results

DEFAULT_DATA_ROOT = Path(r"D:\Users\arthur\Data\MST_birads4")
DEFAULT_MAPPING_CSV = DEFAULT_DATA_ROOT / "new_penn_mapping_v4.csv"
DEFAULT_DATASPLIT_CSV = DEFAULT_DATA_ROOT / "new_penn_datasplit_v4.csv"
DEFAULT_LABEL_TABLE_XLSX = PROJECT_ROOT / "tables" / "matches_birads4_all_v4.xlsx"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results-pretrained-oldpenn-on-newpenn" / "runs" / "PENN"
DEFAULT_TRAIN_RUNS_ROOT = PROJECT_ROOT / "runs" / "PENN"

LABEL_TABLE_MERGE_COLS = [
    "brca",
    "brca1",
    "brca2",
    "PathCode",
    "histologicTypeIcdO3Description",
    "histologictypeicdo3description",
    "gradeclinicaldescription",
    "gradepathologicaldescription",
]

REASON_FOR_EXAM_COLS = ("ReasonforExam", "ReasonForExam", "reasonforexam")
_BRCA1_RE = re.compile(r"brca\s*[- ]?\s*1\b|brca1\b", re.IGNORECASE)
_BRCA2_RE = re.compile(r"brca\s*[- ]?\s*2\b|brca2\b", re.IGNORECASE)

MAPPING_MERGE_COLS = [
    "PatientID",
    "lat",
    "label",
    "bc_prior",
    "studylevelassessment",
    "studyleveloutcome",
    "componentleveloutcome",
    "componentlevelassessment",
    "PennChart_EpicPatientId",
    "recommendation",
    "BreastMriIndicationText",
]

SPLIT_MERGE_COLS = [
    "dummy_acc",
    "Split",
    "Malignant",
    "label",
]


def _uid_from_accession_lat(accession: str, lat: str) -> str:
    return f"{str(accession).strip()}_{str(lat).strip()}"


def _reason_for_exam_column(df: pd.DataFrame) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    for candidate in REASON_FOR_EXAM_COLS:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def brca_flags_from_reason(text) -> tuple[int, int]:
    """Return (brca1, brca2) flags parsed from ReasonforExam free text.

    Gene-specific mentions only (e.g. ``BRCA1+``, ``BRCA2 gene mutation positive``).
    Generic ``BRCA positive`` without a gene number sets neither flag.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return 0, 0
    s = str(text).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return 0, 0
    brca1 = 1 if _BRCA1_RE.search(s) else 0
    brca2 = 1 if _BRCA2_RE.search(s) else 0
    return brca1, brca2


def add_brca_gene_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add integer ``brca1`` / ``brca2`` columns from ReasonforExam when present."""
    df = df.copy()
    reason_col = _reason_for_exam_column(df)
    if reason_col is None:
        df["brca1"] = 0
        df["brca2"] = 0
        return df
    flags = df[reason_col].map(brca_flags_from_reason)
    df["brca1"] = [f[0] for f in flags]
    df["brca2"] = [f[1] for f in flags]
    return df


def _load_mapping(mapping_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(mapping_csv, dtype=str)
    if "PatientID" not in df.columns:
        raise KeyError(f"Mapping CSV missing PatientID: {mapping_csv}")
    if "lat" not in df.columns:
        raise KeyError(f"Mapping CSV missing lat: {mapping_csv}")

    df = df.copy()
    df["PatientID"] = df["PatientID"].astype(str).str.strip()
    df["lat"] = df["lat"].astype(str).str.strip()
    df["UID"] = df.apply(lambda r: _uid_from_accession_lat(r["PatientID"], r["lat"]), axis=1)

    keep = ["UID"] + [c for c in MAPPING_MERGE_COLS if c in df.columns]
    df = df[keep].drop_duplicates(subset=["UID"])
    rename = {c: f"mapping_{c}" for c in df.columns if c not in {"UID"}}
    return df.rename(columns=rename)


def _load_label_table(label_table_xlsx: Path) -> pd.DataFrame:
    """BI-RADS-4 workbook rows keyed by UID = <newaccession>_<lat>."""
    df = pd.read_excel(label_table_xlsx, dtype=str, keep_default_na=False)
    acc_col = "newaccession" if "newaccession" in df.columns else "dummy_acc"
    lat_col = "lat" if "lat" in df.columns else "laterality"
    if acc_col not in df.columns:
        raise KeyError(f"Label table missing accession column ({acc_col}): {label_table_xlsx}")
    if lat_col not in df.columns:
        raise KeyError(f"Label table missing laterality column ({lat_col}): {label_table_xlsx}")

    df = df.copy()
    df[acc_col] = df[acc_col].astype(str).str.strip()
    df[lat_col] = df[lat_col].astype(str).str.strip()
    df["UID"] = df.apply(lambda r: _uid_from_accession_lat(r[acc_col], r[lat_col]), axis=1)
    df = add_brca_gene_flags(df)

    missing = [c for c in LABEL_TABLE_MERGE_COLS if c not in df.columns]
    if missing:
        print(f"Warning: label table missing columns {missing} ({label_table_xlsx.name})")

    keep = ["UID"] + [c for c in LABEL_TABLE_MERGE_COLS if c in df.columns]
    return df[keep].drop_duplicates(subset=["UID"])


def _nonempty_label_values(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    return text.ne("") & text.str.lower().ne("nan")


def merge_label_table(
    predict_df: pd.DataFrame,
    label_table_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = predict_df.merge(label_table_df, on="UID", how="left")
    n = len(merged)
    for col in LABEL_TABLE_MERGE_COLS:
        if col not in merged.columns:
            continue
        if col in ("brca1", "brca2"):
            n_hit = int(pd.to_numeric(merged[col], errors="coerce").fillna(0).eq(1).sum())
        else:
            n_hit = int(_nonempty_label_values(merged[col]).sum())
        print(f"  with label {col}: {n_hit} ({n_hit / max(n, 1) * 100:.1f}%)")
    return merged


def _load_datasplit(datasplit_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(datasplit_csv, dtype=str)
    required = {"UID", "Fold", "dummy_acc", "Split", "Malignant"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Datasplit CSV missing columns {sorted(missing)}: {datasplit_csv}")

    df = df.copy()
    df["UID"] = df["UID"].astype(str).str.strip()
    df["Fold"] = df["Fold"].astype(int)
    keep = ["UID", "Fold"] + [c for c in SPLIT_MERGE_COLS if c in df.columns]
    df = df[keep].drop_duplicates(subset=["UID", "Fold"])
    rename = {}
    for col in df.columns:
        if col in {"UID", "Fold"}:
            continue
        rename[col] = "split_label" if col == "label" else f"split_{col}"
    return df.rename(columns=rename)


def _load_predict_parts(
    *,
    cohort: str | None,
    splits: list[str],
    folds: list[int],
    results_root: Path,
    train_runs_root: Path,
    predict_csvs: list[Path] | None,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []

    if predict_csvs:
        for csv_path in predict_csvs:
            df = pd.read_csv(csv_path)
            if "predict_split" not in df.columns:
                stem = csv_path.stem
                if stem.startswith("results_") and stem != "results":
                    df["predict_split"] = stem.removeprefix("results_")
                elif "pooled_" in stem and "_val" in stem:
                    df["predict_split"] = "val"
                elif "pooled_" in stem and "_test" in stem:
                    df["predict_split"] = "test"
            parts.append(df)
    else:
        if not cohort:
            raise ValueError("Provide --predict-csv and/or --cohort.")
        for split in splits:
            paths = find_fold_results(
                cohort=cohort,
                folds=folds,
                results_root=results_root,
                train_runs_root=train_runs_root,
                split=split,
            )
            pool = aggregate(paths, plot="none", cohort=cohort)
            pool = pool.copy()
            pool["predict_split"] = split
            parts.append(pool)

    if not parts:
        raise ValueError("No predict outputs loaded.")

    out = pd.concat(parts, ignore_index=True)
    out["UID"] = out["UID"].astype(str).str.strip()
    if "Fold" in out.columns:
        out["Fold"] = out["Fold"].astype(int)
    return out


def merge_predict_with_metadata(
    predict_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    datasplit_df: pd.DataFrame,
    *,
    deduped: bool = False,
) -> pd.DataFrame:
    merged = predict_df.merge(mapping_df, on="UID", how="left")
    if deduped or "Fold" not in merged.columns:
        ds = datasplit_df.drop_duplicates("UID")
        merged = merged.merge(ds.drop(columns=["Fold"], errors="ignore"), on="UID", how="left")
    else:
        merged = merged.merge(datasplit_df, on=["UID", "Fold"], how="left")

    n = len(merged)
    n_map = int(merged["mapping_bc_prior"].notna().sum()) if "mapping_bc_prior" in merged.columns else 0
    n_split = int(merged["split_Split"].notna().sum()) if "split_Split" in merged.columns else 0
    print(f"Merged rows: {n}")
    if "mapping_bc_prior" in merged.columns:
        print(f"  with mapping bc_prior: {n_map} ({n_map / max(n, 1) * 100:.1f}%)")
    if "split_Split" in merged.columns:
        print(f"  with datasplit Split:  {n_split} ({n_split / max(n, 1) * 100:.1f}%)")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge pooled predict CSVs with new_penn_mapping_v2 and datasplit metadata."
    )
    parser.add_argument(
        "--predict-csv",
        type=Path,
        action="append",
        default=None,
        help="Explicit pooled predict CSV(s). If omitted, use --cohort + --splits.",
    )
    parser.add_argument(
        "--cohort",
        type=str,
        default=None,
        help="Cohort tag in run folder names (used with --splits to discover per-fold results).",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["val"],
        choices=["train", "val", "test"],
        help="Predict splits to load when using --cohort (default: val).",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Fold indices when discovering results via --cohort.",
    )
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=DEFAULT_MAPPING_CSV,
        help=f"new_penn_mapping_v2 path (default: {DEFAULT_MAPPING_CSV}).",
    )
    parser.add_argument(
        "--datasplit-csv",
        type=Path,
        default=DEFAULT_DATASPLIT_CSV,
        help=f"new_penn_datasplit_v2 path (default: {DEFAULT_DATASPLIT_CSV}).",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=f"Predict output root (default: {DEFAULT_RESULTS_ROOT}).",
    )
    parser.add_argument(
        "--train-runs-root",
        type=Path,
        default=DEFAULT_TRAIN_RUNS_ROOT,
        help=f"Training runs root for --cohort discovery (default: {DEFAULT_TRAIN_RUNS_ROOT}).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Output enriched CSV path.",
    )
    parser.add_argument(
        "--dedupe-uid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Collapse duplicate UIDs before merge (default: true; no-op for test).",
    )
    parser.add_argument(
        "--dedupe-strategy",
        choices=["first", "mean"],
        default="first",
        help="How to combine duplicate UID rows (default: first).",
    )
    args = parser.parse_args()

    cohort = args.cohort.strip() if args.cohort else None
    if args.out_csv is not None:
        out_csv = args.out_csv
    elif cohort:
        split_tag = "_".join(args.splits)
        out_csv = (
            PROJECT_ROOT
            / "results-pretrained-oldpenn-on-newpenn"
            / f"enriched_{cohort}_{split_tag}.csv"
        )
    else:
        out_csv = PROJECT_ROOT / "results-pretrained-oldpenn-on-newpenn" / "enriched_predict.csv"

    predict_df = _load_predict_parts(
        cohort=cohort,
        splits=args.splits,
        folds=args.folds,
        results_root=args.results_root,
        train_runs_root=args.train_runs_root,
        predict_csvs=args.predict_csv,
    )
    deduped = False
    if args.dedupe_uid:
        if "predict_split" in predict_df.columns:
            parts = []
            for split_name, group in predict_df.groupby("predict_split", sort=False):
                print(f"Deduping predict_split={split_name!r}:")
                parts.append(dedupe_uid_predictions(group, strategy=args.dedupe_strategy))
            predict_df = pd.concat(parts, ignore_index=True)
        else:
            print("Deduping pooled predictions:")
            predict_df = dedupe_uid_predictions(predict_df, strategy=args.dedupe_strategy)
        deduped = True

    mapping_df = _load_mapping(args.mapping_csv)
    datasplit_df = _load_datasplit(args.datasplit_csv)

    merged = merge_predict_with_metadata(
        predict_df, mapping_df, datasplit_df, deduped=deduped
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    print(f"Wrote enriched table to {out_csv}")


if __name__ == "__main__":
    main()
