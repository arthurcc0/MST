"""Build a manifest CSV for the new PENN data layout.

Pipeline order (mirrors scripts/preprocessing/penn but adapted for .npz inputs):

    1. new_penn_mapping.py        <- this file
    2. step1_npz2nifti.py
    3. step2a_calc_sub.py
    4. step2b_apply_mask_split.py
    5. step3_create_split.py

This script is shared by the current BIRADS-4 workbook and legacy Penn EHR-style
tables. Switch behavior with the **table profile** constants below (column names,
label rules, blank-laterality handling) rather than maintaining a second script,
unless the two pipelines diverge in more than column semantics.

Inputs:
    - VOL_DIR: flat folder of ``<accession>.npz`` files.
    - LABEL_TABLE: spreadsheet with accession key, laterality, and label fields.

Label resolution (``LABEL_SOURCE``):
    - ``precomputed``: read canonical ``label`` from the table (benign /
      malignant / high risk); used for old Penn after ``lat_added_dummy_ehr_chat.csv``
      is built in table_utils.
    - ``outcomes_then_legacy``: FP/TN/FN/TP from outcome columns first, then
      substring rules on free-text ``label`` (BIRADS-4 / legacy exports).

Laterality:
    - Numeric 1=R, 2=L; 0/3 → random side; 4/5 → excluded entirely.
    - ``BLANK_LATERALITY_MODE``: ``holdout`` (reserved for BI-RADS-4 style) vs
      ``random`` (e.g. screening negatives with no lateral finding in EHR exports).

Old Penn exclusions (when ``DISCARD_BENIGN_STUDY_WITH_BC_PRIOR``):
    - Drop rows whose ``studylevelassessment`` contains ``2: Benign`` or
      ``3: Probably Benign`` and ``bc_prior`` equals ``1`` (prior breast cancer).

Outputs always include canonical columns ``lat`` and ``label`` so
``step3_create_split.py`` (which expects ``lat``) stays aligned.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---- Configurable paths -----------------------------------------------------
VOL_DIR = Path(r"\\10.156.155.77\mccarthy_lab\shared\mri_preproc\n4bc")
OUT_ROOT = Path(r"D:\Users\arthur\Data\MST_birads4")

# === Table profile ============================================================
# --- New Penn (BIRADS-4 workbook example) ---
# LABEL_TABLE = Path(r"D:\Users\arthur\Projects\MST\tables\matches_birads4_all.xlsx")
# OUTPUT_CSV = OUT_ROOT / "new_penn_mapping.csv"
# HOLDOUT_CSV = OUT_ROOT / "new_penn_holdout.csv"
# COL_NEWACC = "newaccession"
# COL_LATERALITY = "lat"
# COL_LABEL = "label"
# COL_PATIENT = "PennChart_EpicPatientId"
# LABEL_SOURCE = "outcomes_then_legacy"
# USE_COMPONENT_LEVEL_OUTCOME = False
# USE_STUDY_LEVEL_FALLBACK = False
# BLANK_LATERALITY_MODE = "holdout"
# DISCARD_AMBIGUOUS_LEGACY_WITHOUT_OUTCOME = False

# --- Old Penn (EHR / dummy export example) ---
LABEL_TABLE = Path(r"D:\Users\arthur\Projects\MST\table_utils\lat_added_dummy_ehr_chat_no_birads4.csv")
OUTPUT_CSV = OUT_ROOT / "old_penn_mapping.csv"
HOLDOUT_CSV = OUT_ROOT / "old_penn_holdout.csv"
COL_NEWACC = "dummy_acc"
COL_LATERALITY = "laterality"
COL_LABEL = "label"
COL_PATIENT = "PennChart_EpicPatientId"

# Trust canonical ``label`` from lat_added_dummy_ehr_chat.csv (birads logic applied upstream).
LABEL_SOURCE = "precomputed"  # "precomputed" | "outcomes_then_legacy"

# Only used when LABEL_SOURCE == "outcomes_then_legacy".
COL_COMPONENT_LEVEL = "componentleveloutcome"
COL_STUDY_LEVEL = "studyleveloutcome"
COL_STUDY_LEVEL_ASSESSMENT = "studylevelassessment"
COL_BC_PRIOR = "bc_prior"

# Old Penn: drop study-level BI-RADS 2/3 text when patient has prior breast cancer (bc_prior=1).
DISCARD_BENIGN_STUDY_WITH_BC_PRIOR = True
_BENIGN_STUDY_ASSESSMENT_MARKERS = ("2: Benign", "3: Probably Benign")
USE_COMPONENT_LEVEL_OUTCOME = False
USE_STUDY_LEVEL_FALLBACK = False
DISCARD_AMBIGUOUS_LEGACY_WITHOUT_OUTCOME = False

# "holdout": missing lat → separate holdout CSV (BI-RADS-4 pipeline).
# "random": missing lat → randomize left/right (e.g. true negatives).
BLANK_LATERALITY_MODE = "random"  # "holdout" | "random"

# Canonical columns written for downstream scripts (especially step3).
STANDARD_LAT = "lat"
STANDARD_LABEL = "label"

ALLOWED_LABELS = frozenset({"benign", "malignant", "high risk"})

LAT_RANDOM_SEED = 42

# Debug: only keep the first case (set to False for full runs)
DEBUG_SINGLE = False


def _read_label_table(path: Path) -> pd.DataFrame:
    dtype_base = {COL_NEWACC: str}
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, dtype=dtype_base)
    else:
        df = pd.read_csv(path, dtype=dtype_base)
    df[COL_NEWACC] = df[COL_NEWACC].astype(str).str.strip()
    if COL_PATIENT in df.columns:
        df[COL_PATIENT] = df[COL_PATIENT].astype(str).str.strip()
    else:
        df[COL_PATIENT] = df[COL_NEWACC]
    return df


def _classify_numeric_code(code: int) -> str:
    """Return 'exclude', 'random', 'left', 'right', or 'holdout'."""
    if code in (4, 5):
        return "exclude"
    if code in (0, 3):
        return "random"
    if code == 1:
        return "right"
    if code == 2:
        return "left"
    if code == 9:
        return "holdout"
    return "holdout"


def _blank_lat_class() -> str:
    """How to treat null / empty laterality."""
    return "random" if BLANK_LATERALITY_MODE.strip().lower() == "random" else "holdout"


def _classify_laterality_cell(val) -> str:
    """Classify one label-table cell as exclude / random / left / right / holdout."""
    blank = _blank_lat_class()

    if val is None:
        return blank
    if isinstance(val, float) and np.isnan(val):
        return blank
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("", "null", "nan", "none"):
            return blank
        if s in ("left", "l"):
            return "left"
        if s in ("right", "r"):
            return "right"
        try:
            code = int(float(s))
        except ValueError:
            return "holdout"
        return _classify_numeric_code(code)
    if isinstance(val, (int, np.integer)):
        return _classify_numeric_code(int(val))
    if isinstance(val, (float, np.floating)):
        if np.isnan(val):
            return blank
        return _classify_numeric_code(int(val))
    return "holdout"


def _series_outcome_abbrev(series: pd.Series) -> pd.Series:
    """Map FP/TN/FN/TP strings to benign/malignant; else NA."""

    def _norm_one(v):
        if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
            return pd.NA
        tok = str(v).strip().upper()
        if tok in {"", "NAN", "NONE"}:
            return pd.NA
        if tok in ("FP", "TN"):
            return "benign"
        if tok in ("FN", "TP"):
            return "malignant"
        return pd.NA

    return series.map(_norm_one)


def _legacy_label_to_canonical(val) -> str | pd.NA:
    """Normalize free-text label column toward step3-compatible strings."""
    if val is None or (isinstance(val, float) and np.isnan(val)) or pd.isna(val):
        return pd.NA
    s = str(val).strip()
    if not s:
        return pd.NA
    low = s.lower().replace("-", " ")
    if low in ("malignant", "benign", "high risk"):
        return low

    benign_markers = (
        "probably benign",
        "benign ",
        " benig",
        "negative",
        "no evidence",
        ": benign",
        "birads 2",
        "birads 3",
    )
    malignant_markers = (
        "malignancy",
        "malignant",
        "biopsy proven",
        "biopsy proven malignancy",
        "known biopsy",
        "invasive ductal",
        " dcis",
        "carcinoma",
        "cancer",
        "highly suggestive",
        ": malignancy",
    )
    high_risk_markers = ("suspicious", "need additional", "birads 4", "birads 5")

    if any(k in low for k in malignant_markers):
        return "malignant"
    if any(k in low for k in high_risk_markers):
        return "high risk"
    if any(k in low for k in benign_markers):
        return "benign"
    # Short labels sometimes appear alone
    if low in {"benign", "negative"}:
        return "benign"

    return pd.NA


def _legacy_ambiguous_without_outcome(val) -> bool:
    """True if free-text label is BI-RADS-ambiguous without FP/TN/FN/TP outcomes."""
    if val is None or (isinstance(val, float) and np.isnan(val)) or pd.isna(val):
        return False
    low = str(val).strip().lower()
    if not low:
        return False
    if "probably benign" in low:
        return True
    if "need additional imaging evaluation" in low:
        return True
    if re.search(r"(?i)\bsuspicious\b", low):
        if re.search(
            r"(?i)(?:non|not)[-.\s]*suspicious|no\s+suspicious|without\s+suspicious",
            low,
        ):
            return False
        return True
    return False


def _normalize_precomputed_label(val) -> str | pd.NA:
    """Pass through canonical labels from the label table (strip + validate)."""
    if val is None or (isinstance(val, float) and np.isnan(val)) or pd.isna(val):
        return pd.NA
    norm = str(val).strip().lower().replace("-", " ")
    if norm in ALLOWED_LABELS:
        return norm
    return pd.NA


def _precomputed_standard_labels(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if COL_LABEL not in df.columns:
        raise ValueError(
            f"LABEL_SOURCE='precomputed' requires column {COL_LABEL!r} in {LABEL_TABLE}."
        )
    out = df[COL_LABEL].map(_normalize_precomputed_label).astype("string")
    discard = pd.Series(False, index=df.index)
    return out, discard


def _resolved_standard_labels(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Resolve labels; second return marks rows to drop (ambiguous legacy path only)."""
    if LABEL_SOURCE.strip().lower() == "precomputed":
        return _precomputed_standard_labels(df)
    if LABEL_SOURCE.strip().lower() != "outcomes_then_legacy":
        raise ValueError(
            f"Invalid LABEL_SOURCE={LABEL_SOURCE!r}; "
            "use 'precomputed' or 'outcomes_then_legacy'."
        )

    idx = df.index
    out = pd.Series(pd.NA, index=idx, dtype="string")

    if USE_COMPONENT_LEVEL_OUTCOME and COL_COMPONENT_LEVEL in df.columns:
        cmp_labels = _series_outcome_abbrev(df[COL_COMPONENT_LEVEL])
        out = cmp_labels.astype("string")
    if USE_STUDY_LEVEL_FALLBACK and COL_STUDY_LEVEL in df.columns:
        st_labels = _series_outcome_abbrev(df[COL_STUDY_LEVEL])
        out = out.fillna(st_labels)

    need_legacy = out.isna()
    discard = pd.Series(False, index=idx)
    if COL_LABEL in df.columns:
        ambig_raw = df[COL_LABEL].map(_legacy_ambiguous_without_outcome).fillna(False)
        if DISCARD_AMBIGUOUS_LEGACY_WITHOUT_OUTCOME:
            discard = need_legacy & ambig_raw
        leg = df[COL_LABEL].map(_legacy_label_to_canonical)
        leg = leg.where(~discard, pd.NA)
        out = out.fillna(leg.astype("string"))
    elif not USE_COMPONENT_LEVEL_OUTCOME and not USE_STUDY_LEVEL_FALLBACK:
        raise ValueError("No label source enabled and no COL_LABEL column present.")

    return out, discard


def _bc_prior_equals_one(val) -> bool:
    if val is None or (isinstance(val, float) and np.isnan(val)) or pd.isna(val):
        return False
    return str(val).strip() in ("1", "1.0")


def _mask_discard_benign_study_with_bc_prior(df: pd.DataFrame) -> pd.Series:
    """True for rows to drop: study assessment is BI-RADS 2/3 benign and bc_prior is 1."""
    if not DISCARD_BENIGN_STUDY_WITH_BC_PRIOR:
        return pd.Series(False, index=df.index)
    if COL_STUDY_LEVEL_ASSESSMENT not in df.columns or COL_BC_PRIOR not in df.columns:
        return pd.Series(False, index=df.index)

    sla = df[COL_STUDY_LEVEL_ASSESSMENT].fillna("").astype(str)
    benign_study = pd.Series(False, index=df.index)
    for marker in _BENIGN_STUDY_ASSESSMENT_MARKERS:
        benign_study |= sla.str.contains(marker, regex=False)
    prior_one = df[COL_BC_PRIOR].map(_bc_prior_equals_one)
    return benign_study & prior_one


def create_mapping() -> None:
    if not VOL_DIR.is_dir():
        raise FileNotFoundError(f"Volume directory not found: {VOL_DIR}")
    if not LABEL_TABLE.is_file():
        raise FileNotFoundError(f"Label table not found: {LABEL_TABLE}")

    df_labels = _read_label_table(LABEL_TABLE)
    extra = [
        COL_COMPONENT_LEVEL,
        COL_STUDY_LEVEL,
        COL_STUDY_LEVEL_ASSESSMENT,
        COL_BC_PRIOR,
    ]
    keep_candidates = [COL_NEWACC, COL_LATERALITY, COL_LABEL, COL_PATIENT]
    keep_cols = [c for c in keep_candidates if c in df_labels.columns]
    keep_cols += [c for c in extra if c in df_labels.columns]
    df_labels = df_labels[keep_cols].drop_duplicates(subset=[COL_NEWACC])

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

    if COL_PATIENT not in df_merged.columns:
        df_merged[COL_PATIENT] = df_merged["PatientID"]
    df_merged[COL_PATIENT] = df_merged[COL_PATIENT].fillna(df_merged["PatientID"]).astype(str).str.strip()

    print(f"Label source: {LABEL_SOURCE!r} (table: {LABEL_TABLE.name})")
    df_merged[STANDARD_LABEL], discard_ambig = _resolved_standard_labels(df_merged)
    discard_benign_prior = _mask_discard_benign_study_with_bc_prior(df_merged)
    discard = discard_ambig | discard_benign_prior
    n_discard_ambig = int(discard_ambig.sum())
    n_discard_benign_prior = int(discard_benign_prior.sum())
    if n_discard_ambig:
        print(
            f"Discarding {n_discard_ambig} rows (ambiguous legacy label, no "
            f"{COL_COMPONENT_LEVEL} / {COL_STUDY_LEVEL} outcome)."
        )
    if n_discard_benign_prior:
        print(
            f"Discarding {n_discard_benign_prior} rows "
            f"({COL_STUDY_LEVEL_ASSESSMENT} contains BI-RADS 2/3 benign text and "
            f"{COL_BC_PRIOR}=1)."
        )
    df_merged = df_merged.loc[~discard].copy()

    n_unresolved = int(df_merged[STANDARD_LABEL].isna().sum())
    if n_unresolved:
        if LABEL_SOURCE.strip().lower() == "precomputed":
            print(
                f"Dropping {n_unresolved} rows with missing or invalid precomputed "
                f"{COL_LABEL} (expected one of {sorted(ALLOWED_LABELS)})."
            )
            df_merged = df_merged[df_merged[STANDARD_LABEL].notna()].copy()
        else:
            print(
                f"WARNING: {n_unresolved} rows have unresolved '{STANDARD_LABEL}' "
                "after outcome + legacy rules."
            )

    print(f"Label table rows: {len(df_labels)}; .npz on disk for those: {len(records)}; missing on disk: {n_missing_on_disk}")

    if COL_LATERALITY not in df_merged.columns:
        df_merged[COL_LATERALITY] = np.nan

    lat_cls = df_merged[COL_LATERALITY].map(_classify_laterality_cell)
    exclude_mask = lat_cls == "exclude"
    n_excluded = int(exclude_mask.sum())
    df_merged = df_merged.loc[~exclude_mask].copy()
    lat_cls = lat_cls.loc[~exclude_mask]

    holdout_mask = lat_cls == "holdout"
    df_holdout = df_merged.loc[holdout_mask].copy()
    df_main = df_merged.loc[~holdout_mask].copy()
    cls_main = lat_cls.loc[~holdout_mask]

    rng = np.random.default_rng(LAT_RANDOM_SEED)
    rnd_mask = cls_main == "random"
    n_random_lat = int(rnd_mask.sum())
    sides = cls_main.astype(str).copy()
    sides.loc[rnd_mask] = rng.choice(np.array(["left", "right"], dtype=str), size=n_random_lat)

    df_main = df_main.copy()
    if COL_LATERALITY != STANDARD_LAT and COL_LATERALITY in df_main.columns:
        df_main = df_main.drop(columns=[COL_LATERALITY])
    df_holdout = df_holdout.copy()
    if COL_LATERALITY != STANDARD_LAT and COL_LATERALITY in df_holdout.columns:
        df_holdout = df_holdout.drop(columns=[COL_LATERALITY])

    df_main[STANDARD_LAT] = sides.values

    # Single canonical label column on disk for step3 / training
    for frame in (df_main, df_holdout):
        if COL_LABEL != STANDARD_LABEL and COL_LABEL in frame.columns:
            frame.drop(columns=[COL_LABEL], inplace=True, errors="ignore")

    if DEBUG_SINGLE:
        df_main = df_main.head(1)
        print(f"[DEBUG_SINGLE] keeping only the first case: {df_main['PatientID'].tolist()}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_main.to_csv(OUTPUT_CSV, index=False)
    if len(df_holdout):
        df_holdout.to_csv(HOLDOUT_CSV, index=False)

    n_total_kept = len(df_merged)
    n_main = len(df_main)
    n_holdout = len(df_holdout)
    n_with_label = df_main[STANDARD_LABEL].notna().sum()
    print(f"Mapping written to {OUTPUT_CSV}")
    if n_discard_ambig:
        print(f"  discarded (ambiguous legacy, no outcome): {n_discard_ambig}")
    if n_discard_benign_prior:
        print(f"  discarded (BI-RADS 2/3 study + bc_prior=1): {n_discard_benign_prior}")
    print(f"  post-exclude cases (still in label+disk intersection): {n_total_kept}")
    print(f"  excluded (lat codes 4/5):          {n_excluded}")
    print(f"  kept (assigned left/right):          {n_main}  "
          f"(random lat from codes 0/3 / blank-as-random: {n_random_lat})")
    print(f"  reserved for test (holdout lat):      {n_holdout}  -> {HOLDOUT_CSV}")
    print(f"  in main mapping with resolved label:  {int(n_with_label)}")
    print(f"  in main mapping with missing label:   {int(n_main - n_with_label)}")


if __name__ == "__main__":
    create_mapping()
