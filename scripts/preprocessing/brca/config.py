"""Shared paths and column names for the Basser BRCA MRI cohort."""

from pathlib import Path

# ---- Data roots -----------------------------------------------------------
OUT_ROOT = Path(r"D:\Users\arthur\Data\Basser_MRI")
VOL_DIR = Path(r"\\10.156.155.77\mccarthy_lab\shared\mri_preproc\n4bc")
MASK_DIR = Path(r"\\10.156.155.77\mccarthy_lab\shared\mri_masks\breast")

# Old Penn pretrain split (accessions to exclude for leakage audit).
OLD_PENN_DATASPLIT = Path(r"D:\Users\arthur\Data\MST_birads4\old_penn_datasplit_v2.csv")

# ---- Label table ----------------------------------------------------------
LABEL_TABLE = OUT_ROOT / "basser_mri_classification_include_after_diagnosis.xlsx"
FILTERED_LABEL_TABLE = OUT_ROOT / "basser_mri_classification_filtered_no_oldpenn.csv"
EXCLUDED_OVERLAP_CSV = OUT_ROOT / "basser_excluded_old_penn_overlap.csv"

# ---- Pipeline artifacts ---------------------------------------------------
MAPPING_CSV = OUT_ROOT / "brca_mapping.csv"
DATASPLIT_CSV = OUT_ROOT / "brca_datasplit.csv"
PREPROCESSED_DATA = OUT_ROOT / "preprocessed" / "data"
FINAL_SLAB_DIR = OUT_ROOT / "final_cropped_and_masked_slabs_n32_s3_o0"

# ---- Table columns --------------------------------------------------------
COL_ACCESSION = "dummyaccession"
COL_PATIENT = "uniqueid"
COL_LABEL = "label"
COL_BREASTS = "breasts_to_consider"
COL_MRI_DATE = "mridate"
COL_BC_FIRST_DATE = "bc_overall_firstdate"
COL_SPLIT = "split"

# When bc_overall=1, downgrade to benign if MRI is more than this many years
# before bc_overall_firstdate (or if mridate is after bc_overall_firstdate).
# Set ENABLE_MRI_TIMING_LABEL_RULE = False to use raw bc_overall only.
ENABLE_MRI_TIMING_LABEL_RULE = False
MAX_YEARS_MRI_BEFORE_CANCER = 3.0 # Adjusted to exclude exams more than 3 years before diagnosis.

# Canonical names written to mapping / split (match PENN_Dataset3D expectations).
STANDARD_LABEL = "label"
STANDARD_LAT = "lat"

DEFAULT_DEPTH = 32

# Split strategy for step3:
#   USE_PROVIDED_SPLIT=True  -> keep train/val/test from label table ``split`` column
#   USE_PROVIDED_SPLIT=False -> StratifiedGroupKFold by ``uniqueid`` (``N_FOLDS`` folds)
USE_PROVIDED_SPLIT = True
PROVIDED_SPLIT_FOLD = 0
SPLIT_EXCLUDE_VALUES = frozenset({"excluded", "exclude"})

N_FOLDS = 5
RANDOM_STATE_OUTER = 0
RANDOM_STATE_INNER = 42
