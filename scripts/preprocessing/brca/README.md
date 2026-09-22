# Basser BRCA cohort preprocessing

Data root: `D:\Users\arthur\Data\Basser_MRI`

Label table: `basser_mri_classification_include_after_diagnosis.xlsx`

| Column | Use |
|--------|-----|
| `dummyaccession` | Accession / `PatientID` for npz + NIfTI |
| `uniqueid` | Patient ID for CV grouping (no leakage across exams) |
| `bc_overall` | Binary label (0=benign, 1=malignant) |
| `mridate` | MRI date (used for timing-based label rule) |
| `bc_overall_firstdate` | First breast-cancer date (used for timing rule) |
| `breasts_to_consider` | Optional: `1`=right only, `2`=left only, blank=both |
| `split` | Provided train/val/test assignment (used when `USE_PROVIDED_SPLIT=True`) |

**Provided split** (`config.py`): set `USE_PROVIDED_SPLIT = True` to keep the
label-table `split` column instead of k-fold CV. Rows with `split=excluded` are
dropped. Output uses `Fold=0` for all rows — train with `--fold 0`. Use
`USE_PROVIDED_SPLIT = False` (or `--kfold-split`) for patient-level
`StratifiedGroupKFold` via `N_FOLDS`.

**Label timing rule** (`config.py`): when `bc_overall=1`, rows are malignant
only if `mridate` is on or before `bc_overall_firstdate` and at most
`MAX_YEARS_MRI_BEFORE_CANCER` years earlier (default 1.0). Otherwise they are
downgraded to benign. Toggle with `ENABLE_MRI_TIMING_LABEL_RULE` or
`--disable-mri-timing-label`; override the window with
`--max-years-before-cancer`.

Unlike BI-RADS-4, **both breasts** are kept as separate training samples
(`<accession>_left`, `<accession>_right`) when both are included. Predict
scores can be pooled back to accession level with `aggregate_accession_scores.py`.

## Pipeline order

```powershell
cd D:\Users\arthur\Projects\MST

# 0. Exclude accessions used in old-Penn pretrain split
python .\scripts\preprocessing\brca\step0_filter_old_penn_overlap.py

# 1. Mapping manifest (one row per accession)
python .\scripts\preprocessing\brca\brca_mapping.py

# 2–4. Reuse new_penn image ops with BRCA paths (thin wrappers)
python .\scripts\preprocessing\brca\run_step1_npz2nifti.py
python .\scripts\preprocessing\brca\run_step2a_calc_sub.py
# IMPORTANT: use the BRCA wrapper (not new_penn/step2b directly).
# Slabs -> OUT_ROOT/final_cropped_and_masked_slabs_n32_s3_o0/
python .\scripts\preprocessing\brca\run_step2b_apply_mask_split.py

# 5. Bilateral datasplit (provided split or k-fold; see config.py)
python .\scripts\preprocessing\brca\step3_create_split.py
```

## Outputs

| File | Description |
|------|-------------|
| `basser_mri_classification_filtered_no_oldpenn.csv` | Label table after step0 |
| `basser_excluded_old_penn_overlap.csv` | Audit of excluded accessions |
| `brca_mapping.csv` | Accession manifest for step1 |
| `preprocessed/data/<accession>/` | pre, post, sub NIfTIs |
| `final_cropped_and_masked_slabs_n32_s3_o0/` | `<accession>_left`, `_right` slabs (step2b) |
| `brca_datasplit.csv` | CV split for `PENN_Dataset3D` |

## Training / predict

```powershell
python .\scripts\train\main_train.py `
  --dataset PENN `
  --fold 0 `
  --penn_split_csv D:\Users\arthur\Data\Basser_MRI\brca_datasplit.csv `
  --path_root_data D:\Users\arthur\Data\Basser_MRI\final_cropped_and_masked_slabs_n32_s3_o0 `
  --ckpt_path <pretrained_ckpt> `
  --cohort "cohort" `
  ...

python .\scripts\predict\main_predict.py `
  --penn_split_csv D:\Users\arthur\Data\Basser_MRI\brca_datasplit.csv `
  --path_root_data D:\Users\arthur\Data\Basser_MRI\final_cropped_and_masked_slabs_n32_s3_o0 `
  ...
```

## Accession-level scores after predict

```powershell
python .\scripts\preprocessing\brca\aggregate_accession_scores.py `
  --predict-csv <path_to_results.csv> `
  --output-csv D:\Users\arthur\Data\Basser_MRI\brca_accession_scores.csv `
  --method mean
```

Shared npz source and breast masks are the same as new Penn (`config.py`).
