# Scripts layout

| Directory | Purpose |
|-----------|---------|
| `train/` | Model training (`main_train.py`) |
| `predict/` | Inference, ROC, attention (`main_predict.py`) |
| `analysis/` | Post-hoc AUC, suspicion bands, result merges |
| `preprocessing/` | Dataset-specific NIfTI / split pipelines (`new_penn/`, `brca/`, …) |
| `debug/` | One-off debugging utilities (attention, fold hangs) |
| `segmentation/` | Generate, evaluate, reconstruct segmentations |
| `reports/` | Imaging findings report generation |

## Common commands

```powershell
# Train
python .\scripts\train\main_train.py --dataset PENN --model_name DinoClassifierSliceV3 ...

# Predict
python .\scripts\predict\main_predict.py --run_folder PENN\<run_name> ...

# Aggregate CV AUC
python .\scripts\analysis\aggregate_auc.py --cohort <tag> --plot both

# Reports
python .\scripts\reports\reporte_gen.py
```

Local fold sweeps and machine-specific paths live in gitignored `*.ps1` files under `scripts/`.

Penn BI-RADS-4 label-table scripts live with that dataset (`Jan2026_BreastMRI\label_scripts`), not in this repo.

Shared DICOM helpers live in `mst/utils/dicom_common.py` (formerly repo-root `common.py`).

Tests: `tests/test_transform.py`, `tests/test_h5.py`.
