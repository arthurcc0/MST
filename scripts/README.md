# Scripts layout

| Directory | Purpose |
|-----------|---------|
| `train/` | Model training (`main_train.py`) |
| `predict/` | Inference, ROC, attention (`main_predict.py`) |
| `analysis/` | Post-hoc AUC, suspicion bands, result merges |
| `preprocessing/` | Dataset-specific NIfTI / split pipelines |

## Common commands

```powershell
# Train
python .\scripts\train\main_train.py --dataset PENN --model_name DinoClassifierSliceV3 ...

# Predict
python .\scripts\predict\main_predict.py --run_folder PENN\<run_name> ...

# Aggregate CV AUC
python .\scripts\analysis\aggregate_auc.py --cohort <tag> --plot both
```

Local fold sweeps and machine-specific paths live in gitignored `*.ps1` files under `scripts/`.
