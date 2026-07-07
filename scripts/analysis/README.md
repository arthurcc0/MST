# Analysis scripts

Post-training utilities (run after `main_predict.py`):

| Script | Purpose |
|--------|---------|
| `aggregate_auc.py` | Pool per-fold `results_*.csv`, CV AUC summary + ROC plots |
| `calibrate_suspicion_bands.py` | Fit suspicion bands on val, apply to test |
| `merge_new_penn_predict_results.py` | Enrich predict CSVs with mapping / label-table columns |
| `merge_datasplit_studylevelassessment.py` | Add `studylevelassessment` to Penn datasplit CSVs |

## Usage

From repo root:

```powershell
python .\scripts\analysis\aggregate_auc.py --cohort <tag> --plot both
python .\scripts\analysis\calibrate_suspicion_bands.py --cohort <tag> --merge-metadata --plot
python .\scripts\analysis\merge_new_penn_predict_results.py --cohort <tag> --splits val test
```

Update local `.ps1` / `terminal_scripts.txt` paths from `scripts\<name>.py` to `scripts\analysis\<name>.py`.
