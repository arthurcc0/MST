# Run main_predict.py on the validation split for all CV folds (one fine-tuned cohort).
#
# Writes results_val.csv per fold (does not overwrite results.csv from test predict).
#
# Usage (PowerShell, from repo root):
#   .\scripts\run_new_penn_val_predict.ps1
#
# Optional env:
#   $env:PREDICT_COHORT = "birads4_pennPretrained_seed123_frozebckbn_headonly_notBalancedCE_v4_lr1x10-6"
#   $env:DATA_ROOT = "D:\Users\arthur\Data\MST_birads4\final_cropped_and_masked_slabs_n32_s3_o0"
#   $env:RESULTS_ROOT = ".\results-pretrained-oldpenn-on-newpenn\runs\PENN"

$ErrorActionPreference = "Stop"

$DATA = if ($env:DATA_ROOT) { $env:DATA_ROOT } else {
    "D:\Users\arthur\Data\MST_birads4\final_cropped_and_masked_slabs_n32_s3_o0"
}
$COHORT = if ($env:PREDICT_COHORT) { $env:PREDICT_COHORT } else {
    "birads4_pennPretrained123_frozebckbn_headonly_notBalancedCE_v4_lr1x10-6"
}
$SPLIT_CSV = "new_penn_datasplit_v2.csv"
$RUNS_DIR = ".\runs\PENN"
$FOLDS = 0..4

function Resolve-RunFolders {
    param([string]$Cohort)
    $runs = @()
    foreach ($f in $FOLDS) {
        $match = Get-ChildItem $RUNS_DIR -Directory |
            Where-Object { $_.Name -like "*${Cohort}_f${f}_reg*" } |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
        if (-not $match) {
            throw "No run folder found for cohort '$Cohort' fold $f (pattern '*${Cohort}_f${f}_reg*' under $RUNS_DIR)"
        }
        $runs += "PENN/$($match.Name)"
        Write-Host "  fold $f -> $($match.Name)" -ForegroundColor DarkGray
    }
    return $runs
}

Write-Host ""
Write-Host "========== VAL PREDICT: $COHORT ==========" -ForegroundColor Cyan

$RUNS = Resolve-RunFolders -Cohort $COHORT

foreach ($f in $FOLDS) {
    Write-Host ""
    Write-Host "--- predict val fold $f ---" -ForegroundColor Yellow

    & python .\scripts\main_predict.py `
        --run_dir .\runs `
        --run_folder $RUNS[$f] `
        --penn_split_csv $SPLIT_CSV `
        --path_root_data $DATA `
        --fold $f `
        --split val `
        --use_registers

    if ($LASTEXITCODE -ne 0) {
        throw "Val predict failed: cohort=$COHORT fold=$f (exit $LASTEXITCODE)"
    }
}

Write-Host ""
Write-Host "Val predict complete. Next steps:" -ForegroundColor Green
Write-Host @"
  # Pooled val ROC + save merged predictions
  python .\scripts\aggregate_auc.py `
    --cohort $COHORT `
    --split val `
    --dedupe-uid `
    --results-root .\results-pretrained-oldpenn-on-newpenn\runs\PENN `
    --plot both `
    --plot-out .\results-pretrained-oldpenn-on-newpenn\roc_${COHORT}_val.png `
    --save-pooled .\results-pretrained-oldpenn-on-newpenn\pooled_${COHORT}_val_deduped.csv

  # Enrich with mapping (bc_prior, label, studylevelassessment, ...); dedupes by default
  python .\scripts\merge_new_penn_predict_results.py `
    --cohort $COHORT `
    --splits val

  # ROC-based suspicion bands: calibrate on deduped val, apply to test
  python .\scripts\calibrate_suspicion_bands.py `
    --cohort $COHORT `
    --merge-metadata `
    --plot
"@
