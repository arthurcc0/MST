# Run main_predict.py for multiple new-Penn cohorts x CV folds 0-4.
#
# Usage (PowerShell, from repo root):
#   .\scripts\run_new_penn_predict_sweep.ps1
# Optional:
#   $env:DATA_ROOT = "D:\Users\arthur\Data\MST_birads4\final_cropped_and_masked_slabs_n32_s3_o0"

$ErrorActionPreference = "Stop"

$DATA = if ($env:DATA_ROOT) { $env:DATA_ROOT } else {
    "D:\Users\arthur\Data\MST_birads4\final_cropped_and_masked_slabs_n32_s3_o0"
}
$SPLIT_CSV = "new_penn_datasplit_v2.csv"
$RUNS_DIR = ".\runs\PENN"
$FOLDS = 0..4

$COHORTS = @(
    "new_penn_from_dino_freezebckbone_headonly_notBalancedCE_v4"
    "new_penn_unfreeze2_lr5e-5"
    "new_penn_unfreeze4_lr5e-5"
    "new_penn_headonly_balancedCE_lr5e-5"
    "new_penn_unfreeze1_balancedCE_lr5e-5"
    "new_penn_unfreeze2_balancedCE_lr5e-5"
    "new_penn_unfreeze4_balancedCE_lr5e-5"
)

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

foreach ($COHORT in $COHORTS) {
    Write-Host ""
    Write-Host "========== COHORT: $COHORT ==========" -ForegroundColor Cyan

    $RUNS = Resolve-RunFolders -Cohort $COHORT

    foreach ($f in $FOLDS) {
        Write-Host ""
        Write-Host "--- predict fold $f ---" -ForegroundColor Yellow

        & python .\scripts\main_predict.py `
            --run_dir .\runs `
            --run_folder $RUNS[$f] `
            --penn_split_csv $SPLIT_CSV `
            --path_root_data $DATA `
            --fold $f `
            --split test `
            --use_registers

        if ($LASTEXITCODE -ne 0) {
            throw "Predict failed: cohort=$COHORT fold=$f (exit $LASTEXITCODE)"
        }
    }
}

Write-Host ""
Write-Host "Predict sweep complete. Aggregate per cohort, e.g.:" -ForegroundColor Green
Write-Host '  python .\scripts\aggregate_auc.py --cohort new_penn_unfreeze2_balancedCE_lr5e-5 --results-root .\results-pretrained-oldpenn-on-newpenn\runs\PENN --plot both'
