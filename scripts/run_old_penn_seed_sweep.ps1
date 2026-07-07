# Old-Penn full fine-tune: 3 seeds x 5 CV folds on old_penn_datasplit_v2.csv.
#
# Usage (PowerShell, from repo root):
#   .\scripts\run_old_penn_seed_sweep.ps1
# Optional:
#   $env:DATA_ROOT = "D:\Users\arthur\Data\MST_birads4\final_cropped_and_masked_data_v2"
#   $env:LEARNING_RATE = "5e-7"   # omit env var to use model default (1e-6)

$ErrorActionPreference = "Stop"

$DATA = if ($env:DATA_ROOT) { $env:DATA_ROOT } else {
    "D:\Users\arthur\Data\MST_birads4\final_cropped_and_masked_slabs_n32_s3_o0"
}
$SPLIT_CSV = "old_penn_datasplit_v2.csv"
$SEEDS = @(123)
$FOLDS = 1..4
$BATCH_SIZE = 8
$NUM_WORKERS = 2

# Set $LEARNING_RATE env var to override; otherwise main_train.py uses default 1e-6.
$LR_ARGS = @()
if ($env:LEARNING_RATE) {
    $LR_ARGS = @("--learning_rate", $env:LEARNING_RATE)
}

function Get-CohortTag {
    param([int]$Seed)
    if ($env:LEARNING_RATE) {
        $lrTag = ($env:LEARNING_RATE -replace "\.", "p" -replace "e-", "em")
        return "old_penn_balancedCE_seed${Seed}_lr${lrTag}"
    }
    return "old_penn_balancedCE_seed${Seed}"
}

function Resolve-LatestRunFolder {
    param([string]$CohortFoldTag)
    $match = Get-ChildItem ".\runs\PENN" -Directory |
        Where-Object { $_.Name -like "*${CohortFoldTag}_reg*" } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $match) {
        throw "No run folder found matching '*${CohortFoldTag}_reg*' under .\runs\PENN"
    }
    if (-not (Test-Path (Join-Path $match.FullName "best_checkpoint.json"))) {
        throw "Missing best_checkpoint.json in $($match.Name); training may not have finished saving."
    }
    return "PENN/$($match.Name)"
}

foreach ($seed in $SEEDS) {
    $COHORT = Get-CohortTag -Seed $seed
    Write-Host ""
    Write-Host "========== seed=$seed  cohort=$COHORT ==========" -ForegroundColor Cyan

    foreach ($f in $FOLDS) {
        Write-Host ""
        Write-Host "--- fold $f ---" -ForegroundColor Yellow

        $args = @(
            ".\scripts\main_train.py",
            "--dataset", "PENN",
            "--penn_split_csv", $SPLIT_CSV,
            "--path_root_data", $DATA,
            "--cohort", "${COHORT}_f$f",
            "--fold", "$f",
            "--seed", "$seed",
            "--class_weight", "balanced",
            "--use_registers", "true",
            "--batch_size", "$BATCH_SIZE",
            "--num_workers", "$NUM_WORKERS"
        ) + $LR_ARGS

        & python @args
        if ($LASTEXITCODE -ne 0) {
            throw "Training failed: seed=$seed fold=$f (exit $LASTEXITCODE)"
        }

        $cohortFold = "${COHORT}_f$f"
        $runFolder = Resolve-LatestRunFolder -CohortFoldTag $cohortFold
        Write-Host "Predict fold $f test (best ckpt via best_checkpoint.json): $runFolder" -ForegroundColor Yellow

        & python .\scripts\main_predict.py `
            --run_dir .\runs `
            --run_folder $runFolder `
            --penn_split_csv $SPLIT_CSV `
            --path_root_data $DATA `
            --fold $f `
            --split test `
            --use_registers

        if ($LASTEXITCODE -ne 0) {
            throw "Predict failed: seed=$seed fold=$f (exit $LASTEXITCODE)"
        }
    }
}

# Write-Host ""
# Write-Host "Seed sweep complete. Compare fold-0 test AUC per seed, e.g.:" -ForegroundColor Green
# Write-Host '  python .\scripts\aggregate_auc.py --cohort old_penn_balancedCE_seed123 --folds 0 --plot both'
