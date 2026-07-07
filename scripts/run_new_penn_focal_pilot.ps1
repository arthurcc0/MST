# Focal loss fine-tune on new Penn (BI-RADS-4), all CV folds.
# Matches v5 unbalanced-CE setup: head-only, LR 1e-6, old-Penn seed-123 init, v3 split.
#
# Usage (PowerShell, from repo root):
#   .\scripts\run_new_penn_focal_pilot.ps1
#
# Optional:
#   $env:OLD_CKPT = ".\runs\PENN\...\epoch=XX.ckpt"
#   $env:DATA_ROOT = "D:\Users\arthur\Data\MST_birads4\final_cropped_and_masked_slabs_n32_s3_o0"
#   $env:FOCAL_ALPHA = "balanced"   # none | balanced | prevalence
#   $env:FOCAL_GAMMA = "2"

$ErrorActionPreference = "Stop"

$DATA = if ($env:DATA_ROOT) { $env:DATA_ROOT } else {
    "D:\Users\arthur\Data\MST_birads4\final_cropped_and_masked_slabs_n32_s3_o0"
}
$SPLIT_CSV = "new_penn_datasplit_v3.csv"
$LEARNING_RATE = 1e-6
$FOCAL_GAMMA = if ($env:FOCAL_GAMMA) { $env:FOCAL_GAMMA } else { "2" }
$FOCAL_ALPHA = if ($env:FOCAL_ALPHA) { $env:FOCAL_ALPHA } else { "balanced" }
$FOLDS = if ($env:FOLDS) { $env:FOLDS.Split(",") | ForEach-Object { [int]$_.Trim() } } else { 0..4 }
$BATCH_SIZE = 8
$NUM_WORKERS = 2

function Resolve-OldPennSeed123Ckpt {
    if ($env:OLD_CKPT) {
        if (-not (Test-Path $env:OLD_CKPT)) {
            throw "Checkpoint not found: $env:OLD_CKPT"
        }
        return (Resolve-Path $env:OLD_CKPT).Path
    }
    $match = Get-ChildItem ".\runs\PENN" -Directory |
        Where-Object { $_.Name -like "*old_penn_balancedCE_seed123_f0_reg*" } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $match) {
        throw "No old-Penn seed-123 fold-0 run under .\runs\PENN. Set `$env:OLD_CKPT."
    }
    $bestJson = Join-Path $match.FullName "best_checkpoint.json"
    if (-not (Test-Path $bestJson)) {
        throw "Missing best_checkpoint.json in $($match.Name)"
    }
    $best = Get-Content $bestJson -Raw | ConvertFrom-Json
    $ckpt = Join-Path $match.FullName $best.best_model_epoch
    if (-not (Test-Path $ckpt)) {
        throw "Checkpoint missing: $ckpt"
    }
    Write-Host "Init ckpt: $ckpt" -ForegroundColor DarkGray
    return (Resolve-Path $ckpt).Path
}

function Resolve-LatestRunFolder {
    param([string]$CohortFoldTag)
    $match = Get-ChildItem ".\runs\PENN" -Directory |
        Where-Object { $_.Name -like "*${CohortFoldTag}_reg*" } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $match) {
        throw "No run folder matching '*${CohortFoldTag}_reg*' under .\runs\PENN"
    }
    return "PENN/$($match.Name)"
}

$OLD_CKPT = Resolve-OldPennSeed123Ckpt
$COHORT_BASE = "birads4_focal_g${FOCAL_GAMMA}_${FOCAL_ALPHA}_headonly_lr1e-6_v5"

Write-Host ""
Write-Host "========== FOCAL PILOT: $COHORT_BASE ==========" -ForegroundColor Cyan
Write-Host "Folds: $($FOLDS -join ', ')  LR=$LEARNING_RATE  init=seed-123 old Penn" -ForegroundColor Cyan

foreach ($f in $FOLDS) {
    $COHORT = "${COHORT_BASE}_f$f"
    Write-Host ""
    Write-Host "--- train fold $f ---" -ForegroundColor Yellow

    & python .\scripts\main_train.py `
        --dataset PENN `
        --penn_split_csv $SPLIT_CSV `
        --path_root_data $DATA `
        --cohort $COHORT `
        --fold $f `
        --ckpt_path $OLD_CKPT `
        --freeze_backbone `
        --learning_rate $LEARNING_RATE `
        --loss focal `
        --focal-gamma $FOCAL_GAMMA `
        --focal-alpha $FOCAL_ALPHA `
        --class_weight none `
        --use_registers true `
        --batch_size $BATCH_SIZE `
        --num_workers $NUM_WORKERS

    if ($LASTEXITCODE -ne 0) {
        throw "Training failed: fold=$f (exit $LASTEXITCODE)"
    }

    $runFolder = Resolve-LatestRunFolder -CohortFoldTag $COHORT
    Write-Host "Predict fold $f test: $runFolder" -ForegroundColor Yellow

    & python .\scripts\main_predict.py `
        --run_dir .\runs `
        --run_folder $runFolder `
        --penn_split_csv $SPLIT_CSV `
        --path_root_data $DATA `
        --fold $f `
        --split test `
        --use_registers

    if ($LASTEXITCODE -ne 0) {
        throw "Predict failed: fold=$f (exit $LASTEXITCODE)"
    }
}

Write-Host ""
Write-Host "Focal run complete. Aggregate test AUC (compare to CE v3 ~0.67 pooled):" -ForegroundColor Green
Write-Host @"
  python .\scripts\aggregate_auc.py `
    --cohort $COHORT_BASE `
    --split test `
    --results-root .\results-pretrained-oldpenn-on-newpenn\runs\PENN `
    --plot both
"@
