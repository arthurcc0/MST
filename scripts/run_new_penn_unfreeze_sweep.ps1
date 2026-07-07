# New-Penn fine-tune sweep: --unfreeze_encoder_blocks x all CV folds.
# Uses balanced CE, frozen backbone (+ optional last-N ViT blocks unfrozen).
#
# Usage (from repo root, with mst env active) — run in PowerShell, NOT via python:
#   .\scripts\run_new_penn_unfreeze_sweep.ps1
# If execution policy blocks the script:
#   powershell -ExecutionPolicy Bypass -File .\scripts\run_new_penn_unfreeze_sweep.ps1
#
# Optional overrides:
#   $env:OLD_CKPT = ".\runs\PENN\...\epoch=38-step=14195.ckpt"
#   $env:DATA_ROOT = "D:\Users\arthur\Data\MST_birads4\final_cropped_and_masked_slabs_n32_s3_o0"

$ErrorActionPreference = "Stop"

$DATA = if ($env:DATA_ROOT) { $env:DATA_ROOT } else {
    "D:\Users\arthur\Data\MST_birads4\final_cropped_and_masked_slabs_n32_s3_o0"
}
$OLD_CKPT = if ($env:OLD_CKPT) { $env:OLD_CKPT } else {
    ".\runs\PENN\DinoClassifierSlice_2026_06_05_180418_subtraction_old_penn_from_dino_cropv4_reg_multi\epoch=38-step=14195.ckpt"
}

$SPLIT_CSV = "new_penn_datasplit_v2.csv"
$LEARNING_RATE = 5e-5
$FOLDS = 0..4
# 0 = head-only (freeze_backbone, no encoder blocks unfrozen)
$UNFREEZE_BLOCKS = @(0, 1, 2, 4)

if (-not (Test-Path $OLD_CKPT)) {
    throw "Checkpoint not found: $OLD_CKPT. Set `$env:OLD_CKPT to your old-Penn ckpt."
}

function Get-CohortTag {
    param([int]$UnfreezeBlocks)
    if ($UnfreezeBlocks -eq 0) {
        return "new_penn_headonly_balancedCE_lr5e-5"
    }
    return "new_penn_unfreeze${UnfreezeBlocks}_balancedCE_lr5e-5"
}

foreach ($n in $UNFREEZE_BLOCKS) {
    $COHORT = Get-CohortTag -UnfreezeBlocks $n
    Write-Host ""
    Write-Host "========== unfreeze_encoder_blocks=$n  cohort=$COHORT ==========" -ForegroundColor Cyan

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
            "--ckpt_path", $OLD_CKPT,
            "--freeze_backbone",
            "--learning_rate", "$LEARNING_RATE",
            "--class_weight", "balanced",
            "--use_registers", "true",
            "--batch_size", "8",
            "--num_workers", "2"
        )
        if ($n -gt 0) {
            $args += @("--unfreeze_encoder_blocks", "$n")
        }

        & python @args
        if ($LASTEXITCODE -ne 0) {
            throw "Training failed: unfreeze_encoder_blocks=$n fold=$f (exit $LASTEXITCODE)"
        }
    }
}

Write-Host ""
Write-Host "Sweep complete. Aggregate per stage, e.g.:" -ForegroundColor Green
Write-Host '  python .\scripts\aggregate_auc.py --cohort new_penn_unfreeze2_balancedCE_lr5e-5 --results-root .\results-duke-penn-training\runs\PENN --plot both'
