# Generate attention maps for all focal v5 folds (sequential — do not run in parallel on one GPU).
$ErrorActionPreference = "Stop"

$DATA = if ($env:DATA_ROOT) { $env:DATA_ROOT } else {
    "D:\Users\arthur\Data\MST_birads4\final_cropped_and_masked_slabs_n32_s3_o0"
}
$SPLIT = "new_penn_datasplit_v3.csv"
$COHORT_BASE = if ($env:COHORT_BASE) { $env:COHORT_BASE } else {
    "birads4_focal_g2_balanced_headonly_lr1e-6_v5"
}

foreach ($f in 0..4) {
    $tag = "${COHORT_BASE}_f$f"
    $match = Get-ChildItem ".\runs\PENN" -Directory |
        Where-Object { $_.Name -like "*${tag}_reg*" } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $match) {
        throw "No run folder for fold $f matching '*${tag}_reg*'"
    }
    $runFolder = "PENN/$($match.Name)"
    Write-Host ""
    Write-Host "========== fold $f : $runFolder ==========" -ForegroundColor Cyan
    Write-Host "First case may take 1-3 min before progress moves from 0%." -ForegroundColor DarkGray

    & python .\scripts\main_predict.py `
        --run_dir .\runs `
        --run_folder $runFolder `
        --penn_split_csv $SPLIT `
        --path_root_data $DATA `
        --fold $f `
        --split test `
        --use_registers true `
        --get_attention

    if ($LASTEXITCODE -ne 0) {
        throw "main_predict failed for fold $f (exit $LASTEXITCODE)"
    }

    $outDir = ".\results-pretrained-oldpenn-on-newpenn\PENN\${tag}_reg\attention-rotated"
    $n = (Get-ChildItem $outDir -Filter "*_overlay.png" -ErrorAction SilentlyContinue | Measure-Object).Count
    Write-Host "fold $f done: $n overlay PNGs in $outDir" -ForegroundColor Green
}
