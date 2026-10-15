# AtmoFuseNet Multi-Variant Ablation Runner (PowerShell)
# Runs all 4 filter variants: diff, adv, diff_adv, unkP

$ErrorActionPreference = "Stop"

Write-Host "=========================================================="
Write-Host "       AtmoFuseNet Ablation Experiment Suite             "
Write-Host "=========================================================="

# 1. Check dataset prerequisites
$requiredFiles = @("data/station.csv", "data/train.npz", "data/val.npz", "data/test.npz")
foreach ($f in $requiredFiles) {
    if (-not (Test-Path $f)) {
        Write-Host "ERROR: Missing required data file: $f" -ForegroundColor Red
        Write-Host "Please ensure all data files exist before starting full training."
        exit 1
    }
}
Write-Host "All required data files verified in data/" -ForegroundColor Green

# 2. Variants to evaluate
$variants = @("diff", "adv", "diff_adv", "unkP")
$expRepeat = 5

New-Item -ItemType Directory -Force -Path "logs/ablations" | Out-Null

foreach ($v in $variants) {
    Write-Host "`n>>> Running ablation variant: $v with exp_repeat = $expRepeat" -ForegroundColor Cyan
    
    # Create temporary config for variant
    $configContent = Get-Content "config.yaml" -Raw
    $tempConfig = "config_$v.yaml"
    $configContent -replace "filter_type:\s*\w+", "filter_type: $v" | Set-Content $tempConfig
    
    # Run experiment
    python main.py --config_filename $tempConfig --exp_repeat $expRepeat
    
    # Save resulting text summary
    if (Test-Path "logs/atmofusenet_results.txt") {
        $dest = "logs/ablations/atmofusenet_results_$v.txt"
        Copy-Item "logs/atmofusenet_results.txt" $dest -Force
        Write-Host "Saved results to $dest" -ForegroundColor Green
    }
    
    Remove-Item $tempConfig -Force -ErrorAction SilentlyContinue
}

Write-Host "`nAll 4 ablation runs complete! Results collected in logs/ablations/" -ForegroundColor Green
