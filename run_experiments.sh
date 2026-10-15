#!/usr/bin/env bash
# AtmoFuseNet Multi-Variant Ablation Runner (Bash)
set -e

echo "=========================================================="
echo "       AtmoFuseNet Ablation Experiment Suite             "
echo "=========================================================="

# Check dataset files
for f in data/station.csv data/train.npz data/val.npz data/test.npz; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing required data file: $f"
        exit 1
    fi
done

variants=("diff" "adv" "diff_adv" "unkP")
exp_repeat=5

mkdir -p logs/ablations

for v in "${variants[@]}"; do
    echo -e "\n>>> Running ablation variant: $v (exp_repeat = $exp_repeat)"
    temp_config="config_${v}.yaml"
    sed "s/filter_type:.*/filter_type: ${v}/" config.yaml > "$temp_config"
    
    python main.py --config_filename "$temp_config" --exp_repeat "$exp_repeat"
    
    if [ -f "logs/atmofusenet_results.txt" ]; then
        cp logs/atmofusenet_results.txt "logs/ablations/atmofusenet_results_${v}.txt"
        echo "Saved results to logs/ablations/atmofusenet_results_${v}.txt"
    fi
    rm -f "$temp_config"
done

echo -e "\nAll ablation runs completed successfully."
