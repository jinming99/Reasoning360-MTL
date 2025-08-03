#!/bin/bash

# =============================================================================
# Checkpoint Evolution Analysis Script
# =============================================================================
# This script evaluates multiple checkpoints from training to track performance
# evolution across domains and identify optimal stopping points.

set -euo pipefail

echo "🔍 Checkpoint Evolution Analysis"
echo "================================"

# Configuration
PROJECT_ROOT="/home/jinming/Reasoning360-MTL"
CHECKPOINT_BASE="$PROJECT_ROOT/checkpoints/Reasoning360-MTL"
EVAL_SCRIPT="$PROJECT_ROOT/scripts/train/example_multinode_eval_guru7b_mtl.sh"
RESULTS_BASE="$PROJECT_ROOT/evolution_analysis"

# Function to find all checkpoints for an experiment
find_checkpoints() {
    local experiment_pattern="$1"
    local experiment_dir=$(find "$CHECKPOINT_BASE" -name "$experiment_pattern" -type d | sort | tail -1)
    
    if [[ -z "$experiment_dir" ]]; then
        echo "❌ No experiment found matching: $experiment_pattern"
        return 1
    fi
    
    echo "📁 Found experiment: $experiment_dir"
    
    # Find all epoch checkpoints
    local checkpoints=($(find "$experiment_dir" -name "epoch_*" -type d | sort -V))
    
    if [[ ${#checkpoints[@]} -eq 0 ]]; then
        echo "⚠️ No epoch checkpoints found, checking for final checkpoint..."
        if [[ -d "$experiment_dir" ]]; then
            checkpoints=("$experiment_dir")
        fi
    fi
    
    echo "🎯 Found ${#checkpoints[@]} checkpoints:"
    for ckpt in "${checkpoints[@]}"; do
        echo "   $(basename "$ckpt")"
    done
    
    echo "${checkpoints[@]}"
}

# Function to submit evaluation job for a checkpoint
submit_evaluation() {
    local checkpoint_path="$1"
    local model_name="$2"
    local epoch_name="$3"
    
    local job_name="eval-${model_name}-${epoch_name}"
    local output_dir="$RESULTS_BASE/$model_name/$epoch_name"
    
    mkdir -p "$output_dir"
    
    echo "🚀 Submitting evaluation for $checkpoint_path"
    
    local job_id=$(sbatch --job-name="$job_name" \
                          --account=llmalignment \
                          --partition=a100_normal_q \
                          --output="$output_dir/eval_%j.out" \
                          --error="$output_dir/eval_%j.err" \
                          --wrap="
export MODEL_PATH='$checkpoint_path'
export MODEL_NAME='${model_name}_${epoch_name}'
cd $PROJECT_ROOT/scripts/train
bash example_multinode_eval_guru7b_mtl.sh
" | awk '{print $4}')
    
    echo "   Job ID: $job_id"
    echo "$job_id" >> "$RESULTS_BASE/job_ids.txt"
}

# Main execution
echo "🔍 Searching for experiment checkpoints..."

# Create results directory
mkdir -p "$RESULTS_BASE"
echo "" > "$RESULTS_BASE/job_ids.txt"

# Find baseline checkpoints
echo ""
echo "📊 Processing Baseline Experiment..."
baseline_checkpoints=($(find_checkpoints "guru-baseline-*"))

if [[ ${#baseline_checkpoints[@]} -gt 0 ]]; then
    for ckpt in "${baseline_checkpoints[@]}"; do
        epoch_name=$(basename "$ckpt")
        submit_evaluation "$ckpt" "baseline" "$epoch_name"
    done
else
    echo "⚠️ No baseline checkpoints found"
fi

# Find MTL checkpoints  
echo ""
echo "📊 Processing MTL Experiment..."
mtl_checkpoints=($(find_checkpoints "mtl-pcgrad-*"))

if [[ ${#mtl_checkpoints[@]} -gt 0 ]]; then
    for ckpt in "${mtl_checkpoints[@]}"; do
        epoch_name=$(basename "$ckpt")
        submit_evaluation "$ckpt" "mtl" "$epoch_name"
    done
else
    echo "⚠️ No MTL checkpoints found"
fi

# Summary
total_jobs=$(wc -l < "$RESULTS_BASE/job_ids.txt")
echo ""
echo "✅ Evolution Analysis Submitted!"
echo "📈 Total evaluation jobs: $total_jobs"
echo "📁 Results directory: $RESULTS_BASE"
echo "📊 Monitor: squeue -u jinming"
echo ""
echo "⏱️ Expected completion: 2-4 hours"
echo "🔍 Track progress: tail -f $RESULTS_BASE/*/eval_*.out"
echo ""
echo "📈 This will provide:"
echo "   • Performance evolution across training epochs"
echo "   • Domain-specific learning curves"
echo "   • Optimal stopping point identification"
echo "   • Overfitting detection"
echo "   • Comparative analysis between MTL and baseline"
