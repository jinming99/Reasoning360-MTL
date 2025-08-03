#!/bin/bash

# =============================================================================
# Evaluation Comparison Script
# =============================================================================
# This script evaluates both baseline and MTL trained models on all domains
# and generates a comprehensive comparison report.

set -euo pipefail

echo "🔍 Evaluation Comparison for MTL vs Baseline"
echo "============================================="

# Configuration
PROJECT_ROOT="/home/jinming/Reasoning360-MTL"
CHECKPOINT_BASE="$PROJECT_ROOT/checkpoints/Reasoning360-MTL"
EVAL_SCRIPT="$PROJECT_ROOT/scripts/train/example_multinode_eval_guru7b_mtl.sh"
RESULTS_BASE="$PROJECT_ROOT/evaluation_results"

# Find the latest checkpoints
echo "📁 Finding latest checkpoints..."

BASELINE_CKPT=$(find "$CHECKPOINT_BASE" -name "guru-baseline-*" -type d | sort | tail -1)
MTL_CKPT=$(find "$CHECKPOINT_BASE" -name "mtl-pcgrad-*" -type d | sort | tail -1)

if [[ -z "$BASELINE_CKPT" ]]; then
    echo "❌ No baseline checkpoint found in $CHECKPOINT_BASE"
    echo "Available checkpoints:"
    ls -la "$CHECKPOINT_BASE" || echo "Checkpoint directory not found"
    exit 1
fi

if [[ -z "$MTL_CKPT" ]]; then
    echo "❌ No MTL checkpoint found in $CHECKPOINT_BASE"
    echo "Available checkpoints:"
    ls -la "$CHECKPOINT_BASE" || echo "Checkpoint directory not found"
    exit 1
fi

echo "✅ Found checkpoints:"
echo "   Baseline: $BASELINE_CKPT"
echo "   MTL:      $MTL_CKPT"

# Create results directories
mkdir -p "$RESULTS_BASE/baseline" "$RESULTS_BASE/mtl" "$RESULTS_BASE/comparison"

echo ""
echo "🚀 Starting Evaluation..."

# Evaluate Baseline Model
echo "📊 Evaluating Baseline Model..."
export MODEL_PATH="$BASELINE_CKPT"
export MODEL_NAME="baseline"
cd "$PROJECT_ROOT/scripts/train"

echo "Submitting baseline evaluation job..."
BASELINE_JOB=$(sbatch --job-name=eval-baseline \
                      --account=llmalignment \
                      --partition=a100_normal_q \
                      --output="$RESULTS_BASE/baseline/eval_%j.out" \
                      --error="$RESULTS_BASE/baseline/eval_%j.err" \
                      --wrap="
export MODEL_PATH='$BASELINE_CKPT'
export MODEL_NAME='baseline'
bash example_multinode_eval_guru7b_mtl.sh
" | awk '{print $4}')

echo "   Baseline evaluation job: $BASELINE_JOB"

# Evaluate MTL Model
echo "📊 Evaluating MTL Model..."
export MODEL_PATH="$MTL_CKPT"
export MODEL_NAME="mtl"

echo "Submitting MTL evaluation job..."
MTL_JOB=$(sbatch --job-name=eval-mtl \
                 --account=llmalignment \
                 --partition=a100_normal_q \
                 --output="$RESULTS_BASE/mtl/eval_%j.out" \
                 --error="$RESULTS_BASE/mtl/eval_%j.err" \
                 --wrap="
export MODEL_PATH='$MTL_CKPT'
export MODEL_NAME='mtl'
bash example_multinode_eval_guru7b_mtl.sh
" | awk '{print $4}')

echo "   MTL evaluation job: $MTL_JOB"

echo ""
echo "✅ Evaluation jobs submitted!"
echo "📈 Monitor: squeue -u jinming"
echo "📁 Results will be saved to: $RESULTS_BASE"
echo ""
echo "📊 Domain-specific results will include:"
echo "   • Math: aime, math"
echo "   • Codegen: humaneval, mbpp, livecodebench" 
echo "   • Logic: arcagi, zebra_puzzle"
echo "   • STEM: gpqa_diamond, supergpqa"
echo "   • Simulation: codeio, cruxeval-i, cruxeval-o"
echo "   • Table: finqa, hitab, multihier"
echo "   • OOD: livebench_*, ifeval"
echo ""
echo "⏱️ Expected completion: 2-4 hours per model"
echo "🔍 Check progress: tail -f $RESULTS_BASE/*/eval_*.out"
