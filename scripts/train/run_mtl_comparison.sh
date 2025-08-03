#!/bin/bash

# =============================================================================
# MTL vs Baseline Comparison Experiment
# =============================================================================
# This script runs both baseline and MTL experiments with early stopping
# for comprehensive comparison following the GURU paper methodology.

set -euo pipefail

echo "🚀 Starting MTL vs Baseline Comparison Experiment"
echo "================================================="
echo "📅 Start time: $(date)"
echo ""

# Configuration
PROJECT_ROOT="/home/jinming/Reasoning360-MTL"
SCRIPTS_DIR="$PROJECT_ROOT/scripts/train"

cd "$SCRIPTS_DIR"

# Submit baseline experiment
echo "📊 Submitting Baseline Experiment..."
BASELINE_JOB_ID=$(sbatch --parsable run_guru_baseline.sh)
echo "   Job ID: $BASELINE_JOB_ID"
echo "   Experiment: GURU Baseline (MTL disabled)"
echo "   Configuration: 30 epochs, early stopping after 10 epochs"
echo ""

# Submit MTL experiment  
echo "📊 Submitting MTL Experiment..."
MTL_JOB_ID=$(sbatch --parsable run_mtl_pcgrad.sh)
echo "   Job ID: $MTL_JOB_ID"
echo "   Experiment: MTL with PCGrad"
echo "   Configuration: 30 epochs, early stopping after 10 epochs"
echo ""

# Summary
echo "✅ Both experiments submitted successfully!"
echo "📈 Baseline Job ID: $BASELINE_JOB_ID"
echo "📈 MTL Job ID: $MTL_JOB_ID"
echo ""
echo "🔍 Monitor progress:"
echo "   squeue -u jinming"
echo "   tail -f guru_baseline_${BASELINE_JOB_ID}.out"
echo "   tail -f mtl_pcgrad_${MTL_JOB_ID}.out"
echo ""
echo "📊 WandB Dashboard:"
echo "   https://wandb.ai/jin-ming-vt/Reasoning360-MTL"
echo ""
echo "⏱️ Expected completion: 4-8 hours (with early stopping)"
echo ""
echo "📋 Key Features:"
echo "   • Early stopping: minimum 10 epochs, patience 3"
echo "   • Validation every 3 epochs"
echo "   • Checkpoints every 5 epochs"
echo "   • WandB logging enabled"
echo "   • Improved reward functions integrated"
echo ""
echo "🎯 After completion, run evaluation:"
echo "   bash run_evaluation_comparison.sh"
