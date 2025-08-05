#!/bin/bash
#SBATCH --job-name=guru_baseline
#SBATCH --account=llmalignment
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=12:00:00
#SBATCH --output=guru_baseline_%j.out
#SBATCH --error=guru_baseline_%j.err

# =============================================================================
# GURU Baseline Experiment (MTL Disabled)
# =============================================================================
# This runs the exact GURU paper approach without multi-task learning

set -euo pipefail

echo "🚀 Starting GURU Baseline Experiment..."
echo "📅 Started at: $(date)"
echo "🔧 MTL: Disabled (GURU paper approach)"
echo "🎯 Ray Port: 6379 (default)"

# Activate virtual environment
source ~/venv_reasoning360mtl/bin/activate

# Set Python path
export PYTHONPATH=/home/jinming/Reasoning360-MTL:${PYTHONPATH:-}

# Navigate to project directory
cd /home/jinming/Reasoning360-MTL/scripts/train

# Configure experiment
export MTL_ENABLED=false
export WANDB_EXPERIMENT_NAME="guru-baseline-$(date +%Y%m%d-%H%M%S)"

# Run the experiment
echo "▶️  Launching baseline experiment..."
bash example_multinode_mtl_qwen2_7b.sh

echo "✅ Baseline experiment completed at: $(date)"
