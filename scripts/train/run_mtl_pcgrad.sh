#!/bin/bash
#SBATCH --job-name=mtl_pcgrad
#SBATCH --account=llmalignment
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --output=mtl_pcgrad_%j.out
#SBATCH --error=mtl_pcgrad_%j.err

# =============================================================================
# MTL PCGrad Experiment (MTL Enabled)
# =============================================================================
# This runs multi-task learning with PCGrad gradient balancing

set -euo pipefail

echo "🚀 Starting MTL PCGrad Experiment..."
echo "📅 Started at: $(date)"
echo "🔧 MTL: Enabled with PCGrad"
echo "🎯 Ray Port: 6380 (custom to avoid conflicts)"

# Activate virtual environment
source ~/venv_reasoning360mtl/bin/activate

# Set Python path
export PYTHONPATH=/home/jinming/Reasoning360-MTL:${PYTHONPATH:-}

# Navigate to project directory
cd /home/jinming/Reasoning360-MTL/scripts/train

# Configure experiment
export MTL_ENABLED=true
export MTL_METHOD=pcgrad
export WANDB_EXPERIMENT_NAME="mtl-pcgrad-$(date +%Y%m%d-%H%M%S)"

# Use custom Ray port to avoid conflicts with baseline experiment
export RAY_PORT=6380

# Run the experiment
echo "▶️  Launching MTL experiment..."
bash example_multinode_mtl_qwen2_7b.sh

echo "✅ MTL experiment completed at: $(date)"
