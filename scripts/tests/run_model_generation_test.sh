#!/bin/bash
#SBATCH --account=llmalignment
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --job-name=model_gen_reward_test
#SBATCH --output=model_gen_test_%j.out
#SBATCH --error=model_gen_test_%j.err

set -euo pipefail

echo "🚀 Starting Model Generation + Reward Testing"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi -L)"
echo "Time: $(date)"

# Activate virtual environment
echo "📦 Activating virtual environment..."
source ~/venv_reasoning360mtl/bin/activate

# Change to test directory
cd /home/jinming/Reasoning360-MTL/scripts/tests

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Run the comprehensive test
echo "🧪 Running model generation + reward computation test..."
echo "This will test the complete RL pipeline: prompt -> generation -> reward"

python3 test_model_generation_rewards.py

echo "✅ Model generation + reward testing completed!"
echo "📄 Check results in /home/jinming/Reasoning360-MTL/scripts/tests/model_generation_results.json"
echo "Time: $(date)"
