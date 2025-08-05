#!/bin/bash
#SBATCH --account=llmalignment
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --job-name=guru7b_reward_test
#SBATCH --output=guru7b_test_%j.out
#SBATCH --error=guru7b_test_%j.err

set -euo pipefail

echo "🚀 Starting GURU-7B Model + Reward Testing"
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

# Run the GURU-7B test
echo "🧪 Running GURU-7B model + reward computation test..."
echo "This will test if a trained model gets non-zero rewards"

python3 test_guru7b_rewards.py

echo "✅ GURU-7B testing completed!"
echo "📄 Check results in /home/jinming/Reasoning360-MTL/scripts/tests/guru7b_results.json"
echo "Time: $(date)"
