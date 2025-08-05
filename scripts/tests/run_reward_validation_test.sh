#!/bin/bash
#SBATCH --job-name=reward_validation_test
#SBATCH --output=/home/jinming/Reasoning360-MTL/scripts/tests/reward_validation_%j.out
#SBATCH --error=/home/jinming/Reasoning360-MTL/scripts/tests/reward_validation_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

echo "🧪 Starting Reward Improvement Validation Test"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=" * 60

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/home/jinming/llm_models/.cache
export TRANSFORMERS_CACHE=/home/jinming/llm_models/.cache
export HF_DATASETS_CACHE=/home/jinming/llm_models/.cache

# Disable flash attention to avoid compatibility issues
export FLASH_ATTENTION_DISABLE=1
export FLASH_ATTENTION_DISABLED=1

# Clean up any conflicting GPU environment variables
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

# Navigate to project directory
cd /home/jinming/Reasoning360-MTL

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Install required dependencies if not already installed
echo "Checking dependencies..."
pip install --quiet outlines transformers torch pandas numpy

# Check GPU availability
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

# Run the comprehensive validation test
echo "🚀 Running comprehensive reward validation test..."
python scripts/tests/test_reward_improvement_validation.py

echo "✅ Test completed at: $(date)"
echo "Check output files for results:"
echo "  - reward_improvement_validation_results.json"
echo "  - reward_improvement_summary.csv"
