#!/bin/bash
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --job-name=smoke_test_qwen2_7b
#SBATCH --output=smoke_test_%j.out
#SBATCH --account=llmalignment

# Activate virtual environment
source ~/venv_reasoning360mtl/bin/activate

# Set Python path
export PYTHONPATH=/home/jinming/Reasoning360-MTL:$PYTHONPATH

# Navigate to project directory
cd ~/Reasoning360-MTL/scripts/train

# Run the smoke test
bash smoke_test_qwen2_7b.sh
