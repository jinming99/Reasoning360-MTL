#!/bin/bash
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --job-name=multinode_rl_qwen32b
#SBATCH --output=multinode_rl_%j.out
#SBATCH --account=llmalignment

# Activate virtual environment
source ~/venv_reasoning360mtl/bin/activate

# Set Python path
export PYTHONPATH=/home/jinming/Reasoning360-MTL:$PYTHONPATH

# Navigate to project directory
cd ~/Reasoning360-MTL/scripts/train

# Run the multinode RL training
bash example_multinode_rl_qwen32b_base.sh
