#!/bin/bash
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=smoke_test_qwen2_7b
#SBATCH --output=smoke_test_%j.out
#SBATCH --account=llmalignment

# Virtual environment already has required dependencies

# Activate virtual environment
source ~/venv_reasoning360mtl/bin/activate

# Set CUDA environment
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_LOG_TO_DRIVER=1
export HYDRA_FULL_ERROR=1

# Set Python path
export PYTHONPATH=/home/jinming/Reasoning360-MTL:$PYTHONPATH

# Navigate to project directory
cd ~/Reasoning360-MTL/scripts/train

# Run the smoke test
bash smoke_test_qwen2_7b.sh
