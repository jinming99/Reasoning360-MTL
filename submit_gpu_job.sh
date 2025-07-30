#!/bin/bash
#SBATCH --job-name=reasoning360_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

# Load required modules
module purge
module --ignore_cache load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a

# Activate virtual environment
source ~/venv_reasoning360mtl/bin/activate

# Install/upgrade GPU-specific packages
echo "=== Installing/updating GPU packages... ==="
# GPU-enabled PyTorch already provided by module
pip install -e ".[gpu,vllm,sglang]"

# Verify installation
echo "=== Verifying installation... ==="
python -c "\
import torch; \
print(f'PyTorch version: {torch.__version__}'); \
print(f'CUDA available: {torch.cuda.is_available()}'); \
if torch.cuda.is_available(): \
    print(f'Current device: {torch.cuda.current_device()}'); \
    print(f'Device name: {torch.cuda.get_device_name(0)}'); \
    print(f'CUDA version: {torch.version.cuda}'); \
    print(f'cuDNN version: {torch.backends.cudnn.version()}')
"

# Uncomment and modify the line below to run your actual script
# echo "=== Running your script... ==="
# python your_script.py

echo "=== Job completed ==="
