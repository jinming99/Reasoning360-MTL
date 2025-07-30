#!/bin/bash
# Load required modules
module purge
module --ignore_cache load Python/3.11.3-GCCcore-12.3.0

# Create and activate virtual environment
python -m venv ~/venv_reasoning360mtl
source ~/venv_reasoning360mtl/bin/activate

# Upgrade pip and install build dependencies
pip install --upgrade pip setuptools wheel

# Install PyTorch with CPU support (for login node)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install the package with non-GPU dependencies
cd "$(dirname "$0")"
pip install -e ".[test,prime,geo,math]"

echo "CPU environment setup complete!"
echo "To activate the environment, run: source ~/venv_reasoning360mtl/bin/activate"