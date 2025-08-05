#!/bin/bash
set -euo pipefail

echo "🧪 Running Comprehensive Reward Computation Tests"
echo "=================================================="

# Activate virtual environment
echo "📦 Activating virtual environment..."
source ~/venv_reasoning360mtl/bin/activate

# Change to test directory
cd /home/jinming/Reasoning360-MTL/scripts/tests

# Make all test scripts executable
chmod +x *.py

# Run the master test runner
echo "🚀 Starting all reward computation tests..."
python3 run_all_reward_tests.py

echo "✅ Reward computation testing completed!"
echo "📄 Check the results in /home/jinming/Reasoning360-MTL/scripts/tests/"
