#!/bin/bash

# WandB Setup Helper Script
# This script helps you configure WandB for your training runs

echo "🚀 WandB Setup Helper for Reasoning360-MTL"
echo "=========================================="

# Check if WandB is installed
if ! command -v wandb &> /dev/null; then
    echo "❌ WandB is not installed. Installing..."
    pip install wandb
else
    echo "✅ WandB is already installed"
fi

# Get API key
echo ""
echo "📝 WandB API Key Setup:"
echo "1. Go to https://wandb.ai/settings"
echo "2. Copy your API key"
echo "3. Enter it below (or press Enter to skip)"
echo ""

read -p "Enter your WandB API key (or press Enter to skip): " WANDB_API_KEY

if [ -n "$WANDB_API_KEY" ]; then
    # Update smoke test script
    echo "🔧 Updating smoke test script..."
    sed -i "s/export WANDB_API_KEY=\"\"/export WANDB_API_KEY=\"$WANDB_API_KEY\"/" smoke_test_qwen2_7b.sh
    
    # Update multinode script
    echo "🔧 Updating multinode script..."
    sed -i "s/export WANDB_API_KEY=\"\"/export WANDB_API_KEY=\"$WANDB_API_KEY\"/" example_multinode_mtl_qwen2_7b.sh
    
    echo "✅ API key updated in both scripts!"
else
    echo "⏭️  Skipped API key setup. You can manually edit the scripts later."
fi

echo ""
echo "📊 WandB Configuration Summary:"
echo "Project Name: Reasoning360-MTL"
echo "Smoke Test Experiments: smoke-test-qwen2-7b-{JOB_ID}"
echo "Multinode Experiments: {JOB_ID}-{JOB_NAME}-qwen2-7b-mtl"
echo ""

echo "🎯 What gets logged to WandB:"
echo "• Training metrics (loss, rewards, etc.)"
echo "• Hyperparameters and configuration"
echo "• Validation generations (up to 30 samples)"
echo "• System metrics (GPU usage, etc.)"
echo ""

echo "🔧 To disable WandB temporarily:"
echo "Uncomment these lines in your script:"
echo "# export WANDB_MODE=disabled"
echo "# export WANDB_DISABLED=true"
echo ""

echo "✅ WandB setup complete! Your training runs will now be logged to WandB."
echo "Visit https://wandb.ai/your-username/Reasoning360-MTL to view your experiments."
