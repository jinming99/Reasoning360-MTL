# WandB Setup Guide for Reasoning360-MTL

## 🚀 Quick Setup

### Option 1: Use the Helper Script
```bash
cd scripts/train
./setup_wandb.sh
```

### Option 2: Manual Setup
1. Get your API key from https://wandb.ai/settings
2. Edit the training scripts and replace:
   ```bash
   export WANDB_API_KEY=""  # TODO: Add your WandB API key here
   ```
   with:
   ```bash
   export WANDB_API_KEY="your_actual_api_key_here"
   ```

## 📊 What Gets Logged

### Training Metrics
- **Loss values**: Actor loss, critic loss, total loss
- **Reward metrics**: Mean reward, reward distribution
- **PPO metrics**: Policy ratio, KL divergence, entropy
- **Learning metrics**: Learning rate, gradient norms

### Hyperparameters
- **Model settings**: Model path, architecture config
- **Training settings**: Batch sizes, learning rates, epochs
- **PPO settings**: Clip ratios, entropy coefficients
- **MTL settings**: Method (pcgrad/equal), task weights

### Validation Data
- **Generated samples**: Up to 30 validation generations per epoch
- **Task performance**: Per-task accuracy and rewards
- **Comparison data**: Generated vs ground truth responses

### System Metrics
- **GPU usage**: Memory utilization, compute usage
- **Training speed**: Tokens/second, samples/second
- **Resource usage**: CPU, memory, network

## 🎯 Project Organization

### Project Structure
```
WandB Project: Reasoning360-MTL
├── Smoke Tests: smoke-test-qwen2-7b-{JOB_ID}
├── Multinode Training: {JOB_ID}-{JOB_NAME}-qwen2-7b-mtl
└── Experiments: Custom experiment names
```

### Experiment Naming Convention
- **Smoke Tests**: `smoke-test-qwen2-7b-{SLURM_JOB_ID}`
- **Multinode Training**: `{SLURM_JOB_ID}-{SLURM_JOB_NAME}-qwen2-7b-mtl`
- **Custom**: Set `WANDB_EXPERIMENT_NAME` environment variable

## 🔧 Configuration Options

### Enable/Disable WandB
```bash
# Enable WandB (default)
export WANDB_MODE=online

# Disable WandB for testing
export WANDB_MODE=disabled
export WANDB_DISABLED=true
```

### Custom Project/Experiment Names
```bash
# Custom project name
export WANDB_PROJECT="MyCustomProject"

# Custom experiment name
export WANDB_EXPERIMENT_NAME="my-experiment-v1"
```

### Validation Logging Control
```bash
# Number of validation samples to log (default: 30)
+trainer.val_generations_to_log_to_wandb=50
```

### Proxy Settings (if behind firewall)
```bash
# Add to training command
trainer.wandb_proxy="http://your-proxy:port"
```

## 📈 Viewing Your Experiments

### WandB Dashboard
1. Go to https://wandb.ai
2. Navigate to your `Reasoning360-MTL` project
3. View real-time training metrics and logs

### Key Dashboards to Monitor
- **Overview**: Training progress, loss curves
- **System**: GPU/CPU usage, memory consumption
- **Hyperparameters**: Configuration comparison across runs
- **Media**: Validation generation samples

## 🛠️ Troubleshooting

### Common Issues

#### 1. API Key Not Working
```bash
# Test your API key
wandb login
# Enter your API key when prompted
```

#### 2. Network/Proxy Issues
```bash
# Set proxy if needed
export HTTPS_PROXY=http://your-proxy:port
export HTTP_PROXY=http://your-proxy:port
```

#### 3. Disable WandB Temporarily
```bash
# In your script, uncomment:
export WANDB_MODE=disabled
export WANDB_DISABLED=true
```

#### 4. Permission Issues
```bash
# Ensure WandB is installed in your environment
pip install wandb
```

### Debug Commands
```bash
# Check WandB status
wandb status

# Test WandB connection
wandb online

# View WandB logs
wandb sync --help
```

## 📋 Best Practices

### 1. Experiment Organization
- Use descriptive experiment names
- Tag experiments with model size, dataset, method
- Group related experiments in same project

### 2. Hyperparameter Tracking
- Log all important hyperparameters
- Use WandB sweeps for hyperparameter optimization
- Compare runs with different configurations

### 3. Resource Monitoring
- Monitor GPU memory usage
- Track training speed metrics
- Set up alerts for failed runs

### 4. Data Management
- Log validation samples regularly
- Save important checkpoints
- Track data preprocessing parameters

## 🎯 Integration with SLURM

### Automatic Job Tracking
The scripts automatically use SLURM job information:
```bash
WANDB_EXPERIMENT_NAME="${SLURM_JOB_ID}-${SLURM_JOB_NAME}-qwen2-7b-mtl"
```

### Multi-node Coordination
- Only rank 0 process logs to WandB
- Metrics are aggregated across all nodes
- Shared experiment tracking across cluster

### Resource Allocation Tracking
- SLURM job ID and node information logged
- GPU allocation and usage tracked
- Training time and resource efficiency metrics

---

## 🚀 Ready to Train!

After setting up WandB:
1. ✅ API key configured in scripts
2. ✅ Project and experiment names set
3. ✅ Logging enabled in training commands
4. ✅ Ready to submit jobs and monitor progress

Your training runs will now be automatically logged to WandB for easy monitoring and analysis!
