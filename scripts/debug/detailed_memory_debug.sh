#!/bin/bash
#SBATCH --job-name=debug_ppo_memory
#SBATCH --account=llmalignment
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --begin=now+2minutes
#SBATCH --output=/home/jinming/Reasoning360-MTL/logs/debug_memory_%j.out
#SBATCH --error=/home/jinming/Reasoning360-MTL/logs/debug_memory_%j.err

###############################################################################
# DETAILED MEMORY DEBUGGING FOR PPO TRAINING
# This script adds comprehensive logging at every step to identify exactly
# where and why memory allocation fails
###############################################################################

set -euo pipefail

# =================== Logging Functions ===================
log_memory() {
    local step="$1"
    echo "🔍 MEMORY CHECK [$step] - $(date)"
    echo "----------------------------------------"
    
    # GPU Memory
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits | while read line; do
        echo "GPU: $line"
    done
    
    # CPU Memory
    free -h | grep -E "Mem:|Swap:"
    
    # Process Memory
    echo "Top GPU processes:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || echo "No GPU processes"
    
    echo "----------------------------------------"
    echo ""
}

log_step() {
    local step="$1"
    echo "📋 STEP: $step - $(date)"
    echo "========================================"
}

# =================== Environment Setup ===================
log_step "Environment Setup"
export WANDB_API_KEY="$WANDB_API_KEY"
export FLASH_ATTENTION_DISABLE=1
export DISABLE_FLASH_ATTENTION=1
export FLASH_ATTENTION_SKIP_CUDA_CHECK=1
export VLLM_USE_V1=0

# Enable detailed CUDA debugging
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3

log_memory "After Environment Setup"

# =================== GPU Reset ===================
log_step "GPU Memory Reset"
nvidia-smi --gpu-reset || true
sleep 3
log_memory "After GPU Reset"

# =================== Ray Cluster Setup ===================
log_step "Ray Cluster Initialization"
ray stop --force || true
sleep 5

ray start --head --num-gpus=4 --num-cpus=8 --object-store-memory=10000000000 --include-dashboard=false --block &
sleep 10

log_memory "After Ray Cluster Start"

# =================== Pre-Training Analysis ===================
log_step "Pre-Training Memory Analysis"
python /home/jinming/Reasoning360-MTL/scripts/debug/memory_analysis.py
log_memory "After Memory Analysis"

# =================== Model Paths ===================
MODEL_DIR="/home/jinming/Reasoning360-MTL/models/Qwen2.5-7B-Instruct"
TRAIN_FILES="[/home/jinming/Reasoning360-MTL/data/train/guru_18k/math.parquet]"
VAL_FILES="[/home/jinming/Reasoning360-MTL/data/validation/guru_3k/math.parquet]"

# =================== Training with Detailed Logging ===================
log_step "Starting PPO Training with Memory Monitoring"

# Start continuous memory monitoring in background
(
    while true; do
        sleep 10
        log_memory "During Training - $(date +%H:%M:%S)"
    done
) &
MONITOR_PID=$!

cd /home/jinming/Reasoning360-MTL

# Ultra-minimal configuration for debugging
python -m verl.trainer.main_ppo \
  data.train_files="$TRAIN_FILES" \
  data.val_files="$VAL_FILES" \
  data.tokenizer="$MODEL_DIR" \
  data.max_prompt_length=128 \
  data.max_response_length=256 \
  data.train_batch_size=2 \
  data.gen_batch_size=2 \
  \
  actor_rollout_ref.model.path="$MODEL_DIR" \
  critic.model.path="$MODEL_DIR" \
  \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  critic.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.05 \
  ++actor_rollout_ref.rollout.max_model_len=128 \
  ++actor_rollout_ref.rollout.max_num_seqs=1 \
  ++actor_rollout_ref.rollout.enforce_eager=true \
  ++actor_rollout_ref.rollout.tensor_parallel_size=2 \
  \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=4 \
  trainer.total_epochs=1 \
  trainer.val_before_train=false \
  trainer.test_freq=0 \
  trainer.save_freq=1 \
  trainer.project_name="Debug-Memory" \
  \
  ++actor_rollout_ref.fsdp_config.param_offload=true \
  ++actor_rollout_ref.fsdp_config.optimizer_offload=true \
  ++actor_rollout_ref.fsdp_config.activation_offload=true \
  ++critic.fsdp_config.param_offload=true \
  ++critic.fsdp_config.optimizer_offload=true \
  ++critic.fsdp_config.activation_offload=true \
  \
  ++actor_rollout_ref.model.enable_flash_attention=false \
  ++critic.model.enable_flash_attention=false \
  ++actor_rollout_ref.model.enable_gradient_checkpointing=true \
  ++critic.model.enable_gradient_checkpointing=true \
  ++actor_rollout_ref.model.enable_activation_offload=true \
  ++critic.model.enable_activation_offload=true \
  ++actor_rollout_ref.actor.entropy_checkpointing=true \
  ++actor_rollout_ref.ref.entropy_from_logits_with_chunking=true

TRAINING_EXIT_CODE=$?

# Stop memory monitoring
kill $MONITOR_PID 2>/dev/null || true

log_step "Training Completed"
log_memory "After Training Completion"

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
else
    echo "❌ Training failed with exit code: $TRAINING_EXIT_CODE"
    
    # Additional debugging info on failure
    echo "🔍 FAILURE ANALYSIS:"
    echo "Recent GPU processes:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || echo "No GPU processes"
    
    echo "CUDA memory summary:"
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); [print(f'GPU {i}: {torch.cuda.get_device_properties(i)}') for i in range(torch.cuda.device_count())]" 2>/dev/null || echo "Could not get CUDA info"
fi

# =================== Cleanup ===================
log_step "Cleanup"
ray stop --force || true
log_memory "After Cleanup"

echo "🎉 Debugging session completed!"
echo "📄 Check logs at: /home/jinming/Reasoning360-MTL/logs/"
