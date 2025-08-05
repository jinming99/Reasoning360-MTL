#!/bin/bash
#SBATCH --job-name=memory_optimized_ppo
#SBATCH --account=llmalignment
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --begin=now+2minutes
#SBATCH --output=/home/jinming/Reasoning360-MTL/logs/memory_optimized_%j.out
#SBATCH --error=/home/jinming/Reasoning360-MTL/logs/memory_optimized_%j.err

###############################################################################
# MEMORY-OPTIMIZED PPO TRAINING WITH STRATEGIC MODEL PLACEMENT
# 
# KEY INSIGHT: The problem is not just memory size, but MODEL PLACEMENT
# PPO loads 4 models simultaneously:
# - Actor (GPUs 0-1): Policy model with gradients
# - Critic (GPUs 2-3): Value model with gradients  
# - Reference (GPUs 4-5): Frozen reference model
# - Rollout/vLLM (GPUs 6-7): Generation model with KV cache
#
# This distributes memory load instead of cramming everything on same GPUs
###############################################################################

set -euo pipefail

# =================== Environment Setup ===================
echo "🔧 Setting up memory-optimized environment..."
export WANDB_DISABLED=true
export FLASH_ATTENTION_DISABLE=1
export DISABLE_FLASH_ATTENTION=1
export FLASH_ATTENTION_SKIP_CUDA_CHECK=1
export VLLM_USE_V1=0

# Memory debugging
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Unset conflicting GPU environment variables
unset ROCR_VISIBLE_DEVICES 2>/dev/null || true
unset HIP_VISIBLE_DEVICES 2>/dev/null || true

# =================== Memory Analysis ===================
echo "🔍 Running pre-training memory analysis..."
python /home/jinming/Reasoning360-MTL/scripts/debug/memory_analysis.py

# =================== GPU Memory Cleanup ===================
echo "🧹 Cleaning up GPU memory..."
nvidia-smi --gpu-reset || true
sleep 3

# =================== Ray Cluster Setup ===================
echo "🔄 Starting memory-optimized Ray cluster..."
ray stop --force || true
sleep 5

# Conservative Ray settings for memory optimization
ray start --head --num-gpus=8 --num-cpus=16 --object-store-memory=20000000000 --include-dashboard=false --block &
sleep 15

echo "📊 Ray cluster started successfully"

# =================== Model and Data Paths ===================
MODEL_DIR="/home/jinming/llm_models/Qwen2.5-7B"
TRAIN_FILES="[/home/jinming/Reasoning360-MTL/data/train/guru_18k/math.parquet]"
VAL_FILES="[/home/jinming/Reasoning360-MTL/data/validation/guru_3k/math.parquet]"

# =================== Memory-Optimized Training ===================
echo "🚀 Starting memory-optimized PPO training..."
echo "📋 Configuration:"
echo "   • Actor Model: GPUs 0-1 (Tensor Parallel)"
echo "   • Critic Model: GPUs 2-3 (Tensor Parallel)"  
echo "   • Reference Model: GPUs 4-5 (Tensor Parallel)"
echo "   • vLLM Rollout: GPUs 6-7 (Tensor Parallel)"
echo "   • Ultra-conservative memory settings"

cd /home/jinming/Reasoning360-MTL

python -m verl.trainer.main_ppo \
  data.train_files="$TRAIN_FILES" \
  data.val_files="$VAL_FILES" \
  data.tokenizer="$MODEL_DIR" \
  data.max_prompt_length=256 \
  data.max_response_length=512 \
  data.train_batch_size=8 \
  data.gen_batch_size=8 \
  \
  actor_rollout_ref.model.path="$MODEL_DIR" \
  \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
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
  ++actor_rollout_ref.rollout.device_map=[6,7] \
  \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=8 \
  trainer.total_epochs=30 \
  trainer.val_before_train=false \
  trainer.test_freq=0 \
  trainer.save_freq=5 \
  trainer.project_name="Reasoning360-MTL" \
  trainer.logger=['console'] \
  \
  algorithm.adv_estimator=grpo \
  ++actor_rollout_ref.fsdp_config.param_offload=true \
  ++actor_rollout_ref.fsdp_config.optimizer_offload=true \
  ++actor_rollout_ref.fsdp_config.activation_offload=true \
  \
  ++actor_rollout_ref.model.enable_flash_attention=false \
  ++actor_rollout_ref.model.enable_gradient_checkpointing=true \
  ++actor_rollout_ref.model.enable_activation_offload=true \
  ++actor_rollout_ref.actor.entropy_checkpointing=true \
  ++actor_rollout_ref.ref.entropy_from_logits_with_chunking=true \
  \
  ++actor_rollout_ref.rollout.multi_turn.enable=false \
  ++actor_rollout_ref.rollout.mode=sync \
  +trainer.early_stopping_min_epochs=10 \
  +trainer.early_stopping_patience=5

echo "✅ Training completed successfully!"

# =================== Cleanup ===================
echo "🧹 Cleaning up resources..."
ray stop --force || true
echo "🎉 Memory-optimized training finished!"
