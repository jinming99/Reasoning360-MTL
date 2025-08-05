#!/bin/bash
#SBATCH --job-name=qwen2-7b
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --account=llmalignment
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
#SBATCH --time=24:00:00

# This script launches a multi‑task RL training job on Qwen2.5‑7B using
# the custom multi‑task trainer (main_mtl_ppo.py).  It assumes the
# GURU‑18K mixed dataset has been created via scripts/tools/create_guru_18k.py
# and is stored in $SHARED_DATA_PATH/train/guru_18k_mix.parquet.  To
# evaluate differences between simple averaging and PCGrad, set
# MTL_METHOD to either 'equal' or 'pcgrad'.

set -euo pipefail

# =================== Environment Setup ===================
echo "🔧 Setting up environment..."
export FLASH_ATTENTION_DISABLE=1
export DISABLE_FLASH_ATTENTION=1
export FLASH_ATTENTION_SKIP_CUDA_CHECK=1
export VLLM_USE_V1=0

# Unset conflicting GPU environment variables
unset ROCR_VISIBLE_DEVICES 2>/dev/null || true
unset HIP_VISIBLE_DEVICES 2>/dev/null || true

# =================== WandB Configuration ===================
# Set your WandB API key here (get from https://wandb.ai/settings)
export WANDB_API_KEY="dd85f472f958957fc212f43f0947e8a086beb38c"  # TODO: Add your WandB API key here

# WandB project settings
export WANDB_PROJECT="Reasoning360-MTL"
export WANDB_EXPERIMENT_NAME="${SLURM_JOB_ID}-${SLURM_JOB_NAME}-qwen2-7b-mtl"

# Enable WandB logging
export WANDB_MODE=online

# Optional: Disable WandB for testing by uncommenting below:
# export WANDB_MODE=disabled
# export WANDB_DISABLED=true

export NCCL_DEBUG=info
export NCCL_ALGO=NVLSTree
export NCCL_IBEXT_DISABLE=1
export NCCL_NVLS_ENABLE=1
export NCCL_IB_HCA=mlx5
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1

# Paths
SHARED_DATA_PATH=./data

# Configure data files based on MTL mode
if [ "${MTL_ENABLED:-false}" = "true" ]; then
    # MTL requires dict format {task: path}
    TRAIN_FILES="{math: /home/jinming/Reasoning360-MTL/data/train/guru_18k/math.parquet}"
else
    # Baseline uses list format
    TRAIN_FILES="[/home/jinming/Reasoning360-MTL/data/train/guru_18k/math.parquet]"
fi

# Base model (local path)
BASE_MODEL=/home/jinming/llm_models/Qwen2.5-7B

# Choose MTL method: 'equal' or 'pcgrad'
MTL_METHOD=${MTL_METHOD:-pcgrad}

# RL hyperparameters (adjusted for 2-GPU setup)
TRAIN_PROMPT_BSZ=64   # on-policy model update batchsize: train_prompt_bsz * rollout.n
GEN_PROMPT_BSZ=$((TRAIN_PROMPT_BSZ * 1))
N_RESP_PER_PROMPT=4   # Reduced for 2-GPU setup
TRAIN_PROMPT_MINI_BSZ=16  # model grad update batchsize
LR=1e-6
LR_WARMUP_STEPS=10
CLIP_RATIO=0.2
MAX_PROMPT_LENGTH=1024    # Reduced for 2-GPU setup
MAX_RESPONSE_LENGTH=2048  # Reduced for 2-GPU setup

# Algorithm settings
TEMPERATURE=1.0
TOP_P=1.0
TOP_K=-1  # 0 for HF rollout, -1 for vLLM rollout

# Batch size settings (adjusted for 2-GPU setup)
INFER_MICRO_BATCH_SIZE=2      # Micro batch size for inference
TRAIN_MICRO_BATCH_SIZE=2      # Micro batch size for training
USE_DYNAMIC_BSZ=false         # Disable dynamic batch sizing for stability
ACTOR_PPO_MAX_TOKEN_LEN=$(( (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 2))
INFER_PPO_MAX_TOKEN_LEN=$(( (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 2))
OFFLOAD=true

# =================== Install Dependencies ===================
# Install required packages in virtual environment
echo "Installing required dependencies..."
pip install numba

# =================== Start Ray Cluster ===================
nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
head_node=${nodes[0]}
port=6379
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
address_head=$head_node_ip:$port
worker_num=${#nodes[@]}

# Simple single-node Ray startup (SLURM-friendly)
echo "🔄 Starting single-node Ray cluster..."
echo "   Port: $port"
echo "   CPUs: ${SLURM_CPUS_PER_TASK:-32}"
echo "   GPUs: $(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)"

# Stop any existing Ray processes
ray stop --force || true
sleep 2

# Start Ray head node
ray start --head --port=$port --num-cpus=${SLURM_CPUS_PER_TASK:-32} --num-gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l) --include-dashboard=false --block &
sleep 5

echo "✅ Ray cluster started successfully"

# =================== DEBUG GPU ALLOCATION ===================
echo "🔍 DEBUG: GPU Allocation Analysis"
echo "   SLURM_GPUS_PER_NODE: ${SLURM_GPUS_PER_NODE:-'not set'}"
echo "   CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-'not set'}"
echo "   Available GPUs: $(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)"
echo "   Ray Status:"
ray status
echo "   Ray Available Resources:"
ray status --verbose 2>/dev/null | grep -A 10 "Resources" || echo "   (Ray status verbose not available)"
echo "🔍 Configuration being passed to training:"
echo "   trainer.nnodes=1"
echo "   trainer.n_gpus_per_node=8"
echo "   Expected total GPUs: 1 * 8 = 8"
echo "   Actual available GPUs: $(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)"

# Launch training with the standard PPO trainer (MTL is handled via config)
# Simplified training command based on working smoke test
python -u -m verl.trainer.main_ppo \
    --config-name=ppo_trainer \
    actor_rollout_ref.model.path="$BASE_MODEL" \
    data.tokenizer="$BASE_MODEL" \
    data.train_files="$TRAIN_FILES" \
    data.val_files="[/home/jinming/Reasoning360-MTL/data/validation/guru_3k/math.parquet]" \
    data.max_prompt_length=4000 \
    data.max_response_length=8000 \
    data.train_batch_size=64 \
    data.gen_batch_size=64 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    critic.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    ++actor_rollout_ref.rollout.tensor_parallel_size=2 \
    ++actor_rollout_ref.rollout.generation_config.pad_token_id=151643 \
    ++actor_rollout_ref.rollout.pad_token_id=151643 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.model.use_remove_padding=false \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=30 \
    trainer.val_before_train=false \
    trainer.test_freq=0 \
    trainer.logger=['console'] \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${WANDB_EXPERIMENT_NAME} \
    ++trainer.ray_wait_register_center_timeout=300 \
    ++actor_rollout_ref.model.enable_flash_attention=false \
    ++critic.model.enable_flash_attention=false \
    ++actor_rollout_ref.model.enable_gradient_checkpointing=true \
    ++critic.model.enable_gradient_checkpointing=true \
    ++actor_rollout_ref.model.enable_activation_offload=true \
    ++critic.model.enable_activation_offload=true \
    ++actor_rollout_ref.actor.entropy_checkpointing=true \
    ++actor_rollout_ref.ref.entropy_from_logits_with_chunking=true \
    ++actor_rollout_ref.actor.fsdp_config.activation_offload=true \
    ++critic.fsdp_config.param_offload=true \
    ++critic.fsdp_config.optimizer_offload=true \
    ++actor_rollout_ref.rollout.multi_turn.enable=false \
    ++actor_rollout_ref.rollout.mode=sync \
    ++actor_rollout_ref.actor.optim.lr=1e-6 \
    ++actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    ++actor_rollout_ref.actor.clip_ratio=0.2 \
    ++algorithm.adv_estimator=grpo \
    mtl.enabled=${MTL_ENABLED:-false} \
    mtl.method=${MTL_METHOD:-pcgrad}