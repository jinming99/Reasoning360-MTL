#!/bin/bash
#SBATCH --job-name=qwen2-7b
#SBATCH --partition=main
#SBATCH --nodes=20
#SBATCH --ntasks=20
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
#SBATCH --time=240:00:00

# This script launches a multi‑task RL training job on Qwen2.5‑7B using
# the custom multi‑task trainer (main_mtl_ppo.py).  It assumes the
# GURU‑18K mixed dataset has been created via scripts/tools/create_guru_18k.py
# and is stored in $SHARED_DATA_PATH/train/guru_18k_mix.parquet.  To
# evaluate differences between simple averaging and PCGrad, set
# MTL_METHOD to either 'equal' or 'pcgrad'.

set -euo pipefail

# =================== Environment Setup ===================
# Disable flash attention to avoid import errors
export FLASH_ATTENTION_DISABLE=1
export FLASH_ATTENTION_DISABLED=1

# Disable conflicting GPU environment variables
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

# Map tasks to individual dataset files
TRAIN_FILES="{
  'math':'${SHARED_DATA_PATH}/train/guru_18k/math.parquet',
  'codegen':'${SHARED_DATA_PATH}/train/guru_18k/codegen.parquet',
  'logic':'${SHARED_DATA_PATH}/train/guru_18k/logic.parquet',
  'simulation':'${SHARED_DATA_PATH}/train/guru_18k/simulation.parquet',
  'table':'${SHARED_DATA_PATH}/train/guru_18k/table.parquet',
  'stem':'${SHARED_DATA_PATH}/train/guru_18k/stem.parquet'
}"

# Base model
BASE_MODEL=Qwen/Qwen2.5-7B

# Choose MTL method: 'equal' or 'pcgrad'
MTL_METHOD=${MTL_METHOD:-pcgrad}

# RL hyperparameters (consistent with 32B script)
TRAIN_PROMPT_BSZ=512  # on-policy model update batchsize: train_prompt_bsz * rollout.n
GEN_PROMPT_BSZ=$((TRAIN_PROMPT_BSZ * 1))
N_RESP_PER_PROMPT=16
TRAIN_PROMPT_MINI_BSZ=64  # model grad update batchsize
LR=1e-6
LR_WARMUP_STEPS=10
CLIP_RATIO=0.2
MAX_PROMPT_LENGTH=$((1024 * 4))
MAX_RESPONSE_LENGTH=$((1024 * 8))

# Algorithm settings
TEMPERATURE=1.0
TOP_P=1.0
TOP_K=-1  # 0 for HF rollout, -1 for vLLM rollout

# Batch size settings (consistent with 32B script)
INFER_MICRO_BATCH_SIZE=null
TRAIN_MICRO_BATCH_SIZE=null
USE_DYNAMIC_BSZ=true
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

echo "Starting Ray cluster with ${worker_num} nodes..."
echo "Head node: $head_node_ip:$port"

srun --nodes=$worker_num --ntasks=$worker_num --ntasks-per-node=1 ray stop
sleep 10
srun --nodes=$worker_num --ntasks=$worker_num --ntasks-per-node=1 rm -rf /tmp/ray/ray_current_cluster
sleep 3

# Start Ray head node (disable dashboard to avoid startup failures)
srun --nodes=1 --ntasks=1 -w "$head_node" ${CONDA_BIN_PATH:-python} -m ray start --head --node-ip-address="$head_node_ip" --port=$port --num-cpus "$SLURM_CPUS_PER_TASK" --num-gpus 8 --include-dashboard=false --block &
sleep 10

# Start Ray worker nodes
for node in "${nodes[@]:1}"; do
    srun --nodes=1 --ntasks=1 -w "$node" ${CONDA_BIN_PATH:-python} -m ray start --address "$address_head" --num-cpus "$SLURM_CPUS_PER_TASK" --num-gpus 8 --block &
done
sleep 10

echo "Ray cluster started successfully"

# Launch training with the standard PPO trainer (MTL is handled via config)
# Use ppo_trainer.yaml as base config, consistent with smoke test
srun --jobid ${SLURM_JOBID} --kill-on-bad-exit=1 \
    python -u -m verl.trainer.main_ppo \
    --config-name=ppo_trainer \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILES" \
    data.val_files="[/home/jinming/Reasoning360-MTL/data/validation/guru_3k_validation_mix.parquet]" \
    data.train_batch_size=$TRAIN_PROMPT_BSZ \
    data.gen_batch_size=$GEN_PROMPT_BSZ \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.use_dynamic_bsz=$USE_DYNAMIC_BSZ \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ACTOR_PPO_MAX_TOKEN_LEN \
    actor_rollout_ref.actor.strategy="fsdp" \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.optim.lr_warmup_steps=$LR_WARMUP_STEPS \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.optim.min_lr_ratio=0. \
    actor_rollout_ref.actor.clip_ratio_low=$CLIP_RATIO \
    actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_PROMPT_MINI_BSZ \
    actor_rollout_ref.actor.ppo_micro_batch_size=$TRAIN_MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.fsdp_config.param_offload=$OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OFFLOAD \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$USE_DYNAMIC_BSZ \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$INFER_PPO_MAX_TOKEN_LEN \
    actor_rollout_ref.ref.log_prob_micro_batch_size=$INFER_MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=$OFFLOAD \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=$N_RESP_PER_PROMPT \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$USE_DYNAMIC_BSZ \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$INFER_PPO_MAX_TOKEN_LEN \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=$INFER_MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.top_p=$TOP_P \
    actor_rollout_ref.rollout.top_k=$TOP_K \
    actor_rollout_ref.rollout.val_kwargs.top_k=$TOP_K \
    actor_rollout_ref.rollout.val_kwargs.top_p=$TOP_P \
    actor_rollout_ref.rollout.val_kwargs.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.rollout.multi_turn.enable=False \
    +actor_rollout_ref.rollout.mode="sync" \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${#nodes[@]} \
    trainer.total_epochs=3 \
    trainer.val_before_train=false \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${WANDB_EXPERIMENT_NAME} \
    +trainer.val_generations_to_log_to_wandb=30 \
    mtl.enabled=true \
    mtl.method=$MTL_METHOD