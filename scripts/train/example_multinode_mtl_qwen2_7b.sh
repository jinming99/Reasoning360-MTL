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

export NCCL_DEBUG=info
export NCCL_ALGO=NVLSTree
export NCCL_IBEXT_DISABLE=1
export NCCL_NVLS_ENABLE=1
export NCCL_IB_HCA=mlx5
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1

# Paths
SHARED_DATA_PATH=./data

# Map tasks to individual dataset files

TRAIN_FILES='{
  "math":"'"${SHARED_DATA_PATH}"'/train/guru_18k/math.parquet",
  "codegen":"'"${SHARED_DATA_PATH}"'/train/guru_18k/codegen.parquet",
  "logic":"'"${SHARED_DATA_PATH}"'/train/guru_18k/logic.parquet",
  "simulation":"'"${SHARED_DATA_PATH}"'/train/guru_18k/simulation.parquet",
  "table":"'"${SHARED_DATA_PATH}"'/train/guru_18k/table.parquet",
  "stem":"'"${SHARED_DATA_PATH}"'/train/guru_18k/stem.parquet"
}'

# Base model
BASE_MODEL=Qwen/Qwen2.5-7B

# Choose MTL method: 'equal' or 'pcgrad'
MTL_METHOD=${MTL_METHOD:-pcgrad}

# RL hyperparameters (similar to the 32B script)
TRAIN_PROMPT_BSZ=512
N_RESP_PER_PROMPT=16
TRAIN_PROMPT_MINI_BSZ=64
LR=1e-6
LR_WARMUP_STEPS=10
CLIP_RATIO=0.2
MAX_PROMPT_LENGTH=$((1024 * 4))
MAX_RESPONSE_LENGTH=$((1024 * 8))

# Start Ray cluster
nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
head_node=${nodes[0]}
port=6379
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
address_head=$head_node_ip:$port
worker_num=${#nodes[@]}

srun --nodes=$worker_num --ntasks=$worker_num --ntasks-per-node=1 ray stop
sleep 10
srun --nodes=$worker_num --ntasks=$worker_num --ntasks-per-node=1 rm -rf /tmp/ray/ray_current_cluster
sleep 3

srun --nodes=1 --ntasks=1 -w "$head_node" ${CONDA_BIN_PATH:-python} -m ray start --head --node-ip-address="$head_node_ip" --port=$port --num-cpus "$SLURM_CPUS_PER_TASK" --num-gpus 8 --block &
sleep 10
for node in "${nodes[@]:1}"; do
    srun --nodes=1 --ntasks=1 -w "$node" ${CONDA_BIN_PATH:-python} -m ray start --address "$address_head" --num-cpus "$SLURM_CPUS_PER_TASK" --num-gpus 8 --block &
done
sleep 10

# Launch training with the standard PPO trainer (MTL is handled via config)
srun --jobid ${SLURM_JOBID} --kill-on-bad-exit=1 \
    python -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILES \
    data.val_files="[]" \
    data.train_batch_size=$TRAIN_PROMPT_BSZ \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.optim.lr_warmup_steps=$LR_WARMUP_STEPS \
    actor_rollout_ref.actor.clip_ratio_low=$CLIP_RATIO \
    actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO \
    actor_rollout_ref.rollout.n=$N_RESP_PER_PROMPT \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_PROMPT_MINI_BSZ \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${#nodes[@]} \
    trainer.total_epochs=3 \
    trainer.val_before_train=True \
    trainer.logger=['console','wandb'] \
    mtl.enabled=true \
    mtl.method=$MTL_METHOD