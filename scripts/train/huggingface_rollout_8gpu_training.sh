#!/bin/bash
#SBATCH --job-name=hf_rollout
#SBATCH --partition=a100_normal_q
#SBATCH --account=llmalignment
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=/home/jinming/Reasoning360-MTL/logs/hf_rollout_%j.out
#SBATCH --error=/home/jinming/Reasoning360-MTL/logs/hf_rollout_%j.err

echo "=== HuggingFace Rollout Training Script (vLLM Alternative) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Environment setup
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export NCCL_IGNORE_DISABLED_P2P=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Disable flash attention
export FLASH_ATTENTION_DISABLE=1
export DISABLE_FLASH_ATTN=1

# Disable WandB
unset WANDB_API_KEY
export WANDB_DISABLED=true

# Clean up conflicting GPU environment variables
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

# Activate environment
cd /home/jinming/Reasoning360-MTL
source /home/jinming/venv_reasoning360mtl/bin/activate

# Install dependencies
pip install numba

echo "=== GPU Information ==="
nvidia-smi
echo "=== Memory Information ==="
free -h

# Get number of GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)
echo "Detected $NUM_GPUS GPUs"

# Ray configuration
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_DEDUP_LOGS=0

echo "=== Starting Ray cluster ==="
ray stop --force
sleep 5

ray start --head \
    --num-cpus=32 \
    --num-gpus=$NUM_GPUS \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --disable-usage-stats \
    --verbose

sleep 10

echo "=== Ray cluster status ==="
ray status

echo "=== Starting PPO Training with HuggingFace Rollout (No vLLM) ==="

# Model path
MODEL_PATH="/home/jinming/llm_models/Qwen2.5-7B"

# Use HuggingFace rollout instead of vLLM to avoid KV cache issues
python -m verl.trainer.main_ppo \
    data.train_files=/home/jinming/Reasoning360-MTL/data/train/guru_18k/math.parquet \
    data.val_files=/home/jinming/Reasoning360-MTL/data/train/guru_18k/math.parquet \
    data.tokenizer=$MODEL_PATH \
    data.max_prompt_length=256 \
    data.max_response_length=512 \
    data.train_batch_size=8 \
    data.gen_batch_size=8 \
    \
    actor_rollout_ref.model.path=$MODEL_PATH \
    \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.name=hf \
    ++actor_rollout_ref.rollout.tensor_parallel_size=2 \
    ++actor_rollout_ref.rollout.model_hf_config.attn_implementation=eager \
    ++actor_rollout_ref.rollout.model_hf_config.torch_dtype=bfloat16 \
    ++actor_rollout_ref.rollout.model_hf_config.use_cache=false \
    ++actor_rollout_ref.rollout.model_hf_config.trust_remote_code=true \
    ++actor_rollout_ref.rollout.generation_config.do_sample=true \
    ++actor_rollout_ref.rollout.generation_config.temperature=0.7 \
    ++actor_rollout_ref.rollout.generation_config.top_p=0.9 \
    ++actor_rollout_ref.rollout.generation_config.max_new_tokens=512 \
    ++actor_rollout_ref.rollout.generation_config.pad_token_id=151643 \
    ++actor_rollout_ref.rollout.pad_token_id=151643 \
    \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.total_epochs=1 \
    trainer.val_before_train=false \
    trainer.test_freq=0 \
    trainer.save_freq=-1 \
    trainer.project_name="hf_rollout_test" \
    trainer.logger=['console'] \
    \
    algorithm.adv_estimator=grpo

echo "=== Training completed at $(date) ==="

# Clean up
ray stop --force

echo "=== Job finished ==="
