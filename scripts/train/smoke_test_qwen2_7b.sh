#!/usr/bin/env bash
###############################################################################
# Single‑GPU smoke test for multi‑task PPO (Qwen2‑7B).                        #
# Runs 10 optimisation steps on math + logic sub‑datasets, with PCGrad,       #
# and writes checkpoints under ./checkpoints/smoke_test.                      #
# Use identical flags/structure to the full Slurm script so code paths match. #
###############################################################################

set -euo pipefail

# --------------------------------------------------------------------------- #
# 1. User‑editable paths                                                      #
# --------------------------------------------------------------------------- #
# Disable flash attention at the environment level
export FLASH_ATTENTION_DISABLE=1
export FLASH_ATTENTION_DISABLED=1

# Unset conflicting GPU environment variables
unset ROCR_VISIBLE_DEVICES 2>/dev/null || true
unset HIP_VISIBLE_DEVICES 2>/dev/null || true

# Disable WandB completely for smoke test
export WANDB_MODE=disabled
export WANDB_DISABLED=true

# Let SLURM handle GPU allocation - remove restrictive CUDA_VISIBLE_DEVICES setting
# export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Automatically detect number of available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)
echo "Detected $NUM_GPUS GPUs available"

# Absolute paths are safer than $PWD inside YAML interpolations
PROJECT_ROOT=$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")
DATA_DIR="$PROJECT_ROOT/../data/train/guru_18k"  # Go up one more level from scripts/ to project root
MODEL_DIR="/home/jinming/llm_models/Qwen2.5-7B"              # adjust if needed

# Hydra/WandB logs will land here
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints/smoke_test"

# Activate conda environment
# Use current virtual environment (assumes setup_arc_env.sh was run)
PYTHON_BIN="/home/jinming/Reasoning360-MTL/venv_reasoning360mtl/bin/python"
export RAY_PYTHON_EXECUTABLE="$PYTHON_BIN"
RAY_BIN="$(dirname "$PYTHON_BIN")/ray"
if [ ! -x "$RAY_BIN" ]; then
  # Fallback to invoking Ray via Python module
  RAY_BIN="$PYTHON_BIN -m ray"
fi

# Skip Ray startup if user provides external cluster address
if [ -n "${RAY_ADDRESS:-}" ]; then
  SKIP_RAY_START=1
fi

# Check for required Python packages
REQUIRED_PKGS=("numpy" "torch" "transformers" "ray" "pandas" "numba")
PIP_BIN="/home/jinming/Reasoning360-MTL/venv_reasoning360mtl/bin/pip"
for pkg in "${REQUIRED_PKGS[@]}"; do
    if ! $PYTHON_BIN -c "import $pkg" &> /dev/null; then
        echo "Installing missing package: $pkg"
        $PIP_BIN install $pkg || {
            echo "Error: Failed to install $pkg"
            exit 1
        }
    fi
done

# --------------------------------------------------------------------------- #
# 2. Safety checks                                                            #
# --------------------------------------------------------------------------- #
for f in math logic; do
  test -f "$DATA_DIR/$f.parquet" \
    || { echo "Missing $DATA_DIR/$f.parquet"; exit 1; }
done
test -d "$MODEL_DIR" || { echo "Missing model dir $MODEL_DIR"; exit 1; }

# --------------------------------------------------------------------------- #
# 3. Temporary Hydra override file                                            #
# --------------------------------------------------------------------------- # Create a temporary YAML file for train_files
TMP_OVR=$(mktemp /tmp/mtl_smoke.XXXX.yaml)

# Ensure MODEL_DIR is absolute and expand ~ if present
MODEL_DIR=$(realpath -m "$MODEL_DIR")

cat >"$TMP_OVR" <<YAML
# Injected overrides for smoke test
data:
  train_files:
    math: "$DATA_DIR/math.parquet"
    logic: "$DATA_DIR/logic.parquet"
  tokenizer: "$MODEL_DIR"
  trust_remote_code: true
  max_prompt_length: 128
  max_response_length: 128
  train_batch_size: 4
  val_batch_size: 4
  filter_overlong_prompts: false
  return_raw_chat: true
  return_full_prompt: true

actor_rollout_ref:
  model:
    path: "$MODEL_DIR"
    input_tokenizer: "$MODEL_DIR"
    trust_remote_code: true
    use_shm: false
    use_fused_kernels: false
  rollout:
    name: hf
    n: 1
    temperature: 0.7
    top_p: 1.0
    top_k: -1
    max_tokens: 128
    stop_token_ids: null
    stop: []
    tensor_model_parallel_size: 1

trainer:
  n_gpus_per_node: $NUM_GPUS
  nnodes: 1
  total_epochs: 1
  total_training_steps: 10
  default_local_dir: "$CHECKPOINT_DIR"
  logger: ["console"]
  val_before_train: false
  log_freq: 1
  eval_freq: 10
  save_freq: 1000
  balance_batch: true
  device: cuda
  resume_from_path: null
  resume_mode: auto

mtl:
  enabled: true
  method: pcgrad

model:
  actor_weights_name_or_path: "$MODEL_DIR"
  critic_weights_name_or_path: "$MODEL_DIR"
  tokenizer_name_or_path: "$MODEL_DIR"
  trust_remote_code: true

optimizer:
  actor_learning_rate: 2e-6
  critic_learning_rate: 2e-6
  weight_decay: 0.01
  adam_epsilon: 1e-8
  max_grad_norm: 1.0

ray_init:
  address: auto
  num_cpus: 4
  num_gpus: $NUM_GPUS
YAML

# Ensure we remove the temporary file even on Ctrl‑C
# Updated trap to only stop Ray if we started it
trap 'rm -f "$TMP_OVR"; [ -z "${SKIP_RAY_START:-}" ] && $RAY_BIN stop >/dev/null 2>&1 || true' EXIT INT TERM

# --------------------------------------------------------------------------- #
# 4. Start a throw‑away Ray head                                              #
# --------------------------------------------------------------------------- #
# Get the Ethernet IP address (not InfiniBand)
HOST_IP=$(hostname --ip-address | awk '{print $1}')

if [ -z "${SKIP_RAY_START:-}" ]; then
  echo "Starting Ray head on $HOST_IP:6379..."
  $RAY_BIN stop >/dev/null 2>&1 || true
  $RAY_BIN start --head \
                 --node-ip-address="$HOST_IP" \
                 --port=6379 \
                 --num-cpus=4 \
                 --num-gpus=$NUM_GPUS \
                 --include-dashboard=false &
  sleep 5
  export RAY_ADDRESS=${HOST_IP}:6379
  echo "Ray head started at $RAY_ADDRESS"
  
  # Verify Ray is running
  if ! $RAY_BIN status >/dev/null 2>&1; then
    echo "Error: Ray failed to start properly"
    exit 1
  fi
fi

# --------------------------------------------------------------------------- #
# 5. Launch training                                                          #
# --------------------------------------------------------------------------- #
# Below we pass a small set of overrides that are REQUIRED for the trainer to
# start.  The values come from sensible defaults used in
#   - verl/trainer/config/ppo_megatron_trainer_edited.yaml  (micro-batch sizes)
#   - verl/trainer/config/generation.yaml                  (log-prob micro-batch)
# They are kept here in-line so that anyone reading this script sees exactly
# which hyper-parameters differ from the base `ppo_trainer.yaml`.
#   data.train_files / data.val_files  – point to the GURU-18K subsets we created
#   data.train_batch_size             – minimal batch divisible by mini-batches
#   actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu   – per-GPU micro-batch
#   actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu – micro-batch used
#        when re-computing log-probs for KL / PPO loss
#   actor_rollout_ref.rollout.n       – number of responses per prompt (1 for smoke)
#   model & tokenizer paths           – local Qwen2.5-7B checkout
# If you need to change GPUs or dataset size, adjust *train_batch_size* and both
# micro-batch sizes accordingly (they must cleanly divide the global batch).

# Export the config path as an environment variable for Hydra to pick up
export HYDRA_CONFIG_PATH="$TMP_OVR"

# If RAY_ADDRESS is set, pass it to the trainer
RAY_INIT_OVERRIDE=""
if [ -n "${RAY_ADDRESS:-}" ]; then
  RAY_INIT_OVERRIDE="+ray_init.address=$RAY_ADDRESS"
fi

$PYTHON_BIN -m verl.trainer.main_ppo \
  --config-name=ppo_trainer \
  actor_rollout_ref.model.path="$MODEL_DIR" \
  data.tokenizer="$MODEL_DIR" \
  data.train_files="[/home/jinming/Reasoning360-MTL/data/train/guru_18k/math.parquet,/home/jinming/Reasoning360-MTL/data/train/guru_18k/logic.parquet]" \
  data.val_files="[/home/jinming/Reasoning360-MTL/data/train/guru_18k/math.parquet]" \
  data.max_prompt_length=128 data.max_response_length=128 \
  data.train_batch_size=4 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  critic.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.n=1 \
  trainer.n_gpus_per_node=$NUM_GPUS \
  trainer.val_before_train=false \
  trainer.test_freq=0 \
  +wandb.mode=disabled \
  ++trainer.ray_wait_register_center_timeout=300 \
  ++actor_rollout_ref.model.enable_flash_attention=false \
  ++critic.model.enable_flash_attention=false \
  $RAY_INIT_OVERRIDE

echo -e "\nSmoke test finished successfully 🎉"