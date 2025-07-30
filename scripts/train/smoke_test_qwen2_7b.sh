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
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Absolute paths are safer than $PWD inside YAML interpolations
PROJECT_ROOT=$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")
DATA_DIR="$PROJECT_ROOT/../data/train/guru_18k"  # Go up one more level from scripts/ to project root
MODEL_DIR="$HOME/models/Qwen2.5-7B"              # adjust if needed

# Hydra/WandB logs will land here
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints/smoke_test"

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate Reasoning360 || {
    echo "Error: Failed to activate conda environment 'Reasoning360'"
    exit 1
}

# Set paths to conda binaries
RAY_BIN="$CONDA_PREFIX/bin/ray"
PYTHON_BIN="$CONDA_PREFIX/bin/python"

# Check for required Python packages
REQUIRED_PKGS=("numpy" "torch" "transformers" "ray")
for pkg in "${REQUIRED_PKGS[@]}"; do
    if ! $PYTHON_BIN -c "import $pkg" &> /dev/null; then
        echo "Installing missing package: $pkg"
        pip install $pkg || {
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
    n: 1
    temperature: 0.7
    top_p: 1.0
    top_k: -1
    max_tokens: 128
    stop_token_ids: null
    stop: []

trainer:
  n_gpus_per_node: 1
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
  num_gpus: 1
YAML

# Ensure we remove the temporary file even on Ctrl‑C
trap 'rm -f "$TMP_OVR"; $RAY_BIN stop >/dev/null 2>&1 || true' EXIT INT TERM

# --------------------------------------------------------------------------- #
# 4. Start a throw‑away Ray head                                              #
# --------------------------------------------------------------------------- #
$RAY_BIN stop >/dev/null 2>&1 || true
$RAY_BIN start --head --num-cpus=4 --num-gpus=1 --block &   # --block keeps the daemon in background
sleep 5

# --------------------------------------------------------------------------- #
# 5. Launch training                                                          #
# --------------------------------------------------------------------------- #
# Export the config path as an environment variable for Hydra to pick up
export HYDRA_CONFIG_PATH="$TMP_OVR"

$PYTHON_BIN -m verl.trainer.main_ppo \
  --config-name=ppo_trainer \
  data.max_prompt_length=128 data.max_response_length=128 \
  data.train_batch_size=4 \
  actor_rollout_ref.rollout.n=1 \
  +wandb.mode=disabled

echo -e "\nSmoke test finished successfully 🎉"
