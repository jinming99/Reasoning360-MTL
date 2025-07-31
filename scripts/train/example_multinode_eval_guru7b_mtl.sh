#!/bin/bash
#SBATCH --job-name=eval-guru7b-mtl
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err
#SBATCH --time=48:00:00

# This script evaluates a trained Qwen2.5‑7B model on the GURU offline
# benchmarks.  It supports evaluating both the baseline Mix‑All model
# and the MTL variants (equal weighting or PCGrad).  Set
# MODEL_PATH to the checkpoint directory and MODEL_NAME to a
# descriptive name for logging.

set -euo pipefail

n_nodes=1
n_gpus_per_node=8
gpu_ids=0,1,2,3,4,5,6,7

# Set these before running
MODEL_PATH=${MODEL_PATH:?"Please set MODEL_PATH to the path of your trained model"}
MODEL_NAME=${MODEL_NAME:-"guru7b_mtl"}
DATA_FOLDER=/home/jinming/Reasoning360-MTL/data/offline_eval
SAVE_FOLDER=/home/jinming/Reasoning360-MTL/evaluation_results/test_offline_leaderboard_output

# Leaderboard tasks (same as example script)
leaderboard_list=(
  "aime"
  "math"
  "humaneval"
  "mbpp"
  "livecodebench"
  "gpqa_diamond"
  "supergpqa"
  "arcagi"
  "zebra_puzzle"
  "codeio"
  "cruxeval-i"
  "cruxeval-o"
  "finqa"
  "hitab"
  "multihier"
  "livebench_reasoning"
  "livebench_language"
  "livebench_data_analysis"
  "ifeval"
)

# Domain mapping from original script
declare -A domain_mappings
domain_mappings=(
  [aime]=math
  [math]=math
  [humaneval]=codegen
  [mbpp]=codegen
  [livecodebench]=codegen
  [gpqa_diamond]=stem
  [supergpqa]=stem
  [arcagi]=logic
  [zebra_puzzle]=logic
  [codeio]=simulation
  [cruxeval-i]=simulation
  [cruxeval-o]=simulation
  [finqa]=table
  [hitab]=table
  [multihier]=table
  [livebench_reasoning]=ood
  [livebench_language]=ood
  [livebench_data_analysis]=ood
  [ifeval]=ood
)

# Create save directories
mkdir -p "$SAVE_FOLDER/$MODEL_NAME"
mkdir -p logs
logs_dir=logs/

for leaderboard in "${leaderboard_list[@]}"; do
  domain=${domain_mappings[$leaderboard]}
  # Determine sample counts as in example script
  if [[ "$leaderboard" == "aime" || "$leaderboard" == "aime2025" ]]; then
    n_samples=4
  elif [[ "$leaderboard" == "arcagi1" || "$leaderboard" == "livecodebench" || "$leaderboard" == "humaneval" || "$leaderboard" == "zebra_puzzle_dataset" || "$leaderboard" == "multihier" || "$leaderboard" == "codeio" || "$leaderboard" == "gpqa_diamond" ]]; then
    n_samples=4
  else
    n_samples=1
  fi
  # Generation settings
  batch_size=1024
  temperature=1.0
  top_p=0.7
  top_k=-1
  if [[ "$leaderboard" == "arcagi1" ]]; then
    prompt_length=16384
    response_length=16384
  else
    prompt_length=4096
    response_length=28672
  fi
  tensor_model_parallel_size=4
  gpu_memory_utilization=0.7
  gen_log_file="${logs_dir}${MODEL_NAME}_${leaderboard}_gen.log"
  eval_log_file="${logs_dir}${MODEL_NAME}_${leaderboard}_eval.log"
  # Find the data file
  if [[ "$leaderboard" == "aime" || "$leaderboard" == "aime2025" ]]; then
    file_pattern="${domain}__${leaderboard}_repeated_8x_[0-9a-zA-Z]*.parquet"
  else
    file_pattern="${domain}__${leaderboard}_[0-9a-zA-Z]*.parquet"
  fi
  data_file=$(find "$DATA_FOLDER" -name "$file_pattern" -type f | head -n 1)
  echo "Processing $leaderboard: $data_file" | tee -a "$gen_log_file"
  if [[ -z "$data_file" ]]; then
    echo "No file found matching pattern: $file_pattern. Skipping." | tee -a "$gen_log_file"
    continue
  fi
  file_name=$(basename "$data_file")
  save_path="$SAVE_FOLDER/$MODEL_NAME/$file_name"
  # Generation
  export CUDA_VISIBLE_DEVICES=$gpu_ids
  echo "Starting generation for $leaderboard at $(date)" | tee -a "$gen_log_file"
  {
    ${CONDA_BIN_PATH:-python} -m verl.trainer.main_generation \
      trainer.nnodes=$n_nodes \
      trainer.n_gpus_per_node=$n_gpus_per_node \
      data.path="$data_file" \
      data.prompt_key=prompt \
      data.n_samples=$n_samples \
      data.batch_size=$batch_size \
      data.output_path="$save_path" \
      model.path=$MODEL_PATH \
      +model.trust_remote_code=True \
      rollout.temperature=$temperature \
      rollout.top_k=$top_k \
      rollout.top_p=$top_p \
      rollout.prompt_length=$prompt_length \
      rollout.response_length=$response_length \
      rollout.max_num_batched_tokens=$(($prompt_length + $response_length)) \
      rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
      rollout.gpu_memory_utilization=$gpu_memory_utilization
  } 2>&1 | tee -a "$gen_log_file"
  echo "Completed generation for $leaderboard at $(date)" | tee -a "$gen_log_file"
  # Evaluation
  echo "Starting evaluation for $leaderboard at $(date)" | tee -a "$eval_log_file"
  unset LD_LIBRARY_PATH
  {
    ${CONDA_BIN_PATH:-python} -m verl.trainer.main_eval \
      data.path="$save_path" \
      data.prompt_key=prompt \
      data.response_key=responses \
      data.data_source_key=data_source \
      data.reward_model_key=reward_model
  } 2>&1 | tee -a "$eval_log_file"
  echo "Completed evaluation for $leaderboard at $(date)" | tee -a "$eval_log_file"
done