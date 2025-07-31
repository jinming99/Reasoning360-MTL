# Full Multinode Test Readiness

This document summarizes the changes made to prepare for full multinode tests and what still needs to be done.

## Changes Already Made

### 1. Flash Attention Handling

- Environment variables set in smoke test script to disable flash attention:
  ```bash
  export FLASH_ATTENTION_DISABLE=1
  export FLASH_ATTENTION_DISABLED=1
  ```
- Conditional imports with fallbacks implemented in:
  - `verl/workers/critic/dp_critic.py`
  - `verl/workers/actor/dp_actor.py`
- Explicit `attn_implementation="eager"` set in model loading in `verl/workers/fsdp_workers.py`

### 2. Model Paths Updated

- `verl/trainer/config/ppo_trainer.yaml`:
  - `actor_rollout_ref.model.path`: `/home/jinming/llm_models/Qwen2.5-7B`
  - `critic.model.path`: `/home/jinming/llm_models/Qwen2.5-7B`

- `verl/trainer/config/ppo_megatron_trainer.yaml`:
  - `actor_rollout_ref.model.path`: `/home/jinming/llm_models/Qwen2.5-7B`
  - `critic.model.path`: `/home/jinming/llm_models/Qwen2.5-7B`
  - `reward_model.model.path`: `/home/jinming/llm_models/Qwen2.5-7B`

### 3. Documentation Created

- `SMOKE_TEST_MODIFICATIONS.md`: Documents all changes made for smoke test
- `README_MULTINODE.md`: Guidance for running full multinode tests

## Configuration for Full Multinode Tests

### Using ppo_trainer.yaml (Recommended)

The `example_multinode_mtl_qwen2_7b.sh` script now uses `ppo_trainer.yaml` as the base configuration, consistent with the smoke test. The following settings in `verl/trainer/config/ppo_trainer.yaml` need to be adjusted for full multinode tests:

1. **Trainer Settings** (lines 273-274):
   ```yaml
   nnodes: 1          # Change to number of nodes in your cluster
   n_gpus_per_node: 8 # Change to number of GPUs per node
   ```

2. **Tensor Parallelism** (line 117):
   ```yaml
   actor_rollout_ref:
     rollout:
       tensor_model_parallel_size: 1  # Change according to your GPU setup
   ```

3. **Environment Variables**:
   If flash attention is available and you want to use it, remove or comment out:
   ```bash
   # export FLASH_ATTENTION_DISABLE=1
   # export FLASH_ATTENTION_DISABLED=1
   ```

4. **Validation Data**:
   The script now uses a subset of the training data (`math.parquet`) for validation during training, consistent with the smoke test.

### Using ppo_megatron_trainer.yaml (Alternative)

Similar adjustments needed:

1. **Trainer Settings** (lines 310-311):
   ```yaml
   nnodes: 1          # Change to number of nodes in your cluster
   n_gpus_per_node: 8 # Change to number of GPUs per node
   ```

2. **Tensor Parallelism** (multiple locations):
   ```yaml
   actor_rollout_ref:
     actor:
       megatron:
         tensor_model_parallel_size: 1  # Change according to your setup
   
   critic:
     megatron:
       tensor_model_parallel_size: 1  # Change according to your setup
   
   reward_model:
     megatron:
       tensor_model_parallel_size: 1  # Change according to your setup
   ```

## Running Full Multinode Tests

1. Adjust configuration settings as described above
2. Set appropriate environment variables
3. Run the multinode training script:
   ```bash
   sbatch run_multinode_rl_job.sh
   ```

4. For evaluation:
   ```bash
   export MODEL_PATH=/path/to/your/trained/model
   sbatch example_multinode_eval_guru7b_mtl.sh
   ```

## Verification Steps

Before running full tests, verify:

1. Model files exist at `/home/jinming/llm_models/Qwen2.5-7B`
2. Flash attention installation status and desired usage
3. Cluster configuration matches settings in YAML files
4. Required data files are available in the expected locations:
   - Either update data paths in configuration files to point to actual data location
   - Or create symbolic links to make expected paths point to actual data:
     ```bash
     mkdir -p ~/data/rlhf/gsm8k/
     ln -s /home/jinming/Reasoning360-MTL/data/train/guru_18k_mix.parquet ~/data/rlhf/gsm8k/train.parquet
     ln -s /home/jinming/Reasoning360-MTL/data/train/guru_18k_mix.parquet ~/data/rlhf/gsm8k/test.parquet
     ```
5. Environment variables are set appropriately

## Rollback Plan

If issues occur, you can revert to the original configuration:

1. Restore original YAML files from backups (if created)
2. Revert environment variable settings
3. Refer to `SMOKE_TEST_MODIFICATIONS.md` for all changes made
