# Multinode Test Configuration

This document provides guidance for running full multinode tests with the proper configuration.

## Prerequisites

1. Ensure flash attention is properly installed if you want to use it for better performance
2. Verify that model paths are correctly set to point to the actual model locations
3. Configure tensor parallelism settings according to your GPU setup
4. Ensure training data files are available in the expected locations

## Data Files

The configuration files expect training data to be located at `~/data/rlhf/gsm8k/`. However, in the current setup, the data files are located at `/home/jinming/Reasoning360-MTL/data/train/`.

For the full multinode tests, you may want to either:

1. Update the data paths in the configuration files to point to the actual data location:
   ```yaml
   data:
     train_files: /home/jinming/Reasoning360-MTL/data/train/guru_18k_mix.parquet
     val_files: /home/jinming/Reasoning360-MTL/data/train/guru_18k_mix.parquet
   ```

2. Or create symbolic links to make the expected paths point to the actual data:
   ```bash
   mkdir -p ~/data/rlhf/gsm8k/
   ln -s /home/jinming/Reasoning360-MTL/data/train/guru_18k_mix.parquet ~/data/rlhf/gsm8k/train.parquet
   ln -s /home/jinming/Reasoning360-MTL/data/train/guru_18k_mix.parquet ~/data/rlhf/gsm8k/test.parquet
   ```

The project includes various training data files for different tasks in `/home/jinming/Reasoning360-MTL/data/train/`.

## Configuration Files

There are two main configuration files for PPO training:

### 1. verl/trainer/config/ppo_megatron_trainer.yaml

This configuration file is designed for multinode setups using Megatron strategy and already has appropriate default settings:

- Uses `strategy: megatron` for actor, ref, critic, and reward_model
- `tensor_model_parallel_size: 1` (suitable for single-node multi-GPU setups, can be increased for multi-node)
- Model paths need to be updated from `~/models/deepseek-llm-7b-chat` to actual model locations
- Uses distributed optimizer by default (`use_distributed_optimizer: True`)

Example model path updates:
```yaml
actor_rollout_ref:
  model:
    path: /home/jinming/llm_models/Qwen2.5-7B

critic:
  model:
    path: /home/jinming/llm_models/Qwen2.5-7B
```

### 2. verl/trainer/config/ppo_trainer.yaml

This configuration file uses FSDP strategy and was modified for smoke testing:

- Uses `strategy: fsdp` for actor, ref, critic, and reward_model
- `tensor_model_parallel_size` may need to be adjusted based on your GPU setup
- Model paths have been updated to `/home/jinming/llm_models/Qwen2.5-7B`

## Which Configuration File to Use?

For full multinode tests, you can use either configuration file:

### Option 1: ppo_megatron_trainer.yaml (Megatron strategy)

This configuration file is designed for multinode setups with Megatron strategy:

1. It's specifically designed for multinode setups with Megatron strategy
2. It has better support for tensor parallelism and distributed training
3. It uses distributed optimizer by default which is more memory efficient for large models
4. It's more suitable for the Qwen2-7B model size

### Option 2: ppo_trainer.yaml (FSDP strategy) - RECOMMENDED

Since this configuration has already been debugged and tested with the smoke test, it's the recommended option for full multinode tests. It uses FSDP strategy which has been verified to work correctly with the Qwen2-7B model.

The updated `example_multinode_mtl_qwen2_7b.sh` script now uses `ppo_trainer.yaml` as the base configuration, consistent with the smoke test. It also uses a subset of the training data (`math.parquet`) for validation during training.

The main adjustments needed for full multinode tests with this configuration are:

1. Adjust `tensor_model_parallel_size` according to your GPU setup (currently set to 1 for smoke test)
2. Adjust `nnodes` and `n_gpus_per_node` in the trainer section according to your cluster setup (currently set to 1 node with 8 GPUs)
3. Ensure model paths are correctly set (already updated to `/home/jinming/llm_models/Qwen2.5-7B`)
4. Verify environment variables for flash attention

For a typical multinode setup with multiple nodes, you would adjust these settings:

```yaml
trainer:
  nnodes: 2  # or however many nodes you're using
  n_gpus_per_node: 8  # or however many GPUs per node

actor_rollout_ref:
  rollout:
    tensor_model_parallel_size: 2  # or appropriate value for your setup
```

Both options should work, but `ppo_trainer.yaml` with FSDP strategy is recommended since it has already been debugged.

## Environment Variables

For full multinode tests, consider the following environment variable settings:

If flash attention is available and you want to use it:
```bash
# Uncomment or remove these lines to enable flash attention
# export FLASH_ATTENTION_DISABLE=1
# export FLASH_ATTENTION_DISABLED=1
```

If flash attention is not available or you want to disable it:
```bash
export FLASH_ATTENTION_DISABLE=1
export FLASH_ATTENTION_DISABLED=1
```

## Running Tests

### 1. Training

```bash
# For multinode RL training
sbatch run_multinode_rl_job.sh

# For MTL training (if applicable)
# Check if there's a specific MTL training script
```

### 2. Evaluation

```bash
# Set the model path before running evaluation
export MODEL_PATH=/path/to/your/trained/model
sbatch example_multinode_eval_guru7b_mtl.sh
```

## Troubleshooting

### Flash Attention Issues

If you encounter flash attention import errors:

1. Check if flash attention is properly installed in your environment
2. Verify that CUDA versions are compatible
3. Consider disabling flash attention by setting the environment variables mentioned above

### Model Path Issues

If you encounter model loading errors:

1. Verify that the model paths in configuration files point to existing model directories
2. Check that the model files are accessible and not corrupted
3. Ensure that the model architecture matches what's specified in the configuration

### Tensor Parallelism Issues

If you encounter tensor parallelism errors:

1. Verify that the number of GPUs matches the tensor_model_parallel_size setting
2. Check that all GPUs have sufficient memory for the model
3. Consider reducing tensor_model_parallel_size if you have fewer GPUs available
