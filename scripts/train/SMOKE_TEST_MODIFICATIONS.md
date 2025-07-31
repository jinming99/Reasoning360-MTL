# Smoke Test Modifications

This document tracks all modifications made to run the smoke test successfully. These changes may need to be considered or reverted when running full multinode tests.

## 1. Flash Attention Issues

### Problem
The transformers library was eagerly importing flash_attn modules even when flash attention was disabled, causing ImportError when the package was not installed.

### Solutions Applied

1. **Environment-level disabling** in `smoke_test_qwen2_7b.sh`:
   ```bash
   export FLASH_ATTENTION_DISABLE=1
   export FLASH_ATTENTION_DISABLED=1
   ```

2. **Conditional imports with fallbacks** in `verl/workers/critic/dp_critic.py` and `verl/workers/actor/dp_actor.py`:
   - Replaced direct imports with try/except blocks
   - Added fallback functions that raise NotImplementedError when flash_attn is not available

3. **Explicit attn_implementation setting** in `verl/workers/fsdp_workers.py`:
   - Set `attn_implementation="eager"` in all AutoConfig.from_pretrained and model loading calls

## 2. Model Path Issues

### Problem
Configuration file had incorrect model paths pointing to `~/models/deepseek-llm-7b-chat` instead of the actual Qwen2.5-7B model location.

### Solution
Updated `verl/trainer/config/ppo_trainer.yaml`:
- Changed `actor_rollout_ref.model.path` to `/home/jinming/llm_models/Qwen2.5-7B`
- Changed `critic.model.path` to `/home/jinming/llm_models/Qwen2.5-7B`

Note: The `verl/trainer/config/ppo_megatron_trainer.yaml` configuration file also has the same model path issue and may need similar updates for full multinode tests.

## 3. Tensor Parallelism Configuration

### Problem
Configuration had `tensor_model_parallel_size: 2` which requires at least 2 GPUs, but smoke test runs on a single GPU.

### Solution
Changed `tensor_model_parallel_size: 1` in `verl/trainer/config/ppo_trainer.yaml`

Note: The `verl/trainer/config/ppo_megatron_trainer.yaml` configuration file already has `tensor_model_parallel_size: 1` which is appropriate.

## Considerations for Full Multinode Tests

When running full multinode tests (`example_multinode_mtl_qwen2_7b.sh`, `run_multinode_rl_job.sh`, `example_multinode_eval_guru7b_mtl.sh`), consider:

1. **Flash Attention**: If flash attention is available in the full environment, you may want to revert the conditional imports and use the actual flash_attn implementations for better performance.

2. **Tensor Parallelism**: Full multinode tests likely require `tensor_model_parallel_size: 2` or higher, depending on the GPU configuration. The `ppo_megatron_trainer.yaml` configuration already has appropriate settings for single-node multi-GPU setups.

3. **Model Paths**: Ensure model paths in configuration files match the actual model locations in the full test environment. Both `ppo_trainer.yaml` and `ppo_megatron_trainer.yaml` may need updates.

4. **Configuration Consistency**: Ensure all related configuration files are consistent with the full test setup.

5. **Environment Variables**: The full multinode tests may require different environment variable settings than the smoke test. Check if `FLASH_ATTENTION_DISABLE` and `FLASH_ATTENTION_DISABLED` should be set or unset based on the availability of flash attention in the full environment.
