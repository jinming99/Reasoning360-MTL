# vLLM KV Cache Memory Solutions

## 🚨 Critical Issue: vLLM KV Cache Allocation Failure

**Error**: `ValueError: No available memory for the cache blocks. Try increasing gpu_memory_utilization when initializing the engine.`

**Root Cause**: Memory fragmentation and competition between actor model and vLLM rollout on shared GPUs, exacerbated by vLLM V1 engine memory requirements.

## 🔧 Comprehensive Solutions Implemented

### Solution 1: Force vLLM V0 Engine (Primary Fix)

**Key Discovery**: vLLM V1 engine has significantly higher memory requirements than V0 engine.

```bash
# Critical environment variable - forces V0 engine
export VLLM_USE_V1=0
```

**Evidence from Community**:
- GitHub Issue #2248: V1 engine fails where V0 succeeds with same settings
- Discussion #15842: V1 requires higher `gpu_memory_utilization` than V0
- V0 engine is more memory-efficient for shared GPU scenarios

**Implementation**: `vllm_memory_fix_8gpu_training.sh`

### Solution 2: Ultra-Conservative vLLM Settings

```yaml
rollout.vllm.gpu_memory_utilization=0.02  # Extremely low (2%)
rollout.vllm.max_model_len=64             # Minimal context
rollout.vllm.max_num_seqs=1               # Single sequence
rollout.vllm.enforce_eager=true           # Disable CUDA graphs
rollout.vllm.enable_prefix_caching=false  # Disable caching
rollout.vllm.swap_space=0                 # No swap
rollout.vllm.cpu_offload_gb=0             # No CPU offload
```

### Solution 3: Alternative HuggingFace Rollout

**Rationale**: Completely avoid vLLM memory issues by using HuggingFace transformers for rollout.

```yaml
rollout.name=hf  # Switch from vllm to hf
rollout.hf.model_hf_config.use_cache=false
rollout.hf.model_hf_config.attn_implementation=eager
```

**Benefits**:
- No KV cache allocation issues
- More predictable memory usage
- Better integration with FSDP models
- Simpler debugging

**Implementation**: `huggingface_rollout_8gpu_training.sh`

### Solution 4: Tensor Parallelism for Memory Distribution

```yaml
rollout.tensor_model_parallel_size=2  # Split model across 2 GPUs
```

**Benefits**:
- Distributes model weights across multiple GPUs
- Reduces per-GPU memory pressure
- Enables larger models on same hardware

## 📊 Memory Analysis

### Current Memory Distribution (8 GPUs, GRPO):
- **Actor Model**: ~68GB (FSDP distributed)
- **Reference Model**: ~28GB (FSDP distributed) 
- **vLLM Rollout**: ~25GB per GPU (problematic)
- **Total per GPU**: ~15-20GB (without vLLM issues)

### Memory Pressure Points:
1. **GPU Memory Fragmentation**: Actor model loads first, fragments memory
2. **vLLM V1 Engine**: Higher memory overhead than V0
3. **KV Cache Allocation**: Fails when insufficient contiguous memory
4. **Shared GPU Usage**: Actor and rollout compete for same GPU memory

## 🎯 Recommended Testing Order

### 1. Test vLLM V0 Engine Fix (Highest Priority)
```bash
sbatch /home/jinming/Reasoning360-MTL/scripts/train/vllm_memory_fix_8gpu_training.sh
```

**Expected Outcome**: V0 engine should resolve KV cache allocation with same settings that fail in V1.

### 2. Test HuggingFace Rollout Alternative
```bash
sbatch /home/jinming/Reasoning360-MTL/scripts/train/huggingface_rollout_8gpu_training.sh
```

**Expected Outcome**: Complete elimination of vLLM memory issues, stable training.

### 3. Monitor and Compare
- Memory usage patterns
- Training stability
- Performance differences
- Error rates

## 🔍 Debugging Commands

### Check vLLM Version and Engine
```python
import vllm
print(f"vLLM version: {vllm.__version__}")
print(f"Using V1 engine: {os.getenv('VLLM_USE_V1', '1') == '1'}")
```

### Monitor GPU Memory During Training
```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Memory usage logging
nvidia-smi --query-gpu=timestamp,memory.used,memory.free --format=csv -l 1 > gpu_memory.log
```

### Ray Memory Monitoring
```python
import ray
print(ray.cluster_resources())
print(ray.available_resources())
```

## 📈 Success Metrics

### Immediate Success Indicators:
- ✅ vLLM workers initialize without KV cache errors
- ✅ Training progresses beyond rollout phase
- ✅ Stable GPU memory usage (no OOM crashes)
- ✅ Consistent batch processing

### Long-term Success Indicators:
- ✅ Complete training epochs without memory issues
- ✅ Scalable to larger batch sizes
- ✅ Reproducible results across runs
- ✅ Production-ready stability

## 🚀 Next Steps After Resolution

1. **Optimize Performance**: Once stable, tune batch sizes and sequence lengths
2. **Scale Testing**: Test with larger models and datasets
3. **Production Deployment**: Implement monitoring and alerting
4. **Documentation**: Create operational runbooks

## 📚 References

- [vLLM Issue #2248](https://github.com/vllm-project/vllm/issues/2248): V1 engine memory issues
- [vLLM Discussion #15842](https://github.com/vllm-project/vllm/discussions/15842): V0 vs V1 comparison
- [vLLM Optimization Guide](https://docs.vllm.ai/en/latest/configuration/optimization.html): Memory reduction strategies
- VERL Documentation: Rollout configuration options
