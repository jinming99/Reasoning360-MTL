# Performance Optimization Guide for 8 A100 Training

## 🔍 Performance Analysis Results

### Current Bottlenecks Identified:
1. **Model Flops Utilization (MFU)**: 1.7-4.3% (should be 40-60%)
2. **Throughput**: 67-83 tokens/second (should be 2000+ for 8 A100s)
3. **Memory Utilization**: 72GB/94GB (underutilized)

### Root Causes:
1. **Small Batch Sizes**: Original batch_size=4 is too small for A100s
2. **Short Sequences**: 128 tokens too short for efficient A100 utilization
3. **CPU Bottleneck**: Data loading and reward computation overhead

## 🚀 Applied Optimizations

### 1. Batch Size Scaling
```bash
# Original (inefficient):
data.train_batch_size=4
ppo_micro_batch_size_per_gpu=1

# Optimized for 8 A100s:
data.train_batch_size=128        # 32x larger
ppo_micro_batch_size_per_gpu=4   # 4x larger
```

### 2. Sequence Length Optimization
```bash
# Original (too short):
max_prompt_length=128
max_response_length=128

# Optimized for A100 efficiency:
max_prompt_length=256
max_response_length=512
```

### 3. Multi-Response Training
```bash
# Original:
rollout.n=1

# Optimized:
rollout.n=4  # Multiple responses per prompt
```

### 4. Memory Optimization
```bash
# Increased SLURM allocation:
--mem=256G              # Double memory
--cpus-per-task=48      # Double CPUs
```

## 📊 Expected Performance Improvements

| Metric | Original | Optimized | Improvement |
|--------|----------|-----------|-------------|
| **Batch Size** | 4 | 128 | 32x |
| **Sequence Length** | 256 | 768 | 3x |
| **GPU Utilization** | 1.7% | 40-60% | 24-35x |
| **Throughput** | 83 tok/s | 2000+ tok/s | 24x |
| **Training Time** | 4.5 days | 4-6 hours | 18-27x |

## 🎯 Experiment Setup

### Experiment 1: GURU Baseline
- **Approach**: Mixed domains (reproduce paper)
- **MTL**: Disabled
- **Data**: All 6 domains combined
- **Checkpoints**: `/checkpoints/guru_baseline`

### Experiment 2: MTL with PCGrad
- **Approach**: Multi-task learning
- **MTL**: Enabled with PCGrad
- **Data**: Task-specific loading
- **Checkpoints**: `/checkpoints/mtl_pcgrad`

## 🔧 Technical Details

### Ray Configuration
```bash
--num-cpus=48           # Match SLURM allocation
--num-gpus=8            # All available A100s
--include-dashboard=false  # Avoid startup issues
```

### FSDP Configuration
```bash
strategy="fsdp"         # Fully Sharded Data Parallel
enable_gradient_checkpointing=True
```

### WandB Logging
```bash
project_name="Reasoning360-MTL"
experiment_name="guru-baseline-8gpu-TIMESTAMP"
experiment_name="mtl-pcgrad-8gpu-TIMESTAMP"
```

## 📈 Monitoring

### GPU Utilization
```bash
# Check GPU usage:
nvidia-smi

# Expected: >80% GPU utilization
# Expected: >60GB memory per GPU
```

### Training Progress
```bash
# Monitor logs:
tail -f guru_baseline_JOBID.out
tail -f mtl_pcgrad_JOBID.out

# Expected: >1000 tokens/second throughput
# Expected: >40% MFU
```

### WandB Dashboard
- **URL**: https://wandb.ai/jin-ming-vt/Reasoning360-MTL
- **Metrics to watch**:
  - `perf/throughput` (should be >1000)
  - `perf/mfu/actor` (should be >40%)
  - `training/global_step` (progress rate)

## 🚨 Troubleshooting

### If Still Slow:
1. **Check GPU utilization**: `nvidia-smi`
2. **Increase batch size further**: Try 256 or 512
3. **Check CPU bottleneck**: Monitor data loading time
4. **Verify Ray cluster**: `ray status`

### If OOM Errors:
1. **Reduce micro_batch_size**: From 4 to 2
2. **Enable gradient checkpointing**: Already enabled
3. **Reduce sequence length**: Back to 256/384

### If Ray Issues:
1. **Restart Ray**: `ray stop && ray start`
2. **Check network**: Verify node connectivity
3. **Increase timeout**: `ray_wait_register_center_timeout=600`

## 🎯 Success Criteria

### Performance Targets:
- **Throughput**: >1000 tokens/second
- **MFU**: >40% for actor, >30% for critic
- **Training Time**: <6 hours for 3 epochs
- **GPU Utilization**: >80%

### Quality Targets:
- **Convergence**: Stable loss reduction
- **Validation**: Improving task performance
- **Checkpoints**: Successful saving every 500 steps

## 📝 Next Steps

1. **Launch experiments**: `bash launch_parallel_experiments.sh`
2. **Monitor progress**: Check WandB and logs
3. **Compare results**: Baseline vs MTL performance
4. **Scale further**: Consider multi-node if needed
