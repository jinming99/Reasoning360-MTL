#!/usr/bin/env python3
"""
Comprehensive Memory Analysis for PPO Training
Analyzes GPU memory usage patterns in VERL PPO training
"""

import torch
import psutil
import subprocess
import json
import time
from pathlib import Path

def get_gpu_memory_info():
    """Get detailed GPU memory information"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(', ')
                gpu_info.append({
                    'index': int(parts[0]),
                    'name': parts[1],
                    'memory_total_mb': int(parts[2]),
                    'memory_used_mb': int(parts[3]),
                    'memory_free_mb': int(parts[4]),
                    'utilization_percent': int(parts[5])
                })
        return gpu_info
    except Exception as e:
        return [{'error': str(e)}]

def get_process_memory_info():
    """Get CPU memory information"""
    memory = psutil.virtual_memory()
    return {
        'total_gb': round(memory.total / (1024**3), 2),
        'available_gb': round(memory.available / (1024**3), 2),
        'used_gb': round(memory.used / (1024**3), 2),
        'percent': memory.percent
    }

def estimate_model_memory_requirements():
    """Estimate memory requirements for PPO models"""
    # Qwen2.5-7B parameters: ~7.62B
    # bfloat16: 2 bytes per parameter
    # Additional overhead: ~20% for FSDP, gradients, etc.
    
    base_model_size_gb = 7.62 * 2  # ~15.24 GB in bfloat16
    overhead_factor = 1.2
    
    models = {
        'actor': base_model_size_gb * overhead_factor,
        'critic': base_model_size_gb * overhead_factor,  
        'reference': base_model_size_gb,  # No gradients
        'rollout_vllm': base_model_size_gb + 10  # + KV cache estimate
    }
    
    return models

def analyze_vllm_kv_cache_requirements(max_model_len=512, max_num_seqs=2, tensor_parallel_size=4):
    """Analyze vLLM KV cache memory requirements"""
    # Qwen2.5-7B: 28 layers, 28 attention heads, 4 key-value heads
    num_layers = 28
    num_kv_heads = 4
    head_dim = 128  # 3584 hidden_size / 28 heads
    
    # KV cache per sequence: 2 (K+V) * num_layers * num_kv_heads * head_dim * max_seq_len * 2 bytes (bfloat16)
    kv_cache_per_seq_bytes = 2 * num_layers * num_kv_heads * head_dim * max_model_len * 2
    total_kv_cache_bytes = kv_cache_per_seq_bytes * max_num_seqs
    
    # Distributed across tensor parallel GPUs
    kv_cache_per_gpu_gb = (total_kv_cache_bytes / tensor_parallel_size) / (1024**3)
    
    return {
        'kv_cache_per_seq_mb': round(kv_cache_per_seq_bytes / (1024**2), 2),
        'total_kv_cache_gb': round(total_kv_cache_bytes / (1024**3), 2),
        'kv_cache_per_gpu_gb': round(kv_cache_per_gpu_gb, 2),
        'max_model_len': max_model_len,
        'max_num_seqs': max_num_seqs,
        'tensor_parallel_size': tensor_parallel_size
    }

def create_memory_report():
    """Create comprehensive memory analysis report"""
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'gpu_info': get_gpu_memory_info(),
        'cpu_memory': get_process_memory_info(),
        'model_estimates': estimate_model_memory_requirements(),
        'vllm_analysis': analyze_vllm_kv_cache_requirements(),
        'recommendations': []
    }
    
    # Add recommendations based on analysis
    gpu_info = report['gpu_info']
    if gpu_info and 'error' not in gpu_info[0]:
        total_gpu_memory_gb = sum(gpu['memory_total_mb'] for gpu in gpu_info) / 1024
        model_memory_gb = sum(report['model_estimates'].values())
        
        if model_memory_gb > total_gpu_memory_gb * 0.8:
            report['recommendations'].append("CRITICAL: Estimated model memory exceeds 80% of available GPU memory")
            report['recommendations'].append("Consider: Reduce model size, increase tensor parallelism, or use model offloading")
        
        if report['vllm_analysis']['kv_cache_per_gpu_gb'] > 10:
            report['recommendations'].append("WARNING: KV cache per GPU > 10GB, consider reducing max_model_len or max_num_seqs")
    
    return report

def save_memory_report(report, output_file):
    """Save memory report to file"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Memory analysis report saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    print("🔍 Running comprehensive memory analysis...")
    report = create_memory_report()
    
    # Print summary
    print("\n📊 MEMORY ANALYSIS SUMMARY")
    print("=" * 50)
    
    if report['gpu_info'] and 'error' not in report['gpu_info'][0]:
        print(f"GPUs detected: {len(report['gpu_info'])}")
        for gpu in report['gpu_info']:
            print(f"  GPU {gpu['index']}: {gpu['name']} - {gpu['memory_total_mb']/1024:.1f}GB total, {gpu['memory_used_mb']/1024:.1f}GB used")
    
    print(f"\nEstimated model memory requirements:")
    for model, size_gb in report['model_estimates'].items():
        print(f"  {model}: {size_gb:.1f} GB")
    
    print(f"\nvLLM KV Cache Analysis:")
    vllm = report['vllm_analysis']
    print(f"  KV cache per GPU: {vllm['kv_cache_per_gpu_gb']:.1f} GB")
    print(f"  Total KV cache: {vllm['total_kv_cache_gb']:.1f} GB")
    
    if report['recommendations']:
        print(f"\n⚠️  RECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
    
    # Save detailed report
    save_memory_report(report, "/home/jinming/Reasoning360-MTL/logs/memory_analysis.json")
