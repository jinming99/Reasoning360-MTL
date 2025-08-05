#!/usr/bin/env python3
"""
GRPO vs PPO Memory Analysis
Compares memory usage between PPO (with critic) and GRPO (critic-free) training.
"""

def analyze_grpo_memory_savings():
    """Analyze memory savings from switching to GRPO (critic-free) training."""
    
    print("=" * 80)
    print("🎯 GRPO vs PPO Memory Analysis for Qwen2.5-7B")
    print("=" * 80)
    
    # Model memory estimates (7.62B parameters)
    model_params = 7.62e9
    bytes_per_param = 4  # FP32
    base_model_memory_gb = (model_params * bytes_per_param) / (1024**3)
    
    print(f"\n📊 Base Model Memory (Qwen2.5-7B):")
    print(f"   Parameters: {model_params/1e9:.2f}B")
    print(f"   Base Memory: {base_model_memory_gb:.1f}GB")
    
    # PPO (GAE) Memory Usage
    print(f"\n🔴 CURRENT: PPO with GAE (Critic-Based)")
    ppo_components = {
        "Actor Model": base_model_memory_gb * 2.4,  # With gradients, optimizer states
        "Critic Model": base_model_memory_gb * 2.4,  # With gradients, optimizer states  
        "Reference Model": base_model_memory_gb * 1.0,  # Inference only
        "vLLM Rollout": 25.0,  # With KV cache
        "Overhead": 5.0  # Ray, FSDP, misc
    }
    
    ppo_total = sum(ppo_components.values())
    
    for component, memory in ppo_components.items():
        print(f"   {component:15}: {memory:5.1f}GB")
    print(f"   {'TOTAL':15}: {ppo_total:5.1f}GB")
    
    # GRPO Memory Usage
    print(f"\n🟢 PROPOSED: GRPO (Critic-Free)")
    grpo_components = {
        "Actor Model": base_model_memory_gb * 2.4,  # With gradients, optimizer states
        "Critic Model": 0.0,  # ← ELIMINATED!
        "Reference Model": base_model_memory_gb * 1.0,  # Inference only
        "vLLM Rollout": 25.0,  # With KV cache
        "Overhead": 5.0  # Ray, FSDP, misc
    }
    
    grpo_total = sum(grpo_components.values())
    
    for component, memory in grpo_components.items():
        if component == "Critic Model":
            print(f"   {component:15}: {memory:5.1f}GB ← ELIMINATED!")
        else:
            print(f"   {component:15}: {memory:5.1f}GB")
    print(f"   {'TOTAL':15}: {grpo_total:5.1f}GB")
    
    # Savings Analysis
    memory_saved = ppo_total - grpo_total
    percent_saved = (memory_saved / ppo_total) * 100
    
    print(f"\n💰 MEMORY SAVINGS:")
    print(f"   Memory Saved: {memory_saved:.1f}GB")
    print(f"   Percent Saved: {percent_saved:.1f}%")
    print(f"   Reduction: {ppo_total:.1f}GB → {grpo_total:.1f}GB")
    
    # GPU Distribution Analysis
    print(f"\n🖥️  GPU DISTRIBUTION ANALYSIS (8x A100 80GB):")
    gpu_memory = 80.0
    total_gpu_memory = 8 * gpu_memory
    
    print(f"   Total GPU Memory: {total_gpu_memory:.0f}GB")
    print(f"   PPO Usage: {ppo_total:.1f}GB ({ppo_total/total_gpu_memory*100:.1f}%)")
    print(f"   GRPO Usage: {grpo_total:.1f}GB ({grpo_total/total_gpu_memory*100:.1f}%)")
    
    # Memory per GPU
    ppo_per_gpu = ppo_total / 8
    grpo_per_gpu = grpo_total / 8
    
    print(f"\n   Average per GPU:")
    print(f"   PPO: {ppo_per_gpu:.1f}GB/GPU ({ppo_per_gpu/gpu_memory*100:.1f}% utilization)")
    print(f"   GRPO: {grpo_per_gpu:.1f}GB/GPU ({grpo_per_gpu/gpu_memory*100:.1f}% utilization)")
    
    # vLLM Memory Pressure Analysis
    print(f"\n🚀 vLLM MEMORY PRESSURE REDUCTION:")
    vllm_memory = 25.0
    other_ppo_memory = ppo_total - vllm_memory
    other_grpo_memory = grpo_total - vllm_memory
    
    print(f"   vLLM Memory Requirement: {vllm_memory:.1f}GB")
    print(f"   Other Components (PPO): {other_ppo_memory:.1f}GB")
    print(f"   Other Components (GRPO): {other_grpo_memory:.1f}GB")
    print(f"   Pressure Reduction: {other_ppo_memory - other_grpo_memory:.1f}GB")
    
    # Expected Outcomes
    print(f"\n🎯 EXPECTED OUTCOMES:")
    print(f"   ✅ Eliminate critic model loading (~18GB)")
    print(f"   ✅ Reduce GPU memory pressure by {percent_saved:.1f}%")
    print(f"   ✅ More memory available for vLLM KV cache")
    print(f"   ✅ Potential for larger batch sizes")
    print(f"   ✅ More stable training with less OOM risk")
    
    # Algorithm Differences
    print(f"\n📚 ALGORITHM DIFFERENCES:")
    print(f"   PPO (GAE): Uses critic to estimate value function")
    print(f"   GRPO: Uses group-relative advantages (critic-free)")
    print(f"   Performance: GRPO often matches or exceeds PPO performance")
    print(f"   Stability: GRPO can be more stable due to simpler architecture")
    
    print(f"\n" + "=" * 80)
    print(f"🚀 RECOMMENDATION: Switch to GRPO for {memory_saved:.1f}GB memory savings!")
    print(f"=" * 80)

if __name__ == "__main__":
    analyze_grpo_memory_savings()
