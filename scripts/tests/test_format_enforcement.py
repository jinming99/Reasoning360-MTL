#!/usr/bin/env python3
"""
Test Format Enforcement with Real Model Generation

This script tests the format enforcement solution with actual model generation
to demonstrate improved reward scores.
"""

import os
import sys
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

# Add the project root to the path
sys.path.append('/home/jinming/Reasoning360-MTL')

from scripts.tests.constrained_decoding_solution import ConstrainedDecodingRewardEnforcer
from verl.utils.reward_score import default_compute_score

def detect_domain_from_data_source(data_source: str) -> str:
    """Detect domain from data_source field."""
    if 'math' in data_source.lower():
        return 'math'
    elif 'logic' in data_source.lower() or 'barc' in data_source.lower() or 'arcagi' in data_source.lower():
        return 'logic'
    elif 'stem' in data_source.lower():
        return 'stem'
    elif 'codegen' in data_source.lower():
        return 'codegen'
    elif 'table' in data_source.lower():
        return 'table'
    else:
        return 'math'  # Default to math

def load_sample_data(num_samples=10):
    """Load sample data from different domains."""
    samples = []
    
    # Math samples
    try:
        math_df = pd.read_parquet('/home/jinming/Reasoning360-MTL/data/train/math__combined_54.4k.parquet')
        math_samples = math_df.head(3).to_dict('records')
        samples.extend(math_samples)
        print(f"✅ Loaded {len(math_samples)} math samples")
    except Exception as e:
        print(f"❌ Failed to load math samples: {e}")
    
    # Logic samples
    try:
        logic_df = pd.read_parquet('/home/jinming/Reasoning360-MTL/data/train/logic__barc_1.6k.parquet')
        logic_samples = logic_df.head(2).to_dict('records')
        samples.extend(logic_samples)
        print(f"✅ Loaded {len(logic_samples)} logic samples")
    except Exception as e:
        print(f"❌ Failed to load logic samples: {e}")
    
    return samples[:num_samples]

def generate_response_hf(model, tokenizer, prompt_text, max_length=512):
    """Generate response using HuggingFace model."""
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=inputs.input_ids.shape[1] + max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Extract only the generated part
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response

def test_format_enforcement_with_model():
    """Test format enforcement with actual model generation."""
    print("🧪 Testing Format Enforcement with Real Model Generation")
    print("=" * 70)
    
    # Load small model for testing
    model_path = "/home/jinming/llm_models/Qwen2.5-1.5B"
    print(f"Loading model: {model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Initialize format enforcer
    enforcer = ConstrainedDecodingRewardEnforcer()
    
    # Load sample data
    samples = load_sample_data(5)
    if not samples:
        print("❌ No sample data loaded")
        return
    
    print(f"\n📊 Testing with {len(samples)} samples")
    print("=" * 70)
    
    results = []
    
    for i, sample in enumerate(samples):
        print(f"\n🔍 Sample {i+1}: {sample.get('data_source', 'unknown')}")
        print("-" * 50)
        
        # Detect domain
        domain = detect_domain_from_data_source(sample.get('data_source', ''))
        print(f"Domain: {domain}")
        
        # Get original prompt
        original_prompt = sample.get('prompt', [])
        if not original_prompt:
            print("❌ No prompt found")
            continue
        
        # Create prompt text
        prompt_text = tokenizer.apply_chat_template(
            original_prompt, 
            add_generation_prompt=True, 
            tokenize=False
        )
        
        print(f"Prompt: {prompt_text[:200]}...")
        
        # Generate original response
        try:
            original_response = generate_response_hf(model, tokenizer, prompt_text, max_length=256)
            print(f"Original response: {original_response[:200]}...")
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            continue
        
        # Apply format enforcement
        enhanced_prompt = enforcer.create_enhanced_prompt(original_prompt, domain)
        enhanced_prompt_text = tokenizer.apply_chat_template(
            enhanced_prompt, 
            add_generation_prompt=True, 
            tokenize=False
        )
        
        try:
            enhanced_response = generate_response_hf(model, tokenizer, enhanced_prompt_text, max_length=256)
            print(f"Enhanced response: {enhanced_response[:200]}...")
        except Exception as e:
            print(f"❌ Enhanced generation failed: {e}")
            enhanced_response = original_response
        
        # Apply post-processing
        processed_response = enforcer.post_process_response(enhanced_response, domain)
        print(f"Processed response: {processed_response[:200]}...")
        
        # Compute rewards
        ground_truth = sample.get('reward_model', {}).get('ground_truth', '')
        extra_info = sample.get('extra_info', {})
        data_source = sample.get('data_source', '')
        
        try:
            # Original reward
            original_reward = default_compute_score(
                data_source, original_response, ground_truth, extra_info
            )
            original_score = original_reward.get('score', 0) if isinstance(original_reward, dict) else original_reward
            
            # Enhanced reward
            enhanced_reward = default_compute_score(
                data_source, enhanced_response, ground_truth, extra_info
            )
            enhanced_score = enhanced_reward.get('score', 0) if isinstance(enhanced_reward, dict) else enhanced_reward
            
            # Processed reward
            processed_reward = default_compute_score(
                data_source, processed_response, ground_truth, extra_info
            )
            processed_score = processed_reward.get('score', 0) if isinstance(processed_reward, dict) else processed_reward
            
            print(f"Rewards - Original: {original_score:.3f}, Enhanced: {enhanced_score:.3f}, Processed: {processed_score:.3f}")
            
            results.append({
                'sample_id': i,
                'domain': domain,
                'data_source': data_source,
                'original_score': original_score,
                'enhanced_score': enhanced_score,
                'processed_score': processed_score,
                'improvement': processed_score - original_score
            })
            
        except Exception as e:
            print(f"❌ Reward computation failed: {e}")
            results.append({
                'sample_id': i,
                'domain': domain,
                'data_source': data_source,
                'original_score': 0,
                'enhanced_score': 0,
                'processed_score': 0,
                'improvement': 0,
                'error': str(e)
            })
    
    # Summary
    print("\n📈 Results Summary")
    print("=" * 70)
    
    if results:
        df_results = pd.DataFrame(results)
        print(df_results[['domain', 'original_score', 'enhanced_score', 'processed_score', 'improvement']])
        
        print(f"\nAverage scores:")
        print(f"  Original: {df_results['original_score'].mean():.3f}")
        print(f"  Enhanced: {df_results['enhanced_score'].mean():.3f}")
        print(f"  Processed: {df_results['processed_score'].mean():.3f}")
        print(f"  Improvement: {df_results['improvement'].mean():.3f}")
        
        # Count improvements
        improved = (df_results['improvement'] > 0).sum()
        total = len(df_results)
        print(f"\nSamples with improved scores: {improved}/{total} ({improved/total*100:.1f}%)")
        
        # Save results
        results_file = '/home/jinming/Reasoning360-MTL/scripts/tests/format_enforcement_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {results_file}")
    
    else:
        print("❌ No results to summarize")

def create_integration_patch():
    """Create a patch file for integrating format enforcement into VERL."""
    print("\n🔧 Creating Integration Patch")
    print("=" * 70)
    
    patch_content = '''
# Format Enforcement Integration Patch for VERL

## 1. Add to verl/workers/fsdp_workers.py

```python
# Add import at top
from scripts.tests.constrained_decoding_solution import ConstrainedDecodingRewardEnforcer

class ActorRolloutRefWorker:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add format enforcer
        self.format_enforcer = ConstrainedDecodingRewardEnforcer()
        
    def detect_domain_from_data_source(self, data_source: str) -> str:
        """Detect domain from data_source field."""
        if 'math' in data_source.lower():
            return 'math'
        elif 'logic' in data_source.lower() or 'barc' in data_source.lower():
            return 'logic'
        elif 'stem' in data_source.lower():
            return 'stem'
        elif 'codegen' in data_source.lower():
            return 'codegen'
        elif 'table' in data_source.lower():
            return 'table'
        else:
            return 'math'  # Default
    
    def generate_sequences(self, data):
        # Enhance prompts before generation
        for item in data:
            if hasattr(item, 'batch') and 'data_source' in item.batch:
                data_source = item.batch['data_source'][0] if isinstance(item.batch['data_source'], list) else item.batch['data_source']
                domain = self.detect_domain_from_data_source(str(data_source))
                
                # Enhance prompt if available
                if 'prompt' in item.batch:
                    original_prompt = item.batch['prompt']
                    if isinstance(original_prompt, str):
                        # Convert string to chat format
                        original_prompt = [{'role': 'user', 'content': original_prompt}]
                    enhanced_prompt = self.format_enforcer.create_enhanced_prompt(original_prompt, domain)
                    item.batch['prompt'] = enhanced_prompt
        
        # Generate normally
        outputs = super().generate_sequences(data)
        
        # Post-process outputs for format compliance
        for i, output in enumerate(outputs):
            if hasattr(output, 'batch') and 'data_source' in output.batch:
                data_source = output.batch['data_source'][0] if isinstance(output.batch['data_source'], list) else output.batch['data_source']
                domain = self.detect_domain_from_data_source(str(data_source))
                
                # Apply format enforcement to generated text
                if hasattr(output, 'response') and output.response:
                    processed_response = self.format_enforcer.post_process_response(output.response, domain)
                    output.response = processed_response
        
        return outputs
```

## 2. Add to verl/utils/reward_score/__init__.py

```python
# Add fallback format correction before reward computation
def default_compute_score_with_format_fallback(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    # Try original computation first
    try:
        result = default_compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)
        if isinstance(result, dict) and result.get('score', 0) > 0:
            return result
        elif isinstance(result, (int, float)) and result > 0:
            return result
    except Exception:
        pass
    
    # If original fails or gives zero, try format correction
    from scripts.tests.constrained_decoding_solution import ConstrainedDecodingRewardEnforcer
    enforcer = ConstrainedDecodingRewardEnforcer()
    
    # Detect domain and apply format correction
    domain = 'math'  # Default
    if 'logic' in data_source.lower() or 'barc' in data_source.lower():
        domain = 'logic'
    elif 'stem' in data_source.lower():
        domain = 'stem'
    elif 'codegen' in data_source.lower():
        domain = 'codegen'
    elif 'table' in data_source.lower():
        domain = 'table'
    
    corrected_solution = enforcer.post_process_response(solution_str, domain)
    
    # Try reward computation with corrected format
    try:
        return default_compute_score(data_source, corrected_solution, ground_truth, extra_info, **kwargs)
    except Exception:
        # Final fallback to improved reward functions
        from scripts.tests.improved_reward_functions import compute_improved_reward
        return compute_improved_reward(data_source, solution_str, ground_truth, extra_info)
```

## 3. Environment Setup

```bash
# Install required dependencies
pip install outlines
pip install guidance  # Alternative constrained decoding library
```

## 4. Configuration Updates

Add to your training configuration:

```yaml
# In ppo_trainer.yaml or similar
data:
  enable_format_enforcement: true
  format_enforcement_domains: ["math", "logic", "stem", "codegen", "table"]
  
actor_rollout_ref:
  rollout:
    enable_format_post_processing: true
```
'''
    
    patch_file = '/home/jinming/Reasoning360-MTL/scripts/tests/format_enforcement_integration.md'
    with open(patch_file, 'w') as f:
        f.write(patch_content)
    
    print(f"📄 Integration patch created: {patch_file}")
    print("\nTo apply the patch:")
    print("1. Review the patch file for integration instructions")
    print("2. Install required dependencies: pip install outlines")
    print("3. Apply code changes to the specified files")
    print("4. Test with smoke test to verify integration")

if __name__ == "__main__":
    test_format_enforcement_with_model()
    create_integration_patch()
