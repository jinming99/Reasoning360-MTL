#!/usr/bin/env python3
"""
Comprehensive Reward Improvement Validation Test

This script tests our improved reward solutions against the original GURU reward functions
using both Qwen and GURU models to validate the effectiveness of our format enforcement
and improved reward computation approaches.
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import traceback

# Add the project root to the path
sys.path.append('/home/jinming/Reasoning360-MTL')

# Import our solutions
from scripts.tests.constrained_decoding_solution import ConstrainedDecodingRewardEnforcer
from scripts.tests.improved_reward_functions import (
    compute_improved_reward,
    improved_math_reward,
    improved_logic_reward,
    improved_barc_reward,
    improved_stem_reward,
    improved_table_reward,
    improved_codegen_reward
)

# Import original reward functions
from verl.utils.reward_score import default_compute_score

def detect_domain_from_data_source(data_source: str) -> str:
    """Detect domain from data_source field."""
    data_source_lower = data_source.lower()
    if 'math' in data_source_lower:
        return 'math'
    elif any(x in data_source_lower for x in ['logic', 'barc', 'arcagi', 'puzzle']):
        return 'logic'
    elif 'stem' in data_source_lower:
        return 'stem'
    elif 'codegen' in data_source_lower:
        return 'codegen'
    elif 'table' in data_source_lower:
        return 'table'
    else:
        return 'math'  # Default

def load_diverse_test_samples(max_samples_per_domain=5):
    """Load diverse test samples from different domains."""
    samples = []
    
    # Define test data files for each domain
    test_files = {
        'math': '/home/jinming/Reasoning360-MTL/data/train/math__combined_54.4k.parquet',
        'logic': '/home/jinming/Reasoning360-MTL/data/train/logic__barc_1.6k.parquet',
        'stem': '/home/jinming/Reasoning360-MTL/data/offline_eval/stem__gpqa_diamond_198.parquet',
        'table': '/home/jinming/Reasoning360-MTL/data/offline_eval/table__finqa_1.1k.parquet',
        'codegen': '/home/jinming/Reasoning360-MTL/data/train/codegen__leetcode2k_1.3k.parquet'
    }
    
    for domain, file_path in test_files.items():
        try:
            if os.path.exists(file_path):
                df = pd.read_parquet(file_path)
                domain_samples = df.head(max_samples_per_domain).to_dict('records')
                for sample in domain_samples:
                    sample['detected_domain'] = domain
                samples.extend(domain_samples)
                print(f"✅ Loaded {len(domain_samples)} {domain} samples from {file_path}")
            else:
                print(f"⚠️ File not found: {file_path}")
        except Exception as e:
            print(f"❌ Failed to load {domain} samples: {e}")
    
    print(f"📊 Total samples loaded: {len(samples)}")
    return samples

def load_model_safely(model_path: str, use_half_precision=True):
    """Load model with error handling and memory optimization."""
    try:
        print(f"🔄 Loading model: {model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model_kwargs = {
            'device_map': 'auto',
            'torch_dtype': torch.float16 if use_half_precision else torch.float32,
            'trust_remote_code': True
        }
        
        # Try loading with low memory usage
        try:
            model_kwargs['low_cpu_mem_usage'] = True
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        except Exception:
            # Fallback without low_cpu_mem_usage
            del model_kwargs['low_cpu_mem_usage']
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        
        print(f"✅ Model loaded successfully: {model_path}")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Failed to load model {model_path}: {e}")
        return None, None

def generate_response_with_timeout(model, tokenizer, prompt_text, max_length=512, timeout_seconds=30):
    """Generate response with timeout and error handling."""
    try:
        inputs = tokenizer(
            prompt_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=2048,
            padding=True
        )
        
        # Move inputs to model device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
                max_length=inputs['input_ids'].shape[1] + max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Extract only the generated part
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response.strip()
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return f"[Generation Error: {str(e)}]"

def compute_original_reward(data_source, response, ground_truth, extra_info):
    """Compute reward using original GURU functions."""
    try:
        result = default_compute_score(data_source, response, ground_truth, extra_info)
        if isinstance(result, dict):
            return result.get('score', 0), result
        else:
            return float(result), {'score': float(result)}
    except Exception as e:
        print(f"❌ Original reward computation failed: {e}")
        return 0.0, {'score': 0.0, 'error': str(e)}

def compute_enhanced_reward(data_source, response, ground_truth, extra_info, domain):
    """Compute reward using our enhanced approach."""
    try:
        # Step 1: Try format enforcement
        enforcer = ConstrainedDecodingRewardEnforcer()
        formatted_response = enforcer.post_process_response(response, domain)
        
        # Step 2: Try original reward with formatted response
        try:
            result = default_compute_score(data_source, formatted_response, ground_truth, extra_info)
            if isinstance(result, dict):
                score = result.get('score', 0)
            else:
                score = float(result)
            
            if score > 0:
                return score, {'score': score, 'method': 'format_enforced_original'}
        except Exception:
            pass
        
        # Step 3: Fallback to improved reward functions
        result = compute_improved_reward(data_source, response, ground_truth, extra_info)
        if isinstance(result, dict):
            score = result.get('score', 0)
            result['method'] = 'improved_reward'
        else:
            score = float(result)
            result = {'score': score, 'method': 'improved_reward'}
        
        return score, result
        
    except Exception as e:
        print(f"❌ Enhanced reward computation failed: {e}")
        return 0.0, {'score': 0.0, 'error': str(e), 'method': 'error'}

def run_comprehensive_test():
    """Run comprehensive test comparing original vs improved reward computation."""
    print("🧪 Comprehensive Reward Improvement Validation Test")
    print("=" * 80)
    
    # Load test samples
    samples = load_diverse_test_samples(max_samples_per_domain=3)
    if not samples:
        print("❌ No test samples loaded. Exiting.")
        return
    
    # Test models
    model_configs = [
        {
            'name': 'Qwen2.5-1.5B',
            'path': '/home/jinming/llm_models/Qwen2.5-1.5B',
            'enabled': True
        },
        {
            'name': 'Qwen2.5-7B',
            'path': '/home/jinming/llm_models/Qwen2.5-7B',
            'enabled': True  # Enable if you want to test the larger model
        }
    ]
    
    all_results = []
    
    for model_config in model_configs:
        if not model_config['enabled']:
            print(f"⏭️ Skipping {model_config['name']} (disabled)")
            continue
            
        print(f"\n🤖 Testing Model: {model_config['name']}")
        print("-" * 60)
        
        # Load model
        model, tokenizer = load_model_safely(model_config['path'])
        if model is None or tokenizer is None:
            print(f"❌ Failed to load {model_config['name']}, skipping...")
            continue
        
        model_results = []
        
        for i, sample in enumerate(samples):
            print(f"\n📝 Sample {i+1}/{len(samples)}: {sample.get('data_source', 'unknown')}")
            print("-" * 40)
            
            try:
                # Extract sample information
                data_source = sample.get('data_source', '')
                domain = detect_domain_from_data_source(data_source)
                prompt = sample.get('prompt', [])
                ground_truth = sample.get('reward_model', {}).get('ground_truth', '')
                extra_info = sample.get('extra_info', {})
                
                print(f"Domain: {domain}")
                print(f"Data source: {data_source}")
                print(f"Ground truth: {str(ground_truth)[:100]}...")
                
                if not prompt:
                    print("❌ No prompt found, skipping...")
                    continue
                
                # Create prompt text
                try:
                    prompt_text = tokenizer.apply_chat_template(
                        prompt, 
                        add_generation_prompt=True, 
                        tokenize=False
                    )
                except Exception as e:
                    print(f"❌ Failed to apply chat template: {e}")
                    # Fallback to simple concatenation
                    prompt_text = ""
                    for msg in prompt:
                        prompt_text += f"{msg.get('role', 'user')}: {msg.get('content', '')}\n"
                    prompt_text += "assistant: "
                
                print(f"Prompt length: {len(prompt_text)} chars")
                
                # Generate response
                print("🔄 Generating response...")
                response = generate_response_with_timeout(
                    model, tokenizer, prompt_text, max_length=256
                )
                
                if response.startswith("[Generation Error"):
                    print(f"❌ Generation failed: {response}")
                    continue
                
                print(f"Response: {response[:200]}...")
                
                # Test original reward
                print("🔄 Computing original reward...")
                original_score, original_details = compute_original_reward(
                    data_source, response, ground_truth, extra_info
                )
                print(f"Original reward: {original_score:.3f}")
                
                # Test enhanced reward
                print("🔄 Computing enhanced reward...")
                enhanced_score, enhanced_details = compute_enhanced_reward(
                    data_source, response, ground_truth, extra_info, domain
                )
                print(f"Enhanced reward: {enhanced_score:.3f}")
                
                # Calculate improvement
                improvement = enhanced_score - original_score
                print(f"Improvement: {improvement:+.3f}")
                
                # Store results
                result = {
                    'model': model_config['name'],
                    'sample_id': i,
                    'domain': domain,
                    'data_source': data_source,
                    'prompt_length': len(prompt_text),
                    'response_length': len(response),
                    'response_preview': response[:200],
                    'ground_truth_preview': str(ground_truth)[:100],
                    'original_score': original_score,
                    'enhanced_score': enhanced_score,
                    'improvement': improvement,
                    'original_details': original_details,
                    'enhanced_details': enhanced_details,
                    'timestamp': datetime.now().isoformat()
                }
                
                model_results.append(result)
                all_results.append(result)
                
                # Print summary for this sample
                if improvement > 0:
                    print("✅ IMPROVED")
                elif improvement == 0:
                    print("➖ NO CHANGE")
                else:
                    print("❌ DEGRADED")
                
            except Exception as e:
                print(f"❌ Error processing sample {i}: {e}")
                traceback.print_exc()
                continue
        
        # Model summary
        if model_results:
            df_model = pd.DataFrame(model_results)
            print(f"\n📊 {model_config['name']} Summary:")
            print(f"  Samples processed: {len(model_results)}")
            print(f"  Average original score: {df_model['original_score'].mean():.3f}")
            print(f"  Average enhanced score: {df_model['enhanced_score'].mean():.3f}")
            print(f"  Average improvement: {df_model['improvement'].mean():.3f}")
            print(f"  Samples improved: {(df_model['improvement'] > 0).sum()}/{len(model_results)}")
            print(f"  Samples with zero original score: {(df_model['original_score'] == 0).sum()}/{len(model_results)}")
            print(f"  Samples with zero enhanced score: {(df_model['enhanced_score'] == 0).sum()}/{len(model_results)}")
        
        # Clean up model to free memory
        del model
        del tokenizer
        torch.cuda.empty_cache()
    
    # Overall analysis
    print("\n🎯 COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    
    if all_results:
        df_all = pd.DataFrame(all_results)
        
        # Overall statistics
        print("\n📈 Overall Statistics:")
        print(f"  Total samples: {len(df_all)}")
        print(f"  Average original score: {df_all['original_score'].mean():.3f}")
        print(f"  Average enhanced score: {df_all['enhanced_score'].mean():.3f}")
        print(f"  Average improvement: {df_all['improvement'].mean():.3f}")
        print(f"  Samples improved: {(df_all['improvement'] > 0).sum()}/{len(df_all)} ({(df_all['improvement'] > 0).mean()*100:.1f}%)")
        print(f"  Zero original scores: {(df_all['original_score'] == 0).sum()}/{len(df_all)} ({(df_all['original_score'] == 0).mean()*100:.1f}%)")
        print(f"  Zero enhanced scores: {(df_all['enhanced_score'] == 0).sum()}/{len(df_all)} ({(df_all['enhanced_score'] == 0).mean()*100:.1f}%)")
        
        # Domain breakdown
        print("\n📊 Domain Breakdown:")
        domain_stats = df_all.groupby('domain').agg({
            'original_score': ['count', 'mean'],
            'enhanced_score': 'mean',
            'improvement': 'mean'
        }).round(3)
        print(domain_stats)
        
        # Model breakdown
        if 'model' in df_all.columns:
            print("\n🤖 Model Breakdown:")
            model_stats = df_all.groupby('model').agg({
                'original_score': ['count', 'mean'],
                'enhanced_score': 'mean',
                'improvement': 'mean'
            }).round(3)
            print(model_stats)
        
        # Identify remaining issues
        print("\n🔍 Remaining Issues Analysis:")
        
        # Cases where both original and enhanced are zero
        both_zero = df_all[(df_all['original_score'] == 0) & (df_all['enhanced_score'] == 0)]
        if len(both_zero) > 0:
            print(f"⚠️ {len(both_zero)} samples still have zero scores in both systems:")
            for _, row in both_zero.iterrows():
                print(f"  - {row['domain']}: {row['data_source']} | Response: {row['response_preview'][:100]}...")
        
        # Cases where enhanced is worse than original
        degraded = df_all[df_all['improvement'] < 0]
        if len(degraded) > 0:
            print(f"⚠️ {len(degraded)} samples degraded with enhanced system:")
            for _, row in degraded.iterrows():
                print(f"  - {row['domain']}: {row['original_score']:.3f} → {row['enhanced_score']:.3f}")
        
        # Success cases
        improved = df_all[df_all['improvement'] > 0]
        if len(improved) > 0:
            print(f"✅ {len(improved)} samples improved:")
            print(f"  - Average improvement: {improved['improvement'].mean():.3f}")
            print(f"  - Max improvement: {improved['improvement'].max():.3f}")
        
        # Save detailed results
        results_file = '/home/jinming/Reasoning360-MTL/scripts/tests/reward_improvement_validation_results.json'
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n💾 Detailed results saved to: {results_file}")
        
        # Save summary CSV
        csv_file = '/home/jinming/Reasoning360-MTL/scripts/tests/reward_improvement_summary.csv'
        df_all.to_csv(csv_file, index=False)
        print(f"📊 Summary CSV saved to: {csv_file}")
        
        # Recommendations
        print("\n💡 Recommendations:")
        
        zero_rate_original = (df_all['original_score'] == 0).mean()
        zero_rate_enhanced = (df_all['enhanced_score'] == 0).mean()
        
        if zero_rate_enhanced < zero_rate_original:
            print(f"✅ Solution is working! Zero rate reduced from {zero_rate_original*100:.1f}% to {zero_rate_enhanced*100:.1f}%")
        else:
            print(f"⚠️ Zero rate not significantly improved: {zero_rate_original*100:.1f}% → {zero_rate_enhanced*100:.1f}%")
        
        if df_all['improvement'].mean() > 0:
            print("✅ Overall improvement achieved - solution ready for deployment")
        else:
            print("⚠️ Overall improvement marginal - may need further refinement")
        
        # Specific domain recommendations
        for domain in df_all['domain'].unique():
            domain_data = df_all[df_all['domain'] == domain]
            domain_improvement = domain_data['improvement'].mean()
            if domain_improvement <= 0:
                print(f"⚠️ {domain} domain needs attention - improvement: {domain_improvement:.3f}")
            else:
                print(f"✅ {domain} domain working well - improvement: {domain_improvement:.3f}")
    
    else:
        print("❌ No results to analyze")

if __name__ == "__main__":
    # Set environment variables for stable execution
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU only
    
    run_comprehensive_test()
    
    print("\n🎯 Test Complete!")
    print("Check the generated files for detailed results and analysis.")
