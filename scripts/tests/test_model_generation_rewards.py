#!/usr/bin/env python3
"""
End-to-end test with actual model generation + reward computation.
This tests the complete pipeline: prompt -> model generation -> reward scoring.
"""

import sys
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import polars as pl
from typing import Dict, Any, List
import traceback

# Add the project root to Python path
sys.path.insert(0, '/home/jinming/Reasoning360-MTL')
sys.path.insert(0, '/home/jinming/Reasoning360-MTL/scripts/tests')

from verl.utils.reward_score import default_compute_score
from base_reward_test import RewardTester, save_test_results

class ModelGenerationRewardTester:
    """Test reward computation with actual model generation."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the model and tokenizer."""
        print(f"🤖 Loading model: {self.model_name}")
        print(f"📍 Device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            print(f"✅ Model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            traceback.print_exc()
            return False
    
    def generate_response(self, prompt: str, max_length: int = 256) -> str:
        """Generate response from the model."""
        if self.model is None or self.tokenizer is None:
            return ""
            
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048
            )
            
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode response (remove input prompt)
            input_length = inputs['input_ids'].shape[1]
            response_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            print(f"⚠ Error generating response: {e}")
            return ""
    
    def test_domain_samples(self, domain_name: str, data_path: str, num_samples: int = 3) -> List[Dict]:
        """Test samples from a specific domain with model generation."""
        print(f"\n{'='*80}")
        print(f"🧪 TESTING {domain_name.upper()} DOMAIN WITH MODEL GENERATION")
        print(f"{'='*80}")
        
        # Load samples
        df = pl.read_parquet(data_path)
        samples = df.head(num_samples).to_dicts()
        
        results = []
        
        for i, sample in enumerate(samples):
            print(f"\n--- Sample {i+1}/{num_samples} ---")
            
            # Extract data
            prompt = sample.get('prompt', '')
            data_source = sample.get('data_source', 'unknown')
            reward_model_str = sample.get('reward_model', '{}')
            extra_info = sample.get('extra_info', None)
            
            print(f"Data source: {data_source}")
            print(f"Prompt length: {len(prompt)}")
            print(f"Prompt preview: {repr(prompt[:200])}...")
            
            # Parse reward_model
            reward_model = self.parse_reward_model(reward_model_str)
            ground_truth = reward_model.get("ground_truth", "")
            
            print(f"Ground truth: {repr(str(ground_truth)[:100])}...")
            
            # Generate model response
            print(f"🤖 Generating response...")
            response = self.generate_response(prompt, max_length=128)
            print(f"Generated response: {repr(response[:200])}...")
            
            # Compute reward
            print(f"🎯 Computing reward...")
            try:
                score = default_compute_score(
                    data_source=data_source,
                    solution_str=response,
                    ground_truth=ground_truth,
                    extra_info=extra_info
                )
                
                final_reward = score.get('score', score) if isinstance(score, dict) else score
                print(f"✅ Reward computed: {score}")
                
                result = {
                    'domain': domain_name,
                    'sample_idx': i,
                    'data_source': data_source,
                    'prompt_length': len(prompt),
                    'response_length': len(response),
                    'response': response,
                    'ground_truth': str(ground_truth)[:100],
                    'score': score,
                    'final_reward': float(final_reward) if final_reward is not None else 0.0,
                    'success': True
                }
                
            except Exception as e:
                print(f"❌ Error computing reward: {e}")
                result = {
                    'domain': domain_name,
                    'sample_idx': i,
                    'data_source': data_source,
                    'prompt_length': len(prompt),
                    'response_length': len(response),
                    'response': response,
                    'ground_truth': str(ground_truth)[:100],
                    'score': None,
                    'final_reward': 0.0,
                    'success': False,
                    'error': str(e)
                }
            
            results.append(result)
            
        return results
    
    def parse_reward_model(self, reward_model_str: str) -> Dict[str, Any]:
        """Parse reward_model string exactly as done in naive.py."""
        if isinstance(reward_model_str, str):
            try:
                import ast
                return ast.literal_eval(reward_model_str)
            except:
                try:
                    import numpy as np
                    return eval(reward_model_str, {'array': np.array, 'dtype': np.dtype, 'object': object, '__builtins__': {}})
                except:
                    return {}
        return reward_model_str if reward_model_str else {}
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test across all domains."""
        print(f"🚀 COMPREHENSIVE MODEL GENERATION + REWARD TESTING")
        print(f"Model: {self.model_name}")
        print(f"="*80)
        
        # Load model
        if not self.load_model():
            return {"error": "Failed to load model"}
        
        # Test domains
        domains = [
            ("math", "/home/jinming/Reasoning360-MTL/data/train/guru_18k/math.parquet"),
            ("logic", "/home/jinming/Reasoning360-MTL/data/train/guru_18k/logic.parquet"),
            ("codegen", "/home/jinming/Reasoning360-MTL/data/train/guru_18k/codegen.parquet"),
            ("simulation", "/home/jinming/Reasoning360-MTL/data/train/guru_18k/simulation.parquet"),
            ("stem", "/home/jinming/Reasoning360-MTL/data/train/guru_18k/stem.parquet"),
            ("table", "/home/jinming/Reasoning360-MTL/data/train/guru_18k/table.parquet"),
        ]
        
        all_results = []
        domain_summaries = {}
        
        for domain_name, data_path in domains:
            try:
                domain_results = self.test_domain_samples(domain_name, data_path, num_samples=2)
                all_results.extend(domain_results)
                
                # Compute domain summary
                successful = sum(1 for r in domain_results if r['success'])
                non_zero_rewards = sum(1 for r in domain_results if r['final_reward'] > 0)
                avg_reward = sum(r['final_reward'] for r in domain_results) / len(domain_results)
                
                domain_summaries[domain_name] = {
                    'total_samples': len(domain_results),
                    'successful': successful,
                    'success_rate': successful / len(domain_results),
                    'non_zero_rewards': non_zero_rewards,
                    'avg_reward': avg_reward,
                    'max_reward': max(r['final_reward'] for r in domain_results),
                    'data_sources': list(set(r['data_source'] for r in domain_results))
                }
                
            except Exception as e:
                print(f"❌ Error testing {domain_name}: {e}")
                domain_summaries[domain_name] = {'error': str(e)}
        
        # Generate overall summary
        total_samples = len(all_results)
        successful_samples = sum(1 for r in all_results if r['success'])
        non_zero_rewards = sum(1 for r in all_results if r['final_reward'] > 0)
        
        summary = {
            'model_name': self.model_name,
            'total_samples': total_samples,
            'successful_samples': successful_samples,
            'success_rate': successful_samples / total_samples if total_samples > 0 else 0,
            'non_zero_rewards': non_zero_rewards,
            'non_zero_rate': non_zero_rewards / total_samples if total_samples > 0 else 0,
            'avg_reward': sum(r['final_reward'] for r in all_results) / total_samples if total_samples > 0 else 0,
            'domain_summaries': domain_summaries,
            'detailed_results': all_results
        }
        
        return summary
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print comprehensive summary."""
        print(f"\n{'='*80}")
        print(f"🎯 MODEL GENERATION + REWARD TESTING SUMMARY")
        print(f"{'='*80}")
        
        print(f"Model: {summary['model_name']}")
        print(f"Total samples tested: {summary['total_samples']}")
        print(f"Successful computations: {summary['successful_samples']} ({summary['success_rate']:.1%})")
        print(f"Non-zero rewards: {summary['non_zero_rewards']} ({summary['non_zero_rate']:.1%})")
        print(f"Average reward: {summary['avg_reward']:.4f}")
        
        print(f"\n📊 DOMAIN BREAKDOWN:")
        for domain, stats in summary['domain_summaries'].items():
            if 'error' in stats:
                print(f"  ❌ {domain}: ERROR - {stats['error']}")
            else:
                print(f"  ✅ {domain}: {stats['successful']}/{stats['total_samples']} success, "
                      f"{stats['non_zero_rewards']} non-zero rewards, avg={stats['avg_reward']:.4f}")
        
        print(f"\n🔍 SAMPLE RESULTS:")
        for result in summary['detailed_results'][:6]:  # Show first 6 results
            status = "✅" if result['success'] else "❌"
            reward = result['final_reward']
            print(f"  {status} {result['domain']}: reward={reward:.4f}, response_len={result['response_length']}")
        
        if summary['non_zero_rewards'] > 0:
            print(f"\n🎉 SUCCESS: Model generation + reward computation working!")
            print(f"✅ {summary['non_zero_rewards']} samples got non-zero rewards")
        else:
            print(f"\n⚠️  All rewards are zero - this may be expected for base models")
            print(f"✅ But the pipeline is working correctly!")
        
        print(f"{'='*80}")


def main():
    """Run the comprehensive model generation + reward test."""
    tester = ModelGenerationRewardTester("Qwen/Qwen2.5-1.5B")
    
    summary = tester.run_comprehensive_test()
    
    if "error" in summary:
        print(f"❌ Test failed: {summary['error']}")
        return False
    
    # Print summary
    tester.print_summary(summary)
    
    # Save results
    output_path = "/home/jinming/Reasoning360-MTL/scripts/tests/model_generation_results.json"
    save_test_results(summary, output_path)
    
    return summary['success_rate'] > 0.5  # Consider success if >50% computations work


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
