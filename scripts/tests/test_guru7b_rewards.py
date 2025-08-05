#!/usr/bin/env python3
"""
Test reward computation with the trained GURU-7B model.
This will show if a properly trained model gets non-zero rewards.
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
from base_reward_test import save_test_results

class GURU7BRewardTester:
    """Test reward computation with GURU-7B model."""
    
    def __init__(self):
        self.model_name = "LLM360/guru-7B"
        self.cache_dir = "/home/jinming/llm_models"  # Use dedicated model directory
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the GURU-7B model and tokenizer."""
        print(f"🤖 Loading GURU-7B model: {self.model_name}")
        print(f"📍 Device: {self.device}")
        print("⏳ This may take a few minutes to download...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                cache_dir=self.cache_dir
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            print(f"✅ GURU-7B model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error loading GURU-7B model: {e}")
            traceback.print_exc()
            return False
    
    def generate_response(self, prompt: str, max_length: int = 256) -> str:
        """Generate response from GURU-7B model."""
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
            
            # Generate response with settings similar to GURU paper
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=1.0,  # GURU paper setting
                    top_p=1.0,        # GURU paper setting
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
    
    def test_domain_samples(self, domain_name: str, data_path: str, num_samples: int = 3) -> List[Dict]:
        """Test samples from a specific domain with GURU-7B generation."""
        print(f"\n{'='*80}")
        print(f"🧪 TESTING {domain_name.upper()} DOMAIN WITH GURU-7B")
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
            print(f"🤖 Generating GURU-7B response...")
            response = self.generate_response(prompt, max_length=200)
            print(f"Generated response ({len(response)} chars): {repr(response[:200])}...")
            
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
                
                # Analyze if response looks good
                response_quality = self.analyze_response_quality(response, ground_truth, domain_name)
                
                result = {
                    'domain': domain_name,
                    'sample_idx': i,
                    'data_source': data_source,
                    'prompt_length': len(prompt),
                    'response_length': len(response),
                    'response': response,
                    'ground_truth': str(ground_truth)[:200],
                    'score': score,
                    'final_reward': float(final_reward) if final_reward is not None else 0.0,
                    'success': True,
                    'response_quality': response_quality
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
                    'ground_truth': str(ground_truth)[:200],
                    'score': None,
                    'final_reward': 0.0,
                    'success': False,
                    'error': str(e),
                    'response_quality': 'error'
                }
            
            results.append(result)
            
        return results
    
    def analyze_response_quality(self, response: str, ground_truth: str, domain: str) -> str:
        """Analyze if the response looks reasonable even if reward is 0."""
        if not response:
            return "empty"
        
        if domain == "math":
            # Check if response contains numbers or mathematical expressions
            if any(c.isdigit() for c in response) or any(op in response for op in ['+', '-', '*', '/', '=', '^']):
                return "mathematical_content"
        elif domain == "logic":
            # Check if response contains logical structures
            if any(pattern in response for pattern in ['[', ']', ',', '0', '1', '2']):
                return "logical_structure"
        elif domain == "codegen":
            # Check if response contains code
            if any(keyword in response for keyword in ['def', 'import', 'return', 'if', 'for', 'while']):
                return "code_content"
        elif domain == "stem":
            # Check if response contains scientific reasoning
            if len(response) > 50 and any(term in response.lower() for term in ['calculate', 'solve', 'equation', 'formula']):
                return "scientific_reasoning"
        elif domain == "table":
            # Check if response contains numerical data
            if any(c.isdigit() for c in response):
                return "numerical_data"
        
        return "generic_text"
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test with GURU-7B across selected domains."""
        print(f"🚀 COMPREHENSIVE GURU-7B MODEL TESTING")
        print(f"Model: {self.model_name}")
        print(f"="*80)
        
        # Load model
        if not self.load_model():
            return {"error": "Failed to load GURU-7B model"}
        
        # Test selected domains (focus on math and logic for faster testing)
        domains = [
            ("math", "/home/jinming/Reasoning360-MTL/data/train/guru_18k/math.parquet"),
            ("logic", "/home/jinming/Reasoning360-MTL/data/train/guru_18k/logic.parquet"),
            ("stem", "/home/jinming/Reasoning360-MTL/data/train/guru_18k/stem.parquet"),
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
                quality_counts = {}
                for r in domain_results:
                    quality = r.get('response_quality', 'unknown')
                    quality_counts[quality] = quality_counts.get(quality, 0) + 1
                
                domain_summaries[domain_name] = {
                    'total_samples': len(domain_results),
                    'successful': successful,
                    'success_rate': successful / len(domain_results),
                    'non_zero_rewards': non_zero_rewards,
                    'avg_reward': avg_reward,
                    'max_reward': max(r['final_reward'] for r in domain_results),
                    'response_quality_counts': quality_counts,
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
        print(f"🎯 GURU-7B MODEL TESTING SUMMARY")
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
                print(f"      Response quality: {stats['response_quality_counts']}")
        
        print(f"\n🔍 SAMPLE RESULTS:")
        for result in summary['detailed_results']:
            status = "✅" if result['success'] else "❌"
            reward = result['final_reward']
            quality = result.get('response_quality', 'unknown')
            print(f"  {status} {result['domain']}: reward={reward:.4f}, quality={quality}, response_len={result['response_length']}")
        
        if summary['non_zero_rewards'] > 0:
            print(f"\n🎉 SUCCESS: GURU-7B gets non-zero rewards!")
            print(f"✅ This proves our reward computation works correctly")
        else:
            print(f"\n🤔 GURU-7B also gets zero rewards")
            print(f"💡 This suggests our reward evaluation might be too strict")
        
        print(f"{'='*80}")


def main():
    """Run the GURU-7B model testing."""
    tester = GURU7BRewardTester()
    
    summary = tester.run_comprehensive_test()
    
    if "error" in summary:
        print(f"❌ Test failed: {summary['error']}")
        return False
    
    # Print summary
    tester.print_summary(summary)
    
    # Save results
    output_path = "/home/jinming/Reasoning360-MTL/scripts/tests/guru7b_results.json"
    save_test_results(summary, output_path)
    
    return summary['success_rate'] > 0.5


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
