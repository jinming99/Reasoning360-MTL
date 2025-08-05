#!/usr/bin/env python3
"""
Base framework for testing reward computation across all domains.
This mimics exactly how rewards are computed in the RL training pipeline.
"""

import sys
import os
import json
import polars as pl
from typing import Dict, Any, List, Tuple
import traceback

# Add the project root to Python path
sys.path.insert(0, '/home/jinming/Reasoning360-MTL')

from verl.utils.reward_score import default_compute_score

class RewardTester:
    """Base class for testing reward computation for different domains."""
    
    def __init__(self, domain_name: str, data_path: str):
        self.domain_name = domain_name
        self.data_path = data_path
        self.test_results = []
        
    def load_test_samples(self, num_samples: int = 5) -> List[Dict[str, Any]]:
        """Load test samples from the domain dataset."""
        print(f"\n=== Loading {num_samples} test samples from {self.domain_name} domain ===")
        
        df = pl.read_parquet(self.data_path)
        print(f"Total samples in dataset: {len(df)}")
        
        # Get first num_samples for testing
        test_df = df.head(num_samples)
        
        samples = []
        for row in test_df.iter_rows(named=True):
            samples.append(dict(row))
            
        print(f"Loaded {len(samples)} test samples")
        return samples
    
    def parse_reward_model(self, reward_model_str: str) -> Dict[str, Any]:
        """Parse reward_model string exactly as done in naive.py."""
        print(f"Parsing reward_model: {repr(reward_model_str[:100])}...")
        
        if isinstance(reward_model_str, str):
            try:
                # First try ast.literal_eval for simple cases
                import ast
                reward_model = ast.literal_eval(reward_model_str)
                print(f"✓ Parsed with ast.literal_eval")
                return reward_model
            except (ValueError, SyntaxError, TypeError) as e:
                print(f"⚠ ast.literal_eval failed: {e}")
                # Try eval for numpy arrays (safe in this controlled context)
                try:
                    import numpy as np
                    # Make numpy available for eval
                    reward_model = eval(reward_model_str, {'array': np.array, 'dtype': np.dtype, 'object': object, '__builtins__': {}})
                    print(f"✓ Parsed with eval")
                    return reward_model
                except Exception as e2:
                    print(f"✗ eval failed: {e2}")
                    return {}
        else:
            return reward_model_str if reward_model_str else {}
    
    def test_sample(self, sample: Dict[str, Any], sample_idx: int) -> Dict[str, Any]:
        """Test reward computation for a single sample."""
        print(f"\n--- Testing Sample {sample_idx + 1} ---")
        
        # Extract fields exactly as done in training
        data_source = sample.get('data_source', 'unknown')
        prompt = sample.get('prompt', '')
        response = sample.get('response', '')
        reward_model_str = sample.get('reward_model', '{}')
        extra_info = sample.get('extra_info', None)
        
        print(f"Data source: {data_source}")
        print(f"Prompt length: {len(prompt)}")
        print(f"Response length: {len(response)}")
        print(f"Response preview: {repr(response[:100])}...")
        
        # Parse reward_model exactly as in naive.py
        reward_model = self.parse_reward_model(reward_model_str)
        ground_truth = reward_model.get("ground_truth", "")
        
        print(f"Ground truth type: {type(ground_truth)}")
        print(f"Ground truth preview: {repr(str(ground_truth)[:100])}...")
        
        # Check if ground_truth is empty (handle numpy arrays)
        is_empty = False
        try:
            if hasattr(ground_truth, '__len__'):
                is_empty = len(ground_truth) == 0
            else:
                is_empty = not ground_truth
        except:
            is_empty = not ground_truth
            
        if is_empty:
            print("⚠ Warning: No ground_truth found in reward_model")
        
        # Test reward computation
        test_result = {
            'sample_idx': sample_idx,
            'data_source': data_source,
            'has_ground_truth': bool(ground_truth),
            'ground_truth_type': str(type(ground_truth)),
            'response_length': len(response),
            'success': False,
            'score': None,
            'error': None
        }
        
        try:
            print(f"Calling default_compute_score...")
            score = default_compute_score(
                data_source=data_source,
                solution_str=response,
                ground_truth=ground_truth,
                extra_info=extra_info
            )
            
            print(f"✓ Score computed successfully: {score}")
            test_result['success'] = True
            test_result['score'] = score
            test_result['score_type'] = str(type(score))
            
            if isinstance(score, dict):
                test_result['final_reward'] = score.get('score', 0.0)
            else:
                test_result['final_reward'] = float(score)
                
        except Exception as e:
            print(f"✗ Error computing score: {e}")
            traceback.print_exc()
            test_result['error'] = str(e)
            test_result['final_reward'] = 0.0
        
        return test_result
    
    def run_tests(self, num_samples: int = 5) -> Dict[str, Any]:
        """Run tests on multiple samples and return summary."""
        print(f"\n{'='*60}")
        print(f"TESTING REWARD COMPUTATION FOR {self.domain_name.upper()} DOMAIN")
        print(f"{'='*60}")
        
        # Load test samples
        samples = self.load_test_samples(num_samples)
        
        # Test each sample
        for i, sample in enumerate(samples):
            result = self.test_sample(sample, i)
            self.test_results.append(result)
        
        # Generate summary
        summary = self.generate_summary()
        self.print_summary(summary)
        
        return summary
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate test summary statistics."""
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r['success'])
        failed_tests = total_tests - successful_tests
        
        rewards = [r['final_reward'] for r in self.test_results if r['success']]
        non_zero_rewards = [r for r in rewards if r != 0.0]
        
        data_sources = list(set(r['data_source'] for r in self.test_results))
        
        summary = {
            'domain': self.domain_name,
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'failed_tests': failed_tests,
            'success_rate': successful_tests / total_tests if total_tests > 0 else 0,
            'data_sources': data_sources,
            'rewards': {
                'total_computed': len(rewards),
                'non_zero_count': len(non_zero_rewards),
                'zero_count': len(rewards) - len(non_zero_rewards),
                'min_reward': min(rewards) if rewards else None,
                'max_reward': max(rewards) if rewards else None,
                'avg_reward': sum(rewards) / len(rewards) if rewards else None,
            },
            'errors': [r['error'] for r in self.test_results if r['error']],
            'detailed_results': self.test_results
        }
        
        return summary
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print test summary."""
        print(f"\n{'='*60}")
        print(f"TEST SUMMARY FOR {summary['domain'].upper()} DOMAIN")
        print(f"{'='*60}")
        
        print(f"Total tests: {summary['total_tests']}")
        print(f"Successful: {summary['successful_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success rate: {summary['success_rate']:.1%}")
        
        print(f"\nData sources tested: {len(summary['data_sources'])}")
        for ds in summary['data_sources']:
            print(f"  - {ds}")
        
        rewards = summary['rewards']
        print(f"\nReward statistics:")
        print(f"  - Total rewards computed: {rewards['total_computed']}")
        print(f"  - Non-zero rewards: {rewards['non_zero_count']}")
        print(f"  - Zero rewards: {rewards['zero_count']}")
        if rewards['avg_reward'] is not None:
            print(f"  - Average reward: {rewards['avg_reward']:.4f}")
            print(f"  - Min reward: {rewards['min_reward']:.4f}")
            print(f"  - Max reward: {rewards['max_reward']:.4f}")
        
        if summary['errors']:
            print(f"\nErrors encountered:")
            for i, error in enumerate(summary['errors']):
                print(f"  {i+1}. {error}")
        
        print(f"\n{'='*60}")


def save_test_results(results: Dict[str, Any], output_path: str):
    """Save test results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Test results saved to: {output_path}")


if __name__ == "__main__":
    print("Base reward testing framework loaded successfully!")
