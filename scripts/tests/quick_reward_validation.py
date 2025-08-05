#!/usr/bin/env python3
"""
Quick Reward Validation Test

A lightweight version of the comprehensive test that can run locally
to quickly validate our reward improvement solutions.
"""

import os
import sys
import pandas as pd
import json
from typing import Dict, List, Any

# Add the project root to the path
sys.path.append('/home/jinming/Reasoning360-MTL')

# Import our solutions
from scripts.tests.constrained_decoding_solution import ConstrainedDecodingRewardEnforcer
from scripts.tests.improved_reward_functions import compute_improved_reward

# Import original reward functions
from verl.utils.reward_score import default_compute_score

def load_quick_test_samples():
    """Load a few test samples for quick validation."""
    samples = []
    
    # Math sample
    math_sample = {
        'data_source': 'math__test',
        'prompt': [{'role': 'user', 'content': 'What is 2 + 2? Please put your final answer within \\boxed{}.'}],
        'reward_model': {'ground_truth': '4', 'style': 'rule'},
        'extra_info': {'index': 1},
        'domain': 'math'
    }
    samples.append(math_sample)
    
    # Logic sample
    logic_sample = {
        'data_source': 'logic__test',
        'prompt': [{'role': 'user', 'content': 'List the first three prime numbers. Please put your answer within <answer> and </answer> tags.'}],
        'reward_model': {'ground_truth': '[2, 3, 5]', 'style': 'rule'},
        'extra_info': {'index': 1},
        'domain': 'logic'
    }
    samples.append(logic_sample)
    
    # STEM sample
    stem_sample = {
        'data_source': 'stem__test',
        'prompt': [{'role': 'user', 'content': 'What is the speed of light in vacuum? Please put your final answer within \\boxed{}.'}],
        'reward_model': {'ground_truth': '299792458 m/s', 'style': 'rule'},
        'extra_info': {'index': 1},
        'domain': 'stem'
    }
    samples.append(stem_sample)
    
    return samples

def test_sample_responses():
    """Test with various sample responses to validate our improvements."""
    
    # Test cases: (response, expected_improvement)
    test_cases = [
        # Math domain tests
        {
            'domain': 'math',
            'data_source': 'math__test',
            'ground_truth': '4',
            'responses': [
                ('The answer is 4.', 'Should improve with format enforcement'),
                ('After calculation, I get 4', 'Should improve with format enforcement'),
                ('\\boxed{4}', 'Should work with both systems'),
                ('2 + 2 = 4, so the final answer is 4.', 'Should improve'),
                ('The result is four.', 'Should improve with normalization'),
            ]
        },
        # Logic domain tests
        {
            'domain': 'logic',
            'data_source': 'logic__test',
            'ground_truth': '[2, 3, 5]',
            'responses': [
                ('The first three primes are 2, 3, 5.', 'Should improve'),
                ('<answer>[2, 3, 5]</answer>', 'Should work with both'),
                ('2, 3, and 5 are the first three prime numbers.', 'Should improve'),
                ('The answer is [2,3,5]', 'Should improve with format enforcement'),
                ('Prime numbers: 2, 3, 5', 'Should improve'),
            ]
        },
        # STEM domain tests
        {
            'domain': 'stem',
            'data_source': 'stem__test',
            'ground_truth': '299792458 m/s',
            'responses': [
                ('The speed of light is 299792458 m/s.', 'Should improve'),
                ('\\boxed{299792458 m/s}', 'Should work with both'),
                ('Approximately 3 × 10^8 m/s', 'Should improve with flexible matching'),
                ('299,792,458 meters per second', 'Should improve'),
                ('c = 299792458 m/s', 'Should improve'),
            ]
        }
    ]
    
    print("🧪 Quick Reward Validation Test")
    print("=" * 60)
    
    enforcer = ConstrainedDecodingRewardEnforcer()
    all_results = []
    
    for test_case in test_cases:
        domain = test_case['domain']
        data_source = test_case['data_source']
        ground_truth = test_case['ground_truth']
        
        print(f"\n📝 Testing {domain.upper()} Domain")
        print("-" * 40)
        print(f"Ground truth: {ground_truth}")
        
        for response, expectation in test_case['responses']:
            print(f"\n🔍 Response: {response}")
            print(f"Expected: {expectation}")
            
            # Test original reward
            try:
                original_result = default_compute_score(
                    data_source, response, ground_truth, {'index': 1}
                )
                if isinstance(original_result, dict):
                    original_score = original_result.get('score', 0)
                else:
                    original_score = float(original_result)
            except Exception as e:
                print(f"❌ Original reward failed: {e}")
                original_score = 0.0
            
            # Test format enforcement
            try:
                formatted_response = enforcer.post_process_response(response, domain)
                format_result = default_compute_score(
                    data_source, formatted_response, ground_truth, {'index': 1}
                )
                if isinstance(format_result, dict):
                    format_score = format_result.get('score', 0)
                else:
                    format_score = float(format_result)
            except Exception as e:
                print(f"❌ Format-enforced reward failed: {e}")
                format_score = 0.0
            
            # Test improved reward
            try:
                improved_result = compute_improved_reward(
                    data_source, response, ground_truth, {'index': 1}
                )
                if isinstance(improved_result, dict):
                    improved_score = improved_result.get('score', 0)
                else:
                    improved_score = float(improved_result)
            except Exception as e:
                print(f"❌ Improved reward failed: {e}")
                improved_score = 0.0
            
            # Calculate improvements
            format_improvement = format_score - original_score
            improved_improvement = improved_score - original_score
            
            print(f"  Original:  {original_score:.3f}")
            print(f"  Formatted: {format_score:.3f} ({format_improvement:+.3f})")
            print(f"  Improved:  {improved_score:.3f} ({improved_improvement:+.3f})")
            
            # Show formatted response if different
            if formatted_response != response:
                print(f"  Formatted response: {formatted_response}")
            
            # Determine result
            if format_improvement > 0 or improved_improvement > 0:
                print("  ✅ IMPROVED")
                result_status = "improved"
            elif original_score > 0:
                print("  ➖ MAINTAINED")
                result_status = "maintained"
            else:
                print("  ❌ STILL ZERO")
                result_status = "still_zero"
            
            # Store result
            all_results.append({
                'domain': domain,
                'response': response,
                'ground_truth': ground_truth,
                'original_score': original_score,
                'format_score': format_score,
                'improved_score': improved_score,
                'format_improvement': format_improvement,
                'improved_improvement': improved_improvement,
                'result_status': result_status,
                'expectation': expectation
            })
    
    # Summary analysis
    print("\n📊 SUMMARY ANALYSIS")
    print("=" * 60)
    
    df = pd.DataFrame(all_results)
    
    print(f"Total test cases: {len(df)}")
    print(f"Originally zero scores: {(df['original_score'] == 0).sum()}")
    print(f"Improved by format enforcement: {(df['format_improvement'] > 0).sum()}")
    print(f"Improved by improved rewards: {(df['improved_improvement'] > 0).sum()}")
    print(f"Any improvement: {((df['format_improvement'] > 0) | (df['improved_improvement'] > 0)).sum()}")
    
    print(f"\nAverage scores:")
    print(f"  Original: {df['original_score'].mean():.3f}")
    print(f"  Format enforced: {df['format_score'].mean():.3f}")
    print(f"  Improved: {df['improved_score'].mean():.3f}")
    
    print(f"\nDomain breakdown:")
    domain_summary = df.groupby('domain').agg({
        'original_score': 'mean',
        'format_score': 'mean',
        'improved_score': 'mean',
        'format_improvement': 'mean',
        'improved_improvement': 'mean'
    }).round(3)
    print(domain_summary)
    
    # Identify issues
    print(f"\n🔍 Issue Analysis:")
    
    still_zero = df[df['result_status'] == 'still_zero']
    if len(still_zero) > 0:
        print(f"⚠️ {len(still_zero)} cases still have zero scores:")
        for _, row in still_zero.iterrows():
            print(f"  - {row['domain']}: '{row['response']}'")
    
    # Success cases
    improved_cases = df[df['result_status'] == 'improved']
    if len(improved_cases) > 0:
        print(f"✅ {len(improved_cases)} cases improved:")
        print(f"  - Format enforcement helped: {(improved_cases['format_improvement'] > 0).sum()} cases")
        print(f"  - Improved rewards helped: {(improved_cases['improved_improvement'] > 0).sum()} cases")
    
    # Save results
    results_file = '/home/jinming/Reasoning360-MTL/scripts/tests/quick_validation_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n💾 Results saved to: {results_file}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    
    improvement_rate = ((df['format_improvement'] > 0) | (df['improved_improvement'] > 0)).mean()
    if improvement_rate > 0.7:
        print("✅ Solution working well - ready for full integration")
    elif improvement_rate > 0.4:
        print("⚠️ Solution partially working - may need refinement")
    else:
        print("❌ Solution needs significant improvement")
    
    zero_reduction = (df['original_score'] == 0).sum() - ((df['format_score'] == 0) & (df['improved_score'] == 0)).sum()
    if zero_reduction > 0:
        print(f"✅ Reduced zero scores by {zero_reduction} cases")
    else:
        print("⚠️ No reduction in zero scores achieved")

if __name__ == "__main__":
    test_sample_responses()
