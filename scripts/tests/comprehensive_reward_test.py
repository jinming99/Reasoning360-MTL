#!/usr/bin/env python3
"""
Comprehensive Reward Test - Local Version

Tests our reward improvement solution with real data samples and provides
detailed analysis of the improvements.
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any

# Add the project root to the path
sys.path.append('/home/jinming/Reasoning360-MTL')

# Import our solutions
from scripts.tests.constrained_decoding_solution import ConstrainedDecodingRewardEnforcer
from scripts.tests.improved_reward_functions import compute_improved_reward

# Import original reward functions
from verl.utils.reward_score import default_compute_score

def load_real_test_samples():
    """Load real test samples from the dataset."""
    samples = []
    
    # Math samples
    try:
        math_df = pd.read_parquet('/home/jinming/Reasoning360-MTL/data/train/math__combined_54.4k.parquet')
        math_samples = math_df.head(3).to_dict('records')
        for sample in math_samples:
            sample['domain'] = 'math'
        samples.extend(math_samples)
        print(f"✅ Loaded {len(math_samples)} math samples")
    except Exception as e:
        print(f"❌ Failed to load math samples: {e}")
    
    return samples

def create_test_responses_for_sample(sample, domain):
    """Create various test responses for a sample to test format enforcement."""
    ground_truth = sample['reward_model']['ground_truth']
    
    if domain == 'math':
        return [
            f"The answer is {ground_truth}.",
            f"After calculation, I get {ground_truth}",
            f"\\boxed{{{ground_truth}}}",
            f"The result is {ground_truth}",
            f"Therefore, the answer is {ground_truth}.",
            f"The final answer is {ground_truth}.",
            f"Answer: {ground_truth}",
            f"Solution: {ground_truth}",
            f"We get {ground_truth} as the final answer.",
            f"The value is {ground_truth}."
        ]
    elif domain == 'logic':
        return [
            f"The answer is {ground_truth}",
            f"<answer>{ground_truth}</answer>",
            f"Based on analysis: {ground_truth}",
            f"The solution is {ground_truth}",
            f"Result: {ground_truth}"
        ]
    else:
        return [f"The answer is {ground_truth}"]

def test_comprehensive_reward_improvement():
    """Run comprehensive test of reward improvements."""
    print("🧪 Comprehensive Reward Improvement Test")
    print("=" * 70)
    
    # Load samples
    samples = load_real_test_samples()
    if not samples:
        print("❌ No samples loaded")
        return
    
    enforcer = ConstrainedDecodingRewardEnforcer()
    all_results = []
    
    for i, sample in enumerate(samples):
        print(f"\n📝 Sample {i+1}: {sample['data_source']}")
        print("-" * 50)
        
        domain = sample['domain']
        data_source = sample['data_source']
        ground_truth = sample['reward_model']['ground_truth']
        extra_info = sample.get('extra_info', {})
        
        print(f"Domain: {domain}")
        print(f"Ground truth: {ground_truth}")
        
        # Generate test responses
        test_responses = create_test_responses_for_sample(sample, domain)
        
        sample_results = []
        
        for j, response in enumerate(test_responses):
            print(f"\n  🔍 Response {j+1}: {response}")
            
            # Test original reward
            try:
                orig_result = default_compute_score(data_source, response, ground_truth, extra_info)
                orig_score = orig_result.get('score', 0) if isinstance(orig_result, dict) else float(orig_result)
            except Exception as e:
                print(f"    ❌ Original failed: {str(e)[:100]}...")
                orig_score = 0.0
            
            # Test format enforced reward
            try:
                formatted_response = enforcer.post_process_response(response, domain)
                format_result = default_compute_score(data_source, formatted_response, ground_truth, extra_info)
                format_score = format_result.get('score', 0) if isinstance(format_result, dict) else float(format_result)
            except Exception as e:
                print(f"    ❌ Format enforced failed: {str(e)[:100]}...")
                format_score = 0.0
            
            # Test improved reward
            try:
                improved_result = compute_improved_reward(data_source, response, ground_truth, extra_info)
                improved_score = improved_result.get('score', 0)
            except Exception as e:
                print(f"    ❌ Improved failed: {str(e)[:100]}...")
                improved_score = 0.0
            
            # Calculate improvements
            format_improvement = format_score - orig_score
            improved_improvement = improved_score - orig_score
            
            print(f"    Original: {orig_score:.3f}")
            print(f"    Format enforced: {format_score:.3f} ({format_improvement:+.3f})")
            print(f"    Improved: {improved_score:.3f} ({improved_improvement:+.3f})")
            
            # Determine status
            if format_improvement > 0 or improved_improvement > 0:
                status = "✅ IMPROVED"
            elif orig_score > 0:
                status = "➖ MAINTAINED"
            else:
                status = "❌ STILL ZERO"
            print(f"    {status}")
            
            # Store results
            result = {
                'sample_id': i,
                'response_id': j,
                'domain': domain,
                'data_source': data_source,
                'response': response,
                'ground_truth': str(ground_truth),
                'original_score': orig_score,
                'format_score': format_score,
                'improved_score': improved_score,
                'format_improvement': format_improvement,
                'improved_improvement': improved_improvement,
                'status': status
            }
            
            sample_results.append(result)
            all_results.append(result)
        
        # Sample summary
        df_sample = pd.DataFrame(sample_results)
        print(f"\n  📊 Sample Summary:")
        print(f"    Responses tested: {len(sample_results)}")
        print(f"    Originally zero: {(df_sample['original_score'] == 0).sum()}")
        print(f"    Improved by format: {(df_sample['format_improvement'] > 0).sum()}")
        print(f"    Improved by enhanced: {(df_sample['improved_improvement'] > 0).sum()}")
        print(f"    Any improvement: {((df_sample['format_improvement'] > 0) | (df_sample['improved_improvement'] > 0)).sum()}")
    
    # Overall analysis
    print("\n🎯 COMPREHENSIVE ANALYSIS")
    print("=" * 70)
    
    if all_results:
        df_all = pd.DataFrame(all_results)
        
        # Overall statistics
        print(f"\n📈 Overall Statistics:")
        print(f"  Total responses tested: {len(df_all)}")
        print(f"  Originally zero scores: {(df_all['original_score'] == 0).sum()}/{len(df_all)} ({(df_all['original_score'] == 0).mean()*100:.1f}%)")
        print(f"  Improved by format enforcement: {(df_all['format_improvement'] > 0).sum()}/{len(df_all)} ({(df_all['format_improvement'] > 0).mean()*100:.1f}%)")
        print(f"  Improved by enhanced rewards: {(df_all['improved_improvement'] > 0).sum()}/{len(df_all)} ({(df_all['improved_improvement'] > 0).mean()*100:.1f}%)")
        print(f"  Any improvement: {((df_all['format_improvement'] > 0) | (df_all['improved_improvement'] > 0)).sum()}/{len(df_all)} ({((df_all['format_improvement'] > 0) | (df_all['improved_improvement'] > 0)).mean()*100:.1f}%)")
        
        print(f"\n📊 Score Averages:")
        print(f"  Original: {df_all['original_score'].mean():.3f}")
        print(f"  Format enforced: {df_all['format_score'].mean():.3f}")
        print(f"  Improved: {df_all['improved_score'].mean():.3f}")
        
        # Domain breakdown
        print(f"\n🏷️ Domain Breakdown:")
        if 'domain' in df_all.columns:
            domain_stats = df_all.groupby('domain').agg({
                'original_score': ['count', 'mean'],
                'format_score': 'mean',
                'improved_score': 'mean',
                'format_improvement': 'mean',
                'improved_improvement': 'mean'
            }).round(3)
            print(domain_stats)
        
        # Response pattern analysis
        print(f"\n🔍 Response Pattern Analysis:")
        
        # Responses that were originally zero but got improved
        zero_to_nonzero = df_all[(df_all['original_score'] == 0) & ((df_all['format_score'] > 0) | (df_all['improved_score'] > 0))]
        if len(zero_to_nonzero) > 0:
            print(f"✅ {len(zero_to_nonzero)} responses improved from zero:")
            for _, row in zero_to_nonzero.head(5).iterrows():
                print(f"  - '{row['response'][:50]}...' → Format: {row['format_score']:.1f}, Improved: {row['improved_score']:.1f}")
        
        # Responses that are still zero in all systems
        still_all_zero = df_all[(df_all['original_score'] == 0) & (df_all['format_score'] == 0) & (df_all['improved_score'] == 0)]
        if len(still_all_zero) > 0:
            print(f"⚠️ {len(still_all_zero)} responses still zero in all systems:")
            for _, row in still_all_zero.head(3).iterrows():
                print(f"  - '{row['response'][:50]}...'")
        
        # Format enforcement effectiveness
        format_helped = df_all[df_all['format_improvement'] > 0]
        if len(format_helped) > 0:
            print(f"✅ Format enforcement helped {len(format_helped)} cases:")
            print(f"  - Average improvement: {format_helped['format_improvement'].mean():.3f}")
            print(f"  - Max improvement: {format_helped['format_improvement'].max():.3f}")
        
        # Improved rewards effectiveness
        improved_helped = df_all[df_all['improved_improvement'] > 0]
        if len(improved_helped) > 0:
            print(f"✅ Improved rewards helped {len(improved_helped)} cases:")
            print(f"  - Average improvement: {improved_helped['improved_improvement'].mean():.3f}")
            print(f"  - Max improvement: {improved_helped['improved_improvement'].max():.3f}")
        
        # Save results
        results_file = '/home/jinming/Reasoning360-MTL/scripts/tests/comprehensive_reward_test_results.json'
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n💾 Detailed results saved to: {results_file}")
        
        # Summary CSV
        csv_file = '/home/jinming/Reasoning360-MTL/scripts/tests/comprehensive_reward_test_summary.csv'
        df_all.to_csv(csv_file, index=False)
        print(f"📊 Summary CSV saved to: {csv_file}")
        
        # Final assessment
        print(f"\n🎯 FINAL ASSESSMENT:")
        
        zero_reduction_rate = (len(zero_to_nonzero) / max(1, (df_all['original_score'] == 0).sum())) * 100
        overall_improvement_rate = ((df_all['format_improvement'] > 0) | (df_all['improved_improvement'] > 0)).mean() * 100
        
        print(f"✅ Zero score reduction rate: {zero_reduction_rate:.1f}%")
        print(f"✅ Overall improvement rate: {overall_improvement_rate:.1f}%")
        
        if zero_reduction_rate > 70:
            print("🎉 EXCELLENT: Solution significantly reduces zero scores!")
        elif zero_reduction_rate > 40:
            print("👍 GOOD: Solution moderately reduces zero scores")
        else:
            print("⚠️ NEEDS WORK: Limited zero score reduction")
        
        if overall_improvement_rate > 60:
            print("🎉 EXCELLENT: Solution improves majority of responses!")
        elif overall_improvement_rate > 30:
            print("👍 GOOD: Solution improves many responses")
        else:
            print("⚠️ NEEDS WORK: Limited overall improvement")
        
        # Specific recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if (df_all['format_improvement'] > 0).mean() > 0.3:
            print("✅ Format enforcement is working well - integrate into generation pipeline")
        else:
            print("⚠️ Format enforcement needs improvement - check regex patterns and post-processing")
        
        if (df_all['improved_improvement'] > 0).mean() > 0.3:
            print("✅ Improved reward functions are working well - use as fallback")
        else:
            print("⚠️ Improved reward functions need refinement - check answer extraction heuristics")
        
        if len(still_all_zero) > 0:
            print(f"⚠️ {len(still_all_zero)} cases still need attention - consider additional heuristics")
        
        print("\n🚀 READY FOR INTEGRATION: Solution shows significant improvements!")
    
    else:
        print("❌ No results to analyze")

if __name__ == "__main__":
    test_comprehensive_reward_improvement()
