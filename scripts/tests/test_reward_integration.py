#!/usr/bin/env python3
"""
Test script to validate the integration of improved reward functions into the RL pipeline.
This tests the fallback mechanism in the default_compute_score function.
"""

import sys
import os

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from verl.utils.reward_score import default_compute_score

def test_reward_integration():
    """Test that our improved rewards are properly integrated as fallback."""
    
    print("🧪 Testing Reward Integration...")
    print("=" * 60)
    
    # Test cases that should trigger fallback (poorly formatted responses)
    test_cases = [
        {
            'data_source': 'math__test',
            'solution_str': 'The answer is 26.',  # Missing \boxed{} format
            'ground_truth': '26',
            'extra_info': {'reward_metric': 'naive_dapo'},
            'expected_fallback': True
        },
        {
            'data_source': 'logic__barc',
            'solution_str': 'The sequence is [1, 2, 3].',  # Missing <answer> tags
            'ground_truth': '[1, 2, 3]',
            'extra_info': {},
            'expected_fallback': True
        },
        {
            'data_source': 'math__test',
            'solution_str': 'The answer is \\boxed{26}.',  # Properly formatted
            'ground_truth': '26',
            'extra_info': {'reward_metric': 'naive_dapo'},
            'expected_fallback': False
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\n📝 Test Case {i+1}:")
        print(f"   Data Source: {test_case['data_source']}")
        print(f"   Solution: {test_case['solution_str'][:50]}...")
        print(f"   Expected Fallback: {test_case['expected_fallback']}")
        
        try:
            result = default_compute_score(
                data_source=test_case['data_source'],
                solution_str=test_case['solution_str'],
                ground_truth=test_case['ground_truth'],
                extra_info=test_case['extra_info']
            )
            
            # Check if fallback was used
            fallback_used = False
            if isinstance(result, dict):
                fallback_used = result.get('fallback_used', False)
                score = result.get('score', 0.0)
            else:
                score = float(result)
            
            print(f"   ✅ Result: {result}")
            print(f"   📊 Score: {score}")
            print(f"   🔄 Fallback Used: {fallback_used}")
            
            # Validate expectations
            if test_case['expected_fallback'] and fallback_used:
                print(f"   ✅ PASS: Fallback used as expected")
                results.append(True)
            elif not test_case['expected_fallback'] and not fallback_used:
                print(f"   ✅ PASS: Original reward used as expected")
                results.append(True)
            else:
                print(f"   ❌ FAIL: Fallback behavior didn't match expectation")
                results.append(False)
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 INTEGRATION TEST SUMMARY:")
    print(f"   Total Tests: {len(results)}")
    print(f"   Passed: {sum(results)}")
    print(f"   Failed: {len(results) - sum(results)}")
    
    if all(results):
        print("   🎉 ALL TESTS PASSED! Integration is working correctly.")
        return True
    else:
        print("   ❌ Some tests failed. Check the integration.")
        return False

if __name__ == "__main__":
    success = test_reward_integration()
    sys.exit(0 if success else 1)
