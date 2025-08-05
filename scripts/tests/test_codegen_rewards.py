#!/usr/bin/env python3
"""
Test reward computation for Codegen domain.
Tests: codegen__leetcode2k, codegen__primeintellect, codegen__livecodebench, codegen__taco
"""

import sys
import os
sys.path.insert(0, '/home/jinming/Reasoning360-MTL/scripts/tests')

from base_reward_test import RewardTester, save_test_results

class CodegenRewardTester(RewardTester):
    """Test reward computation for Codegen domain."""
    
    def __init__(self):
        super().__init__(
            domain_name="codegen",
            data_path="/home/jinming/Reasoning360-MTL/data/train/guru_18k/codegen.parquet"
        )

def main():
    print("💻 Testing Codegen Domain Reward Computation")
    print("Expected reward modules: coder1/")
    
    tester = CodegenRewardTester()
    results = tester.run_tests(num_samples=10)
    
    # Save results
    output_path = "/home/jinming/Reasoning360-MTL/scripts/tests/results_codegen.json"
    save_test_results(results, output_path)
    
    # Check if all tests passed
    if results['success_rate'] == 1.0 and results['rewards']['non_zero_count'] > 0:
        print("\n✅ Codegen domain reward computation: PASSED")
        return True
    else:
        print("\n❌ Codegen domain reward computation: FAILED")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
