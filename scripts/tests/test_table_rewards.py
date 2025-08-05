#!/usr/bin/env python3
"""
Test reward computation for Table domain.
Tests: table__hitab, table__multihier
"""

import sys
import os
sys.path.insert(0, '/home/jinming/Reasoning360-MTL/scripts/tests')

from base_reward_test import RewardTester, save_test_results

class TableRewardTester(RewardTester):
    """Test reward computation for Table domain."""
    
    def __init__(self):
        super().__init__(
            domain_name="table",
            data_path="/home/jinming/Reasoning360-MTL/data/train/guru_18k/table.parquet"
        )

def main():
    print("📊 Testing Table Domain Reward Computation")
    print("Expected reward modules: tablereason.py")
    
    tester = TableRewardTester()
    results = tester.run_tests(num_samples=10)
    
    # Save results
    output_path = "/home/jinming/Reasoning360-MTL/scripts/tests/results_table.json"
    save_test_results(results, output_path)
    
    # Check if all tests passed
    if results['success_rate'] == 1.0 and results['rewards']['non_zero_count'] > 0:
        print("\n✅ Table domain reward computation: PASSED")
        return True
    else:
        print("\n❌ Table domain reward computation: FAILED")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
