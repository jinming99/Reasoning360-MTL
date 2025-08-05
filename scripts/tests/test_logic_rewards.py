#!/usr/bin/env python3
"""
Test reward computation for Logic domain.
Tests: simulation__arcagi2, simulation__barc, logic__graph_logical_dataset, 
       logic__zebra_puzzle_dataset, logic__ordering_puzzle_dataset, simulation__arcagi1
"""

import sys
import os
sys.path.insert(0, '/home/jinming/Reasoning360-MTL/scripts/tests')

from base_reward_test import RewardTester, save_test_results

class LogicRewardTester(RewardTester):
    """Test reward computation for Logic domain."""
    
    def __init__(self):
        super().__init__(
            domain_name="logic",
            data_path="/home/jinming/Reasoning360-MTL/data/train/guru_18k/logic.parquet"
        )
    
    def run_tests(self, num_samples: int = 15):
        """Run tests with more samples to cover all data sources."""
        print(f"\n{'='*60}")
        print(f"TESTING REWARD COMPUTATION FOR {self.domain_name.upper()} DOMAIN")
        print(f"Expected reward modules: arcagi.py, zebra_puzzle.py, puzzles_dataset.py, graph_dataset.py")
        print(f"{'='*60}")
        
        # Load test samples
        samples = self.load_test_samples(num_samples)
        
        # Group samples by data source to ensure we test all
        data_source_samples = {}
        for sample in samples:
            ds = sample.get('data_source', 'unknown')
            if ds not in data_source_samples:
                data_source_samples[ds] = []
            data_source_samples[ds].append(sample)
        
        print(f"\nData sources found: {list(data_source_samples.keys())}")
        
        # Test samples from each data source
        for ds, ds_samples in data_source_samples.items():
            print(f"\n--- Testing {ds} ({len(ds_samples)} samples) ---")
            for i, sample in enumerate(ds_samples[:3]):  # Test up to 3 per data source
                result = self.test_sample(sample, len(self.test_results))
                self.test_results.append(result)
        
        # Generate summary
        summary = self.generate_summary()
        self.print_summary(summary)
        
        return summary

def main():
    print("🧠 Testing Logic Domain Reward Computation")
    print("Expected reward modules: arcagi.py, zebra_puzzle.py, puzzles_dataset.py, graph_dataset.py")
    
    tester = LogicRewardTester()
    results = tester.run_tests(num_samples=15)
    
    # Save results
    output_path = "/home/jinming/Reasoning360-MTL/scripts/tests/results_logic.json"
    save_test_results(results, output_path)
    
    # Check if all tests passed
    if results['success_rate'] >= 0.8 and results['rewards']['non_zero_count'] > 0:
        print("\n✅ Logic domain reward computation: PASSED")
        return True
    else:
        print("\n❌ Logic domain reward computation: FAILED")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
