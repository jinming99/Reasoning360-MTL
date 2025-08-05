#!/usr/bin/env python3
"""
Master test runner for all domain reward computations.
Runs comprehensive tests across all 6 domains to ensure reward computation works correctly.
"""

import sys
import os
import subprocess
import json
from datetime import datetime

# Test scripts for each domain
TEST_SCRIPTS = [
    ("Math", "test_math_rewards.py"),
    ("Codegen", "test_codegen_rewards.py"), 
    ("Logic", "test_logic_rewards.py"),
    ("Simulation", "test_simulation_rewards.py"),
    ("STEM", "test_stem_rewards.py"),
    ("Table", "test_table_rewards.py"),
]

def run_single_test(domain_name: str, script_name: str) -> dict:
    """Run a single domain test and return results."""
    print(f"\n{'='*80}")
    print(f"🧪 RUNNING {domain_name.upper()} DOMAIN TEST")
    print(f"{'='*80}")
    
    script_path = f"/home/jinming/Reasoning360-MTL/scripts/tests/{script_name}"
    
    try:
        # Run the test script
        result = subprocess.run([
            "python3", script_path
        ], capture_output=True, text=True, cwd="/home/jinming/Reasoning360-MTL/scripts/tests")
        
        success = result.returncode == 0
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return {
            'domain': domain_name,
            'script': script_name,
            'success': success,
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except Exception as e:
        print(f"❌ Error running {domain_name} test: {e}")
        return {
            'domain': domain_name,
            'script': script_name,
            'success': False,
            'returncode': -1,
            'stdout': '',
            'stderr': str(e)
        }

def load_detailed_results() -> dict:
    """Load detailed results from individual test JSON files."""
    detailed_results = {}
    
    for domain_name, script_name in TEST_SCRIPTS:
        domain_key = domain_name.lower()
        result_file = f"/home/jinming/Reasoning360-MTL/scripts/tests/results_{domain_key}.json"
        
        try:
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    detailed_results[domain_key] = json.load(f)
            else:
                detailed_results[domain_key] = None
        except Exception as e:
            print(f"Warning: Could not load detailed results for {domain_name}: {e}")
            detailed_results[domain_key] = None
    
    return detailed_results

def generate_summary_report(test_results: list, detailed_results: dict) -> dict:
    """Generate comprehensive summary report."""
    total_domains = len(test_results)
    passed_domains = sum(1 for r in test_results if r['success'])
    failed_domains = total_domains - passed_domains
    
    # Collect statistics from detailed results
    total_samples_tested = 0
    total_rewards_computed = 0
    total_non_zero_rewards = 0
    data_sources_tested = set()
    
    for domain_key, details in detailed_results.items():
        if details:
            total_samples_tested += details.get('total_tests', 0)
            rewards_info = details.get('rewards', {})
            total_rewards_computed += rewards_info.get('total_computed', 0)
            total_non_zero_rewards += rewards_info.get('non_zero_count', 0)
            data_sources_tested.update(details.get('data_sources', []))
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'overview': {
            'total_domains': total_domains,
            'passed_domains': passed_domains,
            'failed_domains': failed_domains,
            'success_rate': passed_domains / total_domains if total_domains > 0 else 0
        },
        'statistics': {
            'total_samples_tested': total_samples_tested,
            'total_rewards_computed': total_rewards_computed,
            'total_non_zero_rewards': total_non_zero_rewards,
            'unique_data_sources': len(data_sources_tested),
            'data_sources_list': sorted(list(data_sources_tested))
        },
        'domain_results': test_results,
        'detailed_results': detailed_results
    }
    
    return summary

def print_final_report(summary: dict):
    """Print comprehensive final report."""
    print(f"\n{'='*80}")
    print(f"🎯 FINAL REWARD COMPUTATION TEST REPORT")
    print(f"{'='*80}")
    
    overview = summary['overview']
    stats = summary['statistics']
    
    print(f"Test completed at: {summary['timestamp']}")
    print(f"\n📊 OVERVIEW:")
    print(f"  • Total domains tested: {overview['total_domains']}")
    print(f"  • Domains passed: {overview['passed_domains']}")
    print(f"  • Domains failed: {overview['failed_domains']}")
    print(f"  • Overall success rate: {overview['success_rate']:.1%}")
    
    print(f"\n📈 STATISTICS:")
    print(f"  • Total samples tested: {stats['total_samples_tested']}")
    print(f"  • Total rewards computed: {stats['total_rewards_computed']}")
    print(f"  • Non-zero rewards: {stats['total_non_zero_rewards']}")
    print(f"  • Unique data sources: {stats['unique_data_sources']}")
    
    print(f"\n🔍 DOMAIN BREAKDOWN:")
    for result in summary['domain_results']:
        status = "✅ PASSED" if result['success'] else "❌ FAILED"
        print(f"  • {result['domain']}: {status}")
    
    print(f"\n📋 DATA SOURCES TESTED:")
    for i, ds in enumerate(stats['data_sources_list'], 1):
        print(f"  {i:2d}. {ds}")
    
    if overview['success_rate'] == 1.0:
        print(f"\n🎉 ALL REWARD COMPUTATIONS ARE WORKING CORRECTLY!")
        print(f"✅ Ready for full multinode RL training")
    else:
        print(f"\n⚠️  SOME REWARD COMPUTATIONS FAILED")
        print(f"❌ Fix issues before running full RL training")
    
    print(f"\n{'='*80}")

def main():
    """Run all reward computation tests."""
    print("🚀 Starting comprehensive reward computation testing for all 6 domains")
    print("This will test the exact same reward computation pipeline used in RL training")
    
    # Activate virtual environment
    print("\n📦 Activating virtual environment...")
    
    # Run tests for each domain
    test_results = []
    for domain_name, script_name in TEST_SCRIPTS:
        result = run_single_test(domain_name, script_name)
        test_results.append(result)
    
    # Load detailed results
    print("\n📊 Loading detailed test results...")
    detailed_results = load_detailed_results()
    
    # Generate summary report
    summary = generate_summary_report(test_results, detailed_results)
    
    # Save summary report
    summary_path = "/home/jinming/Reasoning360-MTL/scripts/tests/reward_test_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Print final report
    print_final_report(summary)
    
    print(f"\n📄 Detailed results saved to: {summary_path}")
    
    # Return success status
    overall_success = summary['overview']['success_rate'] == 1.0
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
