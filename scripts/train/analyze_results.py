#!/usr/bin/env python3

"""
Results Analysis Script for MTL vs Baseline Comparison
=====================================================

This script analyzes evaluation results from both baseline and MTL experiments
and generates comprehensive comparison reports with domain-specific metrics.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Tuple

def load_evaluation_results(results_dir: str) -> Dict:
    """Load evaluation results from log files."""
    results = {}
    
    # Domain mapping
    domain_mapping = {
        'aime': 'math', 'math': 'math',
        'humaneval': 'codegen', 'mbpp': 'codegen', 'livecodebench': 'codegen',
        'gpqa_diamond': 'stem', 'supergpqa': 'stem',
        'arcagi': 'logic', 'zebra_puzzle': 'logic',
        'codeio': 'simulation', 'cruxeval-i': 'simulation', 'cruxeval-o': 'simulation',
        'finqa': 'table', 'hitab': 'table', 'multihier': 'table',
        'livebench_reasoning': 'ood', 'livebench_language': 'ood', 
        'livebench_data_analysis': 'ood', 'ifeval': 'ood'
    }
    
    # Find all evaluation log files
    log_files = list(Path(results_dir).glob("**/logs/*_eval.log"))
    
    for log_file in log_files:
        task_name = log_file.stem.replace('_eval', '').split('_', 1)[1]  # Remove model prefix
        domain = domain_mapping.get(task_name, 'unknown')
        
        # Parse evaluation results from log file
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                # Extract metrics (this would need to be adapted based on actual log format)
                # For now, placeholder structure
                results[task_name] = {
                    'domain': domain,
                    'score': 0.0,  # Would extract from logs
                    'accuracy': 0.0,  # Would extract from logs
                    'samples': 0,   # Would extract from logs
                }
        except Exception as e:
            print(f"Warning: Could not parse {log_file}: {e}")
    
    return results

def compare_models(baseline_results: Dict, mtl_results: Dict) -> Dict:
    """Compare baseline and MTL results."""
    comparison = {}
    
    # Domain-level aggregation
    domains = set()
    for task_data in baseline_results.values():
        domains.add(task_data['domain'])
    
    for domain in domains:
        baseline_domain_scores = []
        mtl_domain_scores = []
        
        for task, data in baseline_results.items():
            if data['domain'] == domain:
                baseline_domain_scores.append(data['score'])
                if task in mtl_results:
                    mtl_domain_scores.append(mtl_results[task]['score'])
        
        if baseline_domain_scores and mtl_domain_scores:
            comparison[domain] = {
                'baseline_avg': np.mean(baseline_domain_scores),
                'mtl_avg': np.mean(mtl_domain_scores),
                'improvement': np.mean(mtl_domain_scores) - np.mean(baseline_domain_scores),
                'improvement_pct': ((np.mean(mtl_domain_scores) - np.mean(baseline_domain_scores)) / 
                                  np.mean(baseline_domain_scores) * 100) if np.mean(baseline_domain_scores) > 0 else 0,
                'tasks': len(baseline_domain_scores)
            }
    
    return comparison

def generate_report(baseline_results: Dict, mtl_results: Dict, comparison: Dict, output_file: str):
    """Generate comprehensive comparison report."""
    
    report = f"""
# MTL vs Baseline Evaluation Results
=====================================

## Summary
- **Baseline Model**: GURU approach (no MTL)
- **MTL Model**: Multi-task learning with PCGrad
- **Evaluation Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Domain-Level Comparison

| Domain | Baseline Avg | MTL Avg | Improvement | Improvement % | Tasks |
|--------|--------------|---------|-------------|---------------|-------|
"""
    
    for domain, data in comparison.items():
        report += f"| {domain.title()} | {data['baseline_avg']:.3f} | {data['mtl_avg']:.3f} | "
        report += f"{data['improvement']:+.3f} | {data['improvement_pct']:+.1f}% | {data['tasks']} |\n"
    
    report += f"""

## Task-Level Details

### Baseline Results
"""
    for task, data in baseline_results.items():
        report += f"- **{task}** ({data['domain']}): {data['score']:.3f}\n"
    
    report += f"""

### MTL Results
"""
    for task, data in mtl_results.items():
        report += f"- **{task}** ({data['domain']}): {data['score']:.3f}\n"
    
    report += f"""

## Key Findings

### Best Performing Domains (MTL)
"""
    
    # Sort domains by improvement
    sorted_domains = sorted(comparison.items(), key=lambda x: x[1]['improvement'], reverse=True)
    
    for domain, data in sorted_domains[:3]:
        report += f"1. **{domain.title()}**: +{data['improvement']:.3f} ({data['improvement_pct']:+.1f}%)\n"
    
    report += f"""

### Areas for Improvement
"""
    
    for domain, data in sorted_domains[-3:]:
        if data['improvement'] < 0:
            report += f"1. **{domain.title()}**: {data['improvement']:.3f} ({data['improvement_pct']:+.1f}%)\n"
    
    report += f"""

## Recommendations

Based on the evaluation results:

1. **Overall MTL Effectiveness**: {'Positive' if sum(d['improvement'] for d in comparison.values()) > 0 else 'Mixed'}
2. **Strongest Domains**: {', '.join([d[0].title() for d in sorted_domains[:2]])}
3. **Focus Areas**: {', '.join([d[0].title() for d in sorted_domains[-2:] if d[1]['improvement'] < 0])}

## Next Steps

1. Analyze task-specific patterns within successful domains
2. Investigate failure modes in underperforming domains  
3. Consider domain-specific hyperparameter tuning
4. Evaluate on additional out-of-distribution tasks
"""
    
    # Save report
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"📊 Comprehensive report saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze MTL vs Baseline evaluation results')
    parser.add_argument('--baseline-dir', required=True, help='Baseline results directory')
    parser.add_argument('--mtl-dir', required=True, help='MTL results directory')
    parser.add_argument('--output', default='mtl_comparison_report.md', help='Output report file')
    
    args = parser.parse_args()
    
    print("🔍 Loading evaluation results...")
    baseline_results = load_evaluation_results(args.baseline_dir)
    mtl_results = load_evaluation_results(args.mtl_dir)
    
    print(f"📊 Found {len(baseline_results)} baseline tasks, {len(mtl_results)} MTL tasks")
    
    print("⚖️ Comparing models...")
    comparison = compare_models(baseline_results, mtl_results)
    
    print("📝 Generating report...")
    generate_report(baseline_results, mtl_results, comparison, args.output)
    
    print("✅ Analysis complete!")

if __name__ == "__main__":
    main()
