#!/usr/bin/env python3
"""
Analyze if our reward functions are too strict by testing with manually crafted responses.
"""

import sys
import os
sys.path.insert(0, '/home/jinming/Reasoning360-MTL')

from verl.utils.reward_score import default_compute_score
import numpy as np

def test_math_rewards():
    """Test math reward function with various response formats."""
    print("🧮 TESTING MATH REWARD STRICTNESS")
    print("="*60)
    
    ground_truth = "42"
    test_cases = [
        ("42", "Exact match"),
        ("The answer is 42", "Answer in sentence"),
        ("42.", "With decimal point"),
        (" 42 ", "With whitespace"),
        ("42\n", "With newline"),
        ("Answer: 42", "With prefix"),
        ("42 is the answer", "Answer at start"),
        ("forty-two", "Written out"),
        ("43", "Wrong answer"),
    ]
    
    for response, description in test_cases:
        try:
            score = default_compute_score(
                data_source="math__merged_deduped_dapo_or1_dataset",
                solution_str=response,
                ground_truth=ground_truth,
                extra_info={"reward_metric": "default"}
            )
            final_score = score.get('score', score) if isinstance(score, dict) else score
            print(f"  {description:20} | '{response:15}' → {final_score}")
        except Exception as e:
            print(f"  {description:20} | '{response:15}' → ERROR: {e}")

def test_logic_rewards():
    """Test logic reward function with various formats."""
    print("\n🧠 TESTING LOGIC REWARD STRICTNESS")
    print("="*60)
    
    # Test graph logic
    ground_truth = "sodnei"
    test_cases = [
        ("sodnei", "Exact match"),
        ("SODNEI", "Uppercase"),
        ("sodnei\n", "With newline"),
        (" sodnei ", "With whitespace"),
        ("The answer is sodnei", "In sentence"),
        ("wrong", "Wrong answer"),
    ]
    
    print("Graph Logic Tests:")
    for response, description in test_cases:
        try:
            score = default_compute_score(
                data_source="logic__graph_logical_dataset",
                solution_str=response,
                ground_truth=ground_truth,
                extra_info={}
            )
            final_score = score.get('score', score) if isinstance(score, dict) else score
            print(f"  {description:20} | '{response:20}' → {final_score}")
        except Exception as e:
            print(f"  {description:20} | '{response:20}' → ERROR: {e}")
    
    # Test ordering puzzle
    print("\nOrdering Puzzle Tests:")
    ground_truth = [1, 2, 3, 4]
    test_cases = [
        ("[1, 2, 3, 4]", "Exact string match"),
        ("1, 2, 3, 4", "Without brackets"),
        ("1 2 3 4", "Space separated"),
        ("[1,2,3,4]", "No spaces"),
        ([1, 2, 3, 4], "Actual list"),
        ("[2, 1, 3, 4]", "Wrong order"),
    ]
    
    for response, description in test_cases:
        try:
            score = default_compute_score(
                data_source="logic__ordering_puzzle_dataset",
                solution_str=response,
                ground_truth=ground_truth,
                extra_info={}
            )
            final_score = score.get('score', score) if isinstance(score, dict) else score
            print(f"  {description:20} | '{str(response):20}' → {final_score}")
        except Exception as e:
            print(f"  {description:20} | '{str(response):20}' → ERROR: {e}")

def test_barc_rewards():
    """Test BARC (ARC-AGI) reward function."""
    print("\n🎯 TESTING BARC/ARC-AGI REWARD STRICTNESS")
    print("="*60)
    
    # Create a simple 2x2 ground truth array
    ground_truth = np.array([[0, 1], [1, 0]])
    
    test_cases = [
        ("[[0, 1], [1, 0]]", "String representation"),
        ("[0, 1, 1, 0]", "Flattened array"),
        ("0 1\n1 0", "Grid format"),
        (np.array([[0, 1], [1, 0]]), "Exact numpy array"),
        (np.array([[1, 0], [0, 1]]), "Wrong array"),
    ]
    
    for response, description in test_cases:
        try:
            score = default_compute_score(
                data_source="simulation__barc",
                solution_str=response,
                ground_truth=ground_truth,
                extra_info={}
            )
            final_score = score.get('score', score) if isinstance(score, dict) else score
            print(f"  {description:20} | '{str(response)[:20]:20}' → {final_score}")
        except Exception as e:
            print(f"  {description:20} | '{str(response)[:20]:20}' → ERROR: {e}")

def test_stem_rewards():
    """Test STEM reward function."""
    print("\n🔬 TESTING STEM REWARD STRICTNESS")
    print("="*60)
    
    ground_truth = "2.95:1"
    test_cases = [
        ("2.95:1", "Exact match"),
        ("2.95 : 1", "With spaces"),
        ("2.95 to 1", "Different format"),
        ("The ratio is 2.95:1", "In sentence"),
        ("2.95", "Just the number"),
        ("3:1", "Approximate"),
        ("wrong", "Wrong answer"),
    ]
    
    for response, description in test_cases:
        try:
            score = default_compute_score(
                data_source="stem_web",
                solution_str=response,
                ground_truth=ground_truth,
                extra_info={"answer_type": "Float", "category": "Chemistry"}
            )
            final_score = score if isinstance(score, (int, float)) else score.get('score', score)
            print(f"  {description:20} | '{response:20}' → {final_score}")
        except Exception as e:
            print(f"  {description:20} | '{response:20}' → ERROR: {e}")

def main():
    """Run all reward strictness tests."""
    print("🔍 ANALYZING REWARD FUNCTION STRICTNESS")
    print("This tests if our reward functions are too strict")
    print("="*80)
    
    test_math_rewards()
    test_logic_rewards()
    test_barc_rewards()
    test_stem_rewards()
    
    print("\n" + "="*80)
    print("🎯 ANALYSIS COMPLETE")
    print("Look for patterns where reasonable responses get 0.0 rewards")
    print("This will help us understand if evaluation is too strict")
    print("="*80)

if __name__ == "__main__":
    main()
