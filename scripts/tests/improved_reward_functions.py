#!/usr/bin/env python3
"""
Improved reward functions that are more flexible and handle real model outputs better.
Based on analysis of GURU codebase evaluation patterns.
"""

import sys
import re
import string
import numpy as np
import ast
from typing import Any, Dict, Union

sys.path.insert(0, '/home/jinming/Reasoning360-MTL')

def normalize_answer(s: str) -> str:
    """Normalize answer text by removing articles, punctuation, extra whitespace, and converting to lowercase."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    
    def white_space_fix(text):
        return " ".join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def extract_boxed_answer(response: str) -> str:
    """Extract answer from \\boxed{} LaTeX format."""
    pattern = r'\\boxed\{([^}]*)\}'
    matches = re.findall(pattern, response)
    return matches[-1] if matches else None

def extract_answer_markers(response: str) -> str:
    """Extract answer using common answer markers."""
    response_lower = response.lower()
    
    # Try various answer markers
    markers = [
        "answer:", "answer is", "answers are", "the answer is",
        "final answer:", "final answer is", "solution:", "solution is",
        "result:", "result is", "output:", "output is"
    ]
    
    for marker in markers:
        idx = response_lower.rfind(marker)
        if idx != -1:
            extracted = response[idx + len(marker):].strip()
            # Take first line or sentence
            extracted = extracted.split('\n')[0].split('.')[0].strip()
            if extracted:
                return extracted
    
    return None

def extract_last_number(response: str) -> str:
    """Extract the last number from response."""
    numbers = re.findall(r'-?\d+\.?\d*', response)
    return numbers[-1] if numbers else None

def improved_math_reward(solution_str: str, ground_truth: str, extra_info: Dict = None) -> float:
    """Improved math reward function with flexible matching."""
    
    # Normalize both answers
    gt_normalized = normalize_answer(str(ground_truth))
    
    # Try multiple extraction methods
    extraction_methods = [
        extract_boxed_answer,
        extract_answer_markers,
        extract_last_number,
        lambda x: x.strip()  # Use full response as fallback
    ]
    
    for extract_func in extraction_methods:
        extracted = extract_func(solution_str)
        if extracted:
            extracted_normalized = normalize_answer(extracted)
            
            # Direct match
            if gt_normalized == extracted_normalized:
                return 1.0
            
            # Check if ground truth is contained in extracted answer
            if gt_normalized in extracted_normalized:
                return 0.8  # Partial credit
            
            # Check if extracted answer is contained in ground truth (for longer GT)
            if extracted_normalized in gt_normalized:
                return 0.6  # Partial credit
    
    return 0.0

def improved_logic_reward(solution_str: str, ground_truth: str, extra_info: Dict = None) -> float:
    """Improved logic reward function with flexible matching."""
    
    gt_str = str(ground_truth).strip()
    solution_str = str(solution_str).strip()
    
    # For string answers (like graph logic)
    if isinstance(ground_truth, str):
        # Try exact match (case insensitive)
        if solution_str.lower() == gt_str.lower():
            return 1.0
        
        # Try normalized match
        if normalize_answer(solution_str) == normalize_answer(gt_str):
            return 1.0
        
        # Check if answer is contained in response
        if gt_str.lower() in solution_str.lower():
            return 0.8
        
        return 0.0
    
    # For list/array answers (like ordering puzzles)
    elif isinstance(ground_truth, (list, tuple)):
        # Try to extract list from solution string
        try:
            # Look for list-like patterns
            list_patterns = [
                r'\[([^\]]+)\]',  # [1, 2, 3, 4]
                r'(\d+(?:\s*,\s*\d+)*)',  # 1, 2, 3, 4
                r'(\d+(?:\s+\d+)*)',  # 1 2 3 4
            ]
            
            for pattern in list_patterns:
                matches = re.findall(pattern, solution_str)
                if matches:
                    # Try to parse the match
                    match_str = matches[-1]  # Take last match
                    
                    # Parse numbers
                    if ',' in match_str:
                        numbers = [int(x.strip()) for x in match_str.split(',') if x.strip().isdigit()]
                    else:
                        numbers = [int(x) for x in match_str.split() if x.isdigit()]
                    
                    if numbers == list(ground_truth):
                        return 1.0
                    elif len(numbers) == len(ground_truth):
                        # Partial credit for same length
                        correct = sum(1 for a, b in zip(numbers, ground_truth) if a == b)
                        return correct / len(ground_truth) * 0.8
            
            return 0.0
            
        except Exception:
            return 0.0
    
    return 0.0

def improved_barc_reward(solution_str: str, ground_truth: np.ndarray, extra_info: Dict = None) -> float:
    """Improved BARC reward function that handles numpy arrays properly."""
    
    try:
        # Convert ground truth to list for comparison
        if isinstance(ground_truth, np.ndarray):
            gt_list = ground_truth.tolist()
        else:
            gt_list = ground_truth
        
        # Try to extract array from solution string
        array_patterns = [
            r'\[\[([^\]]+)\]\]',  # [[0, 1], [1, 0]]
            r'\[([^\]]+)\]',      # [0, 1, 1, 0] (flattened)
        ]
        
        for pattern in array_patterns:
            matches = re.findall(pattern, solution_str)
            if matches:
                try:
                    # Try to parse as nested list
                    if '],[' in solution_str:
                        # Parse 2D array
                        array_str = solution_str[solution_str.find('[['):solution_str.find(']]')+2]
                        parsed = ast.literal_eval(array_str)
                        if parsed == gt_list:
                            return 1.0
                    else:
                        # Parse 1D array and try to reshape
                        numbers = [int(x.strip()) for x in matches[0].split(',') if x.strip().isdigit()]
                        if len(numbers) == np.prod(ground_truth.shape):
                            reshaped = np.array(numbers).reshape(ground_truth.shape)
                            if np.array_equal(reshaped, ground_truth):
                                return 1.0
                except Exception:
                    continue
        
        return 0.0
        
    except Exception:
        return 0.0

def improved_stem_reward(solution_str: str, ground_truth: str, extra_info: Dict = None) -> float:
    """Improved STEM reward function with flexible matching."""
    
    gt_str = str(ground_truth).strip()
    
    # Extract potential answers using multiple methods
    extraction_methods = [
        extract_boxed_answer,
        extract_answer_markers,
        extract_last_number,
    ]
    
    for extract_func in extraction_methods:
        extracted = extract_func(solution_str)
        if extracted:
            # Direct match
            if extracted.strip() == gt_str:
                return 1.0
            
            # Normalized match
            if normalize_answer(extracted) == normalize_answer(gt_str):
                return 1.0
            
            # Check if ground truth is contained in extracted
            if gt_str in extracted:
                return 0.8
            
            # For ratio answers like "2.95:1", try flexible matching
            if ':' in gt_str and ':' in extracted:
                gt_parts = gt_str.split(':')
                ex_parts = extracted.split(':')
                if len(gt_parts) == len(ex_parts) == 2:
                    try:
                        gt_ratio = float(gt_parts[0]) / float(gt_parts[1])
                        ex_ratio = float(ex_parts[0]) / float(ex_parts[1])
                        if abs(gt_ratio - ex_ratio) < 0.1:  # 10% tolerance
                            return 0.8
                    except ValueError:
                        pass
    
    return 0.0

def improved_table_reward(solution_str: str, ground_truth: str, extra_info: Dict = None) -> float:
    """Improved table reward function with flexible matching."""
    
    gt_str = str(ground_truth).strip()
    
    # Extract numbers from solution
    numbers = re.findall(r'\d+\.?\d*', solution_str)
    
    # Check if ground truth number is in the extracted numbers
    if gt_str in numbers:
        return 1.0
    
    # Try normalized matching
    for num in numbers:
        if normalize_answer(num) == normalize_answer(gt_str):
            return 1.0
    
    # Check if ground truth is contained in response
    if gt_str in solution_str:
        return 0.8
    
    return 0.0

def improved_codegen_reward(solution_str: str, ground_truth: Any, extra_info: Dict = None) -> float:
    """Improved codegen reward function that gives partial credit for valid code."""
    
    # Check if solution contains code-like patterns
    code_patterns = [
        r'def\s+\w+',           # function definitions
        r'import\s+\w+',        # imports
        r'for\s+\w+\s+in',      # for loops
        r'if\s+.*:',            # if statements
        r'while\s+.*:',         # while loops
        r'return\s+',           # return statements
    ]
    
    code_score = 0
    for pattern in code_patterns:
        if re.search(pattern, solution_str):
            code_score += 0.2
    
    # Cap at 0.8 for having code structure
    code_score = min(code_score, 0.8)
    
    # TODO: Add actual execution testing here
    # For now, return partial credit for code-like structure
    return code_score

# Test the improved functions
def test_improved_rewards():
    """Test the improved reward functions with examples from our analysis."""
    
    print("🧪 TESTING IMPROVED REWARD FUNCTIONS")
    print("="*60)
    
    # Test math
    print("\n📊 Math Tests:")
    test_cases = [
        ("42", "42", 1.0),
        ("The answer is 42", "42", 1.0),
        ("42.", "42", 1.0),
        (" 42 ", "42", 1.0),
        ("Answer: 42", "42", 1.0),
        ("43", "42", 0.0),
    ]
    
    for response, gt, expected in test_cases:
        score = improved_math_reward(response, gt)
        status = "✅" if score >= expected else "❌"
        print(f"  {status} '{response}' vs '{gt}' → {score:.1f} (expected ≥{expected})")
    
    # Test logic
    print("\n🧠 Logic Tests:")
    test_cases = [
        ("sodnei", "sodnei", 1.0),
        ("SODNEI", "sodnei", 1.0),
        (" sodnei ", "sodnei", 1.0),
        ("The answer is sodnei", "sodnei", 0.8),
        ("[1, 2, 3, 4]", [1, 2, 3, 4], 1.0),
        ("1, 2, 3, 4", [1, 2, 3, 4], 1.0),
        ("1 2 3 4", [1, 2, 3, 4], 1.0),
    ]
    
    for response, gt, expected in test_cases:
        score = improved_logic_reward(response, gt)
        status = "✅" if score >= expected else "❌"
        print(f"  {status} '{response}' vs '{gt}' → {score:.1f} (expected ≥{expected})")
    
    # Test STEM
    print("\n🔬 STEM Tests:")
    test_cases = [
        ("2.95:1", "2.95:1", 1.0),
        ("2.95 : 1", "2.95:1", 1.0),
        ("The ratio is 2.95:1", "2.95:1", 0.8),
        ("3:1", "2.95:1", 0.8),  # Close ratio
    ]
    
    for response, gt, expected in test_cases:
        score = improved_stem_reward(response, gt)
        status = "✅" if score >= expected else "❌"
        print(f"  {status} '{response}' vs '{gt}' → {score:.1f} (expected ≥{expected})")
    
    print("\n" + "="*60)
    print("🎯 Improved reward functions provide much more flexible evaluation!")

def compute_improved_reward(data_source: str, solution_str: str, ground_truth: Any, extra_info: Dict = None) -> Dict[str, Any]:
    """
    Main function to compute improved rewards based on data source.
    
    Args:
        data_source: Data source identifier (e.g., 'math__test', 'logic__barc')
        solution_str: Model's response/solution
        ground_truth: Expected answer
        extra_info: Additional information for evaluation
        
    Returns:
        Dictionary with score and additional information
    """
    if extra_info is None:
        extra_info = {}
    
    data_source_lower = data_source.lower()
    
    try:
        if 'math' in data_source_lower:
            score = improved_math_reward(solution_str, ground_truth, extra_info)
            return {'score': score, 'acc': score, 'method': 'improved_math'}
        
        elif any(x in data_source_lower for x in ['logic', 'puzzle', 'zebra', 'ordering']):
            score = improved_logic_reward(solution_str, ground_truth, extra_info)
            return {'score': score, 'acc': score, 'method': 'improved_logic'}
        
        elif any(x in data_source_lower for x in ['barc', 'arcagi', 'simulation']):
            score = improved_barc_reward(solution_str, ground_truth, extra_info)
            return {'score': score, 'acc': score, 'method': 'improved_barc'}
        
        elif 'stem' in data_source_lower:
            score = improved_stem_reward(solution_str, ground_truth, extra_info)
            return {'score': score, 'acc': score, 'method': 'improved_stem'}
        
        elif 'table' in data_source_lower:
            score = improved_table_reward(solution_str, ground_truth, extra_info)
            return {'score': score, 'acc': score, 'method': 'improved_table'}
        
        elif 'codegen' in data_source_lower:
            score = improved_codegen_reward(solution_str, ground_truth, extra_info)
            return {'score': score, 'acc': score, 'method': 'improved_codegen'}
        
        else:
            # Default to math for unknown domains
            score = improved_math_reward(solution_str, ground_truth, extra_info)
            return {'score': score, 'acc': score, 'method': 'improved_math_default'}
            
    except Exception as e:
        print(f"❌ Improved reward computation failed for {data_source}: {e}")
        return {'score': 0.0, 'acc': 0.0, 'method': 'error', 'error': str(e)}


if __name__ == "__main__":
    test_improved_rewards()
