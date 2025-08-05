#!/usr/bin/env python3
"""
Constrained Decoding Solution for GURU Reward Model Compatibility

This script demonstrates how to use constrained decoding to enforce output formats
that are compatible with GURU's strict reward evaluation functions.

The main issues identified:
1. Models generate reasonable answers but don't follow strict formatting requirements
2. Math domain expects \\boxed{answer} format
3. Logic domain expects <answer>answer</answer> format
4. STEM domain expects \\boxed{answer} format
5. Other domains have similar strict format requirements

Solutions implemented:
1. Outlines-based constrained decoding for format enforcement
2. Post-processing to add missing format markers
3. Enhanced prompt templates with stronger format instructions
4. Fallback mechanisms for format correction
"""

import re
import json
import ast
from typing import Dict, List, Any, Optional, Union
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

# Try to import outlines for constrained decoding
try:
    import outlines
    OUTLINES_AVAILABLE = True
    print("✅ Outlines library available for constrained decoding")
except ImportError:
    OUTLINES_AVAILABLE = False
    print("❌ Outlines library not available. Install with: pip install outlines")

class ConstrainedDecodingRewardEnforcer:
    """
    Enforces output formats compatible with GURU reward functions using constrained decoding
    and post-processing techniques.
    """
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.outlines_generators = {}
        
        # Domain-specific format patterns
        self.format_patterns = {
            'math': {
                'required_format': r'\\boxed\{[^}]+\}',
                'extraction_pattern': r'\\boxed\{([^}]+)\}',
                'template': '\\boxed{{{answer}}}',
                'instruction': 'Please put your final answer within \\boxed{} format.'
            },
            'logic': {
                'required_format': r'<answer>.*?</answer>',
                'extraction_pattern': r'<answer>(.*?)</answer>',
                'template': '<answer>{answer}</answer>',
                'instruction': 'Please put your answer within <answer> and </answer> tags.'
            },
            'stem': {
                'required_format': r'\\boxed\{[^}]+\}',
                'extraction_pattern': r'\\boxed\{([^}]+)\}',
                'template': '\\boxed{{{answer}}}',
                'instruction': 'Please put your final answer within \\boxed{} format.'
            },
            'barc': {
                'required_format': r'<answer>.*?</answer>',
                'extraction_pattern': r'<answer>(.*?)</answer>',
                'template': '<answer>{answer}</answer>',
                'instruction': 'Please put your answer within <answer> and </answer> tags, your final answer should be only the output grid (2d array).'
            },
            'codegen': {
                'required_format': r'```python.*?```',
                'extraction_pattern': r'```python\n(.*?)\n```',
                'template': '```python\n{answer}\n```',
                'instruction': 'Please put your code within ```python and ``` code blocks.'
            },
            'table': {
                'required_format': r'\\boxed\{[^}]+\}',
                'extraction_pattern': r'\\boxed\{([^}]+)\}',
                'template': '\\boxed{{{answer}}}',
                'instruction': 'Please put your final answer within \\boxed{} format.'
            }
        }
    
    def load_model(self, model_path: str):
        """Load model and tokenizer for constrained decoding."""
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if OUTLINES_AVAILABLE:
            self.model = outlines.models.transformers(model_path)
            print(f"✅ Loaded model with Outlines support: {model_path}")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            print(f"⚠️ Loaded model without constrained decoding: {model_path}")
    
    def create_enhanced_prompt(self, original_prompt: List[Dict], domain: str) -> List[Dict]:
        """
        Enhance the original prompt with stronger format instructions.
        """
        if domain not in self.format_patterns:
            return original_prompt
        
        format_info = self.format_patterns[domain]
        enhanced_prompt = original_prompt.copy()
        
        # Add or enhance system message with format instructions
        system_message = None
        for msg in enhanced_prompt:
            if msg['role'] == 'system':
                system_message = msg
                break
        
        if system_message:
            # Enhance existing system message
            if format_info['instruction'] not in system_message['content']:
                system_message['content'] += f"\n\nIMPORTANT: {format_info['instruction']}"
        else:
            # Add new system message
            enhanced_prompt.insert(0, {
                'role': 'system',
                'content': f"You are a helpful assistant. {format_info['instruction']}"
            })
        
        # Enhance user message if needed
        for msg in enhanced_prompt:
            if msg['role'] == 'user':
                if format_info['instruction'] not in msg['content']:
                    msg['content'] += f"\n\n{format_info['instruction']}"
                break
        
        return enhanced_prompt
    
    def create_format_regex(self, domain: str) -> str:
        """
        Create regex pattern for constrained decoding based on domain.
        """
        if domain == 'math' or domain == 'stem' or domain == 'table':
            # Math answer format: any text followed by \\boxed{answer}
            return r".*\\boxed\{[^}]+\}"
        elif domain == 'logic' or domain == 'barc':
            # Logic answer format: any text followed by <answer>content</answer>
            return r".*<answer>.*?</answer>"
        elif domain == 'codegen':
            # Code format: any text followed by ```python code ```
            return r".*```python\n.*?\n```"
        else:
            # Default: any text (no constraint)
            return r".*"
    
    def generate_with_constraints(self, prompt: str, domain: str, max_tokens: int = 512) -> str:
        """
        Generate text with format constraints using Outlines.
        """
        if not OUTLINES_AVAILABLE or self.model is None:
            raise ValueError("Outlines not available or model not loaded")
        
        # Create or get cached generator for this domain
        if domain not in self.outlines_generators:
            regex_pattern = self.create_format_regex(domain)
            self.outlines_generators[domain] = outlines.generate.regex(
                self.model, regex_pattern
            )
        
        generator = self.outlines_generators[domain]
        response = generator(prompt)
        return response
    
    def post_process_response(self, response: str, domain: str) -> str:
        """
        Post-process response to ensure proper formatting.
        """
        if domain not in self.format_patterns:
            return response
        
        format_info = self.format_patterns[domain]
        
        # Check if response already has correct format
        if re.search(format_info['required_format'], response, re.DOTALL):
            return response
        
        # Try to extract answer and reformat
        answer = self.extract_answer_heuristic(response, domain)
        if answer:
            formatted_answer = format_info['template'].format(answer=answer)
            return response + "\n\n" + formatted_answer
        
        return response
    
    def extract_answer_heuristic(self, response: str, domain: str) -> Optional[str]:
        """
        Extract answer using domain-specific heuristics.
        """
        if domain == 'math' or domain == 'stem' or domain == 'table':
            return self.extract_math_answer(response)
        elif domain == 'logic':
            return self.extract_logic_answer(response)
        elif domain == 'barc':
            return self.extract_barc_answer(response)
        elif domain == 'codegen':
            return self.extract_code_answer(response)
        
        return None
    
    def extract_math_answer(self, response: str) -> Optional[str]:
        """Extract mathematical answer from response."""
        # Look for common answer patterns
        patterns = [
            r'(?:final answer|answer|result|solution)(?:\s*is\s*|\s*:\s*)([^.\n]+)',
            r'(?:therefore|thus|hence)(?:\s*,?\s*)([^.\n]+)',
            r'(?:=\s*)([^.\n]+)(?:\s*$|\s*\.$)',
            r'(\d+(?:\.\d+)?(?:/\d+)?)',  # Numbers
            r'([a-zA-Z]+)',  # Single words
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                # Clean up common artifacts
                answer = re.sub(r'^[^\w\d]+|[^\w\d]+$', '', answer)
                if answer:
                    return answer
        
        return None
    
    def extract_logic_answer(self, response: str) -> Optional[str]:
        """Extract logic answer from response."""
        # Look for list-like answers
        patterns = [
            r'\[([^\]]+)\]',  # [item1, item2, ...]
            r'(?:answer|solution)(?:\s*is\s*|\s*:\s*)([^.\n]+)',
            r'(?:therefore|thus|hence)(?:\s*,?\s*)([^.\n]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                if answer:
                    return answer
        
        return None
    
    def extract_barc_answer(self, response: str) -> Optional[str]:
        """Extract BARC grid answer from response."""
        # Look for 2D array patterns
        patterns = [
            r'\[\[([^\]]+)\]\]',  # [[row1], [row2], ...]
            r'(?:output|result)(?:\s*is\s*|\s*:\s*)(\[\[.*?\]\])',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                answer = match.group(1).strip()
                if answer:
                    return answer
        
        return None
    
    def extract_code_answer(self, response: str) -> Optional[str]:
        """Extract code answer from response."""
        # Look for code blocks or function definitions
        patterns = [
            r'```python\n(.*?)\n```',
            r'def\s+\w+.*?(?=\n\n|\n$|\Z)',
            r'(?:solution|answer)(?:\s*is\s*|\s*:\s*)(.*?)(?=\n\n|\n$|\Z)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                answer = match.group(1).strip()
                if answer:
                    return answer
        
        return None


def create_format_enforced_prompts():
    """
    Create enhanced prompt templates with stronger format enforcement.
    """
    enhanced_prompts = {
        'math': {
            'system': "You are a mathematical problem solver. Always provide your final answer in the exact format \\boxed{answer} where 'answer' is your numerical or algebraic result. Do not include any text after the boxed answer.",
            'user_suffix': "\n\nIMPORTANT: You must put your final answer within \\boxed{} format. For example: \\boxed{42} or \\boxed{x + 1}."
        },
        'logic': {
            'system': "You are a logical reasoning expert. Always provide your final answer in the exact format <answer>your_answer</answer>. Do not include any text after the answer tags.",
            'user_suffix': "\n\nIMPORTANT: You must put your answer within <answer> and </answer> tags. For example: <answer>[1, 2, 3]</answer>."
        },
        'stem': {
            'system': "You are a STEM problem solver. Always provide your final answer in the exact format \\boxed{answer} where 'answer' is your result. Do not include any text after the boxed answer.",
            'user_suffix': "\n\nIMPORTANT: You must put your final answer within \\boxed{} format."
        },
        'barc': {
            'system': "You are a pattern recognition expert. Always provide your final answer as a 2D array within <answer> and </answer> tags. Do not include any text after the answer tags.",
            'user_suffix': "\n\nIMPORTANT: You must put your answer within <answer> and </answer> tags. Your final answer should be only the output grid (2d array)."
        },
        'codegen': {
            'system': "You are a code generation expert. Always provide your final code within ```python and ``` code blocks. Do not include any text after the code block.",
            'user_suffix': "\n\nIMPORTANT: You must put your code within ```python and ``` code blocks."
        },
        'table': {
            'system': "You are a table analysis expert. Always provide your final answer in the exact format \\boxed{answer}. Do not include any text after the boxed answer.",
            'user_suffix': "\n\nIMPORTANT: You must put your final answer within \\boxed{} format."
        }
    }
    return enhanced_prompts


def test_constrained_decoding():
    """
    Test the constrained decoding solution with sample prompts.
    """
    print("🧪 Testing Constrained Decoding Solution")
    print("=" * 60)
    
    enforcer = ConstrainedDecodingRewardEnforcer()
    
    # Test cases for different domains
    test_cases = [
        {
            'domain': 'math',
            'prompt': [{'role': 'user', 'content': 'What is 2 + 2?'}],
            'expected_format': r'\\boxed\{.*\}'
        },
        {
            'domain': 'logic',
            'prompt': [{'role': 'user', 'content': 'List the first three prime numbers.'}],
            'expected_format': r'<answer>.*</answer>'
        },
        {
            'domain': 'stem',
            'prompt': [{'role': 'user', 'content': 'What is the speed of light?'}],
            'expected_format': r'\\boxed\{.*\}'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}: {test_case['domain'].upper()} Domain")
        print("-" * 40)
        
        # Test prompt enhancement
        enhanced_prompt = enforcer.create_enhanced_prompt(
            test_case['prompt'], test_case['domain']
        )
        print("Enhanced prompt:")
        for msg in enhanced_prompt:
            print(f"  {msg['role']}: {msg['content'][:100]}...")
        
        # Test format regex
        regex_pattern = enforcer.create_format_regex(test_case['domain'])
        print(f"Format regex: {regex_pattern}")
        
        # Test post-processing with sample responses
        sample_responses = [
            "The answer is 4.",
            "After calculation, I get 4",
            "Therefore, the result is 4."
        ]
        
        for response in sample_responses:
            processed = enforcer.post_process_response(response, test_case['domain'])
            print(f"  Original: {response}")
            print(f"  Processed: {processed}")
            
            # Check if format is correct
            import re
            if re.search(test_case['expected_format'], processed, re.DOTALL):
                print("  ✅ Format correct")
            else:
                print("  ❌ Format incorrect")
            print()


def integration_example():
    """
    Example of how to integrate constrained decoding into the existing VERL pipeline.
    """
    print("\n🔧 Integration Example")
    print("=" * 60)
    
    print("""
To integrate this solution into the VERL pipeline:

1. **Modify the generation worker** (verl/workers/fsdp_workers.py):
   - Add ConstrainedDecodingRewardEnforcer to the rollout worker
   - Use enhanced prompts during generation
   - Apply post-processing to ensure format compliance

2. **Update the dataset loading** (verl/utils/dataset/rl_dataset.py):
   - Enhance prompts with stronger format instructions
   - Add domain detection based on data_source

3. **Modify the reward computation** (verl/utils/reward_score/__init__.py):
   - Add fallback format correction before reward computation
   - Use improved reward functions as backup

4. **Environment setup**:
   - Install Outlines: pip install outlines
   - Configure constrained decoding parameters

Example code integration:

```python
# In rollout worker
from scripts.tests.constrained_decoding_solution import ConstrainedDecodingRewardEnforcer

class ActorRolloutRefWorker:
    def __init__(self, ...):
        # ... existing code ...
        self.format_enforcer = ConstrainedDecodingRewardEnforcer()
        if hasattr(self, 'model_path'):
            self.format_enforcer.load_model(self.model_path)
    
    def generate_sequences(self, data):
        # ... existing code ...
        
        # Enhance prompts with format instructions
        for item in data:
            domain = self.detect_domain(item.get('data_source', ''))
            if 'prompt' in item:
                item['prompt'] = self.format_enforcer.create_enhanced_prompt(
                    item['prompt'], domain
                )
        
        # Generate with constraints if available
        if self.format_enforcer.model and OUTLINES_AVAILABLE:
            # Use constrained generation
            responses = self.generate_with_constraints(data)
        else:
            # Use standard generation + post-processing
            responses = self.standard_generate(data)
            responses = [
                self.format_enforcer.post_process_response(resp, domain)
                for resp in responses
            ]
        
        return responses
```
    """)


if __name__ == "__main__":
    test_constrained_decoding()
    integration_example()
    
    print("\n🎯 Summary")
    print("=" * 60)
    print("""
This solution addresses the reward evaluation issues by:

✅ **Format Enforcement**: Uses constrained decoding to ensure outputs follow required formats
✅ **Enhanced Prompts**: Adds stronger format instructions to existing prompts  
✅ **Post-Processing**: Applies format correction as fallback
✅ **Domain-Specific**: Handles different format requirements per domain
✅ **Backward Compatible**: Works with existing VERL pipeline

**Next Steps**:
1. Install Outlines library for constrained decoding
2. Integrate format enforcer into rollout workers
3. Test with actual model generation
4. Monitor reward score improvements
5. Fine-tune format patterns based on results

**Expected Impact**:
- Significantly reduce zero reward scores
- Maintain answer quality while improving format compliance
- Enable proper evaluation of model reasoning capabilities
- Support both strict GURU evaluation and flexible fallbacks
    """)
