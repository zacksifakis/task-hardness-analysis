#!/usr/bin/env python3
import sys
from extract_answers import extract_answer_from_response

def test_extract_answers():
    """
    Test the extract_answer_from_response function with different boxed expressions.
    """
    test_cases = [
        # Simple case
        "The answer is \\boxed{32}",
        
        # Fraction case
        "After calculation, we get \\boxed{\\frac{42}{51}}",
        
        # Multiple boxed expressions (should return the last one)
        "First we have \\boxed{10} but later we correct to \\boxed{32}",
        
        # Complex nested expression
        "The final result is \\boxed{\\sqrt{16} = \\frac{4}{1}}",
        
        # Very complex nesting
        "The answer is \\boxed{\\frac{1}{1 + \\frac{1}{2 + \\frac{1}{3}}}}",
    ]
    
    print("Testing extract_answer_from_response function...")
    for i, test_case in enumerate(test_cases):
        result = extract_answer_from_response(test_case)
        print(f"\nTest case {i+1}:")
        print(f"Input: {test_case}")
        print(f"Extracted answer: {result}")
    
    print("\nSpecific test cases from the prompt:")
    
    # Test the specific examples from the prompt
    example1 = "\\boxed{32}"
    example2 = "\\boxed{\\frac{42}{51}}"
    
    result1 = extract_answer_from_response(example1)
    result2 = extract_answer_from_response(example2)
    
    print(f"1. Input: {example1}")
    print(f"   Extracted answer: {result1}")
    print(f"2. Input: {example2}")
    print(f"   Extracted answer: {result2}")

if __name__ == "__main__":
    test_extract_answers()
