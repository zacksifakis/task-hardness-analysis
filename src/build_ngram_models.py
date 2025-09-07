#!/usr/bin/env python3
import os
import json
import argparse
import pickle
from collections import defaultdict
from typing import List, Dict, Set, Tuple



# Import ngram model from the project
import sys
from ngram import NgramModel

def find_result_files(base_dir: str, model_name: str, exclude_year: int, include_gen_ids: Set[int]) -> List[str]:
    """
    Recursively find all result.json files in the specified directory
    
    Args:
        base_dir: The base directory path
        model_name: The name of the model folder
        exclude_year: Year to exclude from the search
        include_gen_ids: Set of generation IDs to include (only these IDs will be processed)
        
    Returns:
        A list of paths to result.json files
    """
    result_files = []
    model_dir = os.path.join(base_dir, model_name)
    for root, _, files in os.walk(model_dir):
        if len(files) == 0:
            continue

        # Check if exclude_year is in path_parts 
        path_parts = root.split('/')
        
        # Extract gen_id if present in the path
        gen_id = None
        for part in path_parts:
            if part.startswith("gen_"):
                try:
                    gen_id = int(part[4:])
                    break
                except ValueError:
                    pass
                    
        # Skip if exclude_year is in path OR gen_id is not in include_gen_ids
        should_skip = False
        skip_reason = []
        
        if str(exclude_year) in path_parts:
            should_skip = True
            skip_reason.append(f"exclusion of year {exclude_year}")
            
        if gen_id is not None and include_gen_ids and gen_id not in include_gen_ids:
            should_skip = True
            skip_reason.append(f"generation ID {gen_id} not in include list {include_gen_ids}")
            
        if should_skip:
            print(f"Skipping directory {root} due to " + " and ".join(skip_reason))
            continue
            
        for file in files:
            if file == "result.json":
                result_files.append(os.path.join(root, file))
    
    return result_files

def extract_token_ids(result_files: List[str]) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Extract token_ids from result files and separate them into correct and incorrect lists
    
    Args:
        result_files: List of paths to result.json files
        
    Returns:
        A tuple containing (correct_token_ids, incorrect_token_ids)
    """
    correct_token_ids = []
    incorrect_token_ids = []
    
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                result_data = json.load(f)
                
            # Ensure required fields exist
            if 'answer' not in result_data or 'extracted_answer' not in result_data or 'token_ids' not in result_data:
                print(f"Warning: Missing required fields in {file_path}")
                continue
            
            # Check if the answer is correct
            is_correct = (result_data['answer'] == result_data['extracted_answer'] and 
                          result_data['answer'] is not None and 
                          result_data['extracted_answer'] is not None)
            
            # Add token_ids to the appropriate list
            if is_correct:
                correct_token_ids.append(result_data['token_ids'])
            else:
                incorrect_token_ids.append(result_data['token_ids'])
                
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error processing {file_path}: {e}")
    
    return correct_token_ids, incorrect_token_ids

def build_ngram_models(correct_token_ids: List[List[int]], 
                      incorrect_token_ids: List[List[int]], 
                      n: int, 
                      output_dir: str,
                      model_name: str,
                      exclude_year: int = None,
                      include_gen_ids: Set[int] = None) -> None:
    """
    Build n-gram models for correct and incorrect token sequences
    
    Args:
        correct_token_ids: List of token ID sequences for correct answers
        incorrect_token_ids: List of token ID sequences for incorrect answers
        n: The order of the n-gram model
        output_dir: Directory to save the n-gram models
        model_name: Name of the model (used for file naming)
        exclude_year: Year excluded from training data
        include_gen_ids: Generation IDs included in training data
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Build n-gram model for correct answers
    print(f"Building n-gram model for {len(correct_token_ids)} correct answers with n={n}")
    correct_model = NgramModel(n=n, smoothing='laplace')
    correct_model.train(correct_token_ids)
    
    # Build n-gram model for incorrect answers
    print(f"Building n-gram model for {len(incorrect_token_ids)} incorrect answers with n={n}")
    incorrect_model = NgramModel(n=n, smoothing='laplace')
    incorrect_model.train(incorrect_token_ids)
    
    # Create filename with exclusion information
    exclusion_info = []
    if exclude_year is not None:
        exclusion_info.append(f"excl_year_{exclude_year}")
    if include_gen_ids and len(include_gen_ids) > 0:
        # Sort IDs for consistent naming
        sorted_ids = sorted(list(include_gen_ids))
        exclusion_info.append(f"incl_ids_{'_'.join(map(str, sorted_ids))}")
    
    exclusion_suffix = "_" + "_".join(exclusion_info) if exclusion_info else ""
    
    # Save the models
    correct_model_path = os.path.join(output_dir, f"{model_name}_correct_ngram_{n}{exclusion_suffix}.pkl")
    incorrect_model_path = os.path.join(output_dir, f"{model_name}_incorrect_ngram_{n}{exclusion_suffix}.pkl")
    
    correct_model.save(correct_model_path)
    incorrect_model.save(incorrect_model_path)
    
    print(f"Models saved to {correct_model_path} and {incorrect_model_path}")
    
    # Print some statistics
    print(f"\nStatistics:")
    print(f"Correct answers: {len(correct_token_ids)}")
    print(f"Incorrect answers: {len(incorrect_token_ids)}")
    print(f"Vocabulary size (correct): {len(correct_model.vocab)}")
    print(f"Vocabulary size (incorrect): {len(incorrect_model.vocab)}")
    
    # Print count of n-grams
    correct_ngram_count = sum(len(counts) for counts in correct_model.counts.values())
    incorrect_ngram_count = sum(len(counts) for counts in incorrect_model.counts.values())
    print(f"Number of distinct {n}-grams (correct): {correct_ngram_count}")
    print(f"Number of distinct {n}-grams (incorrect): {incorrect_ngram_count}")

def main():
    parser = argparse.ArgumentParser(description="Build n-gram models from token ID sequences in result.json files")
    parser.add_argument("--base_dir", type=str, required=True, 
                        help="Base directory path (up to but not including model name)")
    parser.add_argument("--model_name", type=str, required=True, 
                        help="Name of the model folder")
    parser.add_argument("--n", type=int, required=True, 
                        help="Order of the n-gram model")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save the n-gram models")
    parser.add_argument("--exclude_year", type=int, default=2025, 
                        help="Year to exclude from the search (default: 2025)")
    parser.add_argument("--include_gen_ids", type=int, nargs='*', default=[],
                        help="Generation IDs to include in the search (default: [])")

    args = parser.parse_args()

    # Find all result.json files
    print(f"Searching for result.json files in {os.path.join(args.base_dir, args.model_name)}")
    result_files = find_result_files(args.base_dir, args.model_name, args.exclude_year, set(args.include_gen_ids))
    print(f"Found {len(result_files)} result.json files")
    
    # Extract token IDs
    correct_token_ids, incorrect_token_ids = extract_token_ids(result_files)
    print(f"Extracted {len(correct_token_ids)} correct sequences and {len(incorrect_token_ids)} incorrect sequences")
    
    # Build and save n-gram models
    build_ngram_models(correct_token_ids, incorrect_token_ids, args.n, args.output_dir, args.model_name, args.exclude_year, set(args.include_gen_ids))

if __name__ == "__main__":
    main()
