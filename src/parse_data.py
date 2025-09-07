#!/usr/bin/env python3

import os
import json
import glob
import argparse
import numpy as np
import pandas as pd
import re
from pathlib import Path
from ngram import NgramModel
from tqdm import tqdm


# Helper function for n-gram processing
def calc_logprob_diff_single(token_ids, ngram_correct, ngram_incorrect):
    """Calculate log probability difference for a single row."""
    if token_ids:
        result = ngram_correct.log_probability(token_ids) - ngram_incorrect.log_probability(token_ids)
    else:
        result = 0.0
    return result

# Process a batch of rows for parallel processing
def process_batch(batch_df, ngram_correct, ngram_incorrect):
    """Process a batch of DataFrame rows."""
    return [calc_logprob_diff_single(row['token_ids'], ngram_correct, ngram_incorrect) for _, row in batch_df.iterrows()]


# Define confusion markers
CONFUSION_MARKERS = ["wait", "hmm", "maybe", "perhaps", "confused", "unsure", "rethink", "?", 
                     "oops", "hold on", "stuck", "messy", "complicated", "unclear", "digress"]

def calculate_confusion_metric(text):
    """
    Calculate confusion metric based on frequency of confusion markers in text.
    
    Args:
        text: String containing the response text
        
    Returns:
        Float value representing confusion density (# of confusion markers / # of words)
    """
    if not text or not isinstance(text, str):
        return 0.0
    
    # Convert text to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Count total words
    total_words = len(re.findall(r'\b\w+\b', text_lower))
    if total_words == 0:
        return 0.0
    
    # Count occurrences of confusion markers
    confusion_count = 0
    for marker in CONFUSION_MARKERS:
        # For the "?" marker, count directly
        if marker == "?":
            confusion_count += text_lower.count(marker)
        # For word markers, use regex to match whole words only
        else:
            confusion_count += len(re.findall(r'\b' + re.escape(marker) + r'\b', text_lower))
    
    # Calculate confusion density
    confusion_density = confusion_count #/ total_words
    
    return confusion_density

def load_result_data(base_dir, dataset_name='aime', model_name=None):
    """
    Load result data from nested directory structure.
    
    Base directory structure is expected to be:
    results/self_classification/<dataset_name>/<model_name>/<temp_val>/<year>/<problem_number>/gen_<generation_i>/result.json
    
    Args:
        base_dir: Base directory containing the results directory structure
        dataset_name: Name of the dataset ('aime' or 'hmmt')
        model_name: Optional name of the model to filter results
    
    Returns:
        List of result dictionaries
    """
    results = []
    
    # Adjust pattern based on whether a specific model is requested
    if model_name:
        pattern = os.path.join(base_dir, f"results/self_classification/{dataset_name}/{model_name}/**/*/gen_*/result.json")
    else:
        pattern = os.path.join(base_dir, f"results/self_classification/{dataset_name}/**/*/gen_*/result.json")
    
    result_files = glob.glob(pattern, recursive=True)
    
    if not result_files:
        print(f"No results found matching pattern: {pattern}")
        return results
    
    print(f"Found {len(result_files)} result files for {dataset_name}")

    for result_file in tqdm(result_files, desc="Processing result files"):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # Extract path components to get metadata
            path_parts = Path(result_file).parts    
            if len(path_parts) == 9:
                idx = path_parts.index("self_classification")
                model_name = path_parts[idx + 2]
                temp_val = path_parts[idx + 3].replace("temp_", "")
                year = path_parts[idx + 4]
                problem_number = path_parts[idx + 5]
                generation = path_parts[idx + 6].replace("gen_", "")
                part = "I"
            elif len(path_parts) == 10:
                idx = path_parts.index("self_classification")
                model_name = path_parts[idx + 2]
                temp_val = path_parts[idx + 3].replace("temp_", "")
                year = path_parts[idx + 4]
                part = path_parts[idx + 5]
                problem_number = path_parts[idx + 6]
                generation = path_parts[idx + 7].replace("gen_", "")
            else:
                raise ValueError(f"Unexpected path structure: {result_file}")
            
            
            # Extract needed data
            result_info = {
                'model_name': model_name,
                'temperature': float(temp_val),
                'year': year,
                'problem_number': problem_number,
                'generation': int(generation),
                'response_length': data.get('response_length', 0),
                'avg_neg_logprob': data.get('avg_neg_logprob', 0),
                'var_neg_logprob': np.var(data.get('token_neg_logprobs', []), ddof=1) if 'token_neg_logprobs' in data else 0,
                'answer': data.get('answer'),
                'extracted_answer': data.get('extracted_answer'),
                'response_text': data.get('response_text', ''),
                'dataset': dataset_name,  # Add dataset name to identify the source
                'part': part,  # Add part information
                'character_count': len(data.get('response_text', '')),
                'token_ids': data.get('token_ids', []),
            }
            
            
            # Add dataset-specific fields
            if dataset_name.lower() == 'hmmt':
                # For HMMT, x_value is problem_idx
                result_info['problem_idx'] = problem_number  # Store problem_idx for HMMT
            
            results.append(result_info)
        except KeyboardInterrupt:
            raise KeyboardInterrupt("Process interrupted by user.")
        except Exception as e:
            print(f"Error processing {result_file}: {e}")
            #terminate
            exit(1)
            
    
    return results

def create_dataframe(results, dataset_name='aime', ngram_correct:NgramModel=None, ngram_incorrect:NgramModel=None):
    """
    Create a pandas DataFrame from the results list and add a global_id column.
    
    Args:
        results: List of result dictionaries
        dataset_name: Name of the dataset ('aime' or 'hmmt')
        ngram_correct: N-gram model for correct answers
        ngram_incorrect: N-gram model for incorrect answers

    Returns:
        pandas DataFrame with the results and a global_id column
    """
    print(f"Creating DataFrame for dataset: {dataset_name} with {len(results)} results...")
    # Convert results list to DataFrame
    df = pd.DataFrame(results)
    
    if len(df) == 0:
        print(f"No data found for dataset: {dataset_name}")
        return pd.DataFrame()
    
    # Add global_id as an increasing unique identifier
    df['global_id'] = range(1, len(df) + 1)
    
    # Add is_correct column (true if answer matches extracted_answer, false otherwise)
    df['is_correct'] = df['answer'] == df['extracted_answer']
    
    # Calculate confusion_metric for each row
    # df['confusion_metric'] = df['response_text'].apply(calculate_confusion_metric)
    
    # # Add contains_chinese_character column
    # df['contains_chinese_character'] = df['response_text'].apply(lambda x: bool(re.search(r'[\u4e00-\u9fff]', x)))
    
    if dataset_name.lower() == 'aime':
        # Correct - incorrect logprobs from ngram models
        print(f"Calculating n-gram log probabilities for AIME dataset sequentially for {len(df)} rows...")
        
        # Process all rows at once with a single progress bar
        res_logprob_diff = []
        results_pos_perplexity = []
        results_neg_perplexity = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
            res_logprob_diff.append(calc_logprob_diff_single(row['token_ids'], ngram_correct, ngram_incorrect))
            results_pos_perplexity.append(ngram_correct.perplexity(row['token_ids']))
            results_neg_perplexity.append(ngram_incorrect.perplexity(row['token_ids']))

        # Assign res_logprob_diff back to the DataFrame
        df['ngram_logprob_diff'] = res_logprob_diff

        df['ngram_n'] = ngram_correct.n
        
        df['ngram_pos_perplexity'] = results_pos_perplexity
        df['ngram_neg_perplexity'] = results_neg_perplexity
    
    # Define columns based on dataset type
    common_columns = ['global_id', 'model_name', 'temperature', 'year', 
                      'generation', 'response_length', 'avg_neg_logprob', 'var_neg_logprob',
                      'answer', 'extracted_answer', 'is_correct', 'confusion_metric', 'response_text', 'dataset']
    
    if dataset_name.lower() == 'aime':
        columns = ['global_id', 'model_name', 'temperature', 'year', 'problem_number', 
                   'generation', 'response_length', 'avg_neg_logprob', 'var_neg_logprob',
                   'answer', 'extracted_answer', 'is_correct', 'response_text', 'dataset', 
                   'part', 'ngram_logprob_diff', 'ngram_n', 'ngram_pos_perplexity', 'ngram_neg_perplexity',
                   'confusion_metric']
    elif dataset_name.lower() == 'hmmt':
        columns = ['global_id', 'model_name', 'temperature', 'year', 'problem_number', 
                   'problem_idx', 'generation', 'response_length', 'avg_neg_logprob', 'var_neg_logprob',
                   'answer', 'extracted_answer', 'is_correct', 'confusion_metric', 'response_text', 'dataset', 'character_count', 'contains_chinese_character']
    else:
        # For unknown datasets, use only common columns
        columns = common_columns
    
    # Filter columns that exist in the DataFrame
    columns = [col for col in columns if col in df.columns]
    
    # Reorder columns
    df = df[columns]
    
    return df

def save_dataframe(df, output_dir, filename=None, dataset_name=None, model_name=None):
    """
    Save DataFrame to a parquet file.
    
    Args:
        df: pandas DataFrame to save
        output_dir: Directory to save the file
        filename: Base filename (without extension). If None, will use dataset_name_generations
        dataset_name: Name of the dataset ('aime' or 'hmmt')
        model_name: Name of the model to include in the filename
    """
    # If filename is not provided, create one based on dataset and model name
    if filename is None:
        if dataset_name is not None and model_name is not None:
            filename = f"{dataset_name}_{model_name}_generations"
        elif dataset_name is not None:
            filename = f"{dataset_name}_generations"
        else:
            # Default to 'all_generations' if neither filename nor dataset_name is provided
            filename = "all_generations"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save DataFrame to parquet file
    parquet_path = os.path.join(output_dir, f"{filename}.parquet")
    df.to_parquet(parquet_path, index=False)
    print(f"DataFrame saved to {parquet_path}")
    
    # Print some statistics
    print(f"\nDataFrame statistics:")
    print(f"Total entries: {len(df)}")
    print(f"Unique models: {df['model_name'].nunique()}")
    print(f"Models: {', '.join(sorted(df['model_name'].unique()))}")
    print(f"Temperature values: {', '.join(map(str, sorted(df['temperature'].unique())))}")
    
    # Print year statistics if available
    if 'year' in df.columns:
        print(f"Years: {', '.join(sorted(df['year'].unique()))}")
    
    # Print dataset statistics if available
    if 'dataset' in df.columns:
        print(f"Datasets: {', '.join(sorted(df['dataset'].unique()))}")
    
    # Print correct vs incorrect answers statistics
    # Count as correct only when extracted_answer matches the answer
    correct_answers = df[df['answer'] == df['extracted_answer']].shape[0]
    
    # Count all entries when calculating accuracy (treating null extracted_answers as wrong)
    total_entries = len(df)
    
    # Also show statistics for entries where extraction was possible
    total_with_extracted = df[df['extracted_answer'].notna()].shape[0]
    
    # Get count of correct answers based on the is_correct column
    correct_count = df['is_correct'].sum()
    
    if total_entries > 0:
        overall_accuracy = correct_count / total_entries * 100
        print(f"Overall accuracy (null answers counted as wrong): {overall_accuracy:.2f}% ({correct_count}/{total_entries})")
    
    if total_with_extracted > 0:
        extraction_accuracy = correct_count / total_with_extracted * 100
        print(f"Accuracy among extracted answers only: {extraction_accuracy:.2f}% ({correct_count}/{total_with_extracted})")
        
    # Print percentage of responses where answer extraction failed
    if total_entries > 0:
        extraction_failure_rate = (total_entries - total_with_extracted) / total_entries * 100
        print(f"Answer extraction failure rate: {extraction_failure_rate:.2f}% ({total_entries - total_with_extracted}/{total_entries})")
    
    # Print is_correct column stats
    print(f"\nCorrect answers (based on is_correct column): {df['is_correct'].sum()} out of {total_entries}")
    
    # Print confusion metric statistics
    if 'confusion_metric' in df.columns:
        mean_confusion = df['confusion_metric'].mean()
        median_confusion = df['confusion_metric'].median()
        max_confusion = df['confusion_metric'].max()
        
        print(f"\nConfusion metric statistics:")
        print(f"Mean confusion metric: {mean_confusion:.6f}")
        print(f"Median confusion metric: {median_confusion:.6f}")
        print(f"Maximum confusion metric: {max_confusion:.6f}")
        
        # Compare confusion metrics between correct and incorrect answers
        if df['is_correct'].any() and (~df['is_correct']).any():
            correct_confusion = df[df['is_correct']]['confusion_metric'].mean()
            incorrect_confusion = df[~df['is_correct']]['confusion_metric'].mean()
            print(f"Mean confusion metric for correct answers: {correct_confusion:.6f}")
            print(f"Mean confusion metric for incorrect answers: {incorrect_confusion:.6f}")

def main():
    parser = argparse.ArgumentParser(description='Parse result data and save as parquet')
    
    # Leave defaults for base_dir and filename as None
    parser.add_argument('--base_dir', default='.', type=str, help='Base directory containing the results directory structure')
    parser.add_argument('--filename', type=str, default=None, help='Base filename for the output parquet file')
    # Remaining arguments
    parser.add_argument('--output_dir', type=str, default='data', help='Directory to save the output parquet file')
    parser.add_argument('--dataset', type=str, default='all', choices=['all', 'aime', 'hmmt', 'webinstruct'], 
                        help='Dataset to process (aime, hmmt, webinstruct, or all)')
    parser.add_argument('--model', type=str, default=None, help='Model name to include in the output filename')
    parser.add_argument('--ngram', action='store_true', help='Build n-gram models for AIME dataset')
    parser.add_argument('--ngram_n', type=int, default=3, help='Order of the n-gram model (default: 3)')
    # Add new arguments for specifying full paths to ngram models
    parser.add_argument('--correct_ngram_path', type=str, default=None, 
                        help='Full path to the correct n-gram model file')
    parser.add_argument('--incorrect_ngram_path', type=str, default=None, 
                        help='Full path to the incorrect n-gram model file')
    
    args = parser.parse_args()
    
    # Determine which datasets to process
    datasets_to_process = []
    if args.dataset.lower() == 'all':
        datasets_to_process = ['aime', 'hmmt', 'webinstruct']
    else:
        datasets_to_process = [args.dataset.lower()]
    
    # Process each dataset
    for dataset in datasets_to_process:
        print(f"\nProcessing {dataset.upper()} dataset...")
        if args.model:
            print(f"Filtering for model: {args.model}")
        
        results = load_result_data(args.base_dir, dataset, args.model)
        

        if not results:
            print(f"No results found for {dataset}" + (f" with model {args.model}" if args.model else "") + ". Skipping.")
            continue
        
        if dataset == 'aime':
            # Load ngram models for AIME dataset
            if args.correct_ngram_path and args.incorrect_ngram_path:
                # Use explicitly provided paths
                ngram_correct_path = args.correct_ngram_path
                ngram_incorrect_path = args.incorrect_ngram_path
                print(f"Using explicitly provided ngram model paths")
            else:
                # Use the default approach to construct paths
                ngram_incorrect_path = os.path.join("ngrams", f"{args.model}_incorrect_ngram_{args.ngram_n}.pkl")
                ngram_correct_path = os.path.join("ngrams", f"{args.model}_correct_ngram_{args.ngram_n}.pkl")
                print(f"Using default ngram model paths based on model name and n-gram order")
            
            ngram_incorrect = NgramModel.load(ngram_incorrect_path)
            print(f"Loaded incorrect n-gram model from {ngram_incorrect_path}")
            ngram_correct = NgramModel.load(ngram_correct_path)
            print(f"Loaded correct n-gram model from {ngram_correct_path}")
            
        # Create DataFrame for this dataset
        df = create_dataframe(results, dataset, ngram_correct, ngram_incorrect)
        
        if len(df) == 0:
            print(f"No data found for {dataset}. Skipping.")
            continue
        
        # Save this dataset's DataFrame
        filename = args.filename if args.filename else None
        save_dataframe(df, args.output_dir, filename, dataset, args.model)

if __name__ == "__main__":
    main()
