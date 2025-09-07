import numpy as np
import os
import glob
import json
import pandas as pd
from tqdm import tqdm

def aggregate_results(df):
    """
    Aggregate results by x_value, which is either problem_number for AIME or difficulty for WebInstruct
    """
    # Check which column to use for grouping - prefer x_value if present
    group_by = 'x_value' if 'x_value' in df.columns else 'problem_number'
    
    # import pdb; pdb.set_trace()
    
    grouped = df.groupby(group_by).agg(
        mean_length=('response_length', 'mean'),
        std_length=('response_length', 'std'),
        count_length=('response_length', 'count'),
        mean_neg_logprob=('avg_neg_logprob', 'mean'),
        std_neg_logprob=('avg_neg_logprob', 'std'),
        count_neg_logprob=('avg_neg_logprob', 'count')
    ).reset_index()
    
    # Print the count length and count neg_logprob for after aggregation
    print(f"Count length: {grouped['count_length'].sum()}")
    print(f"Count neg_logprob: {grouped['count_neg_logprob'].sum()}")
    
    
    # Add column x_value if it doesn't exist
    if 'x_value' not in grouped.columns:
        grouped['x_value'] = grouped[group_by]
    
    grouped['length_ci'] = 1.96 * grouped['std_length'] / np.sqrt(grouped['count_length'])
    grouped['neg_logprob_ci'] = 1.96 * grouped['std_neg_logprob'] / np.sqrt(grouped['count_neg_logprob'])
    return grouped

def load_results_from_directory(model_name, dataset_name="aime", temperature=None):
    """
    Load results for a model from saved files in dataset-specific directories
    """
    print(f"Loading results for {model_name} from {dataset_name} saved files...")
    model_name_safe = model_name.replace('/', '_')
    if temperature is not None:
        results_dir = f"results/{dataset_name}/{model_name_safe}/temp_{temperature}"
    else:
        results_dir = f"results/{dataset_name}/{model_name_safe}"
        
    if not os.path.exists(results_dir):
        print(f"Error: Results directory {results_dir} not found.")
        return None
        
    results = []
    json_files = list(set(glob.glob(f"{results_dir}/**/**/result.json", recursive=True)))
    
    print(len(json_files), "result files found")
    
    if not json_files:
        print(f"No result files found in {results_dir}")
        return None
        
    for json_file in tqdm(json_files, desc=f"Loading {model_name} results from {dataset_name}"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            # Handle both old entropy format and new negative logprob format
            result_for_df = {}
            for k, v in data.items():
                if k not in ['token_entropies', 'token_neg_logprobs']:
                    result_for_df[k] = v
                    
            # If we have old entropy data but not neg_logprob data, try to convert
            if 'avg_entropy' in result_for_df and 'avg_neg_logprob' not in result_for_df:
                result_for_df['avg_neg_logprob'] = result_for_df['avg_entropy']
                result_for_df['total_neg_logprob'] = result_for_df['total_entropy']
                
            results.append(result_for_df)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            
    print(f"Loaded {len(results)} results for {model_name} from {dataset_name}")
    return pd.DataFrame(results)

def load_results_from_pickle(model_name, dataset_name="aime", temperature=None, difficulty=None):
    """
    Load results from pickle/jsonl files with appropriate dataset-specific naming
    """
    model_name_safe = model_name.replace('/', '_')
    
    filename_parts = [f"results_{model_name_safe}"]
    
    # Add dataset info to filename
    filename_parts.append(f"_{dataset_name}")
    
    # Add temperature if specified
    if temperature is not None:
        filename_parts.append(f"_temp_{temperature}")
        
    # Add difficulty if specified (for WebInstruct only)
    if difficulty is not None and dataset_name.lower() == "webinstruct":
        difficulty_safe = difficulty.replace(" ", "_")
        filename_parts.append(f"_{difficulty_safe}")
        
    base_filename = "".join(filename_parts)
    pickle_filename = f"{base_filename}.pkl"
    jsonl_filename = f"{base_filename}.jsonl"
    
    # Try loading from pickle first
    if os.path.exists(pickle_filename):
        print(f"Loading results from pickle file: {pickle_filename}")
        try:
            df = pd.read_pickle(pickle_filename)
            
            # Handle legacy data: if we have only entropy but not neg_logprob
            if 'avg_entropy' in df.columns and 'avg_neg_logprob' not in df.columns:
                df['avg_neg_logprob'] = df['avg_entropy']
                df['total_neg_logprob'] = df['total_entropy']
                
            return df
        except Exception as e:
            print(f"Error loading pickle file: {e}")
            
    # Then try loading from JSONL
    if os.path.exists(jsonl_filename):
        print(f"Loading results from JSONL file: {jsonl_filename}")
        try:
            results = []
            with open(jsonl_filename, 'r') as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        results.append(json.loads(line))
            
            df = pd.DataFrame(results)
            
            # Handle legacy data: if we have only entropy but not neg_logprob
            if 'avg_entropy' in df.columns and 'avg_neg_logprob' not in df.columns:
                df['avg_neg_logprob'] = df['avg_entropy']
                df['total_neg_logprob'] = df['total_entropy']
                
            return df
        except Exception as e:
            print(f"Error loading JSONL file: {e}")
            
    print(f"No saved pickle or JSONL files found for {model_name} with {dataset_name}")
    return None
