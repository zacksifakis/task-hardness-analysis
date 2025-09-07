

import os
import json
import argparse
from transformers import AutoTokenizer
import glob

# Try to import tqdm for progress bar, use a simple alternative if not available
try:
    from tqdm import tqdm
    def progress_bar(iterable, desc):
        return tqdm(iterable, desc=desc)
except ImportError:
    def progress_bar(iterable, desc):
        print(f"{desc}...")
        return iterable

def parse_args():
    parser = argparse.ArgumentParser(description='Add token IDs to result.json files')
    parser.add_argument('--model', type=str, required=True, help='Model name for loading the tokenizer')
    parser.add_argument('--path', type=str, default='results/self_classification/aime/deepseek-ai_DeepSeek-R1-Distill-Qwen-7B/',
                        help='Root path containing response.txt and result.json files')
    return parser.parse_args()

def load_tokenizer(model_name):
    """Load tokenizer for the given model name"""
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def process_path(root_path, tokenizer):
    """
    Recursively process all leaf paths containing response.txt and result.json files
    
    Args:
        root_path: Root directory to start searching from
        tokenizer: The tokenizer to use for tokenizing responses
    """
    # Find all response.txt files in the directory structure
    response_files = glob.glob(os.path.join(root_path, "**", "response.txt"), recursive=True)
    print(f"Found {len(response_files)} response.txt files")
    
    # Process each response file
    for response_file in progress_bar(response_files, desc="Processing files"):
        dir_path = os.path.dirname(response_file)
        result_file = os.path.join(dir_path, "result.json")
        
        # Check if result.json exists in the same directory
        if not os.path.exists(result_file):
            print(f"Warning: result.json not found for {response_file}")
            continue
        
        # Load response.txt
        with open(response_file, 'r', encoding='utf-8') as f:
            response_text = f.read()
        
        # Tokenize the response
        token_ids = tokenizer.encode(response_text)
        
        # Load result.json
        with open(result_file, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
        
        # Add token_ids to result.json
        result_data['token_ids'] = token_ids
        
        # Save updated result.json
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2)

def main():
    args = parse_args()
    tokenizer = load_tokenizer(args.model)
    process_path(args.path, tokenizer)
    print("Done adding token IDs to result.json files")

if __name__ == "__main__":
    main()