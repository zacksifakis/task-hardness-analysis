#!/usr/bin/env python3
import os
import json
import re
import glob
from pathlib import Path

def extract_answer_from_response(response_text):
    """
    Extract the last boxed answer from the response text, handling nested braces.
    
    Args:
        response_text (str): The response text from the result.json file
        
    Returns:
        str: The extracted answer, or None if no boxed answer is found
    """
    # This approach handles all levels of nesting by counting braces
    boxed_starts = [m.start() for m in re.finditer(r'\\boxed\{', response_text)]
    
    if boxed_starts:
        last_start = boxed_starts[-1]
        # Find the matching closing brace by counting
        count = 0
        for i in range(last_start + 7, len(response_text)):
            if response_text[i] == '{':
                count += 1
            elif response_text[i] == '}':
                if count == 0:
                    return response_text[last_start + 7:i].strip()
                count -= 1
    
    return None

def update_result_files():
    """
    Recursively find all result.json files under results/self_classification/,
    extract answers from the corresponding response.txt files,
    and update the extracted_answer field in result.json files.
    """
    
    # Counters for tracking progress
    total_files_processed = 0
    files_with_answers = 0
    files_without_answers = 0
    error_files = 0
    
    # Build the pattern to find result.json files
    pattern = "results/self_classification/**/result.json"
    result_files = glob.glob(pattern, recursive=True)
    
    print(f"Found {len(result_files)} result.json files to process.")
    
    # Process each result file
    for result_file in result_files:
        try:
            response_file = os.path.join(os.path.dirname(result_file), "response.txt")
            
            # Skip if response file doesn't exist
            if not os.path.exists(response_file):
                print(f"Warning: Response file not found for {result_file}")
                continue
            
            # Read the response text
            with open(response_file, "r", encoding="utf-8") as f:
                response_text = f.read()
            
            # Extract the answer
            extracted_answer = extract_answer_from_response(response_text)
            
            # Track if answer was found
            if extracted_answer is not None:
                files_with_answers += 1
            else:
                files_without_answers += 1
            
            # Read the current result.json
            with open(result_file, "r", encoding="utf-8") as f:
                result_data = json.load(f)
            
            # Update the extracted_answer field
            result_data["extracted_answer"] = extracted_answer
            
            # Write the updated result.json
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=2)
            
            total_files_processed += 1
            
        except Exception as e:
            print(f"Error processing file: {result_file}")
            print(f"Error details: {str(e)}")
            error_files += 1
            exit(1)
        
        # Print progress every 100 files
        if total_files_processed % 100 == 0:
            print(f"Processed {total_files_processed} files...")
    
    # Print summary
    print(f"\nProcessing completed:")
    print(f"Total files processed: {total_files_processed}")
    print(f"Files with extracted answers: {files_with_answers}")
    print(f"Files without extracted answers: {files_without_answers}")

if __name__ == "__main__":
    print("Starting to update extracted_answer fields in result.json files...")
    update_result_files()
    print("Update completed.")
