#!/usr/bin/env python3
"""
Plot histograms of response lengths for correct vs incorrect answers.

This script reads all result.json files in results/aime/<model_name>/<temp_val>/<year>/,
compares the extracted_answer with the ground truth answer, and plots histograms
of response lengths for correct and incorrect responses.
"""

import os
import json
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# No need to calculate response_length as it's already in the JSON file

def main():
    """Main function to collect data and create histograms."""
    base_dir = Path("results/aime")
    
    if not base_dir.exists():
        print(f"Base directory not found: {base_dir}")
        return
    
    # Find all result.json files
    pattern = str(base_dir / "**" / "result.json")
    result_files = glob.glob(pattern, recursive=True)
    
    result_files = [f for f in result_files if 'deepseek' in f and '14' in f]
    
    print(f"Found {len(result_files)} result.json files")
    
    # Lists to store the response lengths
    correct_lengths = []
    incorrect_lengths = []
    
    # Process each file
    processed_count = 0
    for file_path in result_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Skip if any required field is missing
            if 'extracted_answer' not in data or 'answer' not in data or 'response_length' not in data:
                continue
            
            extracted_answer = data['extracted_answer']
            true_answer = data['answer']
            response_length = data['response_length']
            
            # Handle None values
            if extracted_answer is None:
                incorrect_lengths.append(response_length)
            else:
                # Compare answers (after converting to strings and stripping whitespace)
                extracted_str = str(extracted_answer).strip()
                true_str = str(true_answer).strip()
                
                if extracted_str == true_str:
                    correct_lengths.append(response_length)
                else:
                    incorrect_lengths.append(response_length)
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    print(f"Successfully processed {processed_count} files")
    print(f"Found {len(correct_lengths)} correct answers and {len(incorrect_lengths)} incorrect answers")
    
    # Create the histograms
    plt.figure(figsize=(12, 8))
    
    # Calculate bin ranges covering both datasets
    min_length = min(min(correct_lengths) if correct_lengths else float('inf'), 
                    min(incorrect_lengths) if incorrect_lengths else float('inf'))
    max_length = max(max(correct_lengths) if correct_lengths else 0, 
                    max(incorrect_lengths) if incorrect_lengths else 0)
    
    # Create reasonable bins
    if min_length == float('inf'):
        print("No valid data found for histogram")
        return
        
    num_bins = 30
    bins = np.linspace(min_length, max_length, num_bins)
    
    # Plot histograms
    plt.hist(correct_lengths, bins=bins, alpha=0.7, label='Correct Answers', color='green', density=True)
    plt.hist(incorrect_lengths, bins=bins, alpha=0.7, label='Incorrect Answers', color='red', density=True)
    
    plt.xlabel('Response Length (tokens)')
    plt.ylabel('Density')
    plt.title('Distribution of Response Lengths: Correct vs. Incorrect Answers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add summary statistics as text
    correct_mean = np.mean(correct_lengths) if correct_lengths else 0
    incorrect_mean = np.mean(incorrect_lengths) if incorrect_lengths else 0
    
    stats_text = f"Correct answers: {len(correct_lengths)}, Mean length: {correct_mean:.1f}\n"
    stats_text += f"Incorrect answers: {len(incorrect_lengths)}, Mean length: {incorrect_mean:.1f}"
    
    plt.figtext(0.14, 0.85, stats_text, horizontalalignment='left',
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Save the figure
    output_dir = Path("results")
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig("plots/aime/answer_length_histogram_reasoning_14.png", dpi=300)
    plt.savefig("plots/aime/answer_length_histogram_reasoning_14.pdf")
    print(f"Saved plots to results/answer_length_histogram.png and results/answer_length_histogram.pdf")
    
    # Optionally show the plot (comment this out when running in a script)
    # plt.show()

if __name__ == "__main__":
    main()
