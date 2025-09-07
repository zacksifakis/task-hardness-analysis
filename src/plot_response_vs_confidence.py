#!/usr/bin/env python3

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
from pathlib import Path
import argparse

def normalize_data(data):
    """
    Normalize data to have zero mean and unit variance.
    
    Args:
        data: numpy array of shape (n, 2)
    
    Returns:
        Normalized data with zero mean and unit variance for each dimension
    """
    if len(data) == 0:
        return data
        
    # Calculate mean and std for each dimension
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    
    # Handle case where std is 0 (prevent division by zero)
    std[std == 0] = 1.0
    
    # Normalize data
    normalized_data = (data - mean) / std
    
    return normalized_data

def load_result_data(base_dir):
    """
    Load result data from nested directory structure.
    
    Base directory structure is expected to be:
    results/self_classification/aime/<model_name>/<temp_val>/<year>/<problem_number>/gen_<generation_i>/result.json
    """
    results = []
    pattern = os.path.join(base_dir, "results/self_classification/aime/**/*/gen_*/result.json")
    result_files = glob.glob(pattern, recursive=True)
    
    if not result_files:
        print(f"No results found matching pattern: {pattern}")
        return results
    
    print(f"Found {len(result_files)} result files")
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                
            # Extract path components to get metadata
            path_parts = Path(result_file).parts
            idx = path_parts.index("self_classification")
            model_name = path_parts[idx + 2]
            temp_val = path_parts[idx + 3].replace("temp_", "")
            year = path_parts[idx + 4]
            problem_number = path_parts[idx + 5]
            generation = path_parts[idx + 6].replace("gen_", "")
            
            # Extract needed data
            result_info = {
                'model_name': model_name,
                'temperature': float(temp_val),
                'year': year,
                'problem_number': problem_number,
                'generation': int(generation),
                'response_length': data.get('response_length', 0),
                'avg_neg_logprob': data.get('avg_neg_logprob', 0),
                'token_neg_logprobs': data.get('token_neg_logprobs', []),
                'answer': data.get('answer'),
                'extracted_answer': data.get('extracted_answer'),
                'response_text': data.get('response_text', ''),
            }
            
            results.append(result_info)
        except Exception as e:
            print(f"Error processing {result_file}: {e}")
    
    return results

def plot_response_length_vs_confidence(results, output_dir=None):
    """
    Create a scatter plot with:
    - x-axis: response_length
    - y-axis: avg_neg_logprob
    - green points: correct answers (extracted_answer matches answer)
    - red points: incorrect answers (extracted_answer doesn't match answer)
    """
    # Separate correct and incorrect answers
    correct = []
    incorrect = []
    
    for result in results:
        if result['extracted_answer'] is None:
            # Skip if no extracted answer
            continue
            
        if result['answer'] == result['extracted_answer']:
            correct.append(result)
        else:
            incorrect.append(result)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Create arrays for the data points
    if correct:
        correct_data = np.array([(r['response_length'], r['avg_neg_logprob']) for r in correct])
        # correct_data = np.array([(r['response_length'], np.mean(r['token_neg_logprobs'][-5:])) for r in correct])
        # Normalize correct data
        normalized_correct_data = normalize_data(correct_data)
        plt.scatter(correct_data[:, 0], correct_data[:, 1], c='green', alpha=0.6,
                    label=f'Correct answers ({len(correct)})')
    
    # Plot incorrect answers in red
    if incorrect:
        incorrect_data = np.array([(r['response_length'], r['avg_neg_logprob']) for r in incorrect])
        # incorrect_data = np.array([(r['response_length'], np.mean(r['token_neg_logprobs'][-15:])) for r in incorrect])
        # Normalize incorrect data
        normalized_incorrect_data = normalize_data(incorrect_data)
        plt.scatter(incorrect_data[:, 0], incorrect_data[:, 1], c='red', alpha=0.6,
                    label=f'Incorrect answers ({len(incorrect)})')
    
    # Calculate statistics
    if correct:
        correct_mean_length = np.mean([r['response_length'] for r in correct])
        correct_mean_logprob = np.mean([r['avg_neg_logprob'] for r in correct])
    else:
        correct_mean_length, correct_mean_logprob = 0, 0
    
    if incorrect:
        incorrect_mean_length = np.mean([r['response_length'] for r in incorrect])
        incorrect_mean_logprob = np.mean([r['avg_neg_logprob'] for r in incorrect])
    else:
        incorrect_mean_length, incorrect_mean_logprob = 0, 0
    
    # Add statistics to plot title
    accuracy = len(correct) / (len(correct) + len(incorrect)) * 100 if (len(correct) + len(incorrect)) > 0 else 0
    
    # Set labels and title
    plt.xlabel('Response Length (tokens)')
    plt.ylabel('Average Negative Log Probability')
    plt.title(f'Response Length vs. Confidence\n'
              f'Accuracy: {accuracy:.1f}% ({len(correct)}/{len(correct) + len(incorrect)})\n'
              f'Mean length: Correct={correct_mean_length:.1f}, Incorrect={incorrect_mean_length:.1f}\n'
              f'Mean neg logprob: Correct={correct_mean_logprob:.4f}, Incorrect={incorrect_mean_logprob:.4f}')
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        model_names = set(r['model_name'] for r in results)
        temperatures = set(r['temperature'] for r in results)
        
        model_str = '_'.join(model_names) if len(model_names) <= 2 else f"{len(model_names)}_models"
        temp_str = '_'.join(str(t) for t in temperatures) if len(temperatures) <= 2 else f"{len(temperatures)}_temps"
        
        filename = f"response_vs_confidence_{model_str}_{temp_str}.pdf"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, format='pdf', bbox_inches='tight')
        print(f"Plot saved to {filepath}")
    
    plt.show()

def plot_response_length_vs_std_logprob(results, output_dir=None):
    """
    Create a scatter plot with:
    - x-axis: response_length
    - y-axis: standard deviation of token_neg_logprobs
    - green points: correct answers (extracted_answer matches answer)
    - red points: incorrect answers (extracted_answer doesn't match answer)
    """
    # Separate correct and incorrect answers
    correct = []
    incorrect = []
    
    # Calculate std for each result and add it to the result object
    for result in results:
        if result['extracted_answer'] is None or not result['token_neg_logprobs']:
            # Skip if no extracted answer or no token logprobs
            continue
        
        # Calculate standard deviation of token negative log probabilities
        result['std_neg_logprob'] = np.std(result['token_neg_logprobs'])
            
        if result['answer'] == result['extracted_answer']:
            correct.append(result)
        else:
            incorrect.append(result)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Create arrays for the data points
    if correct:
        correct_data = np.array([(r['response_length'], r['std_neg_logprob']) for r in correct])
        plt.scatter(correct_data[:, 0], correct_data[:, 1], c='green', alpha=0.6,
                    label=f'Correct answers ({len(correct)})')
    
    # Plot incorrect answers in red
    if incorrect:
        incorrect_data = np.array([(r['response_length'], r['std_neg_logprob']) for r in incorrect])
        plt.scatter(incorrect_data[:, 0], incorrect_data[:, 1], c='red', alpha=0.6,
                    label=f'Incorrect answers ({len(incorrect)})')
    
    # Calculate statistics
    if correct:
        correct_mean_length = np.mean([r['response_length'] for r in correct])
        correct_mean_std = np.mean([r['std_neg_logprob'] for r in correct])
    else:
        correct_mean_length, correct_mean_std = 0, 0
    
    if incorrect:
        incorrect_mean_length = np.mean([r['response_length'] for r in incorrect])
        incorrect_mean_std = np.mean([r['std_neg_logprob'] for r in incorrect])
    else:
        incorrect_mean_length, incorrect_mean_std = 0, 0
    
    # Add statistics to plot title
    accuracy = len(correct) / (len(correct) + len(incorrect)) * 100 if (len(correct) + len(incorrect)) > 0 else 0
    
    # Set labels and title
    plt.xlabel('Response Length (tokens)')
    plt.ylabel('Standard Deviation of Token Negative Log Probabilities')
    plt.title(f'Response Length vs. Token LogProb StdDev\n'
              f'Accuracy: {accuracy:.1f}% ({len(correct)}/{len(correct) + len(incorrect)})\n'
              f'Mean length: Correct={correct_mean_length:.1f}, Incorrect={incorrect_mean_length:.1f}\n'
              f'Mean std: Correct={correct_mean_std:.4f}, Incorrect={incorrect_mean_std:.4f}')
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        model_names = set(r['model_name'] for r in results)
        temperatures = set(r['temperature'] for r in results)
        
        model_str = '_'.join(model_names) if len(model_names) <= 2 else f"{len(model_names)}_models"
        temp_str = '_'.join(str(t) for t in temperatures) if len(temperatures) <= 2 else f"{len(temperatures)}_temps"
        
        filename = f"response_vs_std_logprob_{model_str}_{temp_str}.pdf"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, format='pdf', bbox_inches='tight')
        print(f"Plot saved to {filepath}")
    
    plt.show()

def plot_avg_vs_std_logprob(results, output_dir=None):
    """
    Create a scatter plot with:
    - x-axis: avg_neg_logprob
    - y-axis: standard deviation of token_neg_logprobs
    - green points: correct answers (extracted_answer matches answer)
    - red points: incorrect answers (extracted_answer doesn't match answer)
    """
    # Separate correct and incorrect answers
    correct = []
    incorrect = []
    
    # Calculate std for each result and add it to the result object
    for result in results:
        if result['extracted_answer'] is None or not result['token_neg_logprobs']:
            # Skip if no extracted answer or no token logprobs
            continue
        
        # Calculate standard deviation of token negative log probabilities
        result['std_neg_logprob'] = np.std(result['token_neg_logprobs'])
            
        if result['answer'] == result['extracted_answer']:
            correct.append(result)
        else:
            incorrect.append(result)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Create arrays for the data points
    if correct:
        correct_data = np.array([(r['avg_neg_logprob'], r['std_neg_logprob']) for r in correct])
        plt.scatter(correct_data[:, 0], correct_data[:, 1], c='green', alpha=0.6,
                    label=f'Correct answers ({len(correct)})')
    
    # Plot incorrect answers in red
    if incorrect:
        incorrect_data = np.array([(r['avg_neg_logprob'], r['std_neg_logprob']) for r in incorrect])
        plt.scatter(incorrect_data[:, 0], incorrect_data[:, 1], c='red', alpha=0.6,
                    label=f'Incorrect answers ({len(incorrect)})')
    
    # Calculate statistics
    if correct:
        correct_mean_avg = np.mean([r['avg_neg_logprob'] for r in correct])
        correct_mean_std = np.mean([r['std_neg_logprob'] for r in correct])
    else:
        correct_mean_avg, correct_mean_std = 0, 0
    
    if incorrect:
        incorrect_mean_avg = np.mean([r['avg_neg_logprob'] for r in incorrect])
        incorrect_mean_std = np.mean([r['std_neg_logprob'] for r in incorrect])
    else:
        incorrect_mean_avg, incorrect_mean_std = 0, 0
    
    # Add statistics to plot title
    accuracy = len(correct) / (len(correct) + len(incorrect)) * 100 if (len(correct) + len(incorrect)) > 0 else 0
    
    # Set labels and title
    plt.xlabel('Average Negative Log Probability')
    plt.ylabel('Standard Deviation of Token Negative Log Probabilities')
    plt.title(f'Avg LogProb vs. Token LogProb StdDev\n'
              f'Accuracy: {accuracy:.1f}% ({len(correct)}/{len(correct) + len(incorrect)})\n'
              f'Mean avg logprob: Correct={correct_mean_avg:.4f}, Incorrect={incorrect_mean_avg:.4f}\n'
              f'Mean std: Correct={correct_mean_std:.4f}, Incorrect={incorrect_mean_std:.4f}')
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        model_names = set(r['model_name'] for r in results)
        temperatures = set(r['temperature'] for r in results)
        
        model_str = '_'.join(model_names) if len(model_names) <= 2 else f"{len(model_names)}_models"
        temp_str = '_'.join(str(t) for t in temperatures) if len(temperatures) <= 2 else f"{len(temperatures)}_temps"
        
        filename = f"avg_vs_std_logprob_{model_str}_{temp_str}.pdf"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, format='pdf', bbox_inches='tight')
        print(f"Plot saved to {filepath}")
    
    plt.show()

def plot_by_year(results, output_dir=None):
    """
    Create scatter plots for each year in the dataset
    """
    # Group by year
    years = {}
    for result in results:
        year = result['year']
        if year not in years:
            years[year] = []
        years[year].append(result)
    
    for year, year_results in years.items():
        plt.figure(figsize=(10, 7))
        
        # Separate correct and incorrect answers
        correct = [r for r in year_results if r['extracted_answer'] is not None and r['answer'] == r['extracted_answer']]
        incorrect = [r for r in year_results if r['extracted_answer'] is not None and r['answer'] != r['extracted_answer']]
        
        # Plot correct answers in green
        if correct:
            correct_data = np.array([(r['response_length'], r['avg_neg_logprob']) for r in correct])
            # Normalize correct data
            # normalized_correct_data = normalize_data(correct_data)
            plt.scatter(correct_data[:, 0], correct_data[:, 1], c='green', alpha=0.6,
                        label=f'Correct answers ({len(correct)})')
        
        # Plot incorrect answers in red
        if incorrect:
            incorrect_data = np.array([(r['response_length'], r['avg_neg_logprob']) for r in incorrect])
            # Normalize incorrect data
            normalized_incorrect_data = normalize_data(incorrect_data)
            plt.scatter(incorrect_data[:, 0], incorrect_data[:, 1], c='red', alpha=0.6,
                        label=f'Incorrect answers ({len(incorrect)})')
        
        # Calculate statistics
        accuracy = len(correct) / (len(correct) + len(incorrect)) * 100 if (len(correct) + len(incorrect)) > 0 else 0
        
        # Set labels and title
        plt.xlabel('Response Length (tokens)')
        plt.ylabel('Average Negative Log Probability')
        plt.title(f'Response Length vs. Confidence - Year {year}\n'
                  f'Accuracy: {accuracy:.1f}% ({len(correct)}/{len(correct) + len(incorrect)})')
        
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save plot if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            model_names = set(r['model_name'] for r in year_results)
            temperatures = set(r['temperature'] for r in year_results)
            
            model_str = '_'.join(model_names) if len(model_names) <= 2 else f"{len(model_names)}_models"
            temp_str = '_'.join(str(t) for t in temperatures) if len(temperatures) <= 2 else f"{len(temperatures)}_temps"
            
            filename = f"response_vs_confidence_year_{year}_{model_str}_{temp_str}.pdf"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, format='pdf', bbox_inches='tight')
            print(f"Plot saved to {filepath}")
        
        plt.close()

def plot_by_problem(results, output_dir=None):
    """
    Create scatter plots for each problem number in the dataset
    """
    # Group by problem number
    problems = {}
    for result in results:
        problem_key = f"{result['year']}_{result['problem_number']}"
        if problem_key not in problems:
            problems[problem_key] = []
        problems[problem_key].append(result)
    
    for problem_key, problem_results in problems.items():
        year, problem_number = problem_key.split('_')
        plt.figure(figsize=(10, 7))
        
        # Separate correct and incorrect answers
        correct = [r for r in problem_results if r['extracted_answer'] is not None and r['answer'] == r['extracted_answer']]
        incorrect = [r for r in problem_results if r['extracted_answer'] is not None and r['answer'] != r['extracted_answer']]
        
        # Plot correct answers in green
        if correct:
            correct_data = np.array([(r['response_length'], r['avg_neg_logprob']) for r in correct])
            plt.scatter(correct_data[:, 0], correct_data[:, 1], c='green', alpha=0.6, 
                        label=f'Correct answers ({len(correct)})')
        
        # Plot incorrect answers in red
        if incorrect:
            incorrect_data = np.array([(r['response_length'], r['avg_neg_logprob']) for r in incorrect])
            plt.scatter(incorrect_data[:, 0], incorrect_data[:, 1], c='red', alpha=0.6, 
                        label=f'Incorrect answers ({len(incorrect)})')
        
        # Calculate statistics
        accuracy = len(correct) / (len(correct) + len(incorrect)) * 100 if (len(correct) + len(incorrect)) > 0 else 0
        
        # Set labels and title
        plt.xlabel('Response Length (tokens)')
        plt.ylabel('Average Negative Log Probability')
        plt.title(f'Response Length vs. Confidence - Year {year}, Problem {problem_number}\n'
                  f'Accuracy: {accuracy:.1f}% ({len(correct)}/{len(correct) + len(incorrect)})')
        
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save plot if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            model_names = set(r['model_name'] for r in problem_results)
            temperatures = set(r['temperature'] for r in problem_results)
            
            model_str = '_'.join(model_names) if len(model_names) <= 2 else f"{len(model_names)}_models"
            temp_str = '_'.join(str(t) for t in temperatures) if len(temperatures) <= 2 else f"{len(temperatures)}_temps"
            
            problem_dir = os.path.join(output_dir, "problems")
            os.makedirs(problem_dir, exist_ok=True)
            
            filename = f"response_vs_confidence_year_{year}_problem_{problem_number}_{model_str}_{temp_str}.pdf"
            filepath = os.path.join(problem_dir, filename)
            plt.savefig(filepath, format='pdf', bbox_inches='tight')
            print(f"Plot saved to {filepath}")
        
        plt.close()

def plot_by_problem_id(results, output_dir=None):
    """
    Create scatter plots for each problem ID (across all years) in the dataset
    """
    # Group by problem ID
    problem_ids = {}
    for result in results:
        problem_id = result['problem_number']
        if problem_id not in problem_ids:
            problem_ids[problem_id] = []
        problem_ids[problem_id].append(result)
    
    for problem_id, problem_results in problem_ids.items():
        plt.figure(figsize=(10, 7))
        
        # Separate correct and incorrect answers
        correct = [r for r in problem_results if r['extracted_answer'] is not None and r['answer'] == r['extracted_answer']]
        incorrect = [r for r in problem_results if r['extracted_answer'] is not None and r['answer'] != r['extracted_answer']]
        
        # Plot correct answers in green
        if correct:
            correct_data = np.array([(r['response_length'], r['avg_neg_logprob']) for r in correct])
            plt.scatter(correct_data[:, 0], correct_data[:, 1], c='green', alpha=0.6, 
                        label=f'Correct answers ({len(correct)})')
        
        # Plot incorrect answers in red
        if incorrect:
            incorrect_data = np.array([(r['response_length'], r['avg_neg_logprob']) for r in incorrect])
            plt.scatter(incorrect_data[:, 0], incorrect_data[:, 1], c='red', alpha=0.6, 
                        label=f'Incorrect answers ({len(incorrect)})')
        
        # Calculate statistics
        accuracy = len(correct) / (len(correct) + len(incorrect)) * 100 if (len(correct) + len(incorrect)) > 0 else 0
        
        # Get all years represented in this problem ID
        years = sorted(set(r['year'] for r in problem_results))
        years_str = ', '.join(years) if len(years) <= 5 else f"{len(years)} different years"
        
        # Set labels and title
        plt.xlabel('Response Length (tokens)')
        plt.ylabel('Average Negative Log Probability')
        plt.title(f'Response Length vs. Confidence - Problem {problem_id} (across years: {years_str})\n'
                  f'Accuracy: {accuracy:.1f}% ({len(correct)}/{len(correct) + len(incorrect)})')
        
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save plot if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            model_names = set(r['model_name'] for r in problem_results)
            temperatures = set(r['temperature'] for r in problem_results)
            
            model_str = '_'.join(model_names) if len(model_names) <= 2 else f"{len(model_names)}_models"
            temp_str = '_'.join(str(t) for t in temperatures) if len(temperatures) <= 2 else f"{len(temperatures)}_temps"
            
            problem_id_dir = os.path.join(output_dir, "problem_ids")
            os.makedirs(problem_id_dir, exist_ok=True)
            
            filename = f"response_vs_confidence_problem_{problem_id}_{model_str}_{temp_str}.pdf"
            filepath = os.path.join(problem_id_dir, filename)
            plt.savefig(filepath, format='pdf', bbox_inches='tight')
            print(f"Plot saved to {filepath}")
        
        plt.close()

def plot_wait_vs_confidence(results, output_dir=None):
    """
    Create a scatter plot with:
    - x-axis: number of 'wait'/'Wait' occurrences (raw count)
    - y-axis: avg_neg_logprob
    - green points: correct answers (extracted_answer matches answer)
    - red points: incorrect answers (extracted_answer doesn't match answer)
    """
    # Separate correct and incorrect answers
    correct = []
    incorrect = []
    
    for result in results:
        if result['extracted_answer'] is None or 'response_text' not in result:
            # Skip if no extracted answer or response text
            continue
            
        # Count occurrences of 'wait' or 'Wait' in the response
        wait_count = result['response_text'].lower().count('wait')
        
        # Add the wait count to the result object
        result['wait_count'] = wait_count
            
        if result['answer'] == result['extracted_answer']:
            correct.append(result)
        else:
            incorrect.append(result)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Create arrays for the data points
    if correct:
        correct_data = np.array([(r['wait_count'], r['avg_neg_logprob']) for r in correct])
        plt.scatter(correct_data[:, 0], correct_data[:, 1], c='green', alpha=0.6,
                    label=f'Correct answers ({len(correct)})')
    
    # Plot incorrect answers in red
    if incorrect:
        incorrect_data = np.array([(r['wait_count'], r['avg_neg_logprob']) for r in incorrect])
        plt.scatter(incorrect_data[:, 0], incorrect_data[:, 1], c='red', alpha=0.6,
                    label=f'Incorrect answers ({len(incorrect)})')
    
    # Calculate statistics
    if correct:
        correct_mean_count = np.mean([r['wait_count'] for r in correct])
        correct_mean_logprob = np.mean([r['avg_neg_logprob'] for r in correct])
    else:
        correct_mean_count, correct_mean_logprob = 0, 0
    
    if incorrect:
        incorrect_mean_count = np.mean([r['wait_count'] for r in incorrect])
        incorrect_mean_logprob = np.mean([r['avg_neg_logprob'] for r in incorrect])
    else:
        incorrect_mean_count, incorrect_mean_logprob = 0, 0
    
    # Add statistics to plot title
    accuracy = len(correct) / (len(correct) + len(incorrect)) * 100 if (len(correct) + len(incorrect)) > 0 else 0
    
    # Set labels and title
    plt.xlabel('Number of "Wait" Occurrences')
    plt.ylabel('Average Negative Log Probability')
    plt.title(f'Wait Occurrences vs. Confidence\n'
              f'Accuracy: {accuracy:.1f}% ({len(correct)}/{len(correct) + len(incorrect)})\n'
              f'Mean wait count: Correct={correct_mean_count:.2f}, Incorrect={incorrect_mean_count:.2f}\n'
              f'Mean neg logprob: Correct={correct_mean_logprob:.4f}, Incorrect={incorrect_mean_logprob:.4f}')
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        model_names = set(r['model_name'] for r in results)
        temperatures = set(r['temperature'] for r in results)
        
        model_str = '_'.join(model_names) if len(model_names) <= 2 else f"{len(model_names)}_models"
        temp_str = '_'.join(str(t) for t in temperatures) if len(temperatures) <= 2 else f"{len(temperatures)}_temps"
        
        filename = f"wait_vs_confidence_{model_str}_{temp_str}.pdf"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, format='pdf', bbox_inches='tight')
        print(f"Plot saved to {filepath}")
    
    plt.show()

def plot_wait_by_year(results, output_dir=None):
    """
    Create scatter plots of wait occurrences vs confidence for each year in the dataset
    """
    # Group by year
    years = {}
    for result in results:
        if 'response_text' not in result:
            continue
            
        year = result['year']
        if year not in years:
            years[year] = []
            
        # Count occurrences of 'wait' or 'Wait' in the response
        wait_count = result['response_text'].lower().count('wait')
        
        # Add the wait count to the result object
        result['wait_count'] = wait_count
        years[year].append(result)
    
    for year, year_results in years.items():
        plt.figure(figsize=(10, 7))
        
        # Separate correct and incorrect answers
        correct = [r for r in year_results if r['extracted_answer'] is not None and r['answer'] == r['extracted_answer']]
        incorrect = [r for r in year_results if r['extracted_answer'] is not None and r['answer'] != r['extracted_answer']]
        
        # Plot correct answers in green
        if correct:
            correct_data = np.array([(r['wait_count'], r['avg_neg_logprob']) for r in correct])
            plt.scatter(correct_data[:, 0], correct_data[:, 1], c='green', alpha=0.6,
                        label=f'Correct answers ({len(correct)})')
        
        # Plot incorrect answers in red
        if incorrect:
            incorrect_data = np.array([(r['wait_count'], r['avg_neg_logprob']) for r in incorrect])
            plt.scatter(incorrect_data[:, 0], incorrect_data[:, 1], c='red', alpha=0.6,
                        label=f'Incorrect answers ({len(incorrect)})')
        
        # Calculate statistics
        accuracy = len(correct) / (len(correct) + len(incorrect)) * 100 if (len(correct) + len(incorrect)) > 0 else 0
        
        # Set labels and title
        plt.xlabel('Number of "Wait" Occurrences')
        plt.ylabel('Average Negative Log Probability')
        plt.title(f'Wait Occurrences vs. Confidence - Year {year}\n'
                  f'Accuracy: {accuracy:.1f}% ({len(correct)}/{len(correct) + len(incorrect)})')
        
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save plot if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            model_names = set(r['model_name'] for r in year_results)
            temperatures = set(r['temperature'] for r in year_results)
            
            model_str = '_'.join(model_names) if len(model_names) <= 2 else f"{len(model_names)}_models"
            temp_str = '_'.join(str(t) for t in temperatures) if len(temperatures) <= 2 else f"{len(temperatures)}_temps"
            
            filename = f"wait_vs_confidence_year_{year}_{model_str}_{temp_str}.pdf"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, format='pdf', bbox_inches='tight')
            print(f"Plot saved to {filepath}")
        
        plt.close()

def plot_3d_combined(results, output_dir=None):
    """
    Create a 3D scatter plot with:
    - x-axis: response_length
    - y-axis: avg_neg_logprob
    - z-axis: number of 'wait'/'Wait' occurrences (raw count)
    - green points: correct answers (extracted_answer matches answer)
    - red points: incorrect answers (extracted_answer doesn't match answer)
    """
    # Separate correct and incorrect answers
    correct = []
    incorrect = []
    
    for result in results:
        if result['extracted_answer'] is None or 'response_text' not in result:
            # Skip if no extracted answer or response text
            continue
            
        # Count occurrences of 'wait' or 'Wait' in the response
        wait_count = result['response_text'].lower().count('wait')
        
        # Add the wait count to the result object
        result['wait_count'] = wait_count
            
        if result['answer'] == result['extracted_answer']:
            correct.append(result)
        else:
            incorrect.append(result)
    
    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create arrays for the data points
    if correct:
        correct_data = np.array([(r['response_length'], r['avg_neg_logprob'], r['wait_count']) for r in correct])
        ax.scatter(correct_data[:, 0], correct_data[:, 1], correct_data[:, 2], c='green', alpha=0.6,
                   label=f'Correct answers ({len(correct)})')
    
    # Plot incorrect answers in red
    if incorrect:
        incorrect_data = np.array([(r['response_length'], r['avg_neg_logprob'], r['wait_count']) for r in incorrect])
        ax.scatter(incorrect_data[:, 0], incorrect_data[:, 1], incorrect_data[:, 2], c='red', alpha=0.6,
                   label=f'Incorrect answers ({len(incorrect)})')
    
    # Calculate statistics
    accuracy = len(correct) / (len(correct) + len(incorrect)) * 100 if (len(correct) + len(incorrect)) > 0 else 0
    
    # Set labels and title
    ax.set_xlabel('Response Length (tokens)')
    ax.set_ylabel('Average Negative Log Probability')
    ax.set_zlabel('Number of "Wait" Occurrences')
    ax.set_title(f'3D: Response Length vs Confidence vs Wait Count\n'
                 f'Accuracy: {accuracy:.1f}% ({len(correct)}/{len(correct) + len(incorrect)})')
    
    # Add a grid
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add rotational view for better visualization
    ax.view_init(elev=20, azim=45)
    
    # Save plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        model_names = set(r['model_name'] for r in results)
        temperatures = set(r['temperature'] for r in results)
        
        model_str = '_'.join(model_names) if len(model_names) <= 2 else f"{len(model_names)}_models"
        temp_str = '_'.join(str(t) for t in temperatures) if len(temperatures) <= 2 else f"{len(temperatures)}_temps"
        
        filename = f"3d_combined_{model_str}_{temp_str}.pdf"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, format='pdf', bbox_inches='tight')
        print(f"3D plot saved to {filepath}")
    
    plt.show()

def plot_wait_count_vs_response_length(results, output_dir=None):
    """
    Create a scatter plot with:
    - x-axis: response_length
    - y-axis: number of 'wait'/'Wait' occurrences (raw count)
    - green points: correct answers (extracted_answer matches answer)
    - red points: incorrect answers (extracted_answer doesn't match answer)
    """
    # Separate correct and incorrect answers
    correct = []
    incorrect = []
    
    for result in results:
        if result['extracted_answer'] is None or 'response_text' not in result:
            # Skip if no extracted answer or response text
            continue
            
        # Count occurrences of 'wait' or 'Wait' in the response
        wait_count = result['response_text'].lower().count('wait')
        
        # Add the wait count to the result object
        result['wait_count'] = wait_count
            
        if result['answer'] == result['extracted_answer']:
            correct.append(result)
        else:
            incorrect.append(result)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Create arrays for the data points
    if correct:
        correct_data = np.array([(r['response_length'], r['wait_count']) for r in correct])
        plt.scatter(correct_data[:, 0], correct_data[:, 1], c='green', alpha=0.6,
                    label=f'Correct answers ({len(correct)})')
    
    # Plot incorrect answers in red
    if incorrect:
        incorrect_data = np.array([(r['response_length'], r['wait_count']) for r in incorrect])
        plt.scatter(incorrect_data[:, 0], incorrect_data[:, 1], c='red', alpha=0.6,
                    label=f'Incorrect answers ({len(incorrect)})')
    
    # Calculate statistics
    if correct:
        correct_mean_length = np.mean([r['response_length'] for r in correct])
        correct_mean_count = np.mean([r['wait_count'] for r in correct])
    else:
        correct_mean_length, correct_mean_count = 0, 0
    
    if incorrect:
        incorrect_mean_length = np.mean([r['response_length'] for r in incorrect])
        incorrect_mean_count = np.mean([r['wait_count'] for r in incorrect])
    else:
        incorrect_mean_length, incorrect_mean_count = 0, 0
    
    # Add statistics to plot title
    accuracy = len(correct) / (len(correct) + len(incorrect)) * 100 if (len(correct) + len(incorrect)) > 0 else 0
    
    # Set labels and title
    plt.xlabel('Response Length (tokens)')
    plt.ylabel('Number of "Wait" Occurrences')
    plt.title(f'Response Length vs. Wait Occurrences\n'
              f'Accuracy: {accuracy:.1f}% ({len(correct)}/{len(correct) + len(incorrect)})\n'
              f'Mean length: Correct={correct_mean_length:.1f}, Incorrect={incorrect_mean_length:.1f}\n'
              f'Mean wait count: Correct={correct_mean_count:.2f}, Incorrect={incorrect_mean_count:.2f}')
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save plot if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        model_names = set(r['model_name'] for r in results)
        temperatures = set(r['temperature'] for r in results)
        
        model_str = '_'.join(model_names) if len(model_names) <= 2 else f"{len(model_names)}_models"
        temp_str = '_'.join(str(t) for t in temperatures) if len(temperatures) <= 2 else f"{len(temperatures)}_temps"
        
        filename = f"response_length_vs_wait_count_{model_str}_{temp_str}.pdf"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, format='pdf', bbox_inches='tight')
        print(f"Plot saved to {filepath}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot response length vs confidence for AIME problems')
    parser.add_argument('--base_dir', type=str, default='.', help='Base directory containing results directory structure')
    parser.add_argument('--output_dir', type=str, default='plots', help='Directory to save plots')
    parser.add_argument('--by_year', action='store_true', help='Generate plots grouped by year')
    parser.add_argument('--by_problem', action='store_true', help='Generate plots for each problem')
    parser.add_argument('--model', type=str, help='Filter by model name')
    parser.add_argument('--temp', type=float, help='Filter by temperature value')
    parser.add_argument('--wait', action='store_true', help='Generate plots of wait occurrence ratio vs confidence')
    parser.add_argument('--only_wait', action='store_true', help='Only generate wait occurrence plots, skip response length plots')
    
    args = parser.parse_args()
    
    # Load data
    results = load_result_data(args.base_dir)
    
    if not results:
        print("No results found. Please check your base directory.")
        return
    
    # Filter results if needed
    if args.model:
        results = [r for r in results if args.model in r['model_name']]
        print(f"Filtered to {len(results)} results for model: {args.model}")
    
    if args.temp is not None:
        results = [r for r in results if r['temperature'] == args.temp]
        print(f"Filtered to {len(results)} results for temperature: {args.temp}")
    
    # If --only_wait is not specified, create standard plots
    if not args.only_wait:
        # Create overall plot
        plot_response_length_vs_confidence(results, args.output_dir)
        
        # Create plot for response length vs standard deviation of token log probabilities
        plot_response_length_vs_std_logprob(results, args.output_dir)
        
        # Create plot for average log probability vs standard deviation of token log probabilities
        plot_avg_vs_std_logprob(results, args.output_dir)
        
        # Create plots by year if requested
        if args.by_year:
            plot_by_year(results, args.output_dir)
        
        # Create plots by problem if requested
        if args.by_problem:
            # plot_by_problem(results, args.output_dir)
            
            # Create plots by problem ID across all years
            plot_by_problem_id(results, args.output_dir)
    
    # Create wait occurrence plots if requested
    if args.wait or args.only_wait:
        # Plot wait count vs confidence
        plot_wait_vs_confidence(results, args.output_dir)
        
        # Plot wait count vs response length
        plot_wait_count_vs_response_length(results, args.output_dir)
        
        if args.by_year:
            plot_wait_by_year(results, args.output_dir)
    
    # Create 3D combined plot
    plot_3d_combined(results, args.output_dir)
    
    # Create plot for wait count vs response length
    plot_wait_count_vs_response_length(results, args.output_dir)


if __name__ == "__main__":
    main()
