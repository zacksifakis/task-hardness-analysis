#!/usr/bin/env python3
import argparse
import torch
import json
import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from data_utils import load_dataset_by_name, ensure_dir
from inference import calculate_entropies, calculate_entropies_vllm
from results_utils import aggregate_results, load_results_from_pickle, load_results_from_directory
from plotting import plot_combined_results

def plot_only(dataset_name="aime", cutoff_year=None, difficulty_filter=None, temperature=None):
    """Generate combined plots from existing result files for a specific dataset"""
    print(f"Generating combined plots from existing {dataset_name} result files...")
    
    # Setup the directories for results
    dataset_dir = f"results/{dataset_name}"
    if not os.path.exists(dataset_dir):
        print(f"No results directory found for dataset {dataset_name}.")
        return
    
    # Find all model directories in the dataset directory
    model_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    if not model_dirs:
        print(f"No model results found in {dataset_dir}")
        return
    
    # Dataset-specific info for file naming
    dataset_info = f"_{dataset_name}"
    if dataset_name.lower() == "aime" and cutoff_year:
        dataset_info += f"_{cutoff_year}+"
    elif dataset_name.lower() == "webinstruct" and difficulty_filter:
        difficulty_safe = difficulty_filter.replace(" ", "_")
        dataset_info += f"_{difficulty_safe}"
    
    # Load results for each model
    all_model_results = {}
    for model_dir in model_dirs:
        # Extract model name
        model_name = model_dir.replace('_', '/', 1)  # Replace just the first underscore
        
        print(f"Loading results for {model_name}...")
        results_df = load_results_from_directory(model_name, dataset_name, temperature)
                
        # Filter results if needed
        if dataset_name.lower() == "aime" and cutoff_year:
            if "year" in results_df.columns:
                results_df = results_df[results_df["year"] >= cutoff_year]
        elif dataset_name.lower() == "webinstruct" and difficulty_filter:
            if "difficulty" in results_df.columns:
                results_df = results_df[results_df["difficulty"] == difficulty_filter]
                
        if not results_df.empty:
            aggregated = aggregate_results(results_df)
            all_model_results[model_name] = aggregated
        else:
            print(f"No valid results found for {model_name} after filtering.")
    
    if all_model_results:
        ensure_dir(f"plots/{dataset_name}")
        plot_combined_results(all_model_results, dataset_name, dataset_info, temperature=temperature)
        print(f"Combined plots have been saved to plots/{dataset_name}/ directory.")
    else:
        print("No results could be loaded.")

def main():
    parser = argparse.ArgumentParser(description='Model comparison for dataset tasks')
    
    # Core arguments
    parser.add_argument('--plot-only', action='store_true', help='Only generate plots from existing results without running experiments')
    parser.add_argument('--models', nargs='+', help='Specific models to process')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for text generation (default: 0.7)')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for inference (cuda or cpu)')
    
    # Dataset selection and filtering
    parser.add_argument('--dataset', type=str, default='aime', choices=['aime', 'webinstruct'], 
                        help='Dataset to use (default: aime)')
    parser.add_argument('--cutoff-year', type=int, help='For AIME dataset: Only include problems from this year onward')
    parser.add_argument('--difficulty', type=str, help='For WebInstruct dataset: Only include questions with this difficulty')
    parser.add_argument('--n-per-difficulty', type=int, help='For WebInstruct dataset: Sample this many questions per difficulty level')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility when sampling examples')
    
    # Performance options
    parser.add_argument('--use-vllm', action='store_true', help='Use vLLM for significantly faster inference')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for batched inference (default: 4)')
    
    args = parser.parse_args()
    
    # If plot-only is specified, just generate plots from existing results and exit
    if args.plot_only:
        plot_only(args.dataset, args.cutoff_year, args.difficulty, args.temperature)
        return
    
    # Otherwise, run experiments and then plot
    models = args.models if args.models else [
        "Qwen/Qwen2.5-Math-1.5B",
        "Qwen/Qwen2.5-Math-7B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    ]
    
    # Dataset info for file naming
    dataset_info = f"_{args.dataset}"
    if args.dataset.lower() == "aime" and args.cutoff_year:
        dataset_info += f"_{args.cutoff_year}+"
    elif args.dataset.lower() == "webinstruct" and args.difficulty:
        difficulty_safe = args.difficulty.replace(" ", "_")
        dataset_info += f"_{difficulty_safe}"
    
    # Load the appropriate dataset with any specified filters
    dataset = load_dataset_by_name(
        dataset_name=args.dataset,
        cutoff_year=args.cutoff_year if args.dataset.lower() == "aime" else None,
        difficulty_filter=args.difficulty if args.dataset.lower() == "webinstruct" else None,
        n_per_difficulty=args.n_per_difficulty if args.dataset.lower() == "webinstruct" else None,
        seed=args.seed  # Pass the seed parameter for reproducibility
    )
    
    # Create directories for results and plots
    ensure_dir(f"results/{args.dataset}")
    ensure_dir(f"plots/{args.dataset}")
    
    # For combined plots, we store all model results
    all_model_results = {}
    
    for model_name in models:
        print(f"\nProcessing model: {model_name}")
        model_name_safe = model_name.replace('/', '_')
        
        if args.use_vllm:
            # Use vLLM with batched processing
            print(f"Using vLLM for inference with batch size {args.batch_size}")
            results = calculate_entropies_vllm(
                model_name=model_name,
                problems=dataset,
                dataset_name=args.dataset,
                temperature=args.temperature,
                batch_size=args.batch_size
            )
        else:
            # Load model and tokenizer using HuggingFace
            print(f"Loading tokenizer for {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Use standard sequential processing
            print(f"Using standard sequential processing")
            print(f"Loading model {model_name} with HuggingFace Transformers...")
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(args.device)
            results = calculate_entropies(
                model=model,
                tokenizer=tokenizer,
                problems=dataset,
                dataset_name=args.dataset,
                temperature=args.temperature, 
                device=args.device
            )
                
            # Clear model from memory
            del model
            torch.cuda.empty_cache()
        
        # Save results
        # JSONL format
        jsonl_filename = f"results_{model_name_safe}{dataset_info}.jsonl"
        print(f"Saving JSONL data to {jsonl_filename}")
        with open(jsonl_filename, 'w') as f:
            for record in results:
                f.write(json.dumps(record) + '\n')
        
        # Pickle format for backward compatibility
        pickle_filename = f"results_{model_name_safe}{dataset_info}.pkl"
        print(f"Saving pickle data to {pickle_filename}")
        pd.DataFrame(results).to_pickle(pickle_filename)
        
        # Store for combined plots
        aggregated = aggregate_results(pd.DataFrame(results))
        all_model_results[model_name] = aggregated
    
    # Create combined plots
    print(f"\nCreating combined plots for all models using {args.dataset} dataset...")
    plot_combined_results(all_model_results, args.dataset, dataset_info, temperature=args.temperature)
        
    print("Analysis complete! Results and plots have been saved.")

if __name__ == "__main__":
    main()