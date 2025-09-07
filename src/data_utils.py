import os
from datasets import load_dataset, concatenate_datasets
import random

def load_aime_dataset(cutoff_year=None):
    """
    Load the AIME dataset from HuggingFace
    
    Args:
        cutoff_year (int, optional): If specified, only include problems from this year onward
    
    Returns:
        Dataset: Filtered AIME dataset
    """
    print("Loading AIME dataset...")
    dataset = load_dataset("di-zhang-fdu/AIME_1983_2024")
    
    # Filter by cutoff year if specified
    if cutoff_year:
        print(f"Filtering dataset to include only problems from {cutoff_year} onward")
        dataset = dataset.filter(lambda example: example["Year"] >= cutoff_year)
    
    # Always load 2025 dataset with proper transformations
    print("Loading AIME 2025 dataset...")
    dataset_2025 = load_dataset("yentinglin/aime_2025")
    
    # Apply transformations to the 2025 dataset
    dataset_2025 = dataset_2025["train"]
    
    # Keep only elements with id >= 15
    dataset_2025 = dataset_2025.filter(lambda example: example["id"] >= 15)
    
    # Transform the dataset to match the schema
    dataset_2025 = dataset_2025.rename_column("__index_level_0__", "Problem Number")
    dataset_2025 = dataset_2025.rename_column("problem", "Question")
    dataset_2025 = dataset_2025.rename_column("answer", "Answer")
    dataset_2025 = dataset_2025.rename_column("year", "Year")
    
    # Add ID column with format 2025-II-<problem number>
    dataset_2025 = dataset_2025.map(
        lambda example: {"ID": f"2025-II-{example['Problem Number']}"}
    )
    
    # Add Part column with constant value "I"
    dataset_2025 = dataset_2025.map(
        lambda example: {"Part": "I"}
    )
    
    # Drop unnecessary columns
    columns_to_drop = ["solution", "url", "id"]
    dataset_2025 = dataset_2025.remove_columns([col for col in columns_to_drop if col in dataset_2025.column_names])
    
    # Only merge the 2025 dataset if it should be included based on cutoff year
    if cutoff_year is None or cutoff_year <= 2025:
        print("Merging 2025 AIME dataset...")
        
        # Get the train split from the original dataset
        dataset_train = dataset["train"]
        
        # Print column names for debugging
        print(f"Original dataset columns: {dataset_train.column_names}")
        print(f"2025 dataset columns: {dataset_2025.column_names}")
        
        # Concatenate the datasets
        merged_dataset = concatenate_datasets([dataset_train, dataset_2025])
        
        # Fix null values in Part column
        print("Checking for null values in Part column...")
        null_parts_count = sum(1 for example in merged_dataset if example.get("Part") is None)
        if null_parts_count > 0:
            print(f"Found {null_parts_count} null values in Part column. Setting them to 'I'...")
            merged_dataset = merged_dataset.map(
                lambda example: {"Part": "I" if example.get("Part") is None else example["Part"]}
            )
        
        return merged_dataset
    
    # Fix null values in Part column for non-merged dataset
    dataset_train = dataset["train"]
    print("Checking for null values in Part column...")
    null_parts_count = sum(1 for example in dataset_train if example.get("Part") is None)
    if null_parts_count > 0:
        print(f"Found {null_parts_count} null values in Part column. Setting them to 'I'...")
        dataset_train = dataset_train.map(
            lambda example: {"Part": "I" if example.get("Part") is None else example["Part"]}
        )
    
    return dataset_train

def load_webinstruct_dataset(difficulty_filter=None, n_per_difficulty=None, seed=None):
    """
    Load the WebInstruct-verified dataset from HuggingFace
    
    Args:
        difficulty_filter (str, optional): If specified, only include questions with this difficulty
        n_per_difficulty (int, optional): If specified, sample this many questions per difficulty level
        seed (int, optional): Random seed for reproducibility when sampling examples
    
    Returns:
        Dataset: Filtered WebInstruct dataset
    """
    print("Loading WebInstruct-verified dataset...")
    dataset = load_dataset("TIGER-Lab/WebInstruct-verified")["train"]
    
    # Filter by difficulty if specified
    if difficulty_filter:
        print(f"Filtering dataset to include only questions with difficulty: {difficulty_filter}")
        dataset = dataset.filter(lambda example: example["difficulty"] == difficulty_filter)
        return dataset
    
    # Sample n examples per difficulty if specified
    if n_per_difficulty is not None and n_per_difficulty > 0:
        print(f"Sampling {n_per_difficulty} questions per difficulty level")

        # Create a separate random number generator for reproducibility
        local_random = random.Random(seed)
        if seed is not None:
            print(f"Using random seed: {seed}")

        # Get all unique difficulty levels in a deterministic order
        all_difficulties = sorted(set(dataset["difficulty"]))
        
        # Convert dataset to a list for stable processing
        dataset_list = list(dataset)
        
        # Create a dictionary to store examples for each difficulty
        examples_by_difficulty = {diff: [] for diff in all_difficulties}
        
        # Group examples by difficulty
        for example in dataset_list:
            examples_by_difficulty[example["difficulty"]].append(example)
                  
        # Sample n examples from each difficulty
        sampled_examples = []
        for difficulty in all_difficulties:  # Process in sorted order for determinism
            examples = examples_by_difficulty[difficulty]
            
            # Sort the examples by __index_level_0__ to ensure consistent ordering
            examples.sort(key=lambda x: x["__index_level_0__"])
            
            if len(examples) <= n_per_difficulty:
                print(f"Warning: Only {len(examples)} examples available for difficulty '{difficulty}', using all of them")
                sampled_examples.extend(examples)
            else:
                # Randomly sample n_per_difficulty examples using the local RNG
                indices = local_random.sample(range(len(examples)), n_per_difficulty)
                # Sort the indices to ensure consistent ordering of selected examples
                indices.sort()
                # Sample the examples based on the indices
                sampled = [examples[i] for i in indices]
                sampled_examples.extend(sampled)
        
        # Convert back to dataset format
        dataset_dict = {key: [example[key] for example in sampled_examples] for key in sampled_examples[0].keys()}
        return dataset.from_dict(dataset_dict)
    
    return dataset

def load_hmmt_2025_dataset():
    """
    Load the HMMT February 2025 dataset from HuggingFace
    
    Returns:
        Dataset: HMMT February 2025 dataset
    """
    print("Loading HMMT February 2025 dataset...")
    dataset = load_dataset("MathArena/hmmt_feb_2025")["train"]
    
    # Add year column with constant value 2025
    dataset = dataset.map(
        lambda example: {"year": 2025}
    )
    
    return dataset

def load_dataset_by_name(dataset_name="aime", cutoff_year=None, difficulty_filter=None, n_per_difficulty=None, seed=None):
    """
    Load a dataset by name
    
    Args:
        dataset_name (str): Name of the dataset to load ("aime" or "webinstruct")
        cutoff_year (int, optional): If specified for aime, only include problems from this year onward
        difficulty_filter (str, optional): If specified for webinstruct, only include questions with this difficulty
        n_per_difficulty (int, optional): If specified for webinstruct, sample this many questions per difficulty level
        seed (int, optional): Random seed for reproducibility when sampling examples
    
    Returns:
        Dataset: The requested dataset, filtered as specified
    """
    if dataset_name.lower() == "aime":
        return load_aime_dataset(cutoff_year)
    elif dataset_name.lower() == "webinstruct":
        return load_webinstruct_dataset(difficulty_filter, n_per_difficulty, seed)
    elif dataset_name.lower() == "hmmt":
        return load_hmmt_2025_dataset()
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported. Currently supported: 'aime', 'webinstruct', 'hmmt'")

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
