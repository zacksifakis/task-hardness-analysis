import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def plot_results(model_results, model_name, temperature=None):
    """
    Create separate plots for response length and negative log probability by problem number
    
    Args:
        model_results: DataFrame with model results
        model_name: Name of the model being plotted
        temperature: Temperature setting used for generation (included in filenames)
    """
    # Set Seaborn style
    sns.set_theme(style="whitegrid")
    
    data = model_results.sort_values('x_value')
    model_name_safe = model_name.replace('/', '_')
    temp_str = f"_temp_{str(temperature).replace('.', '_')}" if temperature is not None else ""
    
    # Plot for response length
    plt.figure()
    # Clean model name for legend - remove anything before '/'
    legend_name = model_name.split('/')[-1] if '/' in model_name else model_name
    plt.plot(data['x_value'], data['mean_length'], 'o-', label=legend_name)
    plt.fill_between(data['x_value'], 
                    data['mean_length'] - data['length_ci'],
                    data['mean_length'] + data['length_ci'],
                    alpha=0.3)
    plt.xlabel('Problem Number')
    plt.ylabel('Average Response Length (tokens)')
    plt.title(f'Response Length vs Problem Number ({model_name})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots_{model_name_safe}{temp_str}_response_length.pdf")
    plt.savefig(f"plots_{model_name_safe}{temp_str}_response_length.png")
    plt.close()
    
    # Convert negative logprobs to positive by taking the negative
    pos_logprobs = data['mean_neg_logprob']
    pos_logprob_ci = data['neg_logprob_ci']
    
    # Plot for positive log probability (- log p)
    plt.figure()
    # Clean model name for legend - remove anything before '/'
    legend_name = model_name.split('/')[-1] if '/' in model_name else model_name
    plt.plot(data['x_value'], pos_logprobs, 'o-', color='orange', label=legend_name)
    plt.fill_between(data['x_value'], 
                    pos_logprobs - pos_logprob_ci,
                    pos_logprobs + pos_logprob_ci,
                    alpha=0.3, color='orange')
    plt.xlabel('Problem Number')
    plt.ylabel('Average Per-token -log(p)')
    plt.title(f'Per-token -log(p) vs Problem Number ({model_name})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots_{model_name_safe}{temp_str}_neg_logprob.pdf")
    plt.savefig(f"plots_{model_name_safe}{temp_str}_neg_logprob.png")
    plt.close()

def plot_combined_results(all_model_results, dataset_name, dataset_info="", temperature=None):
    """
    Create combined plots for all models showing response length and negative log probability
    
    Args:
        all_model_results: Dictionary mapping model names to their aggregated results dataframes
        dataset_name: Name of the dataset being plotted (aime or webinstruct)
        dataset_info: String with additional dataset information to include in filenames
        temperature: Temperature setting used for generation (included in filenames)
    """
    # Set Seaborn style
    sns.set_theme(style="whitegrid")
    sns.set_palette("colorblind")  # Use colorblind-friendly palette for multiple models
    
    # Make sure the plots directory exists for this dataset
    plots_dir = f"plots/{dataset_name}"
    ensure_dir(plots_dir)
    
    # Add temperature to filename if provided
    temp_str = f"_temp_{str(temperature).replace('.', '_')}" if temperature is not None else ""
    
    # Determine x-axis label and rotation based on dataset
    if dataset_name.lower() == "aime":
        x_label = 'Problem Number'
        x_rotation = 0
    else:  # webinstruct dataset
        x_label = 'Difficulty'
        x_rotation = 45
    
    # Get unique x values from all datasets
    all_x_values = set()
    for results in all_model_results.values():
        all_x_values.update(results['x_value'].tolist())
    
    # Sort x values appropriately for the dataset type
    if dataset_name.lower() == "aime":
        # For AIME, sort numerically
        all_x_values = sorted(all_x_values)
    else:
        # For WebInstruct, use the specified order for difficulty levels
        difficulty_order = ['Primary School', 'Junior High School', 'Senior High School', 'University', 'PhD']
        # Filter all_x_values to only include values in difficulty_order and maintain the order
        all_x_values = [x for x in difficulty_order if x in all_x_values]
    
    # Create figure for response length
    plt.figure()
    
    for model_name, results in all_model_results.items():
        # Create a mapping from x values to results for easier lookup
        x_to_results = {}
        for _, row in results.iterrows():
            x_to_results[row['x_value']] = row
            
        # Create lists of data points in the correct x-order
        x_values = []
        y_values = []
        error_values = []
        
        for x_val in all_x_values:
            if x_val in x_to_results:
                x_values.append(x_val)
                y_values.append(x_to_results[x_val]['mean_length'])
                error_values.append(x_to_results[x_val]['length_ci'])
                
        # Clean model name for legend - remove anything before '/'
        legend_name = model_name.split('/')[-1] if '/' in model_name else model_name
        plt.plot(x_values, y_values, 'o-', label=legend_name)
        plt.fill_between(x_values, 
                        np.array(y_values) - np.array(error_values),
                        np.array(y_values) + np.array(error_values),
                        alpha=0.2)
    
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel('Average Response Length (tokens)', fontsize=12)
    plt.title(f'Response Length vs {x_label} - All Models', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=x_rotation)
    plt.tight_layout()
    
    filename_suffix = f"_combined_response_length{dataset_info}{temp_str}"
    plt.savefig(os.path.join(plots_dir, f"plots{filename_suffix}.pdf"))
    plt.savefig(os.path.join(plots_dir, f"plots{filename_suffix}.png"))
    plt.close()
    
    # Create figure for negative log probability
    plt.figure()
    
    for model_name, results in all_model_results.items():
        # Create a mapping from x values to results for easier lookup
        x_to_results = {}
        for _, row in results.iterrows():
            x_to_results[row['x_value']] = row
            
        # Create lists of data points in the correct x-order
        x_values = []
        y_values = []
        error_values = []
        
        for x_val in all_x_values:
            if x_val in x_to_results:
                x_values.append(x_val)
                # Use logprobs directly (they are already negative)
                pos_logprob = x_to_results[x_val]['mean_neg_logprob']
                pos_logprob_ci = x_to_results[x_val]['neg_logprob_ci']
                y_values.append(pos_logprob)
                error_values.append(pos_logprob_ci)
                
        # Clean model name for legend - remove anything before '/'
        legend_name = model_name.split('/')[-1] if '/' in model_name else model_name
        plt.plot(x_values, y_values, 'o-', label=legend_name)
        plt.fill_between(x_values, 
                        np.array(y_values) - np.array(error_values),
                        np.array(y_values) + np.array(error_values),
                        alpha=0.2)
    
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel('Average Per-token -log(p)', fontsize=12)
    plt.title(f'Per-token -log(p) vs {x_label} - All Models', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=x_rotation)
    plt.tight_layout()
    
    filename_suffix = f"_combined_neg_logprob{dataset_info}{temp_str}"
    plt.savefig(os.path.join(plots_dir, f"plots{filename_suffix}.pdf"))
    plt.savefig(os.path.join(plots_dir, f"plots{filename_suffix}.png"))
    plt.close()

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
