#!/usr/bin/env python3

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import pearsonr, spearmanr

# Import necessary functions from classify_generations.py
from classify_generations import load_data, normalize_column_names, prepare_data

def analyze_feature_correlations(X, y, feature_names, output_dir, dataset_type, model_name):
    """
    Analyze correlations between features and the target variable (is_correct).
    
    Args:
        X: Feature matrix
        y: Target variable (is_correct)
        feature_names: List of feature names
        output_dir: Directory to save plots
        dataset_type: Type of dataset ('aime', 'hmmt', 'webinstruct')
        model_name: Name of the model used for data generation
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(X, columns=feature_names)
    df['is_correct'] = y
    
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution: {df['is_correct'].value_counts()}")
    print("\nFeature statistics:")
    print(df.describe())
    
    # 1. Calculate Pearson correlation between each feature and is_correct
    pearson_corrs = {}
    for feature in feature_names:
        corr, p_value = pearsonr(df[feature], df['is_correct'])
        pearson_corrs[feature] = (corr, p_value)
        print(f"Pearson correlation - {feature}: {corr:.4f} (p-value: {p_value:.4e})")
    
    # 2. Calculate Spearman rank correlation
    spearman_corrs = {}
    for feature in feature_names:
        corr, p_value = spearmanr(df[feature], df['is_correct'])
        spearman_corrs[feature] = (corr, p_value)
        print(f"Spearman correlation - {feature}: {corr:.4f} (p-value: {p_value:.4e})")
    
    # 3. Calculate mutual information (for non-linear relationships)
    mi_values = mutual_info_classif(X, y, random_state=42)
    mi_scores = dict(zip(feature_names, mi_values))
    print("\nMutual information scores:")
    for feature, score in mi_scores.items():
        print(f"{feature}: {score:.6f}")
    
    # 4. Generate boxplots for each feature by is_correct
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(feature_names):
        plt.subplot(2, (len(feature_names) + 1) // 2, i + 1)
        sns.boxplot(x='is_correct', y=feature, data=df)
        plt.title(f"{feature} by correctness\nPearson={pearson_corrs[feature][0]:.3f}")
        plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f"{dataset_type}_{model_name}_feature_boxplots.pdf"), 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # 5. Generate correlation heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                linewidths=0.5, fmt=".2f")
    plt.title(f"Feature Correlation Matrix - {dataset_type} {model_name}")
    
    plt.savefig(os.path.join(output_dir, f"{dataset_type}_{model_name}_correlation_heatmap.pdf"), 
                bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(output_dir, f"{dataset_type}_{model_name}_correlation_heatmap.png"), 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # 6. Generate scatterplots for pairs of features if there are only a few features
    if len(feature_names) <= 10:  # Only create pairplot for small number of features
        plt.figure(figsize=(15, 15))
        sns.pairplot(df, hue='is_correct', vars=feature_names)
        plt.suptitle(f"Feature Pairplot - {dataset_type} {model_name}", y=1.02)
        plt.savefig(os.path.join(output_dir, f"{dataset_type}_{model_name}_feature_pairplot.pdf"), 
                    bbox_inches='tight', dpi=300)
        plt.savefig(os.path.join(output_dir, f"{dataset_type}_{model_name}_feature_pairplot.png"), 
                    bbox_inches='tight', dpi=300)
        plt.close()
    
    # 7. Create sorted bar chart of feature importance
    plt.figure(figsize=(10, 6))
    # Sort by absolute Pearson correlation
    sorted_features = sorted(pearson_corrs.items(), key=lambda x: abs(x[1][0]), reverse=True)
    feature_names_sorted = [f"{feat} (p={p_val:.1e})" for feat, (corr, p_val) in sorted_features]
    corrs_sorted = [corr for feat, (corr, p_val) in sorted_features]
    
    bars = plt.bar(feature_names_sorted, corrs_sorted, color=['blue' if c > 0 else 'red' for c in corrs_sorted])
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title(f"Feature Correlation with is_correct - {dataset_type} {model_name}")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Pearson Correlation")
    plt.ylim(-1, 1)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f"{dataset_type}_{model_name}_feature_importance.pdf"), 
                bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(output_dir, f"{dataset_type}_{model_name}_feature_importance.png"), 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # 8. Return sorted features by importance (absolute correlation)
    return sorted_features

def main():
    parser = argparse.ArgumentParser(description='Analyze feature correlations with is_correct')
    parser.add_argument('--train_data_path', type=str, required=True,
                        help='Path to the parquet file with generation data')
    parser.add_argument('--train_dataset_type', type=str, default='aime', choices=['aime', 'hmmt', 'webinstruct'],
                        help='Type of dataset')
    parser.add_argument('--train_model', type=str, required=True,
                        help='Model name for the data')
    parser.add_argument('--features', type=str, nargs='+',
                        default=['response_length', 'avg_neg_logprob', 'ngram_logprob_diff', 'confusion_metric', 'character_count'],
                        help='Features to analyze')
    parser.add_argument('--output_dir', type=str, default='plots/feature_analysis',
                        help='Directory to save results')
    parser.add_argument('--train_years', type=str, nargs='+', 
                        default=[str(year) for year in range(1983, 2025)],
                        help='Years to include in the analysis')
    parser.add_argument('--exclude_gen_ids', type=int, nargs='*', default=[0, 1, 2, 3, 4],
                        help='Generation IDs to exclude from the analysis')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and prepare data using the prepare_data function
    print(f"Loading data from {args.train_data_path}")
    prepare_result = prepare_data(
        train_data_path=args.train_data_path,
        features=args.features,
        train_years=args.train_years,
        train_dataset_type=args.train_dataset_type,
        exclude_gen_ids=args.exclude_gen_ids
    )
    
    # Check if the prepare_data function returned the expected values
    if prepare_result is None or not isinstance(prepare_result, tuple) or len(prepare_result) != 4:
        # If not, load the data manually using the pattern from the prepare_data function
        print("Using fallback data loading method")
        train_df = load_data(args.train_data_path)
        train_df = normalize_column_names(train_df, args.train_dataset_type)
        train_df = train_df[train_df['year'].isin(args.train_years)]
        train_df = train_df.dropna(subset=args.features + ['is_correct'])
        X = train_df[args.features].values
        y = train_df['is_correct'].values
    else:
        # Unpack the returned values
        X_train, X_test, y_train, y_test = prepare_result
        # Combine train and test data for analysis
        X = np.vstack((X_train, X_test))
        y = np.concatenate((y_train, y_test))
    
    print(f"Data shape after combining: {X.shape}")
    
    # Print the 20 random rows of the dataset
    
    # Scale features (optional)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("\nSample data:")
    sample_df = pd.DataFrame(X_scaled, columns=args.features)
    sample_df['is_correct'] = y
    print(sample_df.sample(10))
    sample_df = pd.DataFrame(X, columns=args.features)
    sample_df['is_correct'] = y
    print(sample_df.sample(10))
    # exit()
    
    # Analyze correlations
    print("\nAnalyzing feature correlations...")
    analyze_feature_correlations(X, y, args.features, args.output_dir, 
                                args.train_dataset_type, args.train_model)
    
    # Additionally analyze scaled features
    print("\nAnalyzing scaled feature correlations...")
    analyze_feature_correlations(X_scaled, y, [f"{f}_scaled" for f in args.features], 
                                args.output_dir, args.train_dataset_type, f"{args.train_model}_scaled")
    
    print(f"\nAnalysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
