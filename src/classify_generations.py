#!/usr/bin/env python3

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import random
import math
from itertools import combinations

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import random

sns.set_theme(style="whitegrid")



def load_data(parquet_path):
    """
    Load the parquet file with generation data.
    
    Args:
     plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
    
    # Create the consistent output path
    dataset_path = f"{train_dataset_type}_{train_model}/"
    output_path = os.path.join('plots', 'classification', dataset_path)
    os.makedirs(output_path, exist_ok=True)
    
    # Also create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set balance suffix
    balance_suffix = "_balanced" if balanced else ""
    
    # Save only PDF to the consistent path
    plt.savefig(os.path.join(output_path, f"{filename}{balance_suffix}.pdf"), dpi=300, bbox_inches='tight')
    
    # Check if the output_dir already contains the dataset path
    if os.path.basename(output_dir) == dataset_path.rstrip('/') or output_dir.endswith(dataset_path):
        # If output_dir already includes the dataset path, don't add it again
        plt.savefig(os.path.join(output_dir, f"{filename}{balance_suffix}.pdf"), dpi=300, bbox_inches='tight')
    else:
        # Otherwise, include the dataset path in the output directory
        custom_output_path = os.path.join(output_dir, dataset_path)
        os.makedirs(custom_output_path, exist_ok=True)
        plt.savefig(os.path.join(custom_output_path, f"{filename}{balance_suffix}.pdf"), dpi=300, bbox_inches='tight') Path to the parquet file
        
    Returns:
        pandas DataFrame with the data
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found at: {parquet_path}")
    
    df = pd.read_parquet(parquet_path)
    print(f"Loaded data with {len(df)} rows")
    return df

def normalize_column_names(df, dataset_type):
    """
    Normalize column names based on dataset type to ensure consistent column names.
    
    Args:
        df: Input DataFrame
        dataset_type: Type of dataset ('aime', 'hmmt', or 'webinstruct')
        
    Returns:
        DataFrame with normalized column names
    """
    df = df.copy()  # Create a copy to avoid modifying the original
    
    # Ensure dataset_type is lowercase for case-insensitive comparison
    dataset_type = dataset_type.lower()
    
    # Add is_correct column if it doesn't exist (based on extracted_answer and answer columns)
    if 'is_correct' not in df.columns and 'answer' in df.columns and 'extracted_answer' in df.columns:
        df['is_correct'] = df['extracted_answer'] == df['answer']
    
    # Normalize column names based on dataset type
    if dataset_type == 'aime':
        # AIME dataset uses 'Year' and 'Problem Number'
        if 'Year' in df.columns and 'year' not in df.columns:
            df['year'] = df['Year']
        if 'Problem Number' in df.columns and 'problem_number' not in df.columns:
            df['problem_number'] = df['Problem Number']
            
    elif dataset_type == 'hmmt':
        # HMMT dataset uses 'year' and 'problem_idx'
        if 'year' not in df.columns and 'year' in df.columns:
            df['year'] = df['year']
        if 'problem_idx' in df.columns and 'problem_number' not in df.columns:
            df['problem_number'] = df['problem_idx']
            
    elif dataset_type == 'webinstruct':
        # WebInstruct dataset uses 'difficulty' and 'id'
        if 'difficulty' in df.columns and 'year' not in df.columns:
            # For WebInstruct, we'll use difficulty as a proxy for year for filtering
            df['year'] = df['difficulty']
        if 'id' in df.columns and 'problem_number' not in df.columns:
            df['problem_number'] = df['id']
    
    return df

def prepare_data(
        train_data_path,
        features,
        train_years,
        test_data_path=None,
        test_years=None,
        test_size=0.2,
        random_state=42,
        train_dataset_type="aime",
        test_dataset_type=None,
        balance_data=False,
        exclude_gen_ids=None
    ):
    """
    Prepare data for training and testing from specified datasets.
    
    Args:
        train_data_path: Path to the parquet file for training data
        features: List of feature columns to use
        train_years: List of years to use for training (and validation if test_years is None)
        test_data_path: Path to the parquet file for testing data (if None, will use train_data_path)
        test_years: List of years to use for testing (if None, will use train_test_split)
        test_size: Proportion of data to use for testing if test_years is None
        random_state: Random seed for reproducibility
        train_dataset_type: Type of training dataset ('aime', 'hmmt', or 'webinstruct')
        test_dataset_type: Type of testing dataset (if None, will use train_dataset_type)
        balance_data: Whether to balance the dataset to have equal True/False in is_correct
        
    Returns:
        X_train, X_test, y_train, y_test: Training and testing data
    """
    # Load training data
    print(f"Loading training data from {train_data_path} (type: {train_dataset_type})")
    train_df = load_data(train_data_path)
        
    # Set default test dataset type if not provided
    if test_dataset_type is None:
        test_dataset_type = train_dataset_type
        
    # Normalize column names based on dataset type to ensure consistency
    train_df = normalize_column_names(train_df, train_dataset_type)
    
    # Filter data for specified years
    train_df = train_df[train_df['year'].isin(train_years)]
    
    # Keep only the rows that have `generation` not in `exclude_gen_ids`
    if exclude_gen_ids:
        train_df = train_df[~train_df['generation'].isin(exclude_gen_ids)]

    # Remove rows with NaN values in any of the feature columns or target
    train_df = train_df.dropna(subset=features + ['is_correct'])
    
    if test_years:
        test_df = load_data(test_data_path if test_data_path else train_data_path)
        test_df = normalize_column_names(test_df, test_dataset_type)
    
        # Use specific years for testing
        test_df = test_df[test_df['year'].isin(test_years)]
        test_df = test_df.dropna(subset=features + ['is_correct'])
        
        # Balance datasets if requested
        if balance_data:
            print("Balancing training dataset...")
            train_df = balance_dataset(train_df)
            # print("Balancing test dataset...")
            # test_df = balance_dataset(test_df)
            
        X_train = train_df[features]
        y_train = train_df['is_correct']
        
        X_test = test_df[features]
        y_test = test_df['is_correct']
    else:
        # Balance dataset before splitting if requested
        if balance_data:
            print("Balancing dataset before train/test split...")
            train_df = balance_dataset(train_df)
            
        # Split the train data into train and validation sets
        X = train_df[features]
        y = train_df['is_correct']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    
    print(f"Training data: {X_train.shape[0]} samples")
    print(f"Testing data: {X_test.shape[0]} samples")
    print(f"Class distribution in training: {y_train.value_counts(normalize=True).to_dict()}")
    
    return X_train, X_test, y_train, y_test

def train_knn_classifier(X_train, y_train, n_neighbors=5, param_grid=None, cv=5):
    """
    Train a K-Nearest Neighbors classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_neighbors: Number of neighbors (used if param_grid is None)
        param_grid: Parameter grid for GridSearchCV
        cv: Number of cross-validation folds
        
    Returns:
        Trained KNN classifier
    """
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if param_grid is None:
        # Use default parameters
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train_scaled, y_train)
        model = {'classifier': knn, 'scaler': scaler}
    else:
        # Use GridSearchCV to find optimal parameters
        knn = KNeighborsClassifier()
        grid_search = GridSearchCV(knn, param_grid, cv=cv, scoring='f1')
        grid_search.fit(X_train_scaled, y_train)
        
        # Print best parameters
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        model = {'classifier': grid_search.best_estimator_, 'scaler': scaler}
    
    return model

def train_svm_classifier(X_train, y_train, kernel='rbf', C=1.0, gamma='scale', param_grid=None, cv=5):
    """
    Train a Support Vector Machine classifier with kernel.
    
    Args:
        X_train: Training features
        y_train: Training target
        kernel: Kernel type to be used ('linear', 'poly', 'rbf', 'sigmoid')
        C: Regularization parameter
        gamma: Kernel coefficient (default is 'scale')
        param_grid: Parameter grid for GridSearchCV
        cv: Number of cross-validation folds
        
    Returns:
        Trained SVM classifier model
    """
    # Scale the features
    print(f"Scaling features for {X_train.shape[0]} samples, {X_train.shape[1]} features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if param_grid is None:
        # Use default parameters
        print(f"Training SVM with fixed parameters:")
        print(f"  - Kernel: {kernel}")
        print(f"  - C: {C}")
        print(f"  - Gamma: {gamma}")
        
        # Initialize and train the SVM
        svm = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
        print("Starting SVM training...")
        svm.fit(X_train_scaled, y_train)
        
        # Print additional information
        print("SVM training completed")
        print(f"Number of support vectors: {svm.n_support_}")
        print(f"Support vector distribution: {dict(zip(svm.classes_, svm.n_support_))}")
        
        model = {'classifier': svm, 'scaler': scaler}
    else:
        # Use GridSearchCV to find optimal parameters
        print(f"Starting hyperparameter grid search with {cv}-fold cross-validation")
        print(f"Parameter grid: {param_grid}")
        
        parameter_combinations = 1
        for key, values in param_grid.items():
            parameter_combinations *= len(values)
        print(f"Total parameter combinations to evaluate: {parameter_combinations}")
        
        svm = SVC(probability=True)
        grid_search = GridSearchCV(svm, param_grid, cv=cv, scoring='f1', verbose=2)
        print("Starting grid search...")
        grid_search.fit(X_train_scaled, y_train)
        
        # Print best parameters
        print("\n" + "="*50)
        print(f"Grid search completed")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Print top 3 parameter combinations
        cv_results = pd.DataFrame(grid_search.cv_results_)
        top_results = cv_results.sort_values('rank_test_score').head(3)
        print("\nTop 3 parameter combinations:")
        for i, (_, row) in enumerate(top_results.iterrows(), 1):
            params = {k.replace('param_', ''): v for k, v in row.items() if k.startswith('param_') and not pd.isna(v)}
            print(f"{i}. Parameters: {params}")
            print(f"   Mean test score: {row['mean_test_score']:.4f}, Std: {row['std_test_score']:.4f}")
        
        print("="*50)
        
        model = {'classifier': grid_search.best_estimator_, 'scaler': scaler}
    
    return model

def train_mlp_classifier(X_train, y_train, hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0005, n_iter_no_change=100,
                   learning_rate='constant', learning_rate_init=0.0005, max_iter=200, param_grid=None, cv=5, early_stopping=True):
    """
    Train a Multi-Layer Perceptron classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        hidden_layer_sizes: Tuple with the number of neurons in each hidden layer
        activation: Activation function ('identity', 'logistic', 'tanh', 'relu')
        solver: The solver for weight optimization ('lbfgs', 'sgd', 'adam')
        alpha: L2 regularization parameter
        learning_rate: Learning rate schedule ('constant', 'invscaling', 'adaptive')
        learning_rate_init: Initial learning rate
        max_iter: Maximum number of iterations
        param_grid: Parameter grid for GridSearchCV
        cv: Number of cross-validation folds
        early_stopping: Whether to stop training early if validation loss does not improve

    Returns:
        Trained MLP classifier model
    """
    # Scale the features
    print(f"Scaling features for {X_train.shape[0]} samples, {X_train.shape[1]} features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if param_grid is None:
        # Use default parameters
        print(f"Training MLP with fixed parameters:")
        print(f"  - Hidden layer sizes: {hidden_layer_sizes}")
        print(f"  - Activation: {activation}")
        print(f"  - Solver: {solver}")
        print(f"  - Alpha: {alpha}")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Initial learning rate: {learning_rate_init}")
        print(f"  - Max iterations: {max_iter}")
        
        # Initialize and train the MLP
        mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            random_state=42,
            verbose=True,
            early_stopping=early_stopping,
            n_iter_no_change=n_iter_no_change,
        )
        print("Starting MLP training...")
        mlp.fit(X_train_scaled, y_train)
        
        # Print additional information
        print("MLP training completed")
        print(f"Number of iterations: {mlp.n_iter_}")
        print(f"Loss: {mlp.loss_}")
        
        model = {'classifier': mlp, 'scaler': scaler}
    else:
        # Use GridSearchCV to find optimal parameters
        print(f"Starting hyperparameter grid search with {cv}-fold cross-validation")
        print(f"Parameter grid: {param_grid}")
        
        parameter_combinations = 1
        for key, values in param_grid.items():
            parameter_combinations *= len(values)
        print(f"Total parameter combinations to evaluate: {parameter_combinations}")
        
        mlp = MLPClassifier(random_state=42)
        grid_search = GridSearchCV(mlp, param_grid, cv=cv, scoring='f1', verbose=2)
        print("Starting grid search...")
        grid_search.fit(X_train_scaled, y_train)
        
        # Print best parameters
        print("\n" + "="*50)
        print(f"Grid search completed")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Print top 3 parameter combinations
        cv_results = pd.DataFrame(grid_search.cv_results_)
        top_results = cv_results.sort_values('rank_test_score').head(3)
        print("\nTop 3 parameter combinations:")
        for i, (_, row) in enumerate(top_results.iterrows(), 1):
            params = {k.replace('param_', ''): v for k, v in row.items() if k.startswith('param_') and not pd.isna(v)}
            print(f"{i}. Parameters: {params}")
            print(f"   Mean test score: {row['mean_test_score']:.4f}, Std: {row['std_test_score']:.4f}")
        
        print("="*50)
        
        model = {'classifier': grid_search.best_estimator_, 'scaler': scaler}
    
    return model

def evaluate_classifier(model, X_test, y_test):
    """
    Evaluate classifier performance.
    
    Args:
        model: Trained model dict with classifier and scaler
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Scale the test data
    X_test_scaled = model['scaler'].transform(X_test)
    
    # Get predictions
    y_pred = model['classifier'].predict(X_test_scaled)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    # Print evaluation results
    print("\nEvaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    # Wait 3 seconds
    import time
    time.sleep(3)
    
    return metrics

def plot_confusion_matrix(cm, output_dir, filename='knn_confusion_matrix', train_dataset_type='aime', test_dataset_type=None, balanced=False, train_model=None, test_model=None):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        output_dir: Directory to save the plot
        filename: Base filename for the plot (without extension)
        train_dataset_type: Type of training dataset
        test_dataset_type: Type of testing dataset (if None, will use train_dataset_type)
        balanced: Whether balanced dataset was used
        train_model: Model name used for training
        test_model: Model name used for testing (if None, will use train_model)
    """
    if test_dataset_type is None:
        test_dataset_type = train_dataset_type
        
    if test_model is None:
        test_model = train_model
    
    balance_suffix = "_balanced" if balanced else ""
    model_suffix = ""
    if train_model:
        model_suffix += f"_{train_model}"  # Use the full model name in filenames
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Incorrect', 'Correct'],
                yticklabels=['Incorrect', 'Correct'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    title_text = f'Confusion Matrix ({train_dataset_type.upper()}'
    if train_model:
        title_text += f', {train_model}'
    title_text += f' → {test_dataset_type.upper()}'
    if test_model and test_model != train_model:
        title_text += f', {test_model}'
    title_text += ')'
    
    if balanced:
        title_text += ' [Balanced Data]'
    plt.title(title_text, fontsize=10)  # Reduced fontsize to accommodate longer titles
    
    # Create the consistent output path
    dataset_path = f"{train_dataset_type}_{train_model}/"
    output_path = os.path.join('plots', 'classification', dataset_path)
    os.makedirs(output_path, exist_ok=True)
    
    # Also create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save only PDF to the consistent path
    plt.savefig(os.path.join(output_path, f"{filename}{balance_suffix}.pdf"), dpi=300, bbox_inches='tight')
    
    # Check if the output_dir already contains the dataset path
    if os.path.basename(output_dir) == dataset_path.rstrip('/') or output_dir.endswith(dataset_path):
        # If output_dir already includes the dataset path, don't add it again
        plt.savefig(os.path.join(output_dir, f"{filename}{balance_suffix}.pdf"), dpi=300, bbox_inches='tight')
    else:
        # Otherwise, include the dataset path in the output directory
        custom_output_path = os.path.join(output_dir, dataset_path)
        os.makedirs(custom_output_path, exist_ok=True)
        plt.savefig(os.path.join(custom_output_path, f"{filename}{balance_suffix}.pdf"), dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_decision_boundary(model, X_train, y_train, X_test, y_test, feature_names, output_dir, filename='decision_boundary', test_years=None, train_dataset_type='aime', test_dataset_type=None, balanced=False, train_model=None, test_model=None):
    """
    Plot the decision boundary of a classifier along with the data points.
    
    Args:
        model: Trained model dict with classifier and scaler
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        feature_names: Names of the features (for axis labels)
        output_dir: Directory to save the plot
        filename: Base filename for the plot (without extension)
        test_years: List of test years, used for labeling
        train_dataset_type: Type of training dataset
        test_dataset_type: Type of testing dataset (if None, will use train_dataset_type)
        balanced: Whether balanced dataset was used
        train_model: Model name used for training
        test_model: Model name used for testing (if None, will use train_model)
    """
    if test_dataset_type is None:
        test_dataset_type = train_dataset_type
        
    if test_model is None:
        test_model = train_model

    # This function works best with 2 features
    if X_train.shape[1] != 2:
        print("Warning: Decision boundary plot requires exactly 2 features.")
        # If more than 2 features, use the first 2 for visualization
        if X_train.shape[1] > 2:
            print(f"Using only the first 2 features: {feature_names[0]} and {feature_names[1]}")
            X_train = X_train.iloc[:, :2]
            X_test = X_test.iloc[:, :2]
            feature_names = feature_names[:2]
        else:
            return

    # Scale data
    X_train_scaled = model['scaler'].transform(X_train)
    X_test_scaled = model['scaler'].transform(X_test)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Create meshgrid for decision boundary
    h = 0.02  # Step size in the mesh
    x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
    y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict class for each point in the meshgrid
    Z = model['classifier'].predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlGn)
    plt.contour(xx, yy, Z, colors='k', linewidths=0.5)
    
    # Get original scales for the plot
    scaler = model['scaler']
    xx_orig = scaler.inverse_transform(np.c_[xx.ravel(), np.zeros_like(xx.ravel())])[:, 0].reshape(xx.shape)
    yy_orig = scaler.inverse_transform(np.c_[np.zeros_like(yy.ravel()), yy.ravel()])[:, 1].reshape(yy.shape)
    
    # Plot both training and test points
    if test_years and len(test_years) > 0:
        # Create custom markers for test years
        markers = {year: marker for year, marker in zip(test_years, ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*'])}
        
        # Plot test points with different markers for different years
        for i, year in enumerate(test_years):
            # Filter test data for this year
            year_indices = np.where(np.array(X_test.index.get_level_values('year') if isinstance(X_test.index, pd.MultiIndex) 
                                    else [year] * len(X_test)) == year)[0]
            
            if len(year_indices) > 0:
                # Get data for this year
                X_year = X_test_scaled[year_indices]
                y_year = np.array(y_test)[year_indices]
                
                # Plot correct predictions (green)
                plt.scatter(X_year[y_year == True, 0], X_year[y_year == True, 1], 
                            c='green', edgecolor='k', s=100, marker=markers.get(year, 'o'),
                            label=f'Year {year} - Correct')
                
                # Plot incorrect predictions (red)
                plt.scatter(X_year[y_year == False, 0], X_year[y_year == False, 1], 
                            c='red', edgecolor='k', s=100, marker=markers.get(year, 'o'),
                            label=f'Year {year} - Incorrect')
    else:
        # Plot test points (if no specific years are provided)
        plt.scatter(X_test_scaled[y_test == True, 0], X_test_scaled[y_test == True, 1], 
                    c='green', edgecolor='k', s=100, marker='o', label='Test - Correct')
        plt.scatter(X_test_scaled[y_test == False, 0], X_test_scaled[y_test == False, 1], 
                    c='red', edgecolor='k', s=100, marker='o', label='Test - Incorrect')
    
    # Set axis limits to show all points
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    # Create a secondary axis that shows the original feature values
    # Add an axis at the top of the current figure
    ax1 = plt.gca()
    ax2 = ax1.twiny()
    ax3 = ax1.twinx()
    
    # Set the secondary axis ticks to show original feature values
    x_ticks = np.linspace(x_min, x_max, num=5)
    y_ticks = np.linspace(y_min, y_max, num=5)
    x_tick_labels = [f"{val:.2f}" for val in scaler.inverse_transform(np.column_stack((x_ticks, np.zeros_like(x_ticks))))[:, 0]]
    y_tick_labels = [f"{val:.2f}" for val in scaler.inverse_transform(np.column_stack((np.zeros_like(y_ticks), y_ticks)))[:, 1]]
    
    ax2.set_xlim(ax1.get_xlim())
    ax3.set_ylim(ax1.get_ylim())
    ax2.set_xticks(x_ticks)
    ax3.set_yticks(y_ticks)
    ax2.set_xticklabels(x_tick_labels)
    ax3.set_yticklabels(y_tick_labels)
    
    # Label the axes with original feature names
    ax1.set_xlabel(f"Standardized {feature_names[0]}")
    ax1.set_ylabel(f"Standardized {feature_names[1]}")
    ax2.set_xlabel(f"Original {feature_names[0]}")
    ax3.set_ylabel(f"Original {feature_names[1]}")
    
    # Set appropriate title based on classifier type
    if hasattr(model['classifier'], 'n_neighbors'):
        # KNN classifier
        title = f"Decision Boundary for KNN (k={model['classifier'].n_neighbors})"
    elif hasattr(model['classifier'], 'kernel'):
        # SVM classifier
        kernel_type = model['classifier'].kernel
        C_value = model['classifier'].C
        if hasattr(model['classifier'], 'gamma') and model['classifier'].gamma != 'auto':
            gamma_value = model['classifier'].gamma
            title = f"Decision Boundary for SVM (kernel={kernel_type}, C={C_value}, γ={gamma_value})"
        else:
            title = f"Decision Boundary for SVM (kernel={kernel_type}, C={C_value})"
    else:
        title = "Decision Boundary"
    
    title_text = f"{title}\n{train_dataset_type.upper()}"
    if train_model:
        title_text += f" ({train_model})"
    title_text += f" → {test_dataset_type.upper()}"
    if test_model and test_model != train_model:
        title_text += f" ({test_model})"
    
    if balanced:
        title_text += ' [Balanced Data]'
    plt.title(title_text, fontsize=10)  # Reduced fontsize to accommodate longer titles
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
    
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join('plots', 'classification')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Also create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset-specific filename suffix
    dataset_suffix = f"_{train_dataset_type}_to_{test_dataset_type}"
    model_suffix = ""
    if train_model:
        model_suffix += f"_{train_model.split('_')[0]}"  # Use just the organization part for brevity in filenames
    balance_suffix = "_balanced" if balanced else ""
    
    # Save as PNG
    plt.savefig(os.path.join(plots_dir, f"{filename}{dataset_suffix}{model_suffix}{balance_suffix}.png"), dpi=300, bbox_inches='tight')
    # Save as PDF
    plt.savefig(os.path.join(plots_dir, f"{filename}{dataset_suffix}{model_suffix}{balance_suffix}.pdf"), dpi=300, bbox_inches='tight')
    # Also save to original output directory
    plt.savefig(os.path.join(output_dir, f"{filename}{dataset_suffix}{model_suffix}{balance_suffix}.png"), dpi=300, bbox_inches='tight')
    
    plt.close()

def plot_roc_curve(model, X_test, y_test, output_dir, filename='roc_curve', train_dataset_type='aime', test_dataset_type=None, balanced=False, train_model=None, test_model=None):
    """
    Plot the Receiver Operating Characteristic (ROC) curve for a classifier.
    
    Args:
        model: Trained model dict with classifier and scaler
        X_test: Test features
        y_test: Test target
        output_dir: Directory to save the plot
        filename: Base filename for the plot (without extension)
        train_dataset_type: Type of training dataset
        test_dataset_type: Type of testing dataset (if None, will use train_dataset_type)
        balanced: Whether balanced dataset was used
        train_model: Model name used for training
        test_model: Model name used for testing (if None, will use train_model)
    """
    if test_dataset_type is None:
        test_dataset_type = train_dataset_type
        
    if test_model is None:
        test_model = train_model
    
    # Scale the test data
    X_test_scaled = model['scaler'].transform(X_test)
    
    # Get prediction probabilities
    classifier = model['classifier']
    
    # For models that support probability prediction
    if hasattr(classifier, 'predict_proba'):
        y_pred_proba = classifier.predict_proba(X_test_scaled)[:, 1]
    else:
        # For models that can output decision function (like some SVMs)
        y_pred_proba = classifier.decision_function(X_test_scaled)
    
    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    title_text = f'ROC Curve ({train_dataset_type.upper()}'
    if train_model:
        title_text += f', {train_model}'
    title_text += f' → {test_dataset_type.upper()}'
    if test_model and test_model != train_model:
        title_text += f', {test_model}'
    title_text += ')'
    
    if balanced:
        title_text += ' [Balanced Data]'
    plt.title(title_text, fontsize=10)  # Reduced fontsize to accommodate longer titles
    
    plt.legend(loc="lower right")
    
    # Create the consistent output path
    dataset_path = f"{train_dataset_type}_{train_model}/"
    output_path = os.path.join('plots', 'classification', dataset_path)
    os.makedirs(output_path, exist_ok=True)
    
    # Also create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set balance suffix
    balance_suffix = "_balanced" if balanced else ""
    
    # Save only PDF to the consistent path
    plt.savefig(os.path.join(output_path, f"{filename}{balance_suffix}.pdf"), dpi=300, bbox_inches='tight')
    
    # Check if the output_dir already contains the dataset path
    if os.path.basename(output_dir) == dataset_path.rstrip('/') or output_dir.endswith(dataset_path):
        # If output_dir already includes the dataset path, don't add it again
        plt.savefig(os.path.join(output_dir, f"{filename}{balance_suffix}.pdf"), dpi=300, bbox_inches='tight')
    else:
        # Otherwise, include the dataset path in the output directory
        custom_output_path = os.path.join(output_dir, dataset_path)
        os.makedirs(custom_output_path, exist_ok=True)
        plt.savefig(os.path.join(custom_output_path, f"{filename}{balance_suffix}.pdf"), dpi=300, bbox_inches='tight')
    
    plt.close()
    
    return {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'auc': roc_auc}

def balance_dataset(df, target_column='is_correct'):
    """
    Balance the dataset by randomly sampling from the majority class to match the minority class count.
    
    Args:
        df: Input DataFrame
        target_column: Column to balance on (typically 'is_correct')
        
    Returns:
        Balanced DataFrame
    """
    # Count the occurrences of each class
    value_counts = df[target_column].value_counts()
    minority_class = value_counts.idxmin()
    majority_class = value_counts.idxmax()
    minority_count = value_counts[minority_class]
    
    # Separate the dataset by class
    minority_df = df[df[target_column] == minority_class]
    majority_df = df[df[target_column] == majority_class]
    
    # Randomly sample from the majority class to match the minority class count
    majority_sample = majority_df.sample(n=minority_count, random_state=42)
    
    # Combine the minority class with the sampled majority class
    balanced_df = pd.concat([minority_df, majority_sample])
    
    # Shuffle the final DataFrame
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Original dataset: {len(df)} samples, class distribution: {dict(value_counts)}")
    print(f"Balanced dataset: {len(balanced_df)} samples, class distribution: {dict(balanced_df[target_column].value_counts())}")
    
    return balanced_df

def plot_voting_comparison(model, test_data_path, features, output_dir, filename='voting_comparison', train_dataset_type='aime', test_dataset_type=None, balanced=False, train_model=None, test_model=None, max_generations=19, choose_max=20, test_years=None, classifier_method='majority'):
    """
    Plot a comparison between majority voting and classifier-based approaches for prediction accuracy
    across different numbers of generations, averaging over multiple subsets of generations per problem.
    
    Args:
        model: Trained model dict with classifier and scaler
        test_data_path: Path to the raw test data parquet file
        features: List of feature columns used for classification
        output_dir: Directory to save the plot
        filename: Base filename for the plot (without extension)
        train_dataset_type: Type of training dataset
        test_dataset_type: Type of testing dataset (if None, will use train_dataset_type)
        balanced: Whether balanced dataset was used
        train_model: Model name used for training
        test_model: Model name used for testing (if None, will use train_model)
        max_generations: Maximum number of generations per problem to consider
        choose_max: Maximum number of subsets to average over per problem (if there are more possible subsets, use all available)
        test_years: List of years to filter the test data by (if None, will use all years)
    """
    if test_dataset_type is None:
        test_dataset_type = train_dataset_type
        
    if test_model is None:
        test_model = train_model
    
    # Load and normalize the raw test data
    print(f"Loading raw test data from {test_data_path} (type: {test_dataset_type})")
    test_df = load_data(test_data_path)
    test_df = normalize_column_names(test_df, test_dataset_type)
    
    # Filter data by test_years if provided
    if test_years:
        print(f"Filtering data to include only years: {test_years}")
        test_df = test_df[test_df['year'].isin(test_years)]
        if test_df.empty:
            print(f"Warning: No data found for years {test_years} in the test dataset")
            return
    
    # Prepare data for plotting
    # We'll compare accuracy at different generation counts: 1, 3, 5, 7, 9, 11, 13, 15
    gen_counts = list(range(3, min(max_generations + 1, 16), 2))
    
    # Initialize arrays to store accuracy results and confidence intervals
    majority_voting_acc = []
    classifier_acc = []
    majority_voting_ci = []
    classifier_ci = []
    perfect_classifier_acc = []
    perfect_classifier_ci = []

    
    print(f"Analyzing accuracy for {len(gen_counts)} different generation counts...")
    
    # Group by problem_id to process each problem separately
    problem_groups = test_df.groupby(['problem_number', 'part', 'year'])
    total_problems = len(problem_groups)
    
    
    print(f"Processing {total_problems} unique problems")
    
    # For each generation count, compute accuracy using both approaches
    for k in gen_counts:
        print(f"Processing with k={k} generations...")
        
        # Lists to store accuracy results for each sample
        majority_samples = []
        classifier_samples = []
        perfect_classifier_samples = []
        
        print(f"There are {len(problem_groups)} unique problems to process...")
        
        # Process each problem
        for problem_id, problem_data in problem_groups:
            # Get the answer for this problem
            correct_answer = problem_data['answer'].iloc[0]
            
            # Make sure we have enough generations for the current k
            if len(problem_data) < k:
                continue
                
            # Get all available generations for this problem
            available_generations = problem_data['generation'].unique()
            
            # Calculate total possible combinations
            total_combinations = math.comb(len(available_generations), k)
            
            # Determine how many samples to use for this problem
            num_samples = min(choose_max, total_combinations)
            
            # If there are too many combinations, randomly sample them
            if total_combinations > choose_max:
                # Generate random samples without replacement
                sampled_combinations = []
                for _ in range(num_samples):
                    sample = tuple(sorted(random.sample(list(available_generations), k)))
                    # Ensure we don't have duplicate combinations
                    while sample in sampled_combinations:
                        sample = tuple(sorted(random.sample(list(available_generations), k)))
                    sampled_combinations.append(sample)
            else:
                # Use all possible combinations
                sampled_combinations = list(combinations(available_generations, k))
            
            # Process each sampled combination of generations
            for combo in sampled_combinations:

                # Filter to get only the selected generations for this combination
                selected_generations = problem_data[problem_data['generation'].isin(combo)]
                
                if len(selected_generations) == 0:
                    continue
                
                # Approach 1: Majority Voting
                # Count the occurrences of each extracted answer
                if 'extracted_answer' in selected_generations.columns:
                    answer_counts = selected_generations['extracted_answer'].value_counts()
                    if not answer_counts.empty:
                        majority_answer = answer_counts.index[0]
                        majority_samples.append(1 if majority_answer == correct_answer else 0)
                
                # Approach 2: Classifier-based
                # Prepare features for classification
                X_problem = selected_generations[features]
                
                # Get corresponding labels and answers
                y_problem = selected_generations['is_correct']
                answers_problem = selected_generations['extracted_answer']
                
                if len(X_problem) == 0:
                    continue
                    
                # Scale the features
                X_problem_scaled = model['scaler'].transform(X_problem)
                
                # Predict correctness for each generation
                try:
                    if classifier_method == 'majority':
                        y_pred = model['classifier'].predict(X_problem_scaled)
                        
                        # Filter out the generations that are classified as incorrect
                        correct_generations = selected_generations[y_pred == 1]
                        
                        # Take majority vote among the correct generations
                        if not correct_generations.empty:
                            answer_counts = correct_generations['extracted_answer'].value_counts()
                            if not answer_counts.empty:
                                majority_answer = answer_counts.index[0]
                                classifier_samples.append(1 if majority_answer == correct_answer else 0)
                            else:
                                classifier_samples.append(0)
                        else:
                            classifier_samples.append(0)
                    elif classifier_method == 'top_probability':
                        # Get the top probability predictions
                        top_probs = model['classifier'].predict_proba(X_problem_scaled)
                        # import pdb; pdb.set_trace()
                        # Select the generation with the highest probability and its corresponding answer
                        top_index = np.argmax(top_probs[:,1])
                        top_answer = answers_problem.iloc[top_index]
                        # Check if the top answer matches the correct answer
                        classifier_samples.append(1 if top_answer == correct_answer else 0)
                    elif classifier_method == 'majority_density':
                        # Get the predicted probabilities
                        probs = model['classifier'].predict_proba(X_problem_scaled)
                        # Group by the extracted answer, and take the average of the corresponding probabilities
                        prob_df = pd.DataFrame({'extracted_answer': answers_problem, 'probability': probs[:, 1]})
                        prob_df = prob_df.groupby('extracted_answer').mean().reset_index()
                        # Get the answer with the highest average probability
                        if not prob_df.empty:
                            majority_answer = prob_df.loc[prob_df['probability'].idxmax(), 'extracted_answer']
                            classifier_samples.append(1 if majority_answer == correct_answer else 0)
                        
                    else:
                        raise ValueError(f"Unknown classifier method: {classifier_method}")
                except Exception as e:
                    print(f"Error during prediction for problem {problem_id}: {e}")
                    raise e
                
                # Approach 3: Simulating a perfect classifier
                if classifier_method == 'majority':
                    # Simulate a perfect classifier by checking if the predicted answer matches the correct answer
                    if 'extracted_answer' in selected_generations.columns:
                        perfect_answer_counts = selected_generations['extracted_answer'].value_counts()
                        if not perfect_answer_counts.empty:
                            perfect_majority_answer = perfect_answer_counts.index[0]
                            perfect_classifier_samples.append(1 if perfect_majority_answer == correct_answer else 0)
                    else:   
                        perfect_classifier_samples.append(0)
                elif classifier_method == 'top_probability':
                    # Simulate a perfect classifier: if the correct answer is in the selected generations, append 1, else append 0
                    if correct_answer in selected_generations['extracted_answer'].values:
                        perfect_classifier_samples.append(1)
                    else:
                        perfect_classifier_samples.append(0)
                elif classifier_method == 'majority_density':
                    # Simulate a perfect classifier: if the correct answer is in the selected generations, append 1, else append 0
                    if correct_answer in selected_generations['extracted_answer'].values:
                        perfect_classifier_samples.append(1)
                    else:
                        perfect_classifier_samples.append(0)
                else:
                    raise ValueError(f"Unknown classifier method: {classifier_method}")

        
        # Calculate average accuracy and confidence intervals
        if majority_samples:
            majority_mean = np.mean(majority_samples)
            majority_std = np.std(majority_samples, ddof=1)
            majority_ci_value = 1.96 * majority_std / np.sqrt(len(majority_samples))
            
            majority_voting_acc.append(majority_mean)
            majority_voting_ci.append(majority_ci_value)
        else:
            majority_voting_acc.append(0)
            majority_voting_ci.append(0)
            
        if classifier_samples:
            classifier_mean = np.mean(classifier_samples)
            classifier_std = np.std(classifier_samples, ddof=1)
            classifier_ci_value = 1.96 * classifier_std / np.sqrt(len(classifier_samples))
            
            classifier_acc.append(classifier_mean)
            classifier_ci.append(classifier_ci_value)
        else:
            classifier_acc.append(0)
            classifier_ci.append(0)
            
        if perfect_classifier_samples:
            perfect_classifier_mean = np.mean(perfect_classifier_samples)
            perfect_classifier_std = np.std(perfect_classifier_samples, ddof=1)
            perfect_classifier_ci_value = 1.96 * perfect_classifier_std / np.sqrt(len(perfect_classifier_samples))
            
            # Store the perfect classifier accuracy and CI
            perfect_classifier_acc.append(perfect_classifier_mean)
            perfect_classifier_ci.append(perfect_classifier_ci_value)
        else:
            perfect_classifier_acc.append(0)
            perfect_classifier_ci.append(0)
            
        
        print(f"k={k}: Majority Voting Accuracy = {majority_voting_acc[-1]:.4f} ± {majority_voting_ci[-1]:.4f}, " 
              f"Classifier Accuracy = {classifier_acc[-1]:.4f} ± {classifier_ci[-1]:.4f}, "
              f"Perfect Classifier Accuracy = {perfect_classifier_acc[-1]:.4f} ± {perfect_classifier_ci[-1]:.4f}, "
              f"Samples = {len(majority_samples)}")
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot accuracy with error bands
    plt.errorbar(gen_counts, majority_voting_acc, yerr=majority_voting_ci, fmt='o-', color='blue', 
                 ecolor='lightblue', elinewidth=3, capsize=5, label='Majority Voting')
    plt.errorbar(gen_counts, classifier_acc, yerr=classifier_ci, fmt='s-', color='red', 
                 ecolor='lightcoral', elinewidth=3, capsize=5, label='Classifier-based')
    plt.errorbar(gen_counts, perfect_classifier_acc, yerr=perfect_classifier_ci, fmt='^--', color='green',
                 ecolor='lightgreen', elinewidth=3, capsize=5, label='Perfect Classifier')
    
    plt.xlabel('Number of Generations (k)')
    plt.ylabel('Accuracy')
    plt.xticks(gen_counts)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Format title
    title_text = f'Accuracy Comparison ({train_dataset_type.upper()}'
    if train_model:
        title_text += f', {train_model}'
    title_text += f' → {test_dataset_type.upper()}'
    if test_model and test_model != train_model:
        title_text += f', {test_model}'
    title_text += ')'
    
    # Add information about averaging
    title_text += f'\nAveraged over up to {choose_max} random subsets per problem with 95% CI'
    
    if balanced:
        title_text += ' [Balanced Data]'
    
    plt.title(title_text, fontsize=10)
    plt.legend()
    
    # Create the consistent output path
    dataset_path = f"{train_dataset_type}_{train_model}/"
    output_path = os.path.join('plots', 'classification', dataset_path)
    os.makedirs(output_path, exist_ok=True)
    
    # Also create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set balance suffix
    balance_suffix = "_balanced" if balanced else ""
    
    # Save plots
    plt.savefig(os.path.join(output_path, f"{filename}{balance_suffix}.pdf"), dpi=300, bbox_inches='tight')
    
    # Check if the output_dir already contains the dataset path
    if os.path.basename(output_dir) == dataset_path.rstrip('/') or output_dir.endswith(dataset_path):
        # If output_dir already includes the dataset path, don't add it again
        plt.savefig(os.path.join(output_dir, f"{filename}{balance_suffix}.pdf"), dpi=300, bbox_inches='tight')
    else:
        # Otherwise, include the dataset path in the output directory
        custom_output_path = os.path.join(output_dir, dataset_path)
        os.makedirs(custom_output_path, exist_ok=True)
        plt.savefig(os.path.join(custom_output_path, f"{filename}{balance_suffix}.pdf"), dpi=300, bbox_inches='tight')
    
    plt.close()

def plot_per_problem_voting_comparison(model, test_data_path, features, output_dir, filename='per_problem_voting_comparison', train_dataset_type='aime', test_dataset_type=None, balanced=False, train_model=None, test_model=None, max_generations=19, choose_max=20, test_years=None):
    """
    Plot a comparison between majority voting and classifier-based approaches for prediction accuracy
    across different numbers of generations for each problem individually.
    
    Args:
        model: Trained model dict with classifier and scaler
        test_data_path: Path to the raw test data parquet file
        features: List of feature columns used for classification
        output_dir: Directory to save the plot
        filename: Base filename for the plot (without extension)
        train_dataset_type: Type of training dataset
        test_dataset_type: Type of testing dataset (if None, will use train_dataset_type)
        balanced: Whether balanced dataset was used
        train_model: Model name used for training
        test_model: Model name used for testing (if None, will use train_model)
        max_generations: Maximum number of generations per problem to consider
        choose_max: Maximum number of subsets to average over per problem (if there are more possible subsets, use all available)
        test_years: List of years to filter the test data by (if None, will use all years)
    """
    if test_dataset_type is None:
        test_dataset_type = train_dataset_type
        
    if test_model is None:
        test_model = train_model
    
    # Load and normalize the raw test data
    print(f"Loading raw test data from {test_data_path} (type: {test_dataset_type})")
    test_df = load_data(test_data_path)
    test_df = normalize_column_names(test_df, test_dataset_type)
    
    # Filter data by test_years if provided
    if test_years:
        print(f"Filtering data to include only years: {test_years}")
        test_df = test_df[test_df['year'].isin(test_years)]
        if test_df.empty:
            print(f"Warning: No data found for years {test_years} in the test dataset")
            return
    
    # Prepare data for plotting
    # We'll compare accuracy at different generation counts: 1, 3, 5, 7, 9, 11, 13, 15, 17, 19
    gen_counts = list(range(1, min(max_generations + 1, 20), 2))
    
    # Group by problem_id to process each problem separately
    problem_groups = test_df.groupby('problem_number')
    total_problems = len(problem_groups)
    
    print(f"Processing {total_problems} unique problems")
    
    # Create directories to save problem-specific plots
    dataset_path = f"{train_dataset_type}_{train_model}/"
    # Create per-problem directory
    per_problem_dir = os.path.join('plots', 'classification', dataset_path, 'per_problem')
    os.makedirs(per_problem_dir, exist_ok=True)
    
    # Also create a custom per-problem directory in the output_dir
    if os.path.basename(output_dir) == dataset_path.rstrip('/') or output_dir.endswith(dataset_path):
        custom_per_problem_dir = os.path.join(output_dir, 'per_problem')
    else:
        custom_per_problem_dir = os.path.join(output_dir, dataset_path, 'per_problem')
    os.makedirs(custom_per_problem_dir, exist_ok=True)
    
    # Set balance suffix
    balance_suffix = "_balanced" if balanced else ""
    
    # Process each problem separately
    for problem_id, problem_data in problem_groups:
        print(f"Processing problem {problem_id}...")
        
        # Initialize arrays to store accuracy results and confidence intervals for this problem
        majority_voting_acc = []
        classifier_acc = []
        majority_voting_ci = []
        classifier_ci = []
        
        # Get the answer for this problem
        correct_answer = problem_data['answer'].iloc[0]
        
        # For each generation count, compute accuracy using both approaches
        for k in gen_counts:
            # Check if we have enough generations for the current k
            if len(problem_data) < k:
                print(f"  Not enough generations for k={k} (need {k}, have {len(problem_data)})")
                # Append zeros for missing data points to maintain consistent graph lengths
                majority_voting_acc.append(0)
                classifier_acc.append(0)
                majority_voting_ci.append(0)
                classifier_ci.append(0)
                continue
            
            # Lists to store accuracy results for each sample
            majority_samples = []
            classifier_samples = []
            
            # Get all available generations for this problem
            available_generations = problem_data['generation'].unique()
            
            # Calculate total possible combinations
            total_combinations = math.comb(len(available_generations), k)
            
            # Determine how many samples to use for this problem
            num_samples = min(choose_max, total_combinations)
            
            # If there are too many combinations, randomly sample them
            if total_combinations > choose_max:
                # Generate random samples without replacement
                sampled_combinations = []
                for _ in range(num_samples):
                    sample = tuple(sorted(random.sample(list(available_generations), k)))
                    # Ensure we don't have duplicate combinations
                    while sample in sampled_combinations:
                        sample = tuple(sorted(random.sample(list(available_generations), k)))
                    sampled_combinations.append(sample)
            else:
                # Use all possible combinations
                sampled_combinations = list(combinations(available_generations, k))
            
            # Process each sampled combination of generations
            for combo in sampled_combinations:
                # Filter to get only the selected generations for this combination
                selected_generations = problem_data[problem_data['generation'].isin(combo)]
                
                if len(selected_generations) == 0:
                    continue
                
                # Approach 1: Majority Voting
                # Count the occurrences of each extracted answer
                if 'extracted_answer' in selected_generations.columns:
                    answer_counts = selected_generations['extracted_answer'].value_counts()
                    if not answer_counts.empty:
                        majority_answer = answer_counts.index[0]
                        majority_samples.append(1 if majority_answer == correct_answer else 0)
                
                # Approach 2: Classifier-based
                # Prepare features for classification
                X_problem = selected_generations[features]
                
                # Get corresponding labels and answers
                y_problem = selected_generations['is_correct']
                answers_problem = selected_generations['extracted_answer']
                
                if len(X_problem) == 0:
                    continue
                    
                # Scale the features
                X_problem_scaled = model['scaler'].transform(X_problem)
                
                # Predict correctness for each generation
                try:
                    y_pred = model['classifier'].predict(X_problem_scaled)
                    
                    # Filter out the generations that are classified as incorrect
                    correct_generations = selected_generations[y_pred == 1]
                    
                    # Take majority vote among the correct generations
                    if not correct_generations.empty:
                        answer_counts = correct_generations['extracted_answer'].value_counts()
                        if not answer_counts.empty:
                            majority_answer = answer_counts.index[0]
                            classifier_samples.append(1 if majority_answer == correct_answer else 0)
                        else:
                            classifier_samples.append(0)
                    else:
                        classifier_samples.append(0)
                except Exception as e:
                    print(f"Error during prediction for problem {problem_id}: {e}")
            
            # Calculate average accuracy and confidence intervals
            if majority_samples:
                majority_mean = np.mean(majority_samples)
                majority_std = np.std(majority_samples, ddof=1)
                majority_ci_value = 1.96 * majority_std / np.sqrt(len(majority_samples))
                
                majority_voting_acc.append(majority_mean)
                majority_voting_ci.append(majority_ci_value)
            else:
                majority_voting_acc.append(0)
                majority_voting_ci.append(0)
                
            if classifier_samples:
                classifier_mean = np.mean(classifier_samples)
                classifier_std = np.std(classifier_samples, ddof=1)
                classifier_ci_value = 1.96 * classifier_std / np.sqrt(len(classifier_samples))
                
                classifier_acc.append(classifier_mean)
                classifier_ci.append(classifier_ci_value)
            else:
                classifier_acc.append(0)
                classifier_ci.append(0)
            
            print(f"  k={k}: Majority Voting Accuracy = {majority_voting_acc[-1]:.4f} ± {majority_voting_ci[-1]:.4f}, " 
                  f"Classifier Accuracy = {classifier_acc[-1]:.4f} ± {classifier_ci[-1]:.4f}, "
                  f"Samples = {len(majority_samples)}")
        
        # Create the plot for this problem
        plt.figure(figsize=(10, 6))
        
        # Plot accuracy with error bands
        plt.errorbar(gen_counts, majority_voting_acc, yerr=majority_voting_ci, fmt='o-', color='blue', 
                     ecolor='lightblue', elinewidth=3, capsize=5, label='Majority Voting')
        plt.errorbar(gen_counts, classifier_acc, yerr=classifier_ci, fmt='s-', color='red', 
                     ecolor='lightcoral', elinewidth=3, capsize=5, label='Classifier-based')
        
        plt.xlabel('Number of Generations (k)')
        plt.ylabel('Accuracy')
        plt.xticks(gen_counts)
        plt.ylim([-0.05, 1.05])  # Set y-axis limits to be consistent across all problem plots
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Format title
        title_text = f'Problem {problem_id} Accuracy Comparison ({train_dataset_type.upper()}'
        if train_model:
            title_text += f', {train_model}'
        title_text += f' → {test_dataset_type.upper()}'
        if test_model and test_model != train_model:
            title_text += f', {test_model}'
        title_text += ')'
        
        # Add information about averaging
        title_text += f'\nAveraged over up to {choose_max} random subsets with 95% CI'
        
        if balanced:
            title_text += ' [Balanced Data]'
        
        plt.title(title_text, fontsize=10)
        plt.legend()
        
        # Save the plot for this problem
        problem_filename = f"{filename}_problem_{problem_id}{balance_suffix}"
        plt.savefig(os.path.join(per_problem_dir, f"{problem_filename}.pdf"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(per_problem_dir, f"{problem_filename}.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(custom_per_problem_dir, f"{problem_filename}.pdf"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(custom_per_problem_dir, f"{problem_filename}.png"), dpi=300, bbox_inches='tight')
        
        plt.close()
    
    print(f"Created individual plots for {total_problems} problems in {per_problem_dir} and {custom_per_problem_dir}")

def calculate_correct_ratio(model_name, parquet_path, dataset_type="aime", years=None):
    """
    Calculate the ratio of true 'is_correct' samples for a given model and dataset.
    
    Args:
        model_name: The name of the model to filter by
        parquet_path: Path to the parquet file with generation data
        dataset_type: Type of dataset ('aime', 'hmmt', or 'webinstruct')
        years: List of years to filter by (optional)
        
    Returns:
        float: The ratio of correct answers (true 'is_correct' values)
        int: Total number of samples for the model
        int: Number of correct samples for the model
    """
    # Load data from parquet file
    df = load_data(parquet_path)
    
    # Normalize column names for consistency
    df = normalize_column_names(df, dataset_type)
    
    # Filter data by years if specified
    if years is not None:
        # Convert years to strings if they're not already
        if isinstance(years, list):
            years = [str(year) for year in years]
        else:
            years = [str(years)]
        
        if 'year' in df.columns:
            df = df[df['year'].isin(years)]
            print(f"Filtered data to include only years: {years}")
            if df.empty:
                print(f"Warning: No data found for years {years} in the dataset")
                return 0.0, 0, 0
    
    # Filter data for the specified model
    if 'model_name' in df.columns:
        model_df = df[df['model_name'] == model_name]
    else:
        # Some datasets might use 'model_name' instead of 'model'
        model_df = df[df['model_name'] == model_name]
    
    # Get the total number of samples for this model
    total_samples = len(model_df)
    
    if total_samples == 0:
        print(f"No samples found for model '{model_name}' in the dataset")
        return 0.0, 0, 0
    
    # Calculate the number of correct samples
    correct_samples = model_df['is_correct'].sum()
    
    # Calculate the ratio
    correct_ratio = correct_samples / total_samples
    
    print(f"Model: {model_name}")
    if years:
        print(f"Years: {years}")
    print(f"Total samples: {total_samples}")
    print(f"Correct samples: {correct_samples}")
    print(f"Correct ratio: {correct_ratio:.4f} ({correct_ratio*100:.2f}%)\n")
    
    return correct_ratio, total_samples, correct_samples





def main():
    parser = argparse.ArgumentParser(description='Classify generations using ML models')
    parser.add_argument('--train_data_path', type=str, default='aime_deepseek-ai_DeepSeek-R1-Distill-Qwen-7B_generations.parquet',
                        help='Path to the parquet file with training generation data')
    parser.add_argument('--test_data_path', type=str, default='data/aime_deepseek-ai_DeepSeek-R1-Distill-Qwen-7B_generations.parquet',
                        help='Path to the parquet file with testing generation data (if different from training data)')
    parser.add_argument('--train_dataset_type', type=str, default='aime', choices=['aime', 'hmmt', 'webinstruct'],
                        help='Type of training dataset')
    parser.add_argument('--test_dataset_type', type=str, default='hmmt', choices=['aime', 'hmmt', 'webinstruct'],
                        help='Type of testing dataset (if None, will use train_dataset_type)')
    parser.add_argument('--train_model', type=str, default='deepseek-ai_DeepSeek-R1-Distill-Qwen-7B',
                        help='Model name for training data')
    parser.add_argument('--test_model', type=str, default=None,
                        help='Model name for testing data (if None, will use train_model)')
    parser.add_argument('--output_dir', type=str, default='plots/classification',
                        help='Directory to save results')
    parser.add_argument('--features', type=str, nargs='+', 
                        default=['response_length', 'avg_neg_logprob'], choices=['response_length', 'avg_neg_logprob', 'var_neg_logprob', 'confusion_metric', 'character_count', 'contains_chinese_character', 'ngram_logprob_diff', 'ngram_pos_perplexity', 'ngram_neg_perplexity'],
                        help='Features to use for classification')
    parser.add_argument('--train_years', type=str, nargs='+', 
                        default=[str(year) for year in range(1983, 2025)],
                        help='Years to use for training')
    parser.add_argument('--test_years', type=str, nargs='+', default=['2025'],
                        help='Years to use for testing (if not provided, will use train_test_split)')
    parser.add_argument('--model', type=str, default='knn', choices=['knn', 'svm', 'mlp'],
                        help='Model to use for classification (knn, svm, or mlp)')
    parser.add_argument('--n_neighbors', type=int, default=3,
                        help='Number of neighbors for KNN')
    parser.add_argument('--svm_kernel', type=str, default='rbf', choices=['linear', 'poly', 'rbf', 'sigmoid'],
                        help='Kernel type for SVM (linear, poly, rbf, sigmoid)')
    parser.add_argument('--svm_C', type=float, default=1.0,
                        help='Regularization parameter for SVM')
    parser.add_argument('--svm_gamma', type=str, default='scale', 
                        help='Kernel coefficient for SVM (scale, auto, or float value)')
    parser.add_argument('--mlp_hidden_layer_sizes', type=str, default='100',
                        help='Hidden layer sizes for MLP (comma-separated values, e.g., "100,50,25")')
    parser.add_argument('--mlp_activation', type=str, default='relu', choices=['identity', 'logistic', 'tanh', 'relu'],
                        help='Activation function for MLP')
    parser.add_argument('--mlp_solver', type=str, default='adam', choices=['lbfgs', 'sgd', 'adam'],
                        help='Solver for weight optimization in MLP')
    parser.add_argument('--mlp_alpha', type=float, default=0.001,
                        help='L2 regularization parameter for MLP')
    parser.add_argument('--mlp_max_iter', type=int, default=200,
                        help='Maximum number of iterations for MLP')
    parser.add_argument('--mlp_early_stopping', action='store_true',
                        help='Stop training early if validation loss does not improve')
    parser.add_argument('--mlp_learning_rate', type=str, default='adaptive', choices=['constant', 'invscaling', 'adaptive'],
                        help='Learning rate schedule for MLP')
    parser.add_argument('--tune_hyperparams', action='store_true',
                        help='Tune hyperparameters using GridSearchCV')
    parser.add_argument('--balance_data', action='store_true',
                        help='Balance the dataset to have equal True/False samples in is_correct')
    parser.add_argument('--choose_max', type=int, default=20,
                        help='Maximum number of generation subsets to sample for averaging (default: 20)')
    parser.add_argument('--per_problem_plots', action='store_true',
                        help='Generate separate plots for each problem comparing voting methods')
    parser.add_argument('--classifier_method', type=str, default='top_probability', choices=['majority', 'top_probability', 'majority_density'],
                        help='Method for classifier-based approach (majority or top_probability or majority_density)')
    parser.add_argument('--exclude_gen_ids', type=int, nargs='*', default=[0, 1, 2, 3, 4],
                        help='Generation IDs to exclude from the analysis')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract model names from data paths if not explicitly provided
    if args.train_model is None and '_' in args.train_data_path:
        # Try to extract from the filename
        filename = os.path.basename(args.train_data_path).replace('.parquet', '')
        parts = filename.split('_')
        if len(parts) >= 3:  # Format: dataset_modelorg_modelname_generations
            # Extract by removing dataset prefix and _generations suffix
            dataset_prefix = f"{args.train_dataset_type}_"
            generations_suffix = "_generations"
            model_part = filename
            if model_part.startswith(dataset_prefix):
                model_part = model_part[len(dataset_prefix):]
            if model_part.endswith(generations_suffix):
                model_part = model_part[:-len(generations_suffix)]
            args.train_model = model_part
        else:
            args.train_model = "deepseek-ai_DeepSeek-R1-Distill-Qwen-7B"  # Default
    
    if args.test_model is None:
        if '_' in args.test_data_path:
            # Try to extract from the filename
            filename = os.path.basename(args.test_data_path).replace('.parquet', '')
            parts = filename.split('_')
            if len(parts) >= 3:  # Format: dataset_modelorg_modelname_generations
                # Extract by removing dataset prefix and _generations suffix
                dataset_prefix = f"{args.test_dataset_type}_"
                generations_suffix = "_generations"
                model_part = filename
                if model_part.startswith(dataset_prefix):
                    model_part = model_part[len(dataset_prefix):]
                if model_part.endswith(generations_suffix):
                    model_part = model_part[:-len(generations_suffix)]
                args.test_model = model_part
            else:
                args.test_model = args.train_model
        else:
            args.test_model = args.train_model
    
    print(f"Training model: {args.train_model}")
    print(f"Testing model: {args.test_model}")
    
    # Prepare data
    test_years = args.test_years if args.test_years[0] != 'None' else None
    X_train, X_test, y_train, y_test = prepare_data(
        train_data_path=args.train_data_path,
        features=args.features, 
        train_years=args.train_years, 
        test_data_path=args.test_data_path,
        test_years=test_years,
        train_dataset_type=args.train_dataset_type,
        test_dataset_type=args.test_dataset_type,
        balance_data=args.balance_data,
        exclude_gen_ids=args.exclude_gen_ids,
    )
    
    # Choose the model to train
    if args.model == 'knn':
        # Define parameter grid for KNN hyperparameter tuning
        param_grid = None
        if args.tune_hyperparams:
            param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }
        
        # Train KNN classifier
        print("Training KNN classifier...")
        model = train_knn_classifier(
            X_train, y_train, 
            n_neighbors=args.n_neighbors,
            param_grid=param_grid
        )
        
        model_name = 'knn'
        
    elif args.model == 'svm':
        # Define parameter grid for SVM hyperparameter tuning
        param_grid = None
        if args.tune_hyperparams:
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'poly', 'sigmoid']
            }
        
        # Process gamma parameter (convert to float if needed)
        if args.svm_gamma not in ['scale', 'auto']:
            try:
                gamma_value = float(args.svm_gamma)
                args.svm_gamma = gamma_value
            except ValueError:
                print(f"Warning: Invalid gamma value '{args.svm_gamma}', using 'scale' instead.")
                args.svm_gamma = 'scale'
        
        # Train SVM classifier
        print(f"\n{'='*50}")
        print(f"Training SVM classifier with {args.svm_kernel} kernel...")
        print(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"Features used: {args.features}")
        if args.tune_hyperparams:
            print(f"Hyperparameter tuning enabled with 5-fold cross-validation")
        else:
            print(f"Using fixed parameters: C={args.svm_C}, gamma={args.svm_gamma}, kernel={args.svm_kernel}")
        print(f"{'='*50}\n")
            
        model = train_svm_classifier(
            X_train, y_train,
            kernel=args.svm_kernel,
            C=args.svm_C,
            gamma=args.svm_gamma,
            param_grid=param_grid
        )
        
        model_name = f'svm_{args.svm_kernel}'
    
    elif args.model == 'mlp':
        # Parse hidden layer sizes from string to tuple
        hidden_layer_sizes = tuple(int(x) for x in args.mlp_hidden_layer_sizes.split(','))
        
        # Define parameter grid for MLP hyperparameter tuning
        param_grid = None
        if args.tune_hyperparams:
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,), (150,), (100, 50), (100, 100, 50)],
                'activation': ['logistic', 'tanh', 'relu'],
                'solver': ['lbfgs', 'sgd', 'adam'],
                'alpha': [0.0001, 0.001, 0.01, 0.1],
                'learning_rate': ['constant', 'invscaling', 'adaptive'],
                'max_iter': [100, 200, 300]
            }
        
        # Train MLP classifier
        print(f"\n{'='*50}")
        print(f"Training MLP classifier...")
        print(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"Features used: {args.features}")
        if args.tune_hyperparams:
            print(f"Hyperparameter tuning enabled with 5-fold cross-validation")
        else:
            print(f"Using fixed parameters: hidden_layer_sizes={hidden_layer_sizes}, activation={args.mlp_activation}, solver={args.mlp_solver}")
        print(f"{'='*50}\n")
        
        model = train_mlp_classifier(
            X_train, y_train, 
            hidden_layer_sizes=hidden_layer_sizes,
            activation=args.mlp_activation,
            solver=args.mlp_solver,
            alpha=args.mlp_alpha,
            max_iter=args.mlp_max_iter,
            param_grid=param_grid,
            early_stopping=args.mlp_early_stopping,
            learning_rate=args.mlp_learning_rate
        )
        
        model_name = f'mlp_{args.mlp_activation}'
    
    # Evaluate classifier
    metrics = evaluate_classifier(model, X_test, y_test)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        metrics['confusion_matrix'], 
        args.output_dir,
        filename=f'{model_name}_confusion_matrix',
        train_dataset_type=args.train_dataset_type,
        test_dataset_type=args.test_dataset_type,
        balanced=args.balance_data,
        train_model=args.train_model,
        test_model=args.test_model
    )
    
    # Plot ROC curve
    plot_roc_curve(
        model,
        X_test, y_test,
        args.output_dir,
        filename=f'{model_name}_roc_curve',
        train_dataset_type=args.train_dataset_type,
        test_dataset_type=args.test_dataset_type,
        balanced=args.balance_data,
        train_model=args.train_model,
        test_model=args.test_model
    )
    
    if len(args.features) == 2:
        # Plot decision boundary
        plot_decision_boundary(
            model, 
            X_train, y_train, 
            X_test, y_test, 
            args.features, 
            args.output_dir,
            filename=f'{model_name}_decision_boundary',
            test_years=test_years,
            train_dataset_type=args.train_dataset_type,
            test_dataset_type=args.test_dataset_type,
            balanced=args.balance_data,
            train_model=args.train_model,
            test_model=args.test_model
        )
    
    # Compare majority voting and classifier-based approaches across different numbers of generations
    plot_voting_comparison(
        model,
        args.test_data_path,
        args.features,
        args.output_dir,
        filename=f'{model_name}_voting_comparison',
        train_dataset_type=args.train_dataset_type,
        test_dataset_type=args.test_dataset_type,
        balanced=args.balance_data,
        train_model=args.train_model,
        test_model=args.test_model,
        max_generations=20,
        choose_max=args.choose_max,
        test_years=test_years,
        classifier_method=args.classifier_method
    )
    
    # Generate per-problem plots if requested
    if args.per_problem_plots:
        print("\nGenerating per-problem voting comparison plots...")
        plot_per_problem_voting_comparison(
            model,
            args.test_data_path,
            args.features,
            args.output_dir,
            filename=f'{model_name}_per_problem_voting_comparison',
            train_dataset_type=args.train_dataset_type,
            test_dataset_type=args.test_dataset_type,
            balanced=args.balance_data,
            train_model=args.train_model,
            test_model=args.test_model,
            max_generations=20,
            choose_max=args.choose_max,
            test_years=test_years
        )
    
    print(f"Plots saved to {args.output_dir}")
    
    

def print_correct_ratios():
    model_name = "deepseek-ai_DeepSeek-R1-Distill-Qwen-7B"
    parquet_path = "data/aime_deepseek-ai_DeepSeek-R1-Distill-Qwen-7B_generations.parquet"
    dataset_type = "aime"
    years = [2025]
    correct_ratio, total_samples, correct_samples = calculate_correct_ratio(model_name, parquet_path, dataset_type, years)
    
    # Same for 14B
    model_name = "deepseek-ai_DeepSeek-R1-Distill-Qwen-14B"
    parquet_path = "data/aime_deepseek-ai_DeepSeek-R1-Distill-Qwen-14B_generations.parquet"
    dataset_type = "aime"
    years = [2025]
    correct_ratio, total_samples, correct_samples = calculate_correct_ratio(model_name, parquet_path, dataset_type, years)

    

if __name__ == "__main__":
    main()
    # print_correct_ratios()
