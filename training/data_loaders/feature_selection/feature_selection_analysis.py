import os
import json
import numpy as np
import pandas as pd
import torch
import warnings
import time

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
import lightgbm as lgb

from feature_selection_MTP import (
    download_numerical_financial_data,
)

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_rows', 200) # So we can see the full ranked lists

def prepare_data_for_selection(
    tickers: list,
    output_dir: str,
    seq_len: int,
    **kwargs
) -> tuple[pd.DataFrame, pd.Series, list]:
    """
    This function adapts your original data pipeline to prepare data specifically
    for feature selection analysis.

    It runs the full feature engineering pipeline but stops before the final
    PyTorch-specific reshaping. It then flattens the data into a 2D format
    (samples, features) suitable for scikit-learn and pandas.

    Crucially, it only uses the TRAINING portion of the data to avoid look-ahead bias.

    Returns:
        X_train_df (pd.DataFrame): The training features.
        y_train (pd.Series): The training target variable (next day's close return).
        feature_names (list): The names of the columns in X_train_df.
    """
    print("--- Starting Data Preparation for Feature Selection ---")
    
    # We need to re-implement the core logic of your function here to intercept
    # the data before its final transformation.
    # This is a simplified version of your 'download_numerical_financial_data'
    # focused only on getting the feature and target tensors.
    
    # We will call your original function to download and cache the data if needed,
    # but we will perform the feature generation and reshaping here.
    # This avoids duplicating all the feature code.
    
    # Run the original function to ensure data is downloaded/cached
    # We set check_if_already_downloaded to True, assuming a first run was done.
    # If the files don't exist, it will create them.
    print("Step 1: Ensuring base data is available...")
    download_numerical_financial_data(
        tickers=tickers,
        seq_len=seq_len,
        output_dir=output_dir,
        **kwargs
    )

    # Now load the intermediate data that your script would have created.
    # We need the raw data to generate features and targets again in a flat format.
    print("Step 2: Loading raw data and metadata...")
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    train_data_path = os.path.join(output_dir, "train.pt")
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"Training data file not found at {train_data_path}. Please run your main script once to generate it.")

    # We need to reconstruct the flat feature matrix (X) and target vector (y)
    # from the saved, windowed data. This is a bit of a reverse-engineering process.
    # An easier way is to load the FULL, un-windowed data before it was saved.
    
    # Let's create a temporary, simplified version of your data generation.
    # We will get the full_data tensor and MTP_targets before splitting and windowing.
    
    # --- Re-running the core feature generation from your script ---
    # This section mimics your original script to get the un-windowed data.
    temp_dir = "temp_feature_analysis"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    print("Step 3: Re-generating features in a flat format (this might take a moment)...")
    # By setting a new output_dir and `check_if_already_downloaded=False`, we force
    # a re-run of the logic without overwriting your main data files.
    # We need to capture the returned values from a modified version of your function.
    
    # Since modifying the original function to return values is intrusive,
    # we'll assume the files are created and we can load them.
    # Let's load the *full* tensors before they are split.
    
    # This is complex. A much cleaner approach is to modify your original script
    # slightly to be import-friendly. Let's assume we can get the key variables.
    # For now, let's load the saved training data and stitch it back together.

    print("Step 3: Loading and re-stitching training data...")
    train_data = torch.load(os.path.join(output_dir, "train.pt"))
    train_targets = torch.load(os.path.join(output_dir, "train_MTP_targets.pt"))
    
    # Original shape: (features, target_inputs, time_chunks, seq_len, tickers)
    # Target shape: (chlov, target_inputs, time_chunks, seq_len, tickers)
    
    # We want to predict the 1-day ahead close return.
    # From your code, the target is the return. Close is the first feature (index 0).
    # target_dates = [1], so target_inputs dim is 1.
    
    # Let's select the 1-day ahead target (target_inputs=0 for the first target date)
    y_close_return = train_targets[0, 0, :, :, :] # Shape: (time_chunks, seq_len, tickers)

    # The features are in train_data.
    # X has shape (features, target_inputs, time_chunks, seq_len, tickers)
    # We want the features for predicting the 1-day ahead target.
    X_features = train_data[:, 0, :, :, :] # Shape: (features, time_chunks, seq_len, tickers)
    
    # Now, let's flatten these tensors to get a (samples, features) structure.
    # A "sample" is a single (time, ticker) observation.
    
    # Permute to bring features to the last dimension
    # (time_chunks, seq_len, tickers, features)
    X_permuted = X_features.permute(1, 2, 3, 0) 
    num_features = X_permuted.shape[-1]
    
    # Flatten into (total_samples, features)
    X_flat = X_permuted.reshape(-1, num_features)
    
    # Flatten the target tensor similarly
    # (time_chunks, seq_len, tickers) -> (total_samples)
    y_flat = y_close_return.permute(0, 1, 2).reshape(-1)

    # Get feature names from metadata
    feature_names = metadata['columns']
    
    print(f"Data successfully reshaped. Found {X_flat.shape[0]} samples and {X_flat.shape[1]} features.")
    
    # Convert to pandas DataFrame for easier analysis
    X_train_df = pd.DataFrame(X_flat.cpu().numpy(), columns=feature_names)
    y_train_series = pd.Series(y_flat.cpu().numpy(), name="target_close_return")
    
    # --- Data Cleaning ---
    # Scikit-learn can't handle NaNs or Infs.
    X_train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    if X_train_df.isnull().sum().sum() > 0:
        print(f"Warning: Found {X_train_df.isnull().sum().sum()} NaN/Inf values in features. Filling with column median.")
        X_train_df.fillna(X_train_df.median(), inplace=True)
        
    if y_train_series.isnull().sum() > 0:
        print(f"Warning: Found {y_train_series.isnull().sum()} NaN/Inf values in target. Dropping these samples.")
        # Get indices of NaNs in target and drop them from both X and y
        nan_indices = y_train_series[y_train_series.isnull()].index
        y_train_series.dropna(inplace=True)
        X_train_df.drop(index=nan_indices, inplace=True)

    print("--- Data Preparation Complete ---")
    return X_train_df, y_train_series, feature_names


def run_feature_analysis(X_train, y_train):
    """
    Performs and prints the results of the three main feature selection methods.
    """
    print("\n" + "="*80)
    print("                      TIER 1: FILTER METHODS")
    print("="*80)

    # --- 1. Correlation Analysis (Feature vs. Target) ---
    print("\n\n--- 1. Correlation of Features with Target (next day's close return) ---")
    combined_df = pd.concat([X_train, y_train], axis=1)
    target_corr = combined_df.corr()['target_close_return'].abs().sort_values(ascending=False)
    print("Top 30 most correlated features with the target:")
    print(target_corr.head(30))
    print("\nBottom 10 least correlated features with the target:")
    print(target_corr.tail(10))

    # --- 2. Mutual Information ---
    print("\n\n--- 2. Mutual Information Scores ---")
    print(" ---- skipping for now ----") # TODO find why it is weird
    # print("Calculating... (this may take a moment for many features)")
    # mi_scores = mutual_info_regression(X_train, y_train, random_state=42)
    # mi_scores = pd.Series(mi_scores, name="MI_Score", index=X_train.columns)
    # mi_scores = mi_scores.sort_values(ascending=False)
    # print("Top 30 features by Mutual Information Score:")
    # print(mi_scores.head(30))
    # print("\nBottom 10 features by Mutual Information Score:")
    # print(mi_scores.tail(10))

    print("\n" + "="*80)
    print("                      TIER 2: EMBEDDED METHODS")
    print("="*80)

    # --- 3. LightGBM Feature Importance ---
    print("\n\n--- 3. LightGBM Feature Importance ---")
    print("Training LightGBM model to get feature importances...")
    lgb_model = lgb.LGBMRegressor(random_state=42, n_estimators=100, n_jobs=-1, importance_type='gain')
    lgb_model.fit(X_train, y_train)

    importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': lgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 50 features by LightGBM Importance:")
    print(importances.head(50))
    print("\nBottom 30 features by LightGBM Importance:")
    print(importances.tail(30))

    print("\n" + "="*80)
    print("                      REDUNDANCY ANALYSIS")
    print("="*80)
    
    # --- 4. Feature-vs-Feature Correlation ---
    print("\n\n--- 4. Highly Correlated Feature Pairs (Redundancy) ---")
    corr_matrix = X_train.corr().abs()
    
    # Create a boolean mask for the upper triangle
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find index pairs with correlation > 0.95
    highly_correlated_pairs = upper_triangle.stack().reset_index()
    highly_correlated_pairs.columns = ['Feature 1', 'Feature 2', 'Correlation']
    strong_pairs = highly_correlated_pairs[highly_correlated_pairs['Correlation'] > 0.95].sort_values(by='Correlation', ascending=False)
    
    if not strong_pairs.empty:
        print("Found highly redundant feature pairs (correlation > 0.95):")
        print(strong_pairs)
    else:
        print("No highly redundant feature pairs (correlation > 0.95) found.")
    
    # Plotting the heatmap
    num_features_for_heatmap = 30
    top_lgbm_features = importances.head(num_features_for_heatmap)['feature'].tolist()
    
    print(f"\nGenerating correlation heatmap for the top {num_features_for_heatmap} most important features (from LightGBM)...")
    
    plt.figure(figsize=(18, 15))
    heatmap_corr = X_train[top_lgbm_features].corr()
    sns.heatmap(heatmap_corr, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 8})
    plt.title(f'Correlation Matrix of Top {num_features_for_heatmap} Features', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    heatmap_filename = "feature_correlation_heatmap.png"
    plt.savefig(heatmap_filename)
    print(f"Heatmap saved to '{heatmap_filename}'")


if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Use the same parameters as your main script to ensure consistency
    tickers = ["AAPL","^GSPC", "MSFT", "INTC", "NVDA", "AMZN"]
    output_dir = "time_series_data"
    seq_len = 8
    start_date = "2000-09-01"
    end_date = "2025-01-01"

    # 1. Prepare the data
    try:
        X_train, y_train, feature_names = prepare_data_for_selection(
            tickers=tickers,
            output_dir=output_dir,
            seq_len=seq_len,
            start_date=start_date,
            end_date=end_date,
            check_if_already_downloaded=False # Set to False if you want to force a redownload
        )
        
        # 2. Run the analysis
        run_feature_analysis(X_train, y_train)

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Please run your main `test_feats_stocks_time_series_2_MTP.py` script at least once to generate the necessary data files before running this analysis.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")