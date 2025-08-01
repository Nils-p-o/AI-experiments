import os
import json
import numpy as np
import pandas as pd
import torch
import warnings
import time

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import lightgbm as lgb

from scipy import stats
import statsmodels.api as sm

import argparse

from feature_selection_MTP import (
    download_numerical_financial_data
)

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_rows', 200) # So we can see the full ranked lists

def fast_correlation_matrix(tensor, feature_names):
    """
    Calculates the correlation matrix for a 2D tensor (samples, features)
    using PyTorch, which is highly efficient on a GPU.

    Args:
        tensor (torch.Tensor): The input data tensor of shape (n_samples, n_features).
        feature_names (list): List of feature names for the final DataFrame columns/index.

    Returns:
        pd.DataFrame: The correlation matrix as a pandas DataFrame.
    """
    print("Calculating correlation matrix using PyTorch...")
    # Ensure tensor is on the correct device and is float64 for precision, like pandas
    tensor = tensor.to(torch.float64) 
    n_samples = tensor.shape[0]

    # 1. Center the data (subtract the mean)
    mean = torch.mean(tensor, dim=0, keepdim=True)
    centered_tensor = tensor - mean

    # 2. Calculate the covariance matrix
    # cov(X, Y) = E[(X - E[X])(Y - E[Y])]
    # For matrices, this is (X_centered^T @ X_centered) / (n - 1)
    cov_matrix = (centered_tensor.T @ centered_tensor) / (n_samples - 1)

    # 3. Get standard deviations of each feature
    std_dev = torch.std(tensor, dim=0)
    
    # 4. Create the denominator for the correlation formula
    # The denominator for corr(i, j) is std(i) * std(j)
    denominator = torch.outer(std_dev, std_dev)
    
    # 5. Calculate the correlation matrix
    # Add a small epsilon to avoid division by zero for features with zero variance
    corr_matrix_tensor = cov_matrix / (denominator + 1e-8)

    # For perfect correlation of a feature with itself, clamp values to 1
    corr_matrix_tensor.diagonal().clamp_(-1, 1)

    # Convert back to a labeled pandas DataFrame
    corr_matrix_df = pd.DataFrame(
        corr_matrix_tensor.cpu().numpy(), 
        columns=feature_names, 
        index=feature_names
    )
    print("Calculation complete.")
    return corr_matrix_df


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
    
    print("Step 1: Ensuring base data is available...")
    download_numerical_financial_data(
        tickers=tickers,
        seq_len=seq_len,
        output_dir=output_dir,
        **kwargs
    )

    print("Step 2: Loading raw data and metadata...")
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    train_data_path = os.path.join(output_dir, "train.pt")
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"Training data file not found at {train_data_path}. Please run your main script once to generate it.")

    temp_dir = "temp_feature_analysis"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    print("Step 3: Re-generating features in a flat format (this might take a moment)...")

    print("Step 3: Loading and re-stitching training data...")
    train_data = torch.load(os.path.join(output_dir, "train.pt"))
    train_targets = torch.load(os.path.join(output_dir, "train_MTP_targets.pt"))
    
    # Let's select the 1-day ahead target (target_inputs=0 for the first target date)
    y_close_return = train_targets[0, 0, :, :, :] # Shape: (time_chunks, seq_len, tickers)

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
    ticker_names = metadata['tickers']
    
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
    return X_train_df, y_train_series, feature_names, ticker_names, train_data


def run_feature_analysis(X_train, y_train, args):
    """
    Performs and prints the results of the three main feature selection methods.
    """
    print("\n" + "="*80)
    print("                      TIER 1: FILTER METHODS")
    print("="*80)

    # --- 1. Correlation Analysis (Feature vs. Target) ---
    print("\n\n--- 1. Correlation of Features with Target (next day's close return) ---")
    print("----- skipping for now (seems pointless) ------")
    # combined_df = pd.concat([X_train, y_train], axis=1)
    # target_corr = combined_df.corr()['target_close_return'].abs().sort_values(ascending=False)
    # print("Top 30 most correlated features with the target:")
    # print(target_corr.head(30))
    # print("\nBottom 10 least correlated features with the target:")
    # print(target_corr.tail(10))

    # --- 2. Mutual Information ---
    print("\n\n--- 2. Mutual Information Scores ---") # >2 is high
    print(" skipping for speed reasons")
    # print("Calculating... (this may take a moment for many features)")
    # if args.prediction_type == "regression":
    #     mi_scores = mutual_info_regression(X_train, y_train, random_state=42)
    # elif args.prediction_type == "classification":
    #     mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
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
    if args.prediction_type == "regression":
        lgb_model = lgb.LGBMRegressor(random_state=42, n_estimators=100, n_jobs=-1, importance_type="gain")
    elif args.prediction_type == "classification":
        lgb_model = lgb.LGBMClassifier(random_state=42, n_estimators=100, n_jobs=-1, importance_type="gain")
    lgb_model.fit(X_train, y_train)

    importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': lgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 50 features by LightGBM Importance:")
    print(importances.head(20))
    print("\nBottom 30 features by LightGBM Importance:")
    print(importances.tail(10))

    print("\n" + "="*80)
    print("                      REDUNDANCY ANALYSIS")
    print("="*80)
    
    # --- 4. Feature-vs-Feature Correlation ---
    print("\n\n--- 4. Highly Correlated Feature Pairs (Redundancy) ---")
    # corr_matrix = X_train.corr().abs()
    corr_matrix = fast_correlation_matrix(torch.tensor(X_train.values), X_train.columns)
    
    # Create a boolean mask for the upper triangle
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find index pairs with correlation > 0.95
    highly_correlated_pairs = upper_triangle.stack().reset_index()
    highly_correlated_pairs.columns = ['Feature 1', 'Feature 2', 'Correlation']
    strong_pairs = highly_correlated_pairs[highly_correlated_pairs['Correlation'] > 0.97].sort_values(by='Correlation', ascending=False)
    
    if not strong_pairs.empty:
        print("Found highly redundant feature pairs (correlation > 0.97):")
        print(strong_pairs)
    else:
        print("No highly redundant feature pairs (correlation > 0.97) found.")
    
    # # Plotting the heatmap
    # num_features_for_heatmap = 30
    # top_lgbm_features = importances.head(num_features_for_heatmap)['feature'].tolist()
    
    # print(f"\nGenerating correlation heatmap for the top {num_features_for_heatmap} most important features (from LightGBM)...")
    
    # plt.figure(figsize=(18, 15))
    # heatmap_corr = X_train[top_lgbm_features].corr()
    # sns.heatmap(heatmap_corr, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 8})
    # plt.title(f'Correlation Matrix of Top {num_features_for_heatmap} Features', fontsize=16)
    # plt.xticks(rotation=45, ha='right')
    # plt.yticks(rotation=0)
    # plt.tight_layout()
    # heatmap_filename = "feature_correlation_heatmap.png"
    # plt.savefig(heatmap_filename)
    # print(f"Heatmap saved to '{heatmap_filename}'")

    return importances, corr_matrix

def analyze_feature_distributions_programmatic(
    data_tensor: torch.Tensor, 
    feature_names: list, 
    ticker_names: list,
    top_n: int = 10
):
    """
    Programmatically analyzes feature distributions, identifies the most skewed
    and heavy-tailed features, and generates targeted visualizations.
    """
    if data_tensor.shape[0] != len(feature_names):
        raise ValueError("Feature name count must match tensor's first dimension.")
    if data_tensor.shape[-1] != len(ticker_names):
        raise ValueError("Ticker name count must match tensor's last dimension.")

    results = []

    print("Analyzing feature distributions... This may take a moment.")
    for i, f_name in enumerate(feature_names):
        if any([s in f_name for s in ["is_", "cos_", "sin_"]]):
            continue
        feature_slice = data_tensor[i, ...].cpu()
        global_flat = feature_slice.flatten().numpy()
        global_skew = stats.skew(global_flat)
        global_kurt = stats.kurtosis(global_flat)

        per_ticker_skew = [stats.skew(feature_slice[..., j].flatten().numpy()) for j in range(len(ticker_names))]
        per_ticker_kurt = [stats.kurtosis(feature_slice[..., j].flatten().numpy()) for j in range(len(ticker_names))]

        max_abs_skew_idx = np.argmax(np.abs(per_ticker_skew))
        max_kurt_idx = np.argmax(per_ticker_kurt)

        results.append({
            "feature_name": f_name, "feature_index": i, "global_skew": global_skew,
            "global_kurtosis": global_kurt, "max_abs_skew": per_ticker_skew[max_abs_skew_idx],
            "worst_skew_ticker": ticker_names[max_abs_skew_idx], "max_kurtosis": per_ticker_kurt[max_kurt_idx],
            "worst_kurt_ticker": ticker_names[max_kurt_idx],
        })
    
    df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("                TIER 4: DISTRIBUTION ANALYSIS (SKEW & KURTOSIS)")
    print("="*80)

    print(f"\n--- Top {top_n} Most Skewed Features (Globally) ---")
    df_sorted_skew = df.reindex(df.global_skew.abs().sort_values(ascending=False).index)
    print(df_sorted_skew[['feature_name', 'global_skew', 'max_abs_skew', 'worst_skew_ticker']].head(top_n).to_string())

    print(f"\n--- Top {top_n} Most Heavy-Tailed (Leptokurtic) Features (Globally) ---")
    df_sorted_kurt = df.reindex(df.global_kurtosis.sort_values(ascending=False).index)
    print(df_sorted_kurt[['feature_name', 'global_kurtosis', 'max_kurtosis', 'worst_kurt_ticker']].head(top_n).to_string())

    problematic_indices = set(df_sorted_skew.head(top_n)['feature_index'])
    problematic_indices.update(df_sorted_kurt.head(top_n)['feature_index'])

    print(f"\nGenerating detailed plots for the {len(problematic_indices)} most problematic features...")
    
    for i in problematic_indices:
        f_name = feature_names[i]
        feature_data = data_tensor[i, ...].flatten().cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Distribution Analysis for '{f_name}'", fontsize=16)
        
        sns.histplot(feature_data, kde=True, ax=axes[0], bins=100)
        axes[0].set_title("Histogram and Density Plot (Global)")
        axes[0].set_xlabel("Normalized Value")
        
        sm.qqplot(feature_data, line='s', ax=axes[1])
        axes[1].set_title("Q-Q Plot vs. Normal Distribution (Global)")
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

def propose_final_feature_set(feature_importances: pd.DataFrame, correlation_matrix: pd.DataFrame, column_to_idx: dict, correlation_threshold: float = 0.85):
    """
    Proposes a final feature set by balancing feature importance with redundancy.

    This function implements a greedy approach:
    1. It starts with the most important feature (from LightGBM).
    2. It iterates through the rest of the features in order of importance.
    3. It adds a feature to the final set only if it is NOT highly correlated
       (above the given threshold) with any feature already selected.

    Args:
        feature_importances (pd.DataFrame): DataFrame with 'feature' and 'importance' columns, sorted by importance.
        correlation_matrix (pd.DataFrame): A square DataFrame with feature-vs-feature correlation values.
        correlation_threshold (float): The maximum allowed correlation between any two selected features.

    Returns:
        list: A list of proposed feature names to keep for the final model.
    """
    print("\n" + "="*80)
    print("        TIER 4: AUTOMATED FEATURE SET PROPOSAL (IMPORTANCE - REDUNDANCY)")
    print("="*80)
    print(f"\nUsing LightGBM importance and a correlation threshold of {correlation_threshold} to select features.")
    
    ranked_features = feature_importances['feature'].tolist()
    
    # The first feature is always the best, so we keep it
    selected_features = [ranked_features[0]]
    
    # Keep track of why features are discarded for transparency
    discarded_info = []

    # Iterate from the second-best feature onwards
    for candidate_feature in ranked_features[1:]:
        is_redundant = False
        redundant_with_feature = None
        
        importance = feature_importances[feature_importances['feature'] == candidate_feature]['importance'].iloc[0]
        if importance < 0.01:
            discarded_info.append((candidate_feature, candidate_feature, importance))
            continue

        # Check correlation against all features we've already selected
        for selected_feature in selected_features:
            correlation = correlation_matrix.loc[candidate_feature, selected_feature]
            if correlation > correlation_threshold:
                is_redundant = True
                redundant_with_feature = selected_feature
                break # Found a reason to discard, no need to check further
        
        if not is_redundant:
            selected_features.append(candidate_feature)
        else:
            importance = feature_importances[feature_importances['feature'] == candidate_feature]['importance'].iloc[0]
            discarded_info.append((candidate_feature, redundant_with_feature, importance))

    
    selected_indices = [column_to_idx[feature] for feature in selected_features]

    # selecting top fraction of features
    fraction_to_keep = 0.2

    importances = feature_importances['importance'].to_numpy()
    print(f"total importance of selected features {len(selected_features)}: {np.sum(importances)}")

    selected_features = selected_features[:int(len(selected_features) * fraction_to_keep)]
    selected_indices = selected_indices[:int(len(selected_indices) * fraction_to_keep)]

    print(f"total importance of top {fraction_to_keep} features: {np.sum(importances[:int(len(importances) * fraction_to_keep)])}")


    print(f"\n---> Proposed {len(selected_features)} features to keep:")
    # print(", ".join(selected_features))
    
    print("\n--- Top 20 Important Features Discarded Due to Redundancy/0 Importance ---")
    # Sort discarded features by their original importance
    discarded_info.sort(key=lambda x: x[2], reverse=True)
    for feature, reason, importance in discarded_info[:20]:
        corr_val = correlation_matrix.loc[feature, reason]
        print(f"  - Discarded '{feature}' (Importance: {importance:.4f}) due to high correlation ({corr_val:.2f}) with '{reason}'")
        
    return selected_features, selected_indices

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Use the same parameters as your main script to ensure consistency
    tickers = ["AAPL","^GSPC", "MSFT", "INTC", "NVDA", "AMZN"]
    output_dir = "time_series_data_analysis"
    seq_len = 3
    start_date = "2000-09-01"
    end_date = "2025-01-01"

    args = argparse.Namespace(
        tickers=tickers,
        output_dir=output_dir,
        seq_len=seq_len,
        prediction_type = "regression",
        classification_threshold = 0.01
    )

    # 1. Prepare the data
    try:
        X_train, y_train, feature_names, ticker_names, train_data_tensor = prepare_data_for_selection(
            tickers=tickers,
            output_dir=output_dir,
            seq_len=seq_len,
            start_date=start_date,
            end_date=end_date,
            check_if_already_downloaded=False, # Set to False if you want to force a redownload
            config_args=args
        )
        
        # 2. Run the analysis
        importances, correlations = run_feature_analysis(X_train, y_train, args)

        column_to_idx = {name: i for i, name in enumerate(feature_names)}

        final_features, final_indices = propose_final_feature_set(importances, correlations, column_to_idx, correlation_threshold=0.97)
        print(f"\n---> Final Feature Set: {final_features}")

        # print(f"\n---> final indices: {final_indices}")

        # 3. Run distribution analysis on the original tensor
        analyze_feature_distributions_programmatic(
            data_tensor=train_data_tensor,
            feature_names=feature_names,
            ticker_names=ticker_names,
            top_n=2 # Number of worst offenders to plot
        )

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Please run your main `test_feats_stocks_time_series_2_MTP.py` script at least once to generate the necessary data files before running this analysis.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")