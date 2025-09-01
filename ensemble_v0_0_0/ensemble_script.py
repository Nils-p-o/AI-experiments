# mainly for testing/backtesting ensemble models and ideas

# TODO check if combining multiple good models makes it better, or worse
# (combining logits, or probabilities)
# TODO ensemble


# TODO fix so it produces the same results as during val in training
# could be lack of data lookback
# compare produced input tensors
# maybe logit combination




import json
import os
import argparse
import torch
import torch.nn as nn # For MAE/MSE if used later
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt # For plotting later
import yaml
import math
import time

# Assuming your project structure allows these imports
from money_former_MLA_DINT_cog_attn_2_MTP import Money_former_MLA_DINT_cog_attn_MTP
from test_feats_stocks_time_series_2_MTP_new import (
    align_financial_dataframes,
    download_with_retry,
    calculate_features,
    data_fix_ffill,
    auto_correct_feature_skew_pre_znorm,
    download_numerical_financial_data
)
from bayes_opt import BayesianOptimization

# Global variables for Bayesian Optimization (will be set later)
OPTIMIZATION_PREDS_1DAY = None
OPTIMIZATION_ACTUAL_1D_RETURNS = None
OPTIMIZATION_INITIAL_CAPITAL = 100000.0
OPTIMIZATION_TRANSACTION_COST = 0.0005
OPTIMIZATION_SIGNAL_HORIZON_NAME = "1-day (Opt.)"

def load_config_and_args_from_metadata(metadata_path):
    """Loads experiment configuration from a YAML metadata file using FullLoader."""
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, "r") as f:
        try:
            # FullLoader can reconstruct Python objects tagged in the YAML
            full_config_data = yaml.unsafe_load(f)
        except yaml.YAMLError as exc:
            print(f"Error parsing YAML file: {exc}")
            raise
        except Exception as e: # Catch other potential errors during loading
            print(f"An unexpected error occurred during YAML loading: {e}")
            raise

    # Based on your YAML structure: full_config_data should be a dict
    # with a key 'args' that holds the argparse.Namespace object.
    if not isinstance(full_config_data, dict) or 'args' not in full_config_data:
        raise ValueError("YAML file does not have the expected top-level 'args' key.")

    args_namespace_obj = full_config_data['args']

    if not isinstance(args_namespace_obj, argparse.Namespace):
        raise ValueError("The 'args' key in YAML did not resolve to an argparse.Namespace object.")

    # Convert the loaded Namespace object to a dictionary for easier processing
    args_dict = vars(args_namespace_obj)

    # --- Ensure all required model arguments are present in the loaded dict ---
    required_model_args = [
        'architecture', 'd_model', 'nhead', 'num_layers', 'd_ff', 'dropout',
        'input_features', 'tickers', 'indices_to_predict', 'prediction_type',
        'bias', 'head_dim', 'kv_compression_dim', 'q_compression_dim', 'qk_rope_dim'
    ]
    for req_arg in required_model_args:
        if req_arg not in args_dict:
            raise KeyError(f"Required argument '{req_arg}' not found in loaded configuration.")

    final_args_namespace = argparse.Namespace(**args_dict)


    if hasattr(final_args_namespace, 'indices_to_predict') and \
       not isinstance(final_args_namespace.indices_to_predict, list):
        if isinstance(final_args_namespace.indices_to_predict, int):
            final_args_namespace.indices_to_predict = [final_args_namespace.indices_to_predict]
        else:
            print(f"Warning: indices_to_predict is not a list or int: {final_args_namespace.indices_to_predict}")


    # --- Load and convert normalization_means and normalization_stds ---
    if hasattr(final_args_namespace, 'normalization_means') and \
       hasattr(final_args_namespace, 'normalization_stds'):
        
        norm_means_list = final_args_namespace.normalization_means
        norm_stds_list = final_args_namespace.normalization_stds

        if not isinstance(norm_means_list, list) or not isinstance(norm_stds_list, list):
            raise TypeError("normalization_means/stds from YAML are not lists as expected.")

        final_args_namespace.normalization_means = torch.tensor(norm_means_list, dtype=torch.float32)
        final_args_namespace.normalization_stds = torch.tensor(norm_stds_list, dtype=torch.float32)

    else:
        raise ValueError("MTP model requires normalization_means and normalization_stds in metadata.")

    return final_args_namespace


def load_mtp_model(model_path, args_from_metadata):
    """Loads the trained MTP model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Ensure the architecture name matches your MTP model class
    if args_from_metadata.architecture != "Money_former_MLA_DINT_cog_attn_MTP":
        print(f"Warning: Metadata architecture is {args_from_metadata.architecture}, but loading Money_former_MLA_DINT_cog_attn_MTP.")
    
    model = Money_former_MLA_DINT_cog_attn_MTP(args_from_metadata)
    
    # Loading state_dict. PyTorch Lightning saves the whole experiment, model is under 'state_dict'
    # or sometimes directly if saved via torch.save(experiment.model, ...)
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
    
    if type(model) == type(checkpoint): # TODO check if i got the correct checkpoint
        # If the checkpoint is a model object, just return it
        print(f"Loaded model directly from {model_path}.")
        return checkpoint
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_state_dict[k[len("model."):]] = v
            else:
                if k.startswith("loss_fn."):
                    continue
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    elif isinstance(checkpoint, dict) and not any(key.startswith("model.") for key in checkpoint.keys()):
        # Likely a raw model state_dict saved with torch.save(model.state_dict(), ...)
        model.load_state_dict(checkpoint)
    elif hasattr(checkpoint, 'state_dict'): # Saved the whole LightningModule or model object
         # Try to get state_dict if it's a model object
        if hasattr(checkpoint, 'model') and hasattr(checkpoint.model, 'state_dict'): # PL experiment object
            model.load_state_dict(checkpoint.model.state_dict())
        else: # Direct model object
            model.load_state_dict(checkpoint.state_dict())
    else:
        raise ValueError("Could not determine how to load state_dict from the checkpoint.")

    model.eval()
    print(f"Model {model_path} loaded successfully.")
    return model

def get_auto_correct_skew_processors(args, start_date, end_date): # specifically reffering to the original run training data
    tickers = args.tickers
    target_dates = args.indices_to_predict
    raw_data = download_with_retry(
        args.tickers,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False,
        back_adjust=False,
    )

    if raw_data.empty:
        print("No data downloaded.")
        return None, None, None

    aligned_raw_data = pd.DataFrame(columns=raw_data.columns)
    for i in range(len(tickers)):
        for j in range(len(raw_data.columns.levels[0])):
            aligned_raw_data[raw_data.columns.levels[0][j], tickers[i]] = (
                align_financial_dataframes(
                    {tickers[i]: raw_data[raw_data.columns.levels[0][j]]},
                    target_column=tickers[i],
                    fill_method="ffill",
                    min_date=start_date,
                    max_date=end_date,
                )
            )
    raw_data = aligned_raw_data
    
    indexes = raw_data.index
    df_columns = list(raw_data.columns.levels[0])
    raw_data_tensor = torch.tensor(raw_data.values, dtype=torch.float32).reshape(
        -1, len(df_columns), len(tickers)
    )
    raw_data_tensor = raw_data_tensor.transpose(0, 1)

    raw_data = raw_data_tensor[1:, :, :] # Remove 'Adj Close'
    full_data, columns, local_columns, time_columns = calculate_features(raw_data, tickers, indexes)
    full_data[:len(columns)+len(local_columns)] = data_fix_ffill(full_data[:len(columns)+len(local_columns)])

    data = torch.empty(full_data.shape[0], max(target_dates), full_data.shape[1] - max(target_dates), full_data.shape[2], dtype=torch.float32)
    for i in range(max(target_dates)):
        data[:, i, :, :] = full_data[:, i:-(max(target_dates) - i), :]
    
    # Replicate the lookback removal from the data loader
    min_lookback_to_drop = 20
    data = data[:, :, min_lookback_to_drop:, :]

    columns.extend(local_columns)
    columns.extend(time_columns)

    train_data, fitted_processors = auto_correct_feature_skew_pre_znorm(data, data.shape[2], len(time_columns), args.seq_len)

    return train_data, fitted_processors

def apply_auto_correct_feature_skew_pre_znorm(data, fitted_processors):
    original_dtype = data.dtype

    for idx in fitted_processors.keys():
        scaler = fitted_processors[idx]["scaler"]
        transformer = fitted_processors[idx]["transformer"]

        original_shape = data[idx].shape
        data_flat = data[idx].numpy().flatten().reshape(-1, 1).astype(np.float64)
        data_scaled = scaler.transform(data_flat)

        transformed_data = transformer.transform(data_scaled)
        transformed_data = transformed_data.reshape(original_shape)

        data[idx] = torch.from_numpy(transformed_data).to(dtype=original_dtype)
    return data

def auto_correct_feature_kurtosis_post_znorm(data: torch.Tensor, train_data: torch.Tensor) -> torch.Tensor:
    for idx in range(data.shape[0]):
        lower = torch.quantile(train_data[idx], 0.0025)
        upper = torch.quantile(train_data[idx], 0.9975)
        data[idx] = torch.clip(data[idx], lower, upper)

    return data

def download_and_process_inference_data(args, start_date, end_date, cli_args):
    """
    Downloads and processes data for inference, replicating training feature engineering.
    Returns:
        full_input_features_mtp (Tensor): Shape (features, max_pred_horizon, time_steps, num_tickers)
                                           This is the MTP-style input, globally normalized.
        true_chlov_returns_for_backtest (Tensor): Shape (5, time_steps_for_backtest, num_tickers)
                                                  Raw CHLOV returns for calculating actual P&L.
        all_columns (list): List of all feature names.
    """
    print("Downloading and processing inference data...")
    tickers = args.tickers
    target_dates = args.indices_to_predict

    # --- Data Download and Alignment (same as before) ---
    raw_data = download_with_retry(
        tickers,
        start=start_date,
        end=end_date,
        progress=True,
        auto_adjust=False,
        back_adjust=False,
    )
    if raw_data.empty:
        print("No data downloaded.")
        return None, None, None

    aligned_raw_data = pd.DataFrame(columns=raw_data.columns)
    for i in range(len(tickers)):
        for j in range(len(raw_data.columns.levels[0])):
            aligned_raw_data[raw_data.columns.levels[0][j], tickers[i]] = (
                align_financial_dataframes(
                    {tickers[i]: raw_data[raw_data.columns.levels[0][j]]},
                    target_column=tickers[i],
                    fill_method="ffill",
                    min_date=start_date,
                    max_date=end_date,
                )
            )
    raw_data = aligned_raw_data
    
    indexes = raw_data.index
    df_columns = list(raw_data.columns.levels[0])
    raw_data_tensor = torch.tensor(raw_data.values, dtype=torch.float32).reshape(
        -1, len(df_columns), len(tickers)
    )
    raw_data_tensor = raw_data_tensor.transpose(0, 1)

    # --- Start of Feature Engineering (Replicated from Dataloader) ---
    raw_data = raw_data_tensor[1:, :, :] # Remove 'Adj Close'
    full_data, columns, local_columns, time_columns = calculate_features(raw_data, tickers, indexes)
    full_data[:len(columns)+len(local_columns)] = data_fix_ffill(full_data[:len(columns)+len(local_columns)])

    # --- MTP Input Stacking and Time Features ---
    data = torch.empty(full_data.shape[0], max(target_dates), full_data.shape[1] - max(target_dates), full_data.shape[2], dtype=torch.float32)
    for i in range(max(target_dates)):
        data[:, i, :, :] = full_data[:, i:-(max(target_dates) - i), :]
    
    # Replicate the lookback removal from the data loader
    min_lookback_to_drop = 20
    data = data[:, :, min_lookback_to_drop:, :]

    columns.extend(local_columns)
    columns.extend(time_columns)
    
    # --- True Returns for Backtesting ---
    MTP_full_returns = (raw_data[:, 1:, :] - raw_data[:, :-1, :]) / (raw_data[:, :-1, :])
    MTP_full_returns = data_fix_ffill(MTP_full_returns)
    
    # Align returns with feature data
    MTP_full_returns = MTP_full_returns[:, min_lookback_to_drop:, :]
    if max(target_dates) > 1:
        if max(target_dates) > cli_args.pred_day:
            MTP_full_returns = MTP_full_returns[:, :-(max(target_dates) - cli_args.pred_day), :]

    # --- Global Normalization (using loaded stats) ---
    num_of_non_global_norm_feats = len(local_columns) + len(time_columns)

    train_data, fitted_processors = get_auto_correct_skew_processors(args, start_date="2000-09-01", end_date="2019-10-30") # NOTE: update this if need be

    data = apply_auto_correct_feature_skew_pre_znorm(data, fitted_processors)

    local_features_start_idx = data.shape[0] - num_of_non_global_norm_feats
    
    if (type(args.normalization_means) != torch.Tensor) or (type(args.normalization_stds) != torch.Tensor):
        args.normalization_means = torch.tensor(args.normalization_means).clone().detach().to(data.device)
        args.normalization_stds = torch.tensor(args.normalization_stds).clone().detach().to(data.device)

    means = args.normalization_means.view(data.shape[0] - num_of_non_global_norm_feats, 1, 1, -1).to(data.device)
    stds = args.normalization_stds.view(data.shape[0] - num_of_non_global_norm_feats, 1, 1, -1).to(data.device)
    data[:local_features_start_idx] = (data[:local_features_start_idx] - means) / (stds + 1e-8)

    # local norm for non-global features
    data = data.unfold(2, args.seq_len, 1).contiguous() # shape (features, max_pred_horizon, time_steps, num_tickers, seq_len_needed)
    data = data.permute(0, 1, 2, 4, 3) # (features, max_pred_horizon, time_steps, seq_len_needed, num_tickers)

    if local_features_start_idx != data.shape[0] - len(time_columns):
        local_means = data[local_features_start_idx:-len(time_columns)].mean(dim=3, keepdim=True)
        local_stds = data[local_features_start_idx:-len(time_columns)].std(dim=3, keepdim=True)
        data[local_features_start_idx:-len(time_columns)] = (data[local_features_start_idx:-len(time_columns)] - local_means) / (local_stds + 1e-8)

    data = auto_correct_feature_kurtosis_post_znorm(data, train_data)

    return data, MTP_full_returns

def get_mtp_predictions_for_backtest(model, all_mtp_input_features, args, nr_of_days_to_check, device, cli_args):
    """
    Generates predictions for the backtest period using the MTP model.
    This function now performs local z-normalization on each sequence before prediction.
    Outputs 1-day ahead predictions for the 'Close' feature return.
    """
    model = model.eval()
    if (type(args.normalization_means) != torch.Tensor) or (type(args.normalization_stds) != torch.Tensor):
        args.normalization_means = torch.tensor(args.normalization_means).clone().detach().to(device)
        args.normalization_stds = torch.tensor(args.normalization_stds).clone().detach().to(device)

    seq_len = args.seq_len
    num_pred_horizons_input = all_mtp_input_features.shape[1]

    # Permute all_mtp_input_features for easier slicing
    # Input shape: (features, horizon, time, seq_len, tickers)
    # Permuted shape: (time, horizon, seq_len, tickers, features)
    all_mtp_input_features_permuted = all_mtp_input_features.permute(2, 1, 3, 4, 0)

    model_predictions_1day_list = []
    if cli_args.average_predictions:
        all_mtp_input_features_permuted = torch.cat((all_mtp_input_features_permuted, torch.zeros_like(all_mtp_input_features_permuted[-len(args.indices_to_predict):, :, :]).to(device)), dim=0)
        nr_of_batches = math.ceil((nr_of_days_to_check + len(args.indices_to_predict) - 1) / cli_args.batch_size)
        for batch_i in range(nr_of_batches):
            model_input_batch = all_mtp_input_features_permuted[
                batch_i * cli_args.batch_size : (batch_i + 1) * cli_args.batch_size, :, :, :, :
            ]  # Shape: (batch_size, horizon, seq_len, tickers, features)

            model_input_batch = model_input_batch.to(device)

            tickers_input = torch.arange(len(args.tickers), device=device).unsqueeze(0).unsqueeze(0).repeat(
                model_input_batch.shape[0], num_pred_horizons_input, 1
            )

            with torch.no_grad():
                raw_outputs = model(model_input_batch, tickers_input)

                match args.prediction_type:
                    case "regression":
                        outputs_viewed = raw_outputs.view(
                            model_input_batch.shape[0], max(args.indices_to_predict), (seq_len+1), len(args.tickers), 5
                        )
                    case "classification":
                        if args.use_global_seperator:
                            raw_outputs = raw_outputs[:, :, 1:, :]
                            outputs_viewed = raw_outputs.view(
                                model_input_batch.shape[0], max(args.indices_to_predict), seq_len, len(args.tickers), 5 * args.num_classes
                            )
                        else:
                            outputs_viewed = raw_outputs.view(
                                model_input_batch.shape[0], max(args.indices_to_predict), (seq_len+1), len(args.tickers), 5 * args.num_classes
                            )

            outputs_permuted = outputs_viewed.permute(0, 2, 4, 1, 3)  # Shape: (batch, seq_len+1, 5_chlov, horizons, tickers)

            match args.prediction_type:
                case "regression":
                    outputs_permuted = outputs_permuted[:, -1, 0, :, :]  # Shape: (batch, horizons, num_tickers)
                    model_predictions_1day_list.append(outputs_permuted)
                case "classification":
                    if args.use_global_seperator:
                        outputs_permuted = outputs_permuted.view(
                            model_input_batch.shape[0], seq_len, 5, args.num_classes, max(args.indices_to_predict), len(args.tickers)
                        ).permute(0,3,1,2,4,5)
                    else:
                        outputs_permuted = outputs_permuted.view(
                            model_input_batch.shape[0], seq_len+1, 5, args.num_classes, max(args.indices_to_predict), len(args.tickers)
                        ).permute(0,3,1,2,4,5)
                    outputs_permuted = outputs_permuted[:, :, -1, 0, :, :]  # Shape: (batch, num_classes, horizons, num_tickers)
                    model_predictions_1day_list.append(outputs_permuted.permute(0,2,3,1)) # Permute to (batch, horizons, num_tickers, num_classes)

    else:
        nr_of_batches = math.ceil(nr_of_days_to_check / cli_args.batch_size)
        for batch_i in range(nr_of_batches):
            # Shape: (batch_size, horizon, seq_len, tickers, features)
            model_input_batch = all_mtp_input_features_permuted[
                batch_i * cli_args.batch_size : (batch_i + 1) * cli_args.batch_size, :, :, :, :
            ].to(device)
            
            
            # Prepare separator and tickers for the model
            tickers_input = torch.arange(len(args.tickers), device=device).unsqueeze(0).unsqueeze(0).repeat(
                model_input_batch.shape[0], num_pred_horizons_input, 1
            )

            with torch.no_grad():
                # --- MODEL FORWARD PASS & OUTPUT HANDLING (EXACTLY AS IN TRAINING) ---
                raw_outputs = model(model_input_batch, tickers_input)
                # 2. Reshape the output exactly as in _shared_step
                # Shape becomes: (batch, horizons, seq_len+1, tickers, 5_chlov)
                match args.prediction_type:
                    case "regression":
                        outputs_viewed = raw_outputs.view(model_input_batch.shape[0], max(args.indices_to_predict), (seq_len+1), len(args.tickers), 5)
                    case "classification":
                        outputs_viewed = raw_outputs.view(model_input_batch.shape[0], max(args.indices_to_predict), (seq_len+1), len(args.tickers), 5*args.num_classes)
                
            # 3. Permute the output exactly as in _shared_step
            # Shape becomes: (batch, seq_len+1, 5_chlov, horizons, tickers)
            outputs_permuted = outputs_viewed.permute(0, 2, 4, 1, 3)

            # --- EXTRACT THE DESIRED PREDICTION FROM THE PERMUTED TENSOR
            match args.prediction_type:
                case "regression":
                    # extract only close (index 0)
                    outputs_permuted = outputs_permuted[:, -1, 0, :, :] # Shape: (batch, horizons, num_tickers)
                    model_predictions_1day_list.append(outputs_permuted)
                case "classification":
                    outputs_permuted = outputs_permuted.view(
                        model_input_batch.shape[0], seq_len+1, 5, args.num_classes, max(args.indices_to_predict), len(args.tickers)
                    ).permute(0,3,1,2,4,5)
                    outputs_permuted = outputs_permuted[:, :, -1, 0, :, :]  # Shape: (batch, num_classes, horizons, num_tickers)
                    model_predictions_1day_list.append(outputs_permuted.permute(0,2,3,1))


    # model_predictions_1day_tensor = torch.stack(model_predictions_1day_list, dim=0).to(device)
    model_predictions_1day_tensor = torch.cat(model_predictions_1day_list, dim=0).to(device)
    if cli_args.average_predictions:
        if args.prediction_type == "regression":
            close_feature_original_index = 0
            norm_means_for_close = args.normalization_means[close_feature_original_index, :].to(device)
            norm_stds_for_close = args.normalization_stds[close_feature_original_index, :].to(device)

            aligned_predictions = torch.empty((nr_of_days_to_check, max(args.indices_to_predict), len(args.tickers)), dtype=torch.float64, device=device)
            for i in range(max(args.indices_to_predict)):
                start_day = max(args.indices_to_predict) - i - 1
                aligned_predictions[:, i, :] = model_predictions_1day_tensor[start_day:nr_of_days_to_check+start_day, i, :]
            aligned_predictions = aligned_predictions.mean(dim=1) # Average across the horizons
            denormalized_predictions = aligned_predictions * norm_stds_for_close.unsqueeze(0) + norm_means_for_close.unsqueeze(0)
        elif args.prediction_type == "classification":
            aligned_predictions = torch.empty((nr_of_days_to_check, max(args.indices_to_predict), len(args.tickers), args.num_classes), dtype=torch.float64, device=device)
            for i in range(max(args.indices_to_predict)):
                start_day = max(args.indices_to_predict) - i - 1
                aligned_predictions[:, i, :, :] = model_predictions_1day_tensor[start_day:nr_of_days_to_check+start_day, i, :, :]
            aligned_predictions = aligned_predictions.mean(dim=1) # Average across the horizons
            denormalized_predictions = torch.softmax(aligned_predictions, dim=-1)
            # denormalized_predictions = aligned_predictions

    else:
        if args.prediction_type == "regression":
            # --- De-normalize predictions using loaded stats ---
            close_feature_original_index = 0 # 'close_returns' is the first feature
            norm_means_for_close = args.normalization_means[close_feature_original_index, :].to(device)
            norm_stds_for_close = args.normalization_stds[close_feature_original_index, :].to(device)

            denormalized_predictions = model_predictions_1day_tensor * norm_stds_for_close.unsqueeze(0) + norm_means_for_close.unsqueeze(0)
        elif args.prediction_type == "classification":
            denormalized_predictions = torch.softmax(model_predictions_1day_tensor, dim=-1) # Convert logits to probabilities
            # denormalized_predictions = model_predictions_1day_tensor 
    
    return denormalized_predictions


def plot_predicted_vs_actual_returns(
    predicted_returns: torch.Tensor, # Shape: (num_days, num_tickers) - de-normalized
    actual_returns: torch.Tensor,    # Shape: (num_days, num_tickers)
    ticker_names: list,
    num_days_to_plot: int = 100, # Plot the last N days for clarity
    plot_filename_prefix: str = "returns_comparison"
):
    """
    Plots predicted 1-day returns vs. actual 1-day returns for each ticker.
    """
    predicted_returns_np = predicted_returns.cpu().numpy()
    actual_returns_np = actual_returns.cpu().numpy()

    num_total_days, num_tickers = predicted_returns_np.shape

    if num_total_days == 0 or num_tickers == 0:
        print("No data to plot for returns comparison.")
        return

    days_to_plot = min(num_total_days, num_days_to_plot)
    
    # Use the last 'days_to_plot'
    start_idx = num_total_days - days_to_plot
    
    time_axis = np.arange(days_to_plot) # Simple integer time axis for plotting

    for i in range(num_tickers):
        ticker_name = ticker_names[i] if i < len(ticker_names) else f"Ticker_{i+1}"
        
        plt.figure(figsize=(15, 6))
        plt.plot(time_axis, actual_returns_np[start_idx:, i], label=f"Actual 1-Day Returns", color='blue', alpha=0.7)
        plt.plot(time_axis, predicted_returns_np[start_idx:, i], label=f"Predicted 1-Day Returns", color='red', linestyle='--', alpha=0.7)
        
        plt.title(f"Predicted vs. Actual 1-Day Returns for {ticker_name} (Last {days_to_plot} Days)")
        plt.xlabel(f"Trading Day (Last {days_to_plot})")
        plt.ylabel("1-Day Return")
        plt.legend()
        plt.grid(True)
        plt.axhline(0, color='black', linestyle='-', linewidth=0.5) # Zero line for reference
        
        # Improve y-axis formatting if returns are small percentages
        current_values = plt.gca().get_yticks()
        plt.gca().set_yticklabels(['{:,.3%}'.format(x) for x in current_values])

        plot_filename = f"{plot_filename_prefix}_{ticker_name.replace('^', '')}_last{days_to_plot}days.png"
        plt.savefig(plot_filename)
        print(f"Saved plot: {plot_filename}")
        plt.show() # Display plot
        plt.close() # Close the figure to free memory

def calculate_metrics(actuals, predictions, args):

    confusion_metrics_dict = {}

    if args.prediction_type == "regression":
        mae_loss = torch.mean(torch.abs(predictions - actuals))

        # TODO add confusion matrix

        total_accuracy = torch.mean((predictions.sign() == actuals.sign()).float())

        confusion_metrics_dict["MAE_Loss"] = mae_loss.item()
        confusion_metrics_dict["Total_Accuracy"] = total_accuracy.item()

    elif args.prediction_type == "classification":
        loss_fn = nn.NLLLoss()

        pred_classes = predictions.argmax(dim=-1)
        actual_classes = torch.ones_like(actuals, dtype=torch.long)
        actual_classes[actuals < -args.classification_threshold] = 0
        actual_classes[actuals > args.classification_threshold] = 2

        loss = loss_fn(torch.log(predictions + 1e-9).permute(0, 2, 1), actual_classes)

        confusion_matrix = torch.zeros((args.num_classes, args.num_classes), dtype=torch.int64)
        for i in range(args.num_classes):
            for j in range(args.num_classes):
                confusion_matrix[i, j] = torch.sum((pred_classes == i) & (actual_classes == j)).item()
        
        precisions = []
        recalls = []
        f1s = []
        for i in range(args.num_classes):
            true_positives = confusion_matrix[i, i]
            false_positives = torch.sum(confusion_matrix[:, i]).item() - true_positives
            false_negatives = torch.sum(confusion_matrix[i, :]).item() - true_positives
            
            precisions.append(true_positives / (true_positives + false_positives)) if (true_positives + false_positives) > 0 else precisions.append(0.0)
            recalls.append(true_positives / (true_positives + false_negatives)) if (true_positives + false_negatives) > 0 else recalls.append(0.0)
            f1s.append(2 * precisions[-1] * recalls[-1] / (precisions[-1] + recalls[-1])) if (precisions[-1] + recalls[-1]) > 0 else f1s.append(0.0)

        total_accuracy = torch.mean((pred_classes == actual_classes).float())

        confidence_score_up = (predictions[:, :, 2] * (pred_classes == 2)).sum() / (pred_classes == 2).sum().item() if (pred_classes == 2).sum().item() > 0 else 0.0
        confidence_score_down = (predictions[:, :, 0] * (pred_classes == 0)).sum() / (pred_classes == 0).sum().item() if (pred_classes == 0).sum().item() > 0 else 0.0
        confidence_score_flat = (predictions[:, :, 1] * (pred_classes == 1)).sum() / (pred_classes == 1).sum().item() if (pred_classes == 1).sum().item() > 0 else 0.0

        confusion_metrics_dict["CE_Loss"] = loss.item()
        confusion_metrics_dict["Total_Accuracy"] = total_accuracy.item()

        confusion_metrics_dict["Precision_Down"] = precisions[0]
        confusion_metrics_dict["Precision_Flat"] = precisions[1]
        confusion_metrics_dict["Precision_Up"] = precisions[2]
        confusion_metrics_dict["Total_Precision"] = sum(precisions) / len(precisions)

        confusion_metrics_dict["Recall_Down"] = recalls[0]
        confusion_metrics_dict["Recall_Flat"] = recalls[1]
        confusion_metrics_dict["Recall_Up"] = recalls[2]
        confusion_metrics_dict["Total_Recall"] = sum(recalls) / len(recalls)

        confusion_metrics_dict["F1_Down"] = f1s[0]
        confusion_metrics_dict["F1_Flat"] = f1s[1]
        confusion_metrics_dict["F1_Up"] = f1s[2]
        confusion_metrics_dict["Total_F1"] = sum(f1s) / len(f1s)

        confusion_metrics_dict["Confidence_Score_Down"] = confidence_score_down
        confusion_metrics_dict["Confidence_Score_Flat"] = confidence_score_flat
        confusion_metrics_dict["Confidence_Score_Up"] = confidence_score_up
    
    return confusion_metrics_dict

def ground_truth_strategy_trade(
    predictions: torch.Tensor,
    actual_movements: torch.Tensor,
    trade_threshold_up: float = 0.01,
    trade_threshold_down: float = 0.01,
    initial_capital: float = 100000.0,
    transaction_cost_pct: float = 0.0005,
    prediction_type: str = "regression",  # 'regression' or 'classification'
    decision_type: str = "threshold",  # 'argmax' or 'threshold'
    allocation_strategy: str = "equal",  # 'equal' or 'signal_strength'
) -> tuple[torch.Tensor, torch.Tensor]:
    # predictions: [periods, tickers, classes] or [periods, tickers]
    # actual_movements: [periods, tickers]

    predictions = predictions.cpu().to(torch.float64)
    actual_movements = actual_movements.cpu().to(torch.float64)
    potential_trades = actual_movements.shape[0]
    num_tickers = actual_movements.shape[1]
    portfolio_values_ts = torch.zeros(potential_trades + 1, dtype=torch.float64)
    portfolio_values_ts[0] = initial_capital
    daily_portfolio_returns_ts = torch.zeros(potential_trades, dtype=torch.float64)
    current_capital = initial_capital
    current_positions = torch.zeros(num_tickers, dtype=torch.float64)

    for trade in range(potential_trades):
        # Determine the target positions for the current period based on predictions.
        target_positions = torch.zeros(num_tickers, dtype=torch.float64)
        daily_predictions = predictions[trade]

        if prediction_type == "classification":
            if decision_type == "argmax":
                predicted_classes = torch.argmax(daily_predictions, dim=-1)
                target_positions[predicted_classes == 2] = 1.0
                target_positions[predicted_classes == 0] = -1.0
            elif decision_type == "threshold":
                target_positions[daily_predictions[:, 2] > trade_threshold_up] = 1.0
                target_positions[daily_predictions[:, 0] > trade_threshold_down] = -1.0
        elif prediction_type == "regression":
            if decision_type == "threshold":
                target_positions[daily_predictions > trade_threshold_up] = 1.0
                target_positions[daily_predictions < -trade_threshold_down] = -1.0
            else:  # A simple sign-based decision for regression
                target_positions = torch.sign(daily_predictions)

        # Allocate capital based on the chosen strategy.
        num_trades = torch.count_nonzero(target_positions)
        if num_trades > 0:
            if allocation_strategy == "equal":
                allocations = target_positions / num_trades
            elif allocation_strategy == "signal_strength":
                abs_predictions = torch.abs(daily_predictions)
                total_signal_strength = torch.sum(
                    abs_predictions[target_positions != 0]
                )
                if total_signal_strength > 0:
                    allocations = (
                        target_positions * abs_predictions
                    ) / total_signal_strength
                else:
                    allocations = torch.zeros_like(target_positions)
        else:
            allocations = torch.zeros_like(target_positions)

        # take desired position
        desired_allocations = allocations * current_capital  # Convert to dollar amounts
        # Calculate transaction costs only on the change in positions.
        position_change = desired_allocations - current_positions
        transaction_costs = torch.sum(torch.abs(position_change)) * transaction_cost_pct
        current_capital -= transaction_costs
        actual_allocations = (
            allocations * current_capital
        )  # Actual dollar allocations after costs

        # update portfolio value after price movement, (# this is the value of the portfolio at the end of the trade period, right before the next trade is made)
        # Calculate the portfolio's return for the trade.
        change_in_positions = actual_allocations * actual_movements[trade]

        current_positions = actual_allocations + change_in_positions
        # Update the portfolio's capital.
        current_capital += torch.sum(change_in_positions)
        portfolio_values_ts[trade + 1] = current_capital

        if portfolio_values_ts[trade] > 0:
            daily_portfolio_returns_ts[trade] = (
                current_capital / portfolio_values_ts[trade]
            ) - 1
        # end of discrete trade period/beginning of next period
    
    portfolio_dict = {
        "portfolio_values": portfolio_values_ts,
        "daily_portfolio_returns": daily_portfolio_returns_ts,
    }

    # --- Calculate performance statistics ---
    final_capital = portfolio_values_ts[-1]
    total_return = (final_capital - initial_capital) / initial_capital
    num_winning_days = torch.sum(daily_portfolio_returns_ts > 0)
    num_losing_days = torch.sum(daily_portfolio_returns_ts < 0)
    win_loss_ratio = num_winning_days / num_losing_days if num_losing_days > 0 else 0.0
    num_days = potential_trades

    annualized_return = ((1+ total_return) ** (252.0 / num_days)) - 1.0 if num_days > 0 else 0.0
    annualized_volatility = torch.std(daily_portfolio_returns_ts) * torch.sqrt(torch.tensor(252.0, dtype=torch.float64)) if num_days > 0 else 0.0
    sharpe_ratio = (annualized_return / annualized_volatility) if annualized_volatility > 1e-9 else 0.0
    downside_returns = daily_portfolio_returns_ts[daily_portfolio_returns_ts < 0]
    sortino_ratio = (annualized_return / torch.std(downside_returns)) if len(downside_returns) > 0 else 0.0
    
    running_max = torch.cummax(portfolio_values_ts, dim=0).values
    drawdown = (running_max - portfolio_values_ts) / running_max
    max_drawdown = torch.max(drawdown).item()
    calmar_ratio = (annualized_return / max_drawdown) if max_drawdown > 0.04 else (annualized_return / 0.04)

    stats_dict = {
        "Threshold Up": trade_threshold_up,
        "Threshold Down": trade_threshold_down,
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Calmar Ratio": calmar_ratio,
        "Max Drawdown": max_drawdown,
        "Number of Trading Days": int(num_days),
        "Number of Winning Days": int(num_winning_days),
        "Number of Losing Days": int(num_losing_days),
        "Win/Loss Day Ratio": win_loss_ratio,
        "Final Capital": final_capital,
    }

    return portfolio_dict, stats_dict


def vectorized_per_stock_backtest(
    predictions: torch.Tensor,
    actual_movements: torch.Tensor,
    trade_threshold_up: float = 0.01,
    trade_threshold_down: float = 0.01,
    initial_capital: float = 100000.0,
    transaction_cost_pct: float = 0.0005,
    prediction_type: str = "regression",
    decision_type: str = "threshold",
) -> tuple[torch.Tensor, torch.Tensor]:
    # predictions: [periods, tickers, classes] or [periods, tickers]
    # actual_movements: [periods, tickers]

    predictions = predictions.cpu().to(torch.float64)
    actual_movements = actual_movements.cpu().to(torch.float64)
    potential_trades, num_tickers = actual_movements.shape

    # --- VECTORIZED STATE INITIALIZATION ---
    # Each column represents an independent backtest for one stock.
    # Shape: [periods+1, tickers]
    portfolio_values_ts = torch.zeros((potential_trades + 1, num_tickers), dtype=torch.float64)
    portfolio_values_ts[0] = initial_capital # PyTorch broadcasts the scalar

    # Shape: [periods, tickers]
    daily_portfolio_returns_ts = torch.zeros((potential_trades, num_tickers), dtype=torch.float64)
    
    # Shape: [tickers]
    current_capital = torch.full((num_tickers,), initial_capital, dtype=torch.float64)
    current_positions = torch.zeros(num_tickers, dtype=torch.float64)

    for trade in range(potential_trades):
        # --- VECTORIZED DECISION MAKING ---
        # This part of the logic works on the entire [tickers] dimension already.
        target_positions = torch.zeros(num_tickers, dtype=torch.float64)
        daily_predictions = predictions[trade] # Shape: [tickers, classes] or [tickers]

        # This logic is unchanged but now operates in parallel across all tickers
        if prediction_type == "classification":
            if decision_type == "argmax":
                predicted_classes = torch.argmax(daily_predictions, dim=-1)
                target_positions[predicted_classes == 2] = 1.0
                target_positions[predicted_classes == 0] = -1.0
            else: # threshold
                target_positions[daily_predictions[:, 2] > trade_threshold_up] = 1.0
                target_positions[daily_predictions[:, 0] > trade_threshold_down] = -1.0
        else: # regression
            if decision_type == "threshold":
                target_positions[daily_predictions > trade_threshold_up] = 1.0
                target_positions[daily_predictions < -trade_threshold_down] = -1.0
            else: # sign
                target_positions = torch.sign(daily_predictions)

        # For per-stock backtests, allocation is always 100% or 0.
        # The 'allocations' weights are just the target_positions themselves.
        allocations = target_positions # Shape: [tickers]

        # --- VECTORIZED ACCOUNTING ---
        # Element-wise multiplication of allocation weight by each stock's capital
        desired_allocations = allocations * current_capital  # Shape: [tickers]
        position_change = desired_allocations - current_positions # Shape: [tickers]
        
        # CRITICAL: No sum! Calculate costs for each backtest independently.
        transaction_costs = torch.abs(position_change) * transaction_cost_pct # Shape: [tickers]

        current_capital -= transaction_costs # Element-wise subtraction
        actual_allocations = allocations * current_capital # Element-wise mult

        # Calculate P&L for each stock independently
        change_in_positions = actual_allocations * actual_movements[trade] # Shape: [tickers]
        
        current_positions = actual_allocations + change_in_positions
        
        # CRITICAL: No sum! Add P&L to each stock's capital independently.
        current_capital += change_in_positions

        # Store the vector of portfolio values
        portfolio_values_ts[trade + 1] = current_capital
        
        # Calculate returns for each stock
        # Add a small epsilon to avoid division by zero if a portfolio value is 0
        prev_values = portfolio_values_ts[trade]
        daily_portfolio_returns_ts[trade] = torch.where(
            prev_values > 0,
            (current_capital / prev_values) - 1,
            0.0
        )

    portfolio_dict = {
        "portfolio_values": portfolio_values_ts,
        "daily_portfolio_returns": daily_portfolio_returns_ts,
    }

    # --- VECTORIZED PERFORMANCE STATISTICS ---
    # Each metric will be a tensor of shape [num_tickers]
    final_capital = portfolio_values_ts[-1]
    total_return = (final_capital - initial_capital) / initial_capital
    num_days = potential_trades
    annualized_return = (1 + total_return) ** (252/num_days) - 1.0
    annualized_volatility = torch.std(daily_portfolio_returns_ts, dim=0) * torch.sqrt(torch.tensor(252.0, dtype=torch.float64))

    downside_returns = daily_portfolio_returns_ts[daily_portfolio_returns_ts < 0]
    downside_volatility = torch.std(downside_returns, dim=0) * torch.sqrt(torch.tensor(252.0, dtype=torch.float64))
    sortino_ratio = torch.where(
        downside_volatility > 1e-9,
        annualized_return / downside_volatility,
        0.0
    )
    
    # Use torch.where to avoid division by zero for Sharpe Ratio
    sharpe_ratio = torch.where(
        annualized_volatility > 1e-9,
        annualized_return / annualized_volatility,
        0.0
    )
    
    # --- NEW: Vectorized Max Drawdown and Calmar Ratio ---
    running_max = torch.cummax(portfolio_values_ts, dim=0).values
    drawdown = (running_max - portfolio_values_ts) / running_max
    max_drawdown = torch.max(drawdown, dim=0).values

    # Calculate Calmar Ratio, handling cases where max_drawdown is 0
    calmar_ratio = torch.where(
        max_drawdown > 0.04,
        annualized_return / max_drawdown,
        annualized_return / 0.04
    )

    num_winning_days = torch.sum(daily_portfolio_returns_ts > 0, dim=0)
    num_losing_days = torch.sum(daily_portfolio_returns_ts < 0, dim=0)

    win_loss_ratio = torch.where(
        num_losing_days > 0,
        num_winning_days / num_losing_days,
        0.0 # Define as 0 if there are no losing days
    )

    stats_dict = {
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Calmar Ratio": calmar_ratio,
        "Max Drawdown": max_drawdown,
        "Number of Trading Days": torch.full((num_tickers,), float(num_days), dtype=torch.float32),
        "Number of Winning Days": num_winning_days,
        "Number of Losing Days": num_losing_days,
        "Win/Loss Day Ratio": win_loss_ratio,
        "Final Capital": final_capital,
    }

    return portfolio_dict, stats_dict

def new_trading_metrics(actuals, predictions, args):
    metrics = {}
    portfolio_dict, stats = ground_truth_strategy_trade(
        predictions=predictions,
        actual_movements=actuals,
        initial_capital=1000,
        transaction_cost_pct=0.0005,  # assuming high enough capital and US stocks
        prediction_type=args.prediction_type,
        decision_type="argmax",  # 'argmax' or 'threshold'
        allocation_strategy="equal",  # 'equal' or 'signal_strength'
    )

    metrics["Annualized Return"] = stats["Annualized Return"]
    metrics["Sharpe Ratio"] = stats["Sharpe Ratio"]
    metrics["Sortino Ratio"] = stats["Sortino Ratio"]
    metrics["Calmar Ratio"] = stats["Calmar Ratio"]
    metrics["Max Drawdown"] = stats["Max Drawdown"]
    metrics["WLR"] = stats["Win/Loss Day Ratio"]
    metrics["Days Traded"] = stats["Number of Winning Days"] + stats["Number of Losing Days"]

    return metrics

def new_trading_metrics_individual(
    actuals: torch.Tensor,
    predictions: torch.Tensor,
    args
):
    """
    Calculate trading metrics for each ticker individually.
    """
    metrics = {}
    portfolio_dict, stats = vectorized_per_stock_backtest(
        predictions=predictions,
        actual_movements=actuals,
        initial_capital=1000,
        transaction_cost_pct=0.0005,  # assuming high enough capital and US stocks
        prediction_type=args.prediction_type,
        decision_type="argmax",  # 'argmax' or 'threshold'
    )

    metrics["Annualized Return"] = stats["Annualized Return"]
    metrics["Sharpe Ratio"] = stats["Sharpe Ratio"]
    metrics["Sortino Ratio"] = stats["Sortino Ratio"]
    metrics["Calmar Ratio"] = stats["Calmar Ratio"]
    metrics["Max Drawdown"] = stats["Max Drawdown"]
    metrics["WLR"] = stats["Win/Loss Day Ratio"]
    metrics["Days Traded"] = stats["Number of Winning Days"] + stats["Number of Losing Days"]

    return metrics

def analyse_distributions(actuals, predictions, idx):
    import numpy as np
    prediction_classes = torch.argmax(predictions, dim=-1)

    specific_prediction_classes = prediction_classes == idx

    specific_probs = predictions[specific_prediction_classes][:, idx]
    specific_values = actuals[specific_prediction_classes]
    z = np.polyfit(specific_probs, specific_values, 1)
    p = np.poly1d(z)

    plt.scatter(specific_probs, specific_values)
    plt.plot(specific_probs, p(specific_probs), 'r-')
    plt.show()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions with a trained MTP stock model.")
    # parser.add_argument("--model_path", type=str, default="ensemble/models/AAPL/ver_9_3199.ckpt", help="Path to the trained .pth model file.")
    parser.add_argument("--metadata_path", type=str, default="ensemble/models/hparams.yaml", help="Path to the metadata.json file for the model.")
    parser.add_argument("--days_to_check", type=int, default=-1, help="Number of recent days to generate predictions for and backtest.")
    # parser.add_argument("--start_date_data", type=str, default="2020-01-01", help="Start date")
    # parser.add_argument("--end_date_data", type=str, default="2025-05-25", help="End date.")
    parser.add_argument("--start_date", type=str, default="2019-10-01", help="Start date.")
    parser.add_argument("--end_date", type=str, default="2025-08-10", help="End date.")
    parser.add_argument("--initial_capital", type=float, default=1000.0, help="Initial capital for backtesting.")
    parser.add_argument("--transaction_cost", type=float, default=0.0005, help="Transaction cost percentage.")
    parser.add_argument("--plot_equity", type=bool, default=True, help="Plot the equity curve of the optimized strategy.")
    parser.add_argument("--verbose_strategy", type=int, default=0, help="Verbosity level for strategy run printouts (0: silent, 1: summary).")
    parser.add_argument("--plot_individual_returns", type=bool, default=True, help="Plot predicted vs. actual returns for each ticker.")
    parser.add_argument("--pred_day", type=int, default=1, help="MTP block prediction to use.")
    parser.add_argument("--average_predictions", type=bool, default=True, help="Average predictions across all timesteps of MTP.") # aligned of course
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for model inference.")

    load_data_from_saved_source = True


    cli_args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Config and Model
    args_from_metadata = load_config_and_args_from_metadata(cli_args.metadata_path)
    args_from_metadata.device = device # Add device to args
    args_from_metadata.tickers = sorted(args_from_metadata.tickers)
    # Pass normalization stats to device if they are tensors
    if hasattr(args_from_metadata, 'normalization_means'):
        args_from_metadata.normalization_means = args_from_metadata.normalization_means.to(device)
    if hasattr(args_from_metadata, 'normalization_stds'):
        args_from_metadata.normalization_stds = args_from_metadata.normalization_stds.to(device)
    
    if cli_args.average_predictions:
        cli_args.pred_day = max(args_from_metadata.indices_to_predict)

    # 2. Download and Process Data
    
    # all_mtp_input_features, true_chlov_returns = download_and_process_inference_data(
    #     args_from_metadata,
    #     cli_args.start_date_data,
    #     cli_args.end_date_data,
    #     cli_args=cli_args
    # )

    if not load_data_from_saved_source:
        indexes, data, MTP_targets, means_stds, metadata = download_numerical_financial_data(
            tickers=args_from_metadata.tickers,
            seq_len=args_from_metadata.seq_len,
            target_dates=args_from_metadata.indices_to_predict,
            start_date="2000-09-01",
            end_date=cli_args.end_date,
            config_args=args_from_metadata,
            train_data_length=4997,
            val_data_length=0,
            test_data_length=0
        )
    
        _, new_val_data, _ = data
        _, new_val_MTP_targets, _ = MTP_targets

        if not os.path.exists("ensemble/data/"):
            os.makedirs("ensemble/data/")
        torch.save(new_val_data, "ensemble/data/new_val_data.pt")
        torch.save(new_val_MTP_targets, "ensemble/data/new_val_MTP_targets.pt")
    else:
        new_val_data = torch.load("ensemble/data/new_val_data.pt")
        new_val_MTP_targets = torch.load("ensemble/data/new_val_MTP_targets.pt")
    

    if cli_args.days_to_check == -1:
        cli_args.days_to_check = new_val_data.shape[2]

    all_mtp_input_features = new_val_data[:,:,-cli_args.days_to_check:].to(device)
    actual_close_returns = new_val_MTP_targets[0,cli_args.pred_day-1,:,-1,:]
    
    actual_1d_returns_for_backtest_period = actual_close_returns[-cli_args.days_to_check:, :]

    # START of MSFT testing block

    msft_model_0_path = "ensemble/models/MSFT/ver_54_4099.ckpt" # base model
    msft_model_1_path = "ensemble/models/MSFT/ver_54_4199.ckpt"
    msft_model_2_path = "ensemble/models/MSFT/ver_2_2299.ckpt"
    msft_model_3_path = "ensemble/models/MSFT/ver_9_3199.ckpt"
    msft_model_4_path = "ensemble/models/MSFT/ver_30_1199.ckpt"
    msft_model_5_path = "ensemble/models/MSFT/ver_34_3599.ckpt"

    msft_model_0 = load_mtp_model(msft_model_0_path, args_from_metadata).to(device)
    msft_predictions_0 = get_mtp_predictions_for_backtest(
        msft_model_0, all_mtp_input_features, args_from_metadata, cli_args.days_to_check, device, cli_args
    )
    msft_model_1 = load_mtp_model(msft_model_1_path, args_from_metadata).to(device)
    msft_predictions_1 = get_mtp_predictions_for_backtest(
        msft_model_1, all_mtp_input_features, args_from_metadata, cli_args.days_to_check, device, cli_args
    )
    msft_model_2 = load_mtp_model(msft_model_2_path, args_from_metadata).to(device)
    msft_predictions_2 = get_mtp_predictions_for_backtest(
        msft_model_2, all_mtp_input_features, args_from_metadata, cli_args.days_to_check, device, cli_args
    )
    msft_model_3 = load_mtp_model(msft_model_3_path, args_from_metadata).to(device)
    msft_predictions_3 = get_mtp_predictions_for_backtest(
        msft_model_3, all_mtp_input_features, args_from_metadata, cli_args.days_to_check, device, cli_args
    )
    msft_model_4 = load_mtp_model(msft_model_4_path, args_from_metadata).to(device)
    msft_predictions_4 = get_mtp_predictions_for_backtest(
        msft_model_4, all_mtp_input_features, args_from_metadata, cli_args.days_to_check, device, cli_args
    )
    msft_model_5 = load_mtp_model(msft_model_5_path, args_from_metadata).to(device)
    msft_predictions_5 = get_mtp_predictions_for_backtest(
        msft_model_5, all_mtp_input_features, args_from_metadata, cli_args.days_to_check, device, cli_args
    )

    # maybe 0 and 4 or 5?
    # msft_predictions_denorm = msft_predictions_0[:, 3:4]
    msft_predictions_denorm = torch.cat((msft_predictions_0[:, 3:4], msft_predictions_1[:, 3:4], msft_predictions_2[:, 3:4], msft_predictions_3[:, 3:4], msft_predictions_4[:, 3:4], msft_predictions_5[:, 3:4]), dim=1)
    
    # combining all good model preds at the prob level
    # prob_combined_predictions = torch.cat((msft_predictions_0[:, 3:4], msft_predictions_1[:, 3:4], msft_predictions_2[:, 3:4], msft_predictions_3[:, 3:4], msft_predictions_4[:, 3:4], msft_predictions_5[:, 3:4]), dim=1)
    # prob_combined_predictions = prob_combined_predictions.softmax(dim=-1).mean(dim=1, keepdim=True)
    # msft_predictions_denorm = torch.cat((msft_predictions_denorm, prob_combined_predictions), dim=1)

    # logit_combined_predictions = torch.cat((msft_predictions_0[:, 3:4], msft_predictions_1[:, 3:4], msft_predictions_2[:, 3:4], msft_predictions_3[:, 3:4], msft_predictions_4[:, 3:4], msft_predictions_5[:, 3:4]), dim=1)
    # logit_combined_predictions = logit_combined_predictions.mean(dim=1, keepdim=True).softmax(dim=-1)
    # msft_predictions_denorm = torch.cat((msft_predictions_denorm, logit_combined_predictions), dim=1)
    
    msft_returns_period = actual_1d_returns_for_backtest_period[:,3:4].repeat(1, msft_predictions_denorm.shape[1])

    pre_optim_predictions = msft_predictions_denorm.cpu()
    pre_optim_actuals = msft_returns_period.cpu()

    individual_ground_truth_portfolio_dict, individual_ground_stats_dict = vectorized_per_stock_backtest(
        predictions=pre_optim_predictions,
        actual_movements=pre_optim_actuals,
        initial_capital=cli_args.initial_capital,
        transaction_cost_pct=cli_args.transaction_cost,
        prediction_type=args_from_metadata.prediction_type,
        decision_type="argmax",
    )
    for i in range(pre_optim_predictions.shape[1]):
        print(f"\n--- Running PRE-OPTIMIZATION Backtest for MSFT strategy {i} ---")
        
        plt.figure(figsize=(14, 7)) # Create a new figure for this plot
        
        # Add Buy and Hold for comparison (copied from your existing plot logic)
        buy_and_hold_returns_single_ticker = pre_optim_actuals.cpu()[:,i]
        buy_and_hold_equity_curve_single = (1 + buy_and_hold_returns_single_ticker).cumprod(dim=0)

        print(f"Annualized return: {individual_ground_stats_dict['Annualized Return'][i]:.4f}")
        print(f"Sharpe ratio: {individual_ground_stats_dict['Sharpe Ratio'][i]:.4f}")
        converted_t_value = (individual_ground_stats_dict['Sharpe Ratio'][i] * math.sqrt(pre_optim_actuals.shape[0]))/(252**0.5)
        print(f"Converted t-value: {converted_t_value:.4f}")
        print(f"Win/Loss day ratio: {individual_ground_stats_dict['Win/Loss Day Ratio'][i]:.4f}")
        print(f"Calmar ratio: {individual_ground_stats_dict['Calmar Ratio'][i]:.4f}")

        plt.plot(torch.arange(pre_optim_actuals.shape[0]), (1 + individual_ground_truth_portfolio_dict['daily_portfolio_returns'][:,i]).cumprod(dim=0), label=f"Ground Truth Strategy (|Pred| > TxCost)")
        plt.plot(torch.arange(pre_optim_actuals.shape[0]), buy_and_hold_equity_curve_single, label=f"Buy & Hold {args_from_metadata.tickers[i]}", linestyle=":")

        plt.title(f"Ground Truth Strategy vs. Buy & Hold (Tickers: {', '.join(args_from_metadata.tickers)})")
        plt.xlabel("Trading Day in Backtest Period")
        plt.ylabel("Cumulative Return (Normalized to 1)")
        plt.legend()
        plt.grid(True)
        plt.show()

    # END of MSFT combination testing

    aapl_model_path = "ensemble/models/AAPL/ver_9_3199.ckpt"
    amzn_model_path = "ensemble/models/AMZN/ver_2_2299.ckpt"
    intc_model_path = "ensemble/models/INTC/ver_28_2399.ckpt"
    msft_model_path = "ensemble/models/MSFT/ver_54_4099.ckpt"
    nvda_model_path = "ensemble/models/NVDA/ver_2_1899.ckpt"
    gspc_model_path = "ensemble/models/^GSPC/ver_3_1399.ckpt"

    # model = load_mtp_model(cli_args.model_path, args_from_metadata).to(device)

    # # Generate model predictions for the same period
    # print(f"--- Generating model predictions for backtest period ({cli_args.days_to_check} days) ---")
    # model_1day_predictions_denorm = get_mtp_predictions_for_backtest(
    #     model, all_mtp_input_features, args_from_metadata, cli_args.days_to_check, device, cli_args
    # )

    aapl_model = load_mtp_model(aapl_model_path, args_from_metadata).to(device)
    aapl_1day_predictions_denorm = get_mtp_predictions_for_backtest(
        aapl_model, all_mtp_input_features, args_from_metadata, cli_args.days_to_check, device, cli_args
    )
    amzn_model = load_mtp_model(amzn_model_path, args_from_metadata).to(device)
    amzn_1day_predictions_denorm = get_mtp_predictions_for_backtest(
        amzn_model, all_mtp_input_features, args_from_metadata, cli_args.days_to_check, device, cli_args
    )
    intc_model = load_mtp_model(intc_model_path, args_from_metadata).to(device)
    intc_1day_predictions_denorm = get_mtp_predictions_for_backtest(
        intc_model, all_mtp_input_features, args_from_metadata, cli_args.days_to_check, device, cli_args
    )
    msft_model = load_mtp_model(msft_model_path, args_from_metadata).to(device)
    msft_1day_predictions_denorm = get_mtp_predictions_for_backtest(
        msft_model, all_mtp_input_features, args_from_metadata, cli_args.days_to_check, device, cli_args
    )
    nvda_model = load_mtp_model(nvda_model_path, args_from_metadata).to(device)
    nvda_1day_predictions_denorm = get_mtp_predictions_for_backtest(
        nvda_model, all_mtp_input_features, args_from_metadata, cli_args.days_to_check, device, cli_args
    )
    gspc_model = load_mtp_model(gspc_model_path, args_from_metadata).to(device)
    gspc_1day_predictions_denorm = get_mtp_predictions_for_backtest(
        gspc_model, all_mtp_input_features, args_from_metadata, cli_args.days_to_check, device, cli_args
    )

    model_1day_predictions_denorm = aapl_1day_predictions_denorm
    model_1day_predictions_denorm[:,1] = amzn_1day_predictions_denorm[:,1]
    model_1day_predictions_denorm[:,2] = intc_1day_predictions_denorm[:,2]
    model_1day_predictions_denorm[:,3] = msft_1day_predictions_denorm[:,3]
    model_1day_predictions_denorm[:,4] = nvda_1day_predictions_denorm[:,4]
    model_1day_predictions_denorm[:,5] = gspc_1day_predictions_denorm[:,5]


    print("--- metrics from backtest (no threshold) ---")
    backtest_metrics = calculate_metrics(actual_1d_returns_for_backtest_period, model_1day_predictions_denorm, args_from_metadata)
    for metric_name, metric_value in backtest_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")


    # print("\n--- Plotting Predicted vs. Actual Returns ---")
    # plot_predicted_vs_actual_returns(
    #     predicted_returns=model_1day_predictions_denorm,
    #     actual_returns=actual_1d_returns_for_backtest_period,
    #     ticker_names=args_from_metadata.tickers,
    #     num_days_to_plot=min(200, cli_args.days_to_check), # Plot last 100 days or all if fewer
    #     plot_filename_prefix="MTP_returns_comparison"
    # )
    
    pre_optim_predictions = model_1day_predictions_denorm.cpu()
    pre_optim_actuals = actual_1d_returns_for_backtest_period.cpu()
    if args_from_metadata.prediction_type == "classification":
        simple_trade_threshold = 0.50
    else:
        simple_trade_threshold = 0*cli_args.transaction_cost # Trade if signal is stronger than one-way cost

    if cli_args.plot_individual_returns:
        individual_ground_truth_portfolio_dict, individual_ground_stats_dict = vectorized_per_stock_backtest(
            predictions=pre_optim_predictions,
            actual_movements=pre_optim_actuals,
            initial_capital=cli_args.initial_capital,
            transaction_cost_pct=cli_args.transaction_cost,
            prediction_type=args_from_metadata.prediction_type,
            decision_type="argmax",
        )
        for i in range(pre_optim_predictions.shape[1]):
            print(f"\n--- Running PRE-OPTIMIZATION Backtest for {args_from_metadata.tickers[i]} ---")
            
            plt.figure(figsize=(14, 7)) # Create a new figure for this plot
            
            # Add Buy and Hold for comparison (copied from your existing plot logic)
            buy_and_hold_returns_single_ticker = actual_1d_returns_for_backtest_period.cpu()[:,i]
            buy_and_hold_equity_curve_single = (1 + buy_and_hold_returns_single_ticker).cumprod(dim=0)

            print(f"Annualized return: {individual_ground_stats_dict['Annualized Return'][i]:.4f}")
            print(f"Sharpe ratio: {individual_ground_stats_dict['Sharpe Ratio'][i]:.4f}")
            converted_t_value = (individual_ground_stats_dict['Sharpe Ratio'][i] * math.sqrt(pre_optim_actuals.shape[0]))/(252**0.5)
            print(f"Converted t-value: {converted_t_value:.4f}")
            print(f"Win/Loss day ratio: {individual_ground_stats_dict['Win/Loss Day Ratio'][i]:.4f}")
            print(f"Calmar ratio: {individual_ground_stats_dict['Calmar Ratio'][i]:.4f}")

            plt.plot(torch.arange(pre_optim_actuals.shape[0]), (1 + individual_ground_truth_portfolio_dict['daily_portfolio_returns'][:,i]).cumprod(dim=0), label=f"Ground Truth Strategy (|Pred| > TxCost)")
            plt.plot(torch.arange(pre_optim_actuals.shape[0]), buy_and_hold_equity_curve_single, label=f"Buy & Hold {args_from_metadata.tickers[i]}", linestyle=":")

            plt.title(f"Ground Truth Strategy vs. Buy & Hold (Tickers: {', '.join(args_from_metadata.tickers)})")
            plt.xlabel("Trading Day in Backtest Period")
            plt.ylabel("Cumulative Return (Normalized to 1)")
            plt.legend()
            plt.grid(True)
            plt.show()


    
    tickers_to_include = [0,1,2,3,4,5]
    print("\n--- Running PRE-OPTIMIZATION Backtest (Trade if |PredReturn| > TxCost) ---")
    pre_optim_portfolio, pre_optim_stats = ground_truth_strategy_trade(
        predictions=pre_optim_predictions[:,tickers_to_include],
        actual_movements=pre_optim_actuals[:,tickers_to_include],
        initial_capital=cli_args.initial_capital,
        transaction_cost_pct=cli_args.transaction_cost,
        allocation_strategy="equal", # "signal_strength"
        prediction_type=args_from_metadata.prediction_type,
        decision_type="argmax"
    )

    if cli_args.plot_equity:
        if pre_optim_portfolio is not None:
            plt.figure(figsize=(14, 7)) # Create a new figure for this plot
            
            # Plot Pre-Optimization Strategy
            pre_optim_equity_curve = (1 + pd.Series(pre_optim_portfolio['daily_portfolio_returns'])).cumprod()
            plt.plot(pre_optim_equity_curve.index, pre_optim_equity_curve.values, label=f"Pre-Opt Strategy")
            
            # Add Buy and Hold for comparison (copied from your existing plot logic)
            if len(args_from_metadata.tickers) > 0:
                if len(args_from_metadata.tickers) == 1:
                    buy_and_hold_returns_single_ticker = actual_1d_returns_for_backtest_period.cpu()[:,0]
                    buy_and_hold_equity_curve_single = (1 + buy_and_hold_returns_single_ticker).cumprod(dim=0)
                    plt.plot(pre_optim_equity_curve.index, buy_and_hold_equity_curve_single.numpy(), label=f"Buy & Hold {args_from_metadata.tickers[0]}", linestyle=":")
                else: # Multiple tickers, plot average B&H
                    buy_and_hold_returns_avg_tickers = actual_1d_returns_for_backtest_period.cpu().mean(dim=1)
                    buy_and_hold_equity_curve_avg = (1 + buy_and_hold_returns_avg_tickers).cumprod(dim=0)
                    plt.plot(pre_optim_equity_curve.index, buy_and_hold_equity_curve_avg.numpy(), label=f"Buy & Hold Avg of {len(args_from_metadata.tickers)} Tickers", linestyle=":")

            print(f"Annualized return: {pre_optim_stats['Annualized Return']:.4f}")
            print(f"Sharpe ratio: {pre_optim_stats['Sharpe Ratio']:.4f}")
            converted_t_value = (pre_optim_stats['Sharpe Ratio'] * math.sqrt(actual_1d_returns_for_backtest_period.shape[0]))/(252**0.5)
            print(f"Converted t-value: {converted_t_value:.4f}")
            print(f"Win/Loss day ratio: {pre_optim_stats['Win/Loss Day Ratio']:.4f}")
            print(f"Max drawdown: {pre_optim_stats['Max Drawdown']:.4f}")
            print(f"Calmar ratio: {pre_optim_stats['Calmar Ratio']:.4f}")

            plt.title(f"Pre-Optimization Strategy vs. Buy & Hold (Tickers: {', '.join(args_from_metadata.tickers)})")
            plt.xlabel("Trading Day in Backtest Period")
            plt.ylabel("Cumulative Return (Normalized to 1)")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"pre_optimized_strategy_MTP_{'_'.join(args_from_metadata.tickers)}.png")
            print(f"Pre-optimization equity curve plot saved to pre_optimized_strategy_MTP_{'_'.join(args_from_metadata.tickers)}.png")
            plt.show()


# NOTE s
# current best ensemble model - simply all top performers per stock
# all models seem to learn the same general trend/ when they have predictive power heavily overlaps
# combining several good models prediction using logits or probs averaging does not really work, just more of the same/slightly worse
