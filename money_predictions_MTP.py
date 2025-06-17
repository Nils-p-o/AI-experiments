# money_predictions_MTP.py

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

# Assuming your project structure allows these imports
from transformer_arch.money.money_former_MLA_DINT_cog_attn_2_MTP import Money_former_MLA_DINT_cog_attn_MTP
from training.data_loaders.stocks_time_series_2_MTP import (
    align_financial_dataframes,
    feature_volatility_ret,
    feature_ema,
    feature_ppo,
    calculate_volume_price_trend_standard, # Assuming you used standard VPT
    calculate_close_line_values, # Assuming you have this
    feature_time_data,
)
from bayes_opt import BayesianOptimization

# Global variables for Bayesian Optimization (will be set later)
OPTIMIZATION_PREDS_1DAY = None
OPTIMIZATION_ACTUAL_1D_RETURNS = None
OPTIMIZATION_INITIAL_CAPITAL = 100000.0
OPTIMIZATION_TRANSACTION_COST = 0.0005
OPTIMIZATION_SIGNAL_HORIZON_NAME = "1-day (Opt.)"

# def load_config_and_args_from_metadata(metadata_path):
#     """Loads experiment configuration from metadata.json"""
#     if not os.path.exists(metadata_path):
#         raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
#     with open(metadata_path, "r") as f:
#         metadata = json.load(f)
    
#     # Convert list for indices_to_predict back to list if it's not (JSON might make it so)
#     if "indices_to_predict" in metadata and not isinstance(metadata["indices_to_predict"], list):
#         try:
#             # Attempt to parse if it's a string representation of a list
#             parsed_indices = json.loads(metadata["indices_to_predict"].replace("'", "\""))
#             metadata["indices_to_predict"] = parsed_indices
#         except:
#             # Fallback or raise error if parsing fails
#             print(f"Warning: Could not parse indices_to_predict: {metadata['indices_to_predict']}")
#             # You might want a default or to ensure it's always saved as a proper list
#             if isinstance(metadata["indices_to_predict"], int): # If it was a single int
#                  metadata["indices_to_predict"] = [metadata["indices_to_predict"]]


#     # Reconstruct args Namespace. Some args might be in the root, others in 'args' sub-dict
#     # Be flexible based on how your metadata.json is structured from training.
#     # This is a common pattern if args were saved directly.
#     args_dict = metadata.copy() 
#     if 'args' in metadata: # If training script saved args under an 'args' key
#         args_dict.update(metadata['args'])

#     # Ensure essential args for model instantiation are present
#     required_model_args = ['d_model', 'nhead', 'num_layers', 'd_ff', 'dropout', 
#                            'input_features', 'tickers', 'indices_to_predict', 
#                            'predict_gaussian', 'bias', 'head_dim', 
#                            'kv_compression_dim', 'q_compression_dim', 'qk_rope_dim'] # Add all required by your MTP model
    
#     for req_arg in required_model_args:
#         if req_arg not in args_dict:
#             # Try to find it in the root of metadata if not in args_dict directly
#             if req_arg in metadata:
#                 args_dict[req_arg] = metadata[req_arg]
#             else:
#                 raise KeyError(f"Required argument '{req_arg}' not found in metadata or args dictionary.")

#     # Convert to Namespace
#     args_namespace = argparse.Namespace(**args_dict)
    
#     # Make sure critical fields from metadata are directly on args_namespace if model expects them
#     args_namespace.tickers = metadata.get("tickers", args_namespace.tickers)
#     args_namespace.indices_to_predict = metadata.get("indices_to_predict", args_namespace.indices_to_predict)
#     args_namespace.input_features = metadata.get("input_features", args_namespace.input_features if hasattr(args_namespace, 'input_features') else len(metadata.get("columns", [])))
    
#     # Add normalization stats if saved (CRITICAL for MTP)
#     if "normalization_means" in metadata and "normalization_stds" in metadata:
#         args_namespace.normalization_means = torch.tensor(metadata["normalization_means"])
#         args_namespace.normalization_stds = torch.tensor(metadata["normalization_stds"])
#     else:
#         print("Warning: Normalization stats not found in metadata. Loading raw data...")
#         all_dates = open("time_series_data/train.csv", "r").readlines()
#         args_namespace.start_date = all_dates[0]
#         args_namespace.end_date = all_dates[-1]
#         raw_data = yf.download(args_namespace.tickers, start=args_namespace.start_date, end=args_namespace.end_date, progress=True, auto_adjust=False, back_adjust=False)
#         raw_data = torch.tensor(raw_data.values, dtype=torch.float32).reshape(
#             -1, 6, len(args_namespace.tickers)
#         )  # (Time, Features, tickers)
#         raw_data = raw_data.transpose(0, 1)
#         raw_data = raw_data[1:,:,:]
#         args_namespace.normalization_means = torch.mean(raw_data, dim=1)
#         args_namespace.normalization_stds = torch.std(raw_data, dim=1)

#     return args_namespace

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
        'input_features', 'tickers', 'indices_to_predict', 'predict_gaussian',
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
    
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_state_dict[k[len("model."):]] = v
            else:
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


def download_and_process_inference_data(args, start_date, end_date, seq_len_needed):
    """
    Downloads and processes data for inference, replicating training feature engineering.
    Returns:
        full_input_features_mtp (Tensor): Shape (features, max_pred_horizon, time_steps, num_tickers)
                                           This is the MTP-style input.
        true_prices_for_backtest (Tensor): Shape (time_steps_for_backtest+1, num_tickers)
                                           Raw close prices for calculating actual returns.
        all_columns (list): List of all feature names.
    """
    print("Downloading and processing inference data...")
    tickers = args.tickers
    target_dates = args.indices_to_predict


    raw_data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        progress=True,
        auto_adjust=False,
        back_adjust=False,
    )
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
    if raw_data.empty:
        print("No data downloaded.")

    indexes = raw_data.index
    columns = list(raw_data.columns.levels[0])
    raw_data = torch.tensor(raw_data.values, dtype=torch.float32).reshape(
        -1, len(columns), len(tickers)
    )  # (Time, Features, tickers)
    raw_data = raw_data.transpose(0, 1)  # (Features, Time series, tickers)

    raw_data = raw_data[1:, :, :]
    full_data = (raw_data[:,1:,:] - raw_data[:,:-1,:]) / raw_data[:,:-1,:]  # (features, time series, tickers)
    full_data = torch.cat((torch.zeros_like(full_data[:,0:1,:]), full_data), dim=1)
    # maybe add ffil for any infs or nans? 885
    full_data[4,885, 1] = full_data[4,884, 1]

    vol_data, vol_columns = feature_volatility_ret(returns=full_data[0:1], prefix="close_returns_")
    full_data = torch.cat((full_data, vol_data), dim=0)
    columns.extend(vol_columns)

    full_ema = []
    for i in range(len(tickers)):
        ema_data, ema_columns = feature_ema(full_data[0, :, i], prefix="close_returns_")
        full_ema.append(ema_data.unsqueeze(-1))
    full_data = torch.cat((full_data, torch.cat(full_ema, dim=-1)), dim=0)
    columns.extend(ema_columns)

    full_ema = []
    for i in range(len(tickers)):
        ema_data, ema_columns = feature_ema(full_data[1, :, i], prefix="high_returns_")
        full_ema.append(ema_data.unsqueeze(-1))
    full_data = torch.cat((full_data, torch.cat(full_ema, dim=-1)), dim=0)
    columns.extend(ema_columns)

    full_ema = []
    for i in range(len(tickers)):
        ema_data, ema_columns = feature_ema(full_data[2, :, i], prefix="low_returns_")
        full_ema.append(ema_data.unsqueeze(-1))
    full_data = torch.cat((full_data, torch.cat(full_ema, dim=-1)), dim=0)
    columns.extend(ema_columns)

    full_ema = []
    for i in range(len(tickers)):
        ema_data, ema_columns = feature_ema(full_data[3, :, i], prefix="open_returns_")
        full_ema.append(ema_data.unsqueeze(-1))
    full_data = torch.cat((full_data, torch.cat(full_ema, dim=-1)), dim=0)
    columns.extend(ema_columns)

    full_ema = []
    for i in range(len(tickers)):
        ema_data, ema_columns = feature_ema(full_data[4, :, i], prefix="volume_returns_")
        full_ema.append(ema_data.unsqueeze(-1))
    full_data = torch.cat((full_data, torch.cat(full_ema, dim=-1)), dim=0)
    columns.extend(ema_columns)

    full_vpt = []
    vpt_data = calculate_volume_price_trend_standard(raw_data[:4], raw_data[4:])
    full_vpt.append(vpt_data)
    full_data = torch.cat((full_data, torch.cat(full_vpt, dim=0)), dim=0)
    columns.extend(["vpt_close", "vpt_high", "vpt_low", "vpt_open"])

    full_ppo = []
    for i in range(len(tickers)):
        ppo_data, ppo_columns = feature_ppo(raw_data[0, :, i], prefix="close_")
        full_ppo.append(ppo_data)
    full_data = torch.cat((full_data, torch.stack(full_ppo, dim=-1)), dim=0)
    columns.extend(ppo_columns)

    full_ppo = []
    for i in range(len(tickers)):
        ppo_data, ppo_columns = feature_ppo(raw_data[1, :, i], prefix="high_")
        full_ppo.append(ppo_data)
    full_data = torch.cat((full_data, torch.stack(full_ppo, dim=-1)), dim=0)
    columns.extend(ppo_columns)

    full_ppo = []
    for i in range(len(tickers)):
        ppo_data, ppo_columns = feature_ppo(raw_data[2, :, i], prefix="low_")
        full_ppo.append(ppo_data)
    full_data = torch.cat((full_data, torch.stack(full_ppo, dim=-1)), dim=0)
    columns.extend(ppo_columns)

    full_ppo = []
    for i in range(len(tickers)):
        ppo_data, ppo_columns = feature_ppo(raw_data[3, :, i], prefix="open_")
        full_ppo.append(ppo_data)
    full_data = torch.cat((full_data, torch.stack(full_ppo, dim=-1)), dim=0)
    columns.extend(ppo_columns)

    full_ppo = []
    for i in range(len(tickers)):
        ppo_data, ppo_columns = feature_ppo(raw_data[4, :, i], prefix="volume_")
        full_ppo.append(ppo_data)
    full_data = torch.cat((full_data, torch.stack(full_ppo, dim=-1)), dim=0)
    columns.extend(ppo_columns)

    clv_data = calculate_close_line_values(raw_data[0], raw_data[1], raw_data[2]).unsqueeze(0)
    full_data = torch.cat((full_data, clv_data), dim=0)
    columns.extend(["clv"])

    vix_data = yf.download(
        "^VIX", start=start_date, end=end_date, progress=False, auto_adjust=False
    )
    aligned_vix_data = pd.DataFrame(columns=vix_data.columns)
    for column in vix_data.columns.levels[0]:
        aligned_vix_data[column, "^VIX"] = align_financial_dataframes(
            {column: vix_data},
            target_column=column,
            fill_method="ffill",
            min_date=start_date,
            max_date=end_date,
        )
    vix_data = aligned_vix_data
    vix_data = vix_data.to_numpy()[:, 1:-1]
    vix_data = torch.tensor(vix_data, dtype=torch.float32)
    vix_data = vix_data.transpose(0, 1)

    vix_data = vix_data.unsqueeze(-1)
    vix_data = vix_data.expand(vix_data.shape[0], vix_data.shape[1], len(tickers))
    full_data = torch.cat((full_data, vix_data), dim=0)
    columns.extend(["vix_close", "vix_high", "vix_low", "vix_open"])

    copper = yf.download(
        "HG=F",
        start=start_date,
        end=end_date,
        progress=True,
        auto_adjust=False,
        back_adjust=False,
    )
    aligned_copper = pd.DataFrame(columns=copper.columns)
    for column in copper.columns.levels[0]:
        aligned_copper[column, "HG=F"] = align_financial_dataframes(
            {column: copper},
            target_column=column,
            fill_method="ffill",
            min_date=start_date,
            max_date=end_date,
    )
    copper = aligned_copper
    copper = copper.to_numpy()[:, 1:]
    copper = torch.tensor(copper, dtype=torch.float32)
    copper = copper.transpose(0, 1)
    copper = copper.unsqueeze(-1)
    copper = copper.expand(copper.shape[0], copper.shape[1], len(tickers))
    full_data = torch.cat((full_data, copper), dim=0)
    columns.extend(["copper_close", "copper_open", "copper_high", "copper_low", "copper_volume"])

    data = torch.empty(full_data.shape[0], max(target_dates), full_data.shape[1]-max(target_dates), full_data.shape[2], dtype=torch.float32)
    for i in range(max(target_dates)):
        data[:,i,:,:] = full_data[:,i:-(max(target_dates)-i),:]
    data = data[:, :, 20:, :]  # (features, target_inputs, time series, tickers)

    # time data
    time_data, time_columns = feature_time_data(indexes, target_dates, tickers)
    data = torch.cat((data, time_data), dim=0)
    columns.extend(time_columns)

    # MTP_targets = torch.empty(
    #     (5, max(target_dates), data.shape[2]+20, len(tickers)), dtype=torch.float32
    # ) # (chlov, target_dates, time series, tickers)
    MTP_full = (raw_data[:, 1:, :] - raw_data[:, :-1, :]) / raw_data[:, :-1, :]
    MTP_full[4, 884, 1] = MTP_full[4, 883, 1]
    MTP_full = MTP_full[:, 20:, :]
    if max(target_dates) > 1:
        MTP_full = MTP_full[:, :-(max(target_dates)-1), :]
    # for i in range(max(target_dates)):
    #     if i == max(target_dates) - 1:
    #         MTP_targets[:, i, :, :] = MTP_full[:, i:, :]
    #     else:
    #         MTP_targets[:, i, :, :] = MTP_full[:, i:-(max(target_dates) - i - 1), :]
    # MTP_targets = MTP_targets[:, :, 20:, :]

    # znorm step (need all column means and stds)
    data[:-15] = (data[:-15] - args.normalization_means.view(data.shape[0]-15, 1, 1, -1)) / (args.normalization_stds.view(data.shape[0]-15, 1, 1, -1) + 1e-8)

    return data, MTP_full, columns

    # --- Placeholder Feature Engineering (to be replaced) ---
    print("WARNING: Using placeholder data processing for inference. Implement full feature engineering.")
    
    # Fetching raw data (simplified for example)
    raw_yf_data = yf.download(args.tickers, start=start_date_str, end=end_date_str, progress=False)
    if raw_yf_data.empty or 'Close' not in raw_yf_data:
        raise ValueError("Could not download sufficient data for inference.")
    
    # Assuming 'Close' prices are what we'll use for true_chlov_returns and a dummy feature
    # In reality, you'd align and process all OHLCV for all tickers
    
    # For true_chlov_returns (used to calculate actual 1-day returns for backtest)
    # We need nr_of_days_to_check + 1 prices.
    # The processing length needs to be seq_len + nr_of_days_to_check
    # Let's assume end_date_str is the last day OF the backtest.
    # So we need prices from (end_date - (nr_of_days_to_check) days) to end_date.
    # And for features, we need seq_len before the first prediction day.
    
    all_raw_prices_list = []
    min_len = float('inf')
    for ticker in args.tickers:
        try:
            # Ensure we get OHLCV for each ticker
            ticker_data = raw_yf_data['Close'][ticker].ffill().bfill() # Example, get Close
            if ticker_data.empty:
                raise ValueError(f"No data for {ticker}")
            all_raw_prices_list.append(ticker_data.values)
            min_len = min(min_len, len(ticker_data.values))
        except KeyError:
            raise KeyError(f"Ticker {ticker} not found in downloaded YFinance data. Available: {raw_yf_data['Close'].columns}")


    if not all_raw_prices_list:
        raise ValueError("No price data could be processed.")

    # Trim all series to the minimum common length from the end
    all_raw_prices_np = np.array([s[-min_len:] for s in all_raw_prices_list]).T # (time, tickers)
    true_prices_for_backtest = torch.from_numpy(all_raw_prices_np).float() # (time, tickers)

    # Now, create dummy `full_input_features_mtp` based on `args.input_features`
    # and `max(args.indices_to_predict)`
    # The time dimension for this full_input_features_mtp should be `min_len - max(args.indices_to_predict)`
    # because of how the MTP input is constructed.
    mtp_time_dim = min_len - max(args.indices_to_predict)
    if mtp_time_dim < seq_len_needed : # seq_len_needed is for the *input* to the model
        raise ValueError(f"Not enough historical data to form sequences. Need {seq_len_needed} time steps for features, got {mtp_time_dim} after MTP structuring.")

    # This is where your full feature pipeline from stocks_time_series_2_MTP.py goes
    # For now, a placeholder:
    # Let's assume 'full_data' (features, time, tickers) has been created.
    # For this placeholder, let's just use returns as the only feature for all tickers.
    dummy_full_data_returns = (true_prices_for_backtest[1:] - true_prices_for_backtest[:-1]) / (true_prices_for_backtest[:-1] + 1e-8)
    dummy_full_data_returns = torch.cat([dummy_full_data_returns[0:1,:], dummy_full_data_returns], dim=0) # Pad first return
    
    # Expand to match args.input_features by repeating this single return feature
    dummy_full_data = dummy_full_data_returns.T.unsqueeze(0).repeat(args.input_features, 1, 1)
    # dummy_full_data shape: (input_features, time, num_tickers)

    # Construct MTP-style input
    max_pred_horizon = max(args.indices_to_predict)
    # Ensure dummy_full_data has enough length for the MTP shifts
    if dummy_full_data.shape[1] < max_pred_horizon:
        raise ValueError("Dummy full data too short for MTP processing.")

    # This part mimics the MTP input creation in your dataloader
    num_base_features = dummy_full_data.shape[0]
    num_time_steps_for_mtp_input = dummy_full_data.shape[1] - max_pred_horizon
    
    # The number of "target horizons" in the input is `max_pred_horizon`
    # (i.e., you prepare input features aligned for predicting 1 step ahead, 2 steps ahead... up to max_pred_horizon steps ahead)
    num_target_horizons_in_input = max_pred_horizon 

    mtp_input_features = torch.empty(
        num_base_features,
        num_target_horizons_in_input,
        num_time_steps_for_mtp_input,
        len(args.tickers),
        dtype=torch.float32, device=args_from_metadata.normalization_means.device # Ensure same device
    )
    for i in range(num_target_horizons_in_input):
        # The i-th slice of target_horizons_in_input corresponds to features prepared
        # as if we are `i+1` steps into the future relative to the start of a prediction window.
        # `full_data[:, i : num_time_steps_for_mtp_input + i, :]`
        # This matches the dataloader: `data[:,i,:,:] = full_data[:,i:-(max(target_dates)-i),:]`
        # where loop is `for i in range(max(target_dates))`
        # So, target_horizon_idx `j` uses `full_data[:, j : num_time_steps_for_mtp_input + j, :]`
        mtp_input_features[:, i, :, :] = dummy_full_data[:, i : num_time_steps_for_mtp_input + i, :]
    
    # Apply global Z-normalization (excluding time features, if any were added)
    # Assuming normalization_means/stds are (num_base_features, 1, 1) for broadcasting
    # This placeholder only has return-like features, so all get normalized.
    # In your real code, slice out time features before this.
    norm_means = args_from_metadata.normalization_means.unsqueeze(1).unsqueeze(-1) # (features,1,1,1)
    norm_stds = args_from_metadata.normalization_stds.unsqueeze(1).unsqueeze(-1)   # (features,1,1,1)
    
    # mtp_input_features_norm = (mtp_input_features - norm_means) / (norm_stds + 1e-8)
    # For the MTP structure, normalization_means/stds are (features, 1, num_tickers)
    # The input mtp_input_features is (features, target_horizons, time, num_tickers)
    # norm_means must be broadcastable. Let's assume means/stds are (features, 1, num_tickers)
    # and need to be expanded for the target_horizons and time dimensions.
    norm_means_expanded = args_from_metadata.normalization_means.unsqueeze(1).unsqueeze(2) # (features, 1, 1, num_tickers)
    norm_stds_expanded = args_from_metadata.normalization_stds.unsqueeze(1).unsqueeze(2)   # (features, 1, 1, num_tickers)
    
    mtp_input_features_norm = (mtp_input_features - norm_means_expanded) / (norm_stds_expanded + 1e-8)


    # Placeholder: just returning the last `seq_len_needed` part of the MTP input features
    # and the corresponding segment of true_chlov_returns for backtesting.
    # The actual slicing for rolling predictions will happen in the main loop.
    print(f"Shape of mtp_input_features_norm before returning: {mtp_input_features_norm.shape}")
    print(f"Shape of true_prices_for_backtest before returning: {true_prices_for_backtest.shape}")
    
    # This function should return all processed features that are then sliced in the main loop.
    # Also, need the column names if feature selection is done by name.
    # For now, assume args.columns has the right names.
    
    return mtp_input_features_norm, true_prices_for_backtest, args_from_metadata.columns # Or columns from processing


def run_trading_strategy_1day_signal(
    predictions_1day_ahead: torch.Tensor, # Model's predicted 1-day returns
    actual_1d_returns: torch.Tensor,      # Actual 1-day returns for P&L calculation
    trade_threshold_up: float =0.01,
    trade_threshold_down: float =0.01, # Absolute value for short threshold
    initial_capital: float =100000.0,
    transaction_cost_pct: float =0.0005,
    signal_horizon_name: str ="1-day",
    verbose: int =0
):
    """
    Simulates a trading strategy using 1-day predictions, executing trades
    evaluated on the next 1-day actual return. Allows for asymmetric thresholds.
    """
    # Ensure inputs are on CPU for numpy operations if not already
    predictions_1day_ahead = predictions_1day_ahead.cpu()
    actual_1d_returns = actual_1d_returns.cpu()

    num_days, num_tickers = predictions_1day_ahead.shape
    if num_days != actual_1d_returns.shape[0] or num_tickers != actual_1d_returns.shape[1]:
        raise ValueError(
            "Predictions and actual 1-day returns must have the same shape (num_days, num_tickers)."
        )

    portfolio_values = [initial_capital] # Start with initial capital before day 1
    daily_portfolio_returns = []
    current_capital = initial_capital
    
    if verbose > 0:
        print(f"\n--- Running Trading Strategy (Signal: {signal_horizon_name}) ---")
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"Trade Threshold UP: {trade_threshold_up*100:.3f}%")
        print(f"Trade Threshold DOWN (abs): {trade_threshold_down*100:.3f}%")
        print(f"Transaction Cost (per trade leg): {transaction_cost_pct*100:.4f}%")

    for day_idx in range(num_days):
        capital_at_start_of_day = current_capital
        daily_pnl = 0.0
        num_long_trades_today = 0
        num_short_trades_today = 0

        signals_today = predictions_1day_ahead[day_idx]       # Predictions for today's close, made using data up to yesterday's close
        realized_1d_returns_today = actual_1d_returns[day_idx] # Actual returns for today

        active_signals = []
        for ticker_idx in range(num_tickers):
            if signals_today[ticker_idx] > trade_threshold_up:
                active_signals.append({"action": "long", "ticker_idx": ticker_idx})
            elif signals_today[ticker_idx] < -trade_threshold_down: # trade_threshold_down is positive
                active_signals.append({"action": "short", "ticker_idx": ticker_idx})

        if not active_signals:
            daily_portfolio_returns.append(0.0)
            portfolio_values.append(current_capital)
            if verbose > 1 and (day_idx < 5 or day_idx == num_days -1 ):
                 print(f"Day {day_idx+1}: No trades. Capital: ${current_capital:,.2f}")
            continue

        capital_per_trade = capital_at_start_of_day / len(active_signals)

        for trade in active_signals:
            ticker_idx = trade["ticker_idx"]
            invested_amount = capital_per_trade # Amount allocated to this specific trade leg

            # Cost to enter the position
            cost_entry = invested_amount * transaction_cost_pct
            
            pnl_from_position = 0.0
            if trade["action"] == "long":
                pnl_from_position = invested_amount * realized_1d_returns_today[ticker_idx]
                num_long_trades_today +=1
            elif trade["action"] == "short":
                pnl_from_position = invested_amount * (-realized_1d_returns_today[ticker_idx])
                num_short_trades_today +=1
            
            # Value of position before exit cost
            value_before_exit = invested_amount + pnl_from_position
            cost_exit = value_before_exit * transaction_cost_pct # Cost to exit the position

            net_pnl_for_trade = pnl_from_position - cost_entry - cost_exit
            daily_pnl += net_pnl_for_trade
        
        current_capital += daily_pnl
        day_return_pct = (daily_pnl / capital_at_start_of_day) if capital_at_start_of_day > 1e-9 else 0.0
        
        daily_portfolio_returns.append(day_return_pct)
        portfolio_values.append(current_capital) # Capital at END of day_idx

        if verbose > 1 and (len(active_signals) > 0 or day_idx < 5 or day_idx == num_days - 1):
            print(f"Day {day_idx+1}: Longs: {num_long_trades_today}, Shorts: {num_short_trades_today}. Day P&L: ${daily_pnl:,.2f}. Day Ret: {day_return_pct*100:.3f}%. EOD Capital: ${current_capital:,.2f}")


    # --- Performance Statistics Calculation ---
    # portfolio_values is EOD capital, so its length is num_days + 1.
    # daily_portfolio_returns has length num_days.
    portfolio_df = pd.DataFrame({
        "portfolio_value": portfolio_values[1:], # Use EOD values for alignment with returns
        "daily_return": daily_portfolio_returns
    })
    # If using portfolio_values[0] in the df, then daily_return needs a 0 at the start.
    # For stats, usually just use the daily_returns series.

    if not daily_portfolio_returns: # Handles case where num_days = 0
        total_return = 0.0
        annualized_return = 0.0
        annualized_volatility = 0.0
        sharpe_ratio = 0.0
        max_drawdown = 0.0
        num_winning_days = 0
        num_losing_days = 0
        win_loss_ratio = 0.0
    else:
        daily_returns_np = np.array(daily_portfolio_returns)
        total_return = (current_capital / initial_capital) - 1.0
        
        # Annualization: ( (1 + total_ret) ^ (252/num_days) ) - 1
        if num_days > 0:
            annualized_return = ((1 + total_return) ** (252.0 / num_days)) - 1.0
            annualized_volatility = np.std(daily_returns_np) * np.sqrt(252.0)
            sharpe_ratio = (annualized_return / annualized_volatility) if annualized_volatility > 1e-9 else 0.0
        else:
            annualized_return = 0.0
            annualized_volatility = 0.0
            sharpe_ratio = 0.0

        cumulative_returns_pd = (1 + pd.Series(daily_returns_np)).cumprod()
        # Ensure cumulative_returns_pd is not empty before calling .expanding or .min
        if not cumulative_returns_pd.empty:
            peak = cumulative_returns_pd.expanding(min_periods=1).max()
            drawdown = (cumulative_returns_pd / peak) - 1.0
            max_drawdown = drawdown.min()
        else:
            max_drawdown = 0.0
        
        num_winning_days = (daily_returns_np > 0).sum()
        num_losing_days = (daily_returns_np < 0).sum()
        win_loss_ratio = num_winning_days / num_losing_days if num_losing_days > 0 else float("inf")


    stats_dict = {
        "Signal Horizon Used": signal_horizon_name,
        "Trade Threshold Up": trade_threshold_up,
        "Trade Threshold Down (abs)": trade_threshold_down,
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Number of Trading Days": num_days,
        "Number of Winning Days": int(num_winning_days),
        "Number of Losing Days": int(num_losing_days),
        "Win/Loss Day Ratio": win_loss_ratio,
        "Final Capital": current_capital,
    }

    if verbose > 0:
        print(f"\n--- Strategy Summary (Signal: {signal_horizon_name}) ---")
        for key, value in stats_dict.items():
            if isinstance(value, (float, np.floating)) and key not in ["Sharpe Ratio", "Win/Loss Day Ratio", "Signal Horizon Used", "Trade Threshold Up", "Trade Threshold Down (abs)"]:
                print(f"{key}: {value*100:.2f}%" if "Return" in key or "Drawdown" in key or "Volatility" in key else f"{key}: {value:.2f}")
            elif isinstance(value, (float, np.floating)) and key in ["Sharpe Ratio", "Win/Loss Day Ratio", "Trade Threshold Up", "Trade Threshold Down (abs)"]:
                print(f"{key}: {value:.4f}") # More precision for thresholds and ratios
            else:
                print(f"{key}: {value}")
    
    return portfolio_df, stats_dict


def objective_function_mtp(long_threshold_raw, short_threshold_raw, min_step=0.001):
    """Objective function for Bayesian Optimization for MTP (1-day signal)."""
    long_threshold = round(long_threshold_raw / min_step) * min_step
    short_threshold = round(short_threshold_raw / min_step) * min_step # This is absolute value for short

    # print(f"  Testing long_thresh: {long_threshold:.4f}, short_thresh_abs: {short_threshold:.4f}")

    _, strategy_stats = run_trading_strategy_1day_signal(
        predictions_1day_ahead=OPTIMIZATION_PREDS_1DAY, # Global var
        actual_1d_returns=OPTIMIZATION_ACTUAL_1D_RETURNS, # Global var
        trade_threshold_up=long_threshold,
        trade_threshold_down=short_threshold, 
        initial_capital=OPTIMIZATION_INITIAL_CAPITAL,
        transaction_cost_pct=OPTIMIZATION_TRANSACTION_COST,
        signal_horizon_name=OPTIMIZATION_SIGNAL_HORIZON_NAME,
        verbose=0 # Keep optimization quiet unless debugging
    )
    sharpe = strategy_stats.get("Sharpe Ratio", 0.0)
    if not np.isfinite(sharpe) or sharpe < -5: # Penalize very bad Sharpe ratios
        return -10.0 
    return sharpe
    # total_return = strategy_stats.get("Total Return", 0.0)
    # return total_return


def get_mtp_predictions_for_backtest(model, all_mtp_input_features, args, nr_of_days_to_check, device):
    """
    Generates predictions for the backtest period using the MTP model.
    Outputs 1-day ahead predictions for the 'Close' feature return.
    """
    # all_mtp_input_features shape: (features, max_pred_horizon, time_steps, num_tickers)
    # We need to slice this to create batches of (1, max_pred_horizon, seq_len, num_tickers, features)
    
    seq_len = args.seq_len
    num_pred_horizons_input = all_mtp_input_features.shape[1] # Should be max(args.indices_to_predict)
    # num_input_features_model = args.input_features # From metadata, what the model was trained on
    
    # Permute all_mtp_input_features to (max_pred_horizon, time_steps, num_tickers, features)
    # for easier slicing of (seq_len, num_tickers, features) blocks
    all_mtp_input_features_permuted = all_mtp_input_features.permute(1, 2, 3, 0).to(device)
    # New shape: (max_pred_horizon, time_steps, num_tickers, features)

    model_predictions_1day_list = []

    print(f"Generating predictions for {nr_of_days_to_check} days...")
    for day_i in range(nr_of_days_to_check):
        # The input to the model needs to be for a specific window ending before the day we predict FOR.
        # `day_i` = 0 means we predict for the first day of the backtest.
        # The input features should end at `day_i + seq_len - 1`.
        # The `all_mtp_input_features_permuted` has its time dimension aligned such that
        # index `k` corresponds to the k-th day in the available processed feature history.
        # We need a window of `seq_len` ending at `day_i + seq_len -1` from this history.
        
        # Slice for the current window:
        # `current_window_mtp_input` shape: (max_pred_horizon, seq_len, num_tickers, features)
        current_window_mtp_input = all_mtp_input_features_permuted[:, day_i : day_i + seq_len, :, :]
        
        # Reshape for model: (batch_size=1, max_pred_horizon, seq_len, num_tickers, features)
        model_input_tensor = current_window_mtp_input.unsqueeze(0)
        
        # Prepare separator and tickers for the model (now with the horizon dimension)
        seperator_input = torch.zeros((1, num_pred_horizons_input, 1), dtype=torch.int, device=device)
        tickers_input = torch.arange(len(args.tickers), device=device).unsqueeze(0).unsqueeze(0).repeat(1, num_pred_horizons_input, 1)

        with torch.no_grad():
            outputs = model(model_input_tensor, seperator_input, tickers_input)
        outputs = outputs.view(1, num_pred_horizons_input, (seq_len+1), len(args.tickers), 5)
        # outputs shape: (1, num_pred_horizons_output, seq_len+1, num_tickers, 5_chlov)
        
        # Extract 1-day ahead prediction from the "main model" path (horizon 0)
        # from the last relevant output position (index -1 for seq_len+1, as separator is [0])
        # for the 'Close' feature (assuming index 0 of the 5 CHLOV features)
        pred_1day_close_return_norm = outputs[0, 0, -1, :, 0] # Shape: (num_tickers)
        model_predictions_1day_list.append(pred_1day_close_return_norm)

    model_predictions_1day_tensor = torch.stack(model_predictions_1day_list, dim=0) # (nr_of_days_to_check, num_tickers)
    
    # De-normalize predictions
    # Normalization stats are (num_base_features, 1, num_tickers)
    # We need the stats for the 'Close' feature (assuming it's the first of the 5 CHLOV inputs)
    close_feature_original_index = 0 # If 'Close' return was the first of the 5 base features used for target norm
    
    # Ensure normalization_means/stds are on the correct device
    norm_means_for_close = args.normalization_means[close_feature_original_index, :] # Shape (num_tickers)
    norm_stds_for_close = args.normalization_stds[close_feature_original_index, :]   # Shape (num_tickers)

    denormalized_predictions = model_predictions_1day_tensor * norm_stds_for_close.unsqueeze(0) + \
                               norm_means_for_close.unsqueeze(0)
    
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Make predictions with a trained MTP stock model.")
    parser.add_argument("--model_path", type=str, default="good_models/MTP/aapl_gspc/name=0-epoch=24-val_loss=0.00.ckpt", help="Path to the trained .pth model file.")
    parser.add_argument("--metadata_path", type=str, default="good_models/MTP/aapl_gspc/hparams.yaml", help="Path to the metadata.json file for the model.")
    parser.add_argument("--days_to_check", type=int, default=1300, help="Number of recent days to generate predictions for and backtest.")
    parser.add_argument("--start_date_data", type=str, default="2020-01-01", help="Start date for downloading historical data.")
    parser.add_argument("--end_date_data", type=str, default="2025-05-25", help="End date for downloading historical data (serves as backtest end).")
    parser.add_argument("--initial_capital", type=float, default=100000.0, help="Initial capital for backtesting.")
    parser.add_argument("--transaction_cost", type=float, default=0.0005, help="Transaction cost percentage.")
    parser.add_argument("--plot_equity", type=bool, default=True, help="Plot the equity curve of the optimized strategy.")
    parser.add_argument("--verbose_strategy", type=int, default=0, help="Verbosity level for strategy run printouts (0: silent, 1: summary).")


    cli_args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Config and Model
    args_from_metadata = load_config_and_args_from_metadata(cli_args.metadata_path)
    args_from_metadata.device = device # Add device to args
    # Pass normalization stats to device if they are tensors
    if hasattr(args_from_metadata, 'normalization_means'):
        args_from_metadata.normalization_means = args_from_metadata.normalization_means.to(device)
    if hasattr(args_from_metadata, 'normalization_stds'):
        args_from_metadata.normalization_stds = args_from_metadata.normalization_stds.to(device)

    model = load_mtp_model(cli_args.model_path, args_from_metadata).to(device)

    # 2. Download and Process Data
    # We need enough data for seq_len inputs for the *last* day of `days_to_check`
    # The feature processing itself might require earlier data (e.g. for 50-day MAs)
    # Let's be generous with `seq_len_needed_for_features`
    seq_len_for_model_input = args_from_metadata.seq_len
    max_feature_lookback = 100 # Estimate of max lookback used in any feature
    
    # The MTP input structure means we need `max(indices_to_predict)` extra history
    # for the shifts in `download_and_process_inference_data` (placeholder currently)
    # Effective history needed before first pred day = seq_len_for_model_input + max_feature_lookback + max(indices_to_predict)
    # This is complex due to the MTP input shifting; the placeholder `download_and_process_inference_data`
    # needs to be accurate. For now, assume it handles date ranges internally.
    
    all_mtp_input_features, true_chlov_returns, columns = download_and_process_inference_data(
        args_from_metadata,
        cli_args.start_date_data,
        cli_args.end_date_data,
        seq_len_needed=seq_len_for_model_input # This 'seq_len' is for the model's direct input window
    )
    # Ensure data is on the correct device
    all_mtp_input_features = all_mtp_input_features[:,:,-(cli_args.days_to_check+seq_len_for_model_input):].to(device)
    actual_close_returns = true_chlov_returns[0,:,:]

    # 3. Generate Predictions
    # `true_chlov_returns` shape: (total_time_for_backtest_plus_one, num_tickers)
    # `actual_1d_returns` derived from this will be (total_time_for_backtest, num_tickers)
    # `model_1day_predictions` should also be (total_time_for_backtest, num_tickers)
    
    if actual_close_returns.shape[0] < cli_args.days_to_check + 1:
        raise ValueError(f"Not enough true price data ({actual_close_returns.shape[0]} days) for {cli_args.days_to_check} days of backtesting. Need at least {cli_args.days_to_check + 1} price points.")

    actual_1d_returns_for_backtest_period = actual_close_returns[-cli_args.days_to_check:, :]

    # Generate model predictions for the same period
    model_1day_predictions_denorm = get_mtp_predictions_for_backtest(
        model, all_mtp_input_features, args_from_metadata, cli_args.days_to_check, device
    )
    # Ensure predictions and actuals have the same length for the backtest
    if model_1day_predictions_denorm.shape[0] != actual_1d_returns_for_backtest_period.shape[0]:
        raise ValueError(f"Mismatch in length between predictions ({model_1day_predictions_denorm.shape[0]}) and actuals ({actual_1d_returns_for_backtest_period.shape[0]}) for backtest.")

    # print("\n--- Plotting Predicted vs. Actual Returns ---")
    # plot_predicted_vs_actual_returns(
    #     predicted_returns=model_1day_predictions_denorm,
    #     actual_returns=actual_1d_returns_for_backtest_period,
    #     ticker_names=args_from_metadata.tickers,
    #     num_days_to_plot=min(200, cli_args.days_to_check), # Plot last 100 days or all if fewer
    #     plot_filename_prefix="MTP_returns_comparison"
    # )

    print(f"\n--- Running PRE-OPTIMIZATION Backtest (Trade if |PredReturn| > TxCost) ---")
    
    pre_optim_predictions = model_1day_predictions_denorm.cpu()
    pre_optim_actuals = actual_1d_returns_for_backtest_period.cpu()
    
    simple_trade_threshold = 10*cli_args.transaction_cost # Trade if signal is stronger than one-way cost

    pre_optim_portfolio_df, pre_optim_stats = run_trading_strategy_1day_signal(
        predictions_1day_ahead=pre_optim_predictions,
        actual_1d_returns=pre_optim_actuals,
        trade_threshold_up=simple_trade_threshold, # Go long if pred > cost
        trade_threshold_down=simple_trade_threshold, # Go short if pred < -cost (abs value of pred > cost)
        initial_capital=cli_args.initial_capital,
        transaction_cost_pct=0.0,#cli_args.transaction_cost,
        signal_horizon_name="1-day (Pre-Opt Simple Threshold)",
        verbose=cli_args.verbose_strategy # Use the command-line verbosity
    )

    if cli_args.plot_equity:
        if pre_optim_portfolio_df is not None and not pre_optim_portfolio_df.empty:
            plt.figure(figsize=(14, 7)) # Create a new figure for this plot
            
            # Plot Pre-Optimization Strategy
            pre_optim_equity_curve = (1 + pd.Series(pre_optim_portfolio_df['daily_return'])).cumprod()
            plt.plot(pre_optim_equity_curve.index, pre_optim_equity_curve.values, label=f"Pre-Opt Strategy (|Pred| > TxCost)")
            
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
            converted_t_value = (pre_optim_stats['Sharpe Ratio'] * math.sqrt(len(pre_optim_portfolio_df)))/(252**0.5)
            print(f"Converted t-value: {converted_t_value:.4f}")
            print(f"Win/Loss day ratio: {pre_optim_stats['Win/Loss Day Ratio']:.4f}")

            plt.title(f"Pre-Optimization Strategy vs. Buy & Hold (Tickers: {', '.join(args_from_metadata.tickers)})")
            plt.xlabel("Trading Day in Backtest Period")
            plt.ylabel("Cumulative Return (Normalized to 1)")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"pre_optimized_strategy_MTP_{'_'.join(args_from_metadata.tickers)}.png")
            print(f"Pre-optimization equity curve plot saved to pre_optimized_strategy_MTP_{'_'.join(args_from_metadata.tickers)}.png")
            plt.show()


    # 4. Bayesian Optimization for Thresholds
    optim_period_len = min(1300, cli_args.days_to_check)
    print(f"\n--- Optimizing Trading Thresholds for 1-day Signal (using first {optim_period_len} days of backtest) ---")
    
    OPTIMIZATION_PREDS_1DAY = model_1day_predictions_denorm[:optim_period_len].cpu()
    OPTIMIZATION_ACTUAL_1D_RETURNS = actual_1d_returns_for_backtest_period[:optim_period_len].cpu()
    OPTIMIZATION_INITIAL_CAPITAL = cli_args.initial_capital
    OPTIMIZATION_TRANSACTION_COST = cli_args.transaction_cost
    OPTIMIZATION_SIGNAL_HORIZON_NAME = "1-day (Optimized)"

    pbounds_asymmetric = {
        'long_threshold_raw': (0.000, 0.05), # Search range for long threshold
        'short_threshold_raw': (0.000, 0.05) # Search range for absolute short threshold
    }
    min_step_thresh = 0.0005 # Discretization step for thresholds

    optimizer = BayesianOptimization(
        f=lambda long_threshold_raw, short_threshold_raw: objective_function_mtp(long_threshold_raw, short_threshold_raw, min_step=min_step_thresh),
        pbounds=pbounds_asymmetric,
        random_state=1,
        verbose=2 # 0 (silent), 1 (steps), 2 (all)
    )
    optimizer.maximize(init_points=30, n_iter=10) # Adjust init_points and n_iter as needed

    best_params = optimizer.max['params']
    best_sharpe = optimizer.max['target']
    
    optimal_long_thresh = round(best_params['long_threshold_raw'] / min_step_thresh) * min_step_thresh
    optimal_short_thresh_abs = round(best_params['short_threshold_raw'] / min_step_thresh) * min_step_thresh

    print("\n--- Optimal Asymmetric Thresholds Found ---")
    print(f"Optimal Long Threshold: {optimal_long_thresh:.4f}")
    print(f"Optimal Short Threshold (absolute): {optimal_short_thresh_abs:.4f}")
    print(f"Achieved Sharpe Ratio on optimization period: {best_sharpe:.4f}")

    # 5. Run Final Backtest with Optimized Thresholds on the Full Period
    print("\n--- Running Final Backtest with Optimized Thresholds (Full Period) ---")
    final_portfolio_df, final_stats = run_trading_strategy_1day_signal(
        model_1day_predictions_denorm.cpu(),
        actual_1d_returns_for_backtest_period.cpu(),
        trade_threshold_up=optimal_long_thresh,
        trade_threshold_down=optimal_short_thresh_abs,
        initial_capital=cli_args.initial_capital,
        transaction_cost_pct=cli_args.transaction_cost,
        signal_horizon_name="1-day (Final Optimized)",
        verbose=cli_args.verbose_strategy 
    )

    # 6. Plotting (Optional)
    if cli_args.plot_equity:
        if final_portfolio_df is not None and not final_portfolio_df.empty:
            plt.figure(figsize=(14, 7))
            equity_curve = (1 + pd.Series(final_portfolio_df['daily_return'])).cumprod()
            plt.plot(equity_curve.index, equity_curve.values, label=f"Optimized Strategy (1-day Signal)")
            
            # Add Buy and Hold for comparison
            if len(args_from_metadata.tickers) > 0:
                if len(args_from_metadata.tickers) == 1:
                    buy_and_hold_returns_single_ticker = actual_1d_returns_for_backtest_period.cpu()[:,0]
                    buy_and_hold_equity_curve_single = (1 + buy_and_hold_returns_single_ticker).cumprod(dim=0)
                    plt.plot(equity_curve.index, buy_and_hold_equity_curve_single.numpy(), label=f"Buy & Hold {args_from_metadata.tickers[0]}", linestyle=":")
                else: # Multiple tickers, plot average B&H
                    buy_and_hold_returns_avg_tickers = actual_1d_returns_for_backtest_period.cpu().mean(dim=1)
                    buy_and_hold_equity_curve_avg = (1 + buy_and_hold_returns_avg_tickers).cumprod(dim=0)
                    plt.plot(equity_curve.index, buy_and_hold_equity_curve_avg.numpy(), label=f"Buy & Hold Avg of {len(args_from_metadata.tickers)} Tickers", linestyle=":")
            
            print(f"Annualized return: {final_stats['Annualized Return']:.4f}")
            print(f"Sharpe ratio: {final_stats['Sharpe Ratio']:.4f}")
            converted_t_value = (final_stats['Sharpe Ratio'] * math.sqrt(len(final_portfolio_df)))/(252**0.5)
            print(f"t-value: {converted_t_value:.4f}")
            print(f"Win/Loss day ratio: {final_stats['Win/Loss Day Ratio']:.4f}")

            plt.title(f"Optimized Strategy vs. Buy & Hold (Tickers: {', '.join(args_from_metadata.tickers)})")
            plt.xlabel("Trading Day in Backtest Period")
            plt.ylabel("Cumulative Return (Normalized to 1)")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"optimized_strategy_MTP_{'_'.join(args_from_metadata.tickers)}.png")
            print(f"Equity curve plot saved to optimized_strategy_MTP_{'_'.join(args_from_metadata.tickers)}.png")
            plt.show()