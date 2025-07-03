# money_predictions_MTP.py

# TODO i dont trust this code, but maybe?


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
from transformer_arch.money.money_former_MLA_DINT_cog_attn_2_MTP import Money_former_MLA_DINT_cog_attn_MTP
# from training.data_loaders.stocks_time_series_2_MTP import (
#     align_financial_dataframes,
#     feature_volatility_ret,
#     feature_ema,
#     feature_ppo,
#     calculate_volume_price_trend_standard, # Assuming you used standard VPT
#     calculate_close_line_values, # Assuming you have this
#     feature_time_data,
# )
from training.data_loaders.test_feats_stocks_time_series_2_MTP import (
    align_financial_dataframes,
    feature_volatility_ret,
    feature_ema,
    feature_ppo,
    calculate_volume_price_trend_standard, # Assuming you used standard VPT
    calculate_close_line_values, # Assuming you have this
    feature_time_data,
    feature_bollinger_bands_price_histogram
)
from bayes_opt import BayesianOptimization

# Global variables for Bayesian Optimization (will be set later)
OPTIMIZATION_PREDS_1DAY = None
OPTIMIZATION_ACTUAL_1D_RETURNS = None
OPTIMIZATION_INITIAL_CAPITAL = 100000.0
OPTIMIZATION_TRANSACTION_COST = 0.0005
OPTIMIZATION_SIGNAL_HORIZON_NAME = "1-day (Opt.)"

def download_with_retry(tickers, max_retries=5, delay_seconds=3, **kwargs):
    """
    Downloads data from Yahoo Finance with a robust retry mechanism that handles
    partial failures while preserving the default column structure (OHLCV, Ticker).

    Args:
        tickers (str or list): A single ticker string or a list of ticker strings.
        max_retries (int): The maximum number of download attempts.
        delay_seconds (int): The number of seconds to wait between retries.
        **kwargs: Additional keyword arguments to pass to yf.download().

    Returns:
        pd.DataFrame: A DataFrame containing the data. For multiple tickers,
                      columns are a MultiIndex with OHLCV at level 0 and tickers at level 1.

    Raises:
        Exception: Re-raises the last caught exception if all retries fail.
    """
    requested_tickers = [tickers] if isinstance(tickers, str) else tickers
    last_exception = None

    if 'progress' not in kwargs:
        kwargs['progress'] = False

    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries} to download data for: {requested_tickers}...")
            # DO NOT use group_by='ticker' to preserve the desired (OHLCV, Ticker) structure
            data = yf.download(tickers, **kwargs)

            # --- ROBUST VALIDATION STEP ---

            # 1. Check for a completely empty DataFrame (total failure)
            if data.empty:
                raise ValueError("Downloaded data is empty.")

            # 2. Check for partial failures by validating each ticker's data
            failed_tickers = []
            if len(requested_tickers) > 1:
                # Multi-ticker case: DataFrame has MultiIndex columns
                downloaded_tickers = data.columns.get_level_values(1)
                for ticker in requested_tickers:
                    if ticker not in downloaded_tickers:
                        # Ticker is completely missing from the result
                        failed_tickers.append(ticker)
                    else:
                        # Ticker is present, check if its data is all NaN
                        # Use xs() to select the cross-section for this ticker
                        ticker_data = data.xs(ticker, level=1, axis=1)
                        if ticker_data.isnull().all().all():
                            failed_tickers.append(ticker)
            else:
                # Single-ticker case: DataFrame has simple columns
                if data.isnull().all().all():
                    failed_tickers.append(requested_tickers[0])

            if failed_tickers:
                raise ValueError(f"Data for tickers failed (all NaN or missing): {failed_tickers}")
            
            # If we reach here, all checks passed.
            print("Download successful for all requested tickers.")
            return data

        except Exception as e:
            print(f"Download attempt {attempt + 1} failed: {e}")
            last_exception = e
            if attempt < max_retries - 1:
                print(f"Retrying in {delay_seconds} seconds...")
                time.sleep(delay_seconds)
            else:
                print(f"Failed to download data after {max_retries} attempts.")
                if last_exception:
                    raise last_exception
                raise Exception("All download retries failed.")
    
    return pd.DataFrame()

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
    returns_columns = ["close_returns", "high_returns", "low_returns", "open_returns", "volume_returns"]
    price_columns = ["close", "high", "low", "open", "volume"]
    columns = []
    columns.extend(returns_columns)

    full_data = (raw_data[:, 1:, :] - raw_data[:, :-1, :]) / (raw_data[:, :-1, :] + 1e-9)
    full_data = torch.cat((torch.zeros_like(full_data[:, 0:1, :]), full_data), dim=1)
    full_data = torch.nan_to_num(full_data, nan=0.0, posinf=0.0, neginf=0.0) # Robust NaN/inf handling

    vol_data, vol_columns = feature_volatility_ret(returns=full_data[0:1], prefix="close_returns_")
    full_data = torch.cat((full_data, vol_data), dim=0)
    columns.extend(vol_columns)

    full_ema = []
    full_ema_columns = []
    for i in range(5):
        temp_ema = []
        for j in range(len(tickers)):
            ema_data, ema_columns = feature_ema(full_data[i, :, j], columns[i] + "_")
            temp_ema.append(ema_data.unsqueeze(-1))
        full_ema.append(torch.cat(temp_ema, dim=-1))
        full_ema_columns.extend(ema_columns)
    full_data = torch.cat((full_data, torch.cat(full_ema, dim=0)), dim=0)
    columns.extend(full_ema_columns)

    full_vpt = []
    vpt_data = calculate_volume_price_trend_standard(raw_data[:4], raw_data[4:])
    full_vpt.append(vpt_data)
    full_data = torch.cat((full_data, torch.cat(full_vpt, dim=0)), dim=0)
    columns.extend(["vpt_close", "vpt_high", "vpt_low", "vpt_open"])

    full_ppo = []
    full_ppo_columns = []
    for i in range(len(price_columns)):
        temp_ppo = []
        for j in range(len(tickers)):
            ppo_data, ppo_columns = feature_ppo(raw_data[i, :, j], prefix=price_columns[i] + "_")
            temp_ppo.append(ppo_data.unsqueeze(-1))
        full_ppo.append(torch.cat(temp_ppo, dim=-1))
        full_ppo_columns.extend(ppo_columns)
    full_data = torch.cat((full_data, torch.cat(full_ppo, dim=0)), dim=0)
    columns.extend(full_ppo_columns)

    clv_data = calculate_close_line_values(raw_data[0], raw_data[1], raw_data[2]).unsqueeze(0)
    full_data = torch.cat((full_data, clv_data), dim=0)
    columns.extend(["clv"])

    # --- Local Features Section ---
    local_columns = []
    prices = raw_data
    full_data = torch.cat((full_data, prices), dim=0)
    local_price_columns = ["local_" + s for s in price_columns]
    local_returns_columns = ["local_" + s for s in returns_columns]
    local_columns.extend(local_price_columns)

    full_ema = []
    full_ema_columns = []
    for i in range(len(local_price_columns)):
        temp_ema = []
        for j in range(len(tickers)):
            ema_data, ema_columns = feature_ema(prices[i, :, j], prefix=local_price_columns[i] + "_")
            temp_ema.append(ema_data.unsqueeze(-1))
        full_ema.append(torch.cat(temp_ema, dim=-1))
        full_ema_columns.extend(ema_columns)
    full_data = torch.cat((full_data, torch.cat(full_ema, dim=0)), dim=0)
    local_columns.extend(full_ema_columns)

    full_price_vol = []
    full_price_vol_columns = []
    for i in range(len(local_price_columns)):
        price_vol, vol_columns = feature_volatility_ret(prices[i:i+1], prefix=local_price_columns[i] + "_")
        full_price_vol.append(price_vol)
        full_price_vol_columns.extend(vol_columns)
    full_data = torch.cat((full_data, torch.cat(full_price_vol, dim=0)), dim=0)
    local_columns.extend(full_price_vol_columns)

    full_data = torch.cat((full_data, torch.cat(full_vpt, dim=0)), dim=0)
    local_columns.extend(["local_vpt_close", "local_vpt_high", "local_vpt_low", "local_vpt_open"])

    full_data = torch.cat((full_data, clv_data), dim=0)
    local_columns.extend(["local_clv"])

    full_ppo = []
    full_ppo_columns = []
    for i in range(len(local_price_columns)):
        temp_ppo = []
        for j in range(len(tickers)):
            ppo_data, ppo_columns = feature_ppo(prices[i, :, j], prefix=local_price_columns[i] + "_")
            temp_ppo.append(ppo_data.unsqueeze(-1))
        full_ppo.append(torch.cat(temp_ppo, dim=-1))
        full_ppo_columns.extend(ppo_columns)
    full_data = torch.cat((full_data, torch.cat(full_ppo, dim=0)), dim=0)
    local_columns.extend(full_ppo_columns)

    returns = full_data[:5, :, :]
    full_data = torch.cat((full_data, returns), dim=0)
    local_columns.extend(local_returns_columns)

    full_ema = []
    full_ema_columns = []
    for i in range(len(local_returns_columns)):
        temp_ema = []
        for j in range(len(tickers)):
            ema_data, ema_columns = feature_ema(returns[i, :, j], prefix=local_returns_columns[i] + "_")
            temp_ema.append(ema_data.unsqueeze(-1))
        full_ema.append(torch.cat(temp_ema, dim=-1))
        full_ema_columns.extend(ema_columns)
    full_data = torch.cat((full_data, torch.cat(full_ema, dim=0)), dim=0)
    local_columns.extend(full_ema_columns)

    full_ret_vol = []
    full_ret_vol_columns = []
    for i in range(len(local_returns_columns)):
        ret_vol, vol_columns = feature_volatility_ret(returns[i:i+1], prefix=local_returns_columns[i] + "_")
        full_ret_vol.append(ret_vol)
        full_ret_vol_columns.extend(vol_columns)
    full_data = torch.cat((full_data, torch.cat(full_ret_vol, dim=0)), dim=0)
    local_columns.extend(full_ret_vol_columns)

    bb_data, bb_columns = feature_bollinger_bands_price_histogram(prices[0:1], prefix=local_price_columns[0] + "_")
    full_data = torch.cat((full_data, bb_data), dim=0)
    local_columns.extend(bb_columns)

    bb_data, bb_columns = feature_bollinger_bands_price_histogram(returns[0:1], prefix=local_returns_columns[0] + "_")
    full_data = torch.cat((full_data, bb_data), dim=0)
    local_columns.extend(bb_columns)

    # --- MTP Input Stacking and Time Features ---
    data = torch.empty(full_data.shape[0], max(target_dates), full_data.shape[1] - max(target_dates), full_data.shape[2], dtype=torch.float32)
    for i in range(max(target_dates)):
        data[:, i, :, :] = full_data[:, i:-(max(target_dates) - i), :]
    
    # Replicate the lookback removal from the data loader
    min_lookback_to_drop = 20
    data = data[:, :, min_lookback_to_drop:, :]

    time_data, time_columns = feature_time_data(indexes, target_dates, tickers)
    data = torch.cat((data, time_data), dim=0)
    columns.extend(local_columns)
    columns.extend(time_columns)
    
    # --- True Returns for Backtesting ---
    MTP_full_returns = (raw_data_tensor[1:, 1:, :] - raw_data_tensor[1:, :-1, :]) / (raw_data_tensor[1:, :-1, :] + 1e-8)
    MTP_full_returns = torch.nan_to_num(MTP_full_returns, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Align returns with feature data
    MTP_full_returns = MTP_full_returns[:, min_lookback_to_drop:, :]
    if max(target_dates) > 1:
        MTP_full_returns = MTP_full_returns[:, :-(max(target_dates) - 1), :]

    # --- Global Normalization (using loaded stats) ---
    num_of_non_global_norm_feats = len(local_columns) + len(time_columns)
    local_features_start_idx = data.shape[0] - num_of_non_global_norm_feats
    
    means = args.normalization_means.view(data.shape[0] - num_of_non_global_norm_feats, 1, 1, -1)
    stds = args.normalization_stds.view(data.shape[0] - num_of_non_global_norm_feats, 1, 1, -1)
    data[:local_features_start_idx] = (data[:local_features_start_idx] - means) / (stds + 1e-8)
    
    data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    # Return globally-normalized data. Local norm will happen per-sequence.
    return data, MTP_full_returns, columns

def run_trading_strategy_1day_signal_simple(
    predictions_1day_ahead: torch.Tensor, # Model's predicted 1-day returns
    actual_1d_returns: torch.Tensor,      # Actual 1-day returns for P&L calculation
    trade_threshold_up: float =0.01,
    trade_threshold_down: float =0.01, # Absolute value for short threshold
    initial_capital: float =100000.0,
    transaction_cost_pct: float =0.0005,
    signal_horizon_name: str ="1-day",
    verbose: int = 0
):
    """
    Simulates a trading strategy using 1-day predictions, executing trades
    evaluated on the next 1-day actual return. Allows for asymmetric thresholds.
    """
    # Ensure inputs are on CPU for numpy operations if not already
    predictions_1day_ahead = predictions_1day_ahead.cpu().float()
    actual_1d_returns = actual_1d_returns.cpu().float()

    num_days, num_tickers = predictions_1day_ahead.shape
    if num_days != actual_1d_returns.shape[0] or num_tickers != actual_1d_returns.shape[1]:
        raise ValueError(
            "Predictions and actual 1-day returns must have the same shape (num_days, num_tickers)."
        )

    # portfolio_values = [initial_capital] # Start with initial capital before day 1
    # daily_portfolio_returns = []
    portfolio_values_ts = torch.zeros(num_days + 1, dtype=torch.float64)
    portfolio_values_ts[0] = initial_capital
    daily_portfolio_returns_ts = torch.zeros(num_days, dtype=torch.float64)
    current_capital = initial_capital
    
    if verbose > 0:
        print(f"\n--- Running Trading Strategy (Signal: {signal_horizon_name}) ---")
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"Trade Threshold UP: {trade_threshold_up*100:.3f}%")
        print(f"Trade Threshold DOWN (abs): {trade_threshold_down*100:.3f}%")
        print(f"Transaction Cost (per trade leg): {transaction_cost_pct*100:.4f}%")

    for day_idx in range(num_days):
        # Capital at the start of the day from the tensor
        capital_at_start_of_day = portfolio_values_ts[day_idx] # This is a torch.float64 scalar tensor

        daily_pnl_tensor = torch.tensor(0.0, dtype=torch.float64) # Accumulate P&L as a tensor
        num_long_trades_today = 0
        num_short_trades_today = 0

        signals_today = predictions_1day_ahead[day_idx]       
        realized_1d_returns_today = actual_1d_returns[day_idx] 

        active_signals = []
        for ticker_idx in range(num_tickers):
            # Ensure thresholds are float64 for comparison if signals_today is float64
            if signals_today[ticker_idx] > torch.tensor(trade_threshold_up, dtype=torch.float64):
                active_signals.append({"action": "long", "ticker_idx": ticker_idx})
            elif signals_today[ticker_idx] < -torch.tensor(trade_threshold_down, dtype=torch.float64): 
                active_signals.append({"action": "short", "ticker_idx": ticker_idx})

        if not active_signals:
            daily_portfolio_returns_ts[day_idx] = torch.tensor(0.0, dtype=torch.float64)
            portfolio_values_ts[day_idx + 1] = capital_at_start_of_day # No change
            # current_capital python float is no longer the primary tracker

            if verbose > 1 and (day_idx < 5 or day_idx == num_days -1 ):
                 print(f"Day {day_idx+1}: No trades. Capital: ${capital_at_start_of_day.item():,.2f}")
            continue

        # Ensure capital_per_trade is float64
        capital_per_trade = capital_at_start_of_day / torch.tensor(len(active_signals), dtype=torch.float64)

        for trade in active_signals:
            ticker_idx = trade["ticker_idx"]
            # Ensure invested_amount is float64
            invested_amount = capital_per_trade 
            
            # Ensure transaction_cost_pct is float64
            tc_pct_tensor = torch.tensor(transaction_cost_pct, dtype=torch.float64)

            cost_entry = invested_amount * tc_pct_tensor
            
            pnl_from_position_tensor = torch.tensor(0.0, dtype=torch.float64)
            if trade["action"] == "long":
                # realized_1d_returns_today[ticker_idx] is already float64
                pnl_from_position_tensor = invested_amount * realized_1d_returns_today[ticker_idx]
                num_long_trades_today +=1
            elif trade["action"] == "short":
                pnl_from_position_tensor = invested_amount * (-realized_1d_returns_today[ticker_idx])
                num_short_trades_today +=1
            
            value_before_exit = invested_amount + pnl_from_position_tensor
            cost_exit = value_before_exit * tc_pct_tensor

            net_pnl_for_trade_tensor = pnl_from_position_tensor - cost_entry - cost_exit
            daily_pnl_tensor += net_pnl_for_trade_tensor
        
        # Update portfolio value tensor
        portfolio_values_ts[day_idx + 1] = capital_at_start_of_day + daily_pnl_tensor
        
        # Calculate daily return tensor
        if capital_at_start_of_day.abs().item() > 1e-9:
            daily_portfolio_returns_ts[day_idx] = daily_pnl_tensor / capital_at_start_of_day
        else:
            daily_portfolio_returns_ts[day_idx] = torch.tensor(0.0, dtype=torch.float64)
        
        # For verbose printing, use .item() to get Python floats
        if verbose > 1 and (len(active_signals) > 0 or day_idx < 5 or day_idx == num_days - 1):
            print(f"Day {day_idx+1}: Longs: {num_long_trades_today}, Shorts: {num_short_trades_today}. Day P&L: ${daily_pnl_tensor.item():,.2f}. Day Ret: {daily_portfolio_returns_ts[day_idx].item()*100:.3f}%. EOD Capital: ${portfolio_values_ts[day_idx + 1].item():,.2f}")
    
    
    # --- Performance Statistics Calculation ---
    # portfolio_values is EOD capital, so its length is num_days + 1.
    # daily_portfolio_returns has length num_days.
    portfolio_df = pd.DataFrame({
        "portfolio_value": portfolio_values_ts[1:].numpy(), # Use EOD values for alignment with returns
        "daily_return": daily_portfolio_returns_ts.numpy()
    })
    # If using portfolio_values[0] in the df, then daily_return needs a 0 at the start.
    # For stats, usually just use the daily_returns series.

    if daily_portfolio_returns_ts.numel() == 0: # Handles case where num_days = 0
        total_return = 0.0
        annualized_return = 0.0
        annualized_volatility = 0.0
        sharpe_ratio = 0.0
        max_drawdown = 0.0
        num_winning_days = 0
        num_losing_days = 0
        win_loss_ratio = 0.0
    else:
        # daily_returns_np = np.array(daily_portfolio_returns)
        daily_returns_np = daily_portfolio_returns_ts.numpy()
        total_return = (portfolio_values_ts[-1] / initial_capital) - 1.0
        
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


def run_strategy_simple_with_turnover_costs(
    predictions_1day_ahead: torch.Tensor, # Model's predicted 1-day returns
    actual_1d_returns: torch.Tensor,      # Actual 1-day returns for P&L calculation
    trade_threshold_up: float = 0.01,
    trade_threshold_down: float = 0.01, # Absolute value for short threshold
    initial_capital: float = 100000.0,
    transaction_cost_pct: float = 0.0005,
    signal_horizon_name: str = "1-day signal with turnover costs",
    verbose: int = 0
):
    """
    Simulates a day-trading strategy identical to the 'simple' version in terms of P&L,
    but applies transaction costs only on the net change (turnover) in positions
    from one day to the next.

    This function WILL produce identical results to the simple version if transaction_cost_pct is 0.
    """
    # Ensure inputs are on CPU and have the correct data type for precision
    predictions_1day_ahead = predictions_1day_ahead.cpu().to(torch.float64)
    actual_1d_returns = actual_1d_returns.cpu().to(torch.float64)

    num_days, num_tickers = predictions_1day_ahead.shape
    if num_days != actual_1d_returns.shape[0] or num_tickers != actual_1d_returns.shape[1]:
        raise ValueError(
            "Predictions and actual 1-day returns must have the same shape (num_days, num_tickers)."
        )

    # --- State and Tracking Initialization ---
    portfolio_values_ts = torch.zeros(num_days + 1, dtype=torch.float64)
    portfolio_values_ts[0] = initial_capital
    daily_portfolio_returns_ts = torch.zeros(num_days, dtype=torch.float64)
    
    # State variable to track the dollar value of positions at the END of the previous day.
    # This is used ONLY for calculating turnover.
    positions_at_prev_eod = torch.zeros(num_tickers, dtype=torch.float64)
    
    tc_pct_tensor = torch.tensor(transaction_cost_pct, dtype=torch.float64)
    trade_thresh_up_tensor = torch.tensor(trade_threshold_up, dtype=torch.float64)
    trade_thresh_down_tensor = torch.tensor(trade_threshold_down, dtype=torch.float64)

    if verbose > 0:
        print(f"\n--- Running Trading Strategy ({signal_horizon_name}) ---")
        print(f"Initial Capital: ${initial_capital:,.2f}, T.Cost (on turnover): {transaction_cost_pct*100:.4f}%")

    for day_idx in range(num_days):
        capital_at_start_of_day = portfolio_values_ts[day_idx]
        signals_today = predictions_1day_ahead[day_idx]
        realized_1d_returns_today = actual_1d_returns[day_idx]

        # 1. Determine today's target positions based on today's signals and SOD capital.
        # This mirrors the logic of the 'simple' function exactly.
        long_signals = signals_today > trade_thresh_up_tensor
        short_signals = signals_today < -trade_thresh_down_tensor
        num_active_signals = long_signals.sum() + short_signals.sum()

        target_positions_today = torch.zeros_like(positions_at_prev_eod)
        if num_active_signals > 0:
            capital_per_trade = capital_at_start_of_day / num_active_signals
            target_positions_today[long_signals] = capital_per_trade
            target_positions_today[short_signals] = -capital_per_trade
        
        # 2. Calculate P&L for today based on these target positions.
        # This ensures P&L is identical to the 'simple' model's calculation.
        pnl_today = torch.sum(target_positions_today * realized_1d_returns_today)

        # 3. Calculate turnover and transaction costs.
        # This is the ONLY part that uses memory of the previous day's state.
        trade_delta = target_positions_today - positions_at_prev_eod
        turnover = torch.sum(torch.abs(trade_delta))
        transaction_cost_today = turnover * tc_pct_tensor

        # 4. Calculate final End-of-Day (EOD) portfolio value.
        eod_portfolio_value = capital_at_start_of_day + pnl_today - transaction_cost_today
        portfolio_values_ts[day_idx + 1] = eod_portfolio_value
        
        # 5. Calculate daily return.
        if capital_at_start_of_day.abs().item() > 1e-9:
            daily_portfolio_returns_ts[day_idx] = (eod_portfolio_value / capital_at_start_of_day) - 1.0
        else:
            daily_portfolio_returns_ts[day_idx] = torch.tensor(0.0, dtype=torch.float64)

        # 6. Update state for the next day. The new EOD positions are the target positions
        #    after being marked-to-market with today's returns.
        positions_at_prev_eod = target_positions_today * (1 + realized_1d_returns_today)

        if verbose > 1 and (day_idx < 5 or day_idx == num_days - 1 or turnover > 0):
             print(f"Day {day_idx+1: >3}: Start Cap: ${capital_at_start_of_day:11,.2f} | "
                   f"Day P&L: ${pnl_today:9,.2f} | "
                   f"Turnover: ${turnover:11,.2f} | "
                   f"T.Cost: ${transaction_cost_today:8,.2f} | "
                   f"EOD Cap: ${eod_portfolio_value:11,.2f}")

    # --- Performance Statistics Calculation ---
    # ... (The stats calculation part is the same and can be copied)
    portfolio_df = pd.DataFrame({
        "portfolio_value": portfolio_values_ts[1:].numpy(),
        "daily_return": daily_portfolio_returns_ts.numpy()
    })

    final_capital = portfolio_values_ts[-1]

    if daily_portfolio_returns_ts.numel() == 0: # Handles case where num_days = 0
        total_return = 0.0
        annualized_return = 0.0
        annualized_volatility = 0.0
        sharpe_ratio = 0.0
        max_drawdown = 0.0
        num_winning_days = 0
        num_losing_days = 0
        win_loss_ratio = 0.0
    else:
        # daily_returns_np = np.array(daily_portfolio_returns)
        daily_returns_np = daily_portfolio_returns_ts.numpy()
        total_return = (portfolio_values_ts[-1] / initial_capital) - 1.0
        
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
        "Final Capital": final_capital,
    }

    if verbose > 0:
        print(f"\n--- Strategy Summary ({signal_horizon_name}) ---")
        # (Full stats printing from previous answers)
        for k, v in stats_dict.items(): print(f"{k}: {v}")

    return portfolio_df, stats_dict

def run_strategy_with_flexible_allocation(
    predictions_1day_ahead: torch.Tensor,
    actual_1d_returns: torch.Tensor,
    trade_threshold_up: float = 0.01,
    trade_threshold_down: float = 0.01,
    initial_capital: float = 100000.0,
    transaction_cost_pct: float = 0.0005,
    allocation_strategy: str = 'equal', # 'equal' or 'signal_strength'
    signal_horizon_name: str = "1-day signal with turnover costs",
    verbose: int = 0
):
    """
    Simulates a day-trading strategy with transaction costs on turnover,
    offering multiple capital allocation strategies.

    Args:
        ... (previous args) ...
        allocation_strategy (str): Method for allocating capital.
            - 'equal': Divides capital equally among all active signals.
            - 'signal_strength': Allocates capital proportionally to the
              absolute strength of the signal prediction.
        ...
    """
    # --- Initial Setup (same as before) ---
    predictions_1day_ahead = predictions_1day_ahead.cpu().to(torch.float64)
    actual_1d_returns = actual_1d_returns.cpu().to(torch.float64)
    num_days, num_tickers = predictions_1day_ahead.shape
    portfolio_values_ts = torch.zeros(num_days + 1, dtype=torch.float64)
    portfolio_values_ts[0] = initial_capital
    daily_portfolio_returns_ts = torch.zeros(num_days, dtype=torch.float64)
    positions_at_prev_eod = torch.zeros(num_tickers, dtype=torch.float64)
    tc_pct_tensor = torch.tensor(transaction_cost_pct, dtype=torch.float64)
    trade_thresh_up_tensor = torch.tensor(trade_threshold_up, dtype=torch.float64)
    trade_thresh_down_tensor = torch.tensor(trade_threshold_down, dtype=torch.float64)

    if verbose > 0:
        print(f"\n--- Running Trading Strategy ({signal_horizon_name}) ---")
        print(f"Allocation Strategy: {allocation_strategy}")
        print(f"Initial Capital: ${initial_capital:,.2f}, T.Cost (on turnover): {transaction_cost_pct*100:.4f}%")

    for day_idx in range(num_days):
        capital_at_start_of_day = portfolio_values_ts[day_idx]
        signals_today = predictions_1day_ahead[day_idx]
        realized_1d_returns_today = actual_1d_returns[day_idx]

        long_signals_mask = signals_today > trade_thresh_up_tensor
        short_signals_mask = signals_today < -trade_thresh_down_tensor
        active_signals_mask = long_signals_mask | short_signals_mask
        
        target_positions_today = torch.zeros_like(positions_at_prev_eod)

        if torch.any(active_signals_mask):
            # --- NEW: ALLOCATION LOGIC BLOCK ---
            if allocation_strategy == 'equal':
                num_active_signals = active_signals_mask.sum()
                capital_per_trade = capital_at_start_of_day / num_active_signals
                target_positions_today[long_signals_mask] = capital_per_trade
                target_positions_today[short_signals_mask] = -capital_per_trade

            elif allocation_strategy == 'signal_strength':
                # Get the absolute strength of active signals for weighting
                active_signal_strengths = torch.abs(signals_today[active_signals_mask])
                total_weight = torch.sum(active_signal_strengths)

                if total_weight > 0:
                    # Allocate capital proportionally to signal strength
                    proportions = active_signal_strengths / total_weight
                    dollar_allocations = proportions * capital_at_start_of_day
                    
                    # Apply the correct sign (long/short)
                    # Get original signs of the active signals
                    signs = torch.sign(signals_today[active_signals_mask])
                    target_positions_today[active_signals_mask] = dollar_allocations * signs

            else:
                raise ValueError(f"Unknown allocation_strategy: '{allocation_strategy}'. "
                                 f"Choose from 'equal' or 'signal_strength'.")
        
        # --- P&L and Turnover Calculation (same as before) ---
        pnl_today = torch.sum(target_positions_today * realized_1d_returns_today)
        trade_delta = target_positions_today - positions_at_prev_eod
        turnover = torch.sum(torch.abs(trade_delta))
        transaction_cost_today = turnover * tc_pct_tensor
        eod_portfolio_value = capital_at_start_of_day + pnl_today - transaction_cost_today
        portfolio_values_ts[day_idx + 1] = eod_portfolio_value
        
        if capital_at_start_of_day.abs().item() > 1e-9:
            daily_portfolio_returns_ts[day_idx] = (eod_portfolio_value / capital_at_start_of_day) - 1.0
        else:
            daily_portfolio_returns_ts[day_idx] = torch.tensor(0.0, dtype=torch.float64)

        positions_at_prev_eod = target_positions_today * (1 + realized_1d_returns_today)
        # ... (verbose printing can remain the same) ...

    # --- Performance Statistics Calculation (same as before) ---
    portfolio_df = pd.DataFrame({
        "portfolio_value": portfolio_values_ts[1:].numpy(),
        "daily_return": daily_portfolio_returns_ts.numpy()
    })

    final_capital = portfolio_values_ts[-1]

    if daily_portfolio_returns_ts.numel() == 0: # Handles case where num_days = 0
        total_return = 0.0
        annualized_return = 0.0
        annualized_volatility = 0.0
        sharpe_ratio = 0.0
        max_drawdown = 0.0
        num_winning_days = 0
        num_losing_days = 0
        win_loss_ratio = 0.0
    else:
        # daily_returns_np = np.array(daily_portfolio_returns)
        daily_returns_np = daily_portfolio_returns_ts.numpy()
        total_return = (portfolio_values_ts[-1] / initial_capital) - 1.0
        
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
        "Final Capital": final_capital,
    }

    if verbose > 0:
        print(f"\n--- Strategy Summary ({signal_horizon_name}) ---")
        # (Full stats printing from previous answers)
        for k, v in stats_dict.items(): print(f"{k}: {v}")
    return portfolio_df, stats_dict

def run_strategy_with_risk_adjustment(
    predictions_1day_ahead: torch.Tensor,
    predicted_std_devs: torch.Tensor, # NEW: Pass in the predicted standard deviations
    actual_1d_returns: torch.Tensor,
    trade_threshold_up: float = 0.01,
    trade_threshold_down: float = 0.01,
    initial_capital: float = 100000.0,
    transaction_cost_pct: float = 0.0005,
    allocation_strategy: str = 'risk_adjusted', # 'equal', 'signal_strength', 'risk_adjusted'
    risk_aversion: float = 1.0, # NEW: Controls how much we penalize volatility
    signal_horizon_name: str = "1-day signal with risk adjustment",
    verbose: int = 0
):
    """
    Simulates a trading strategy with position sizing based on predicted risk (std dev).
    """
    # --- Initial Setup (same as before) ---
    predictions_1day_ahead = predictions_1day_ahead.cpu().to(torch.float64)
    predicted_std_devs = predicted_std_devs.cpu().to(torch.float64) # Ensure std_devs are processed
    actual_1d_returns = actual_1d_returns.cpu().to(torch.float64)
    num_days, num_tickers = predictions_1day_ahead.shape
    portfolio_values_ts = torch.zeros(num_days + 1, dtype=torch.float64)
    portfolio_values_ts[0] = initial_capital
    daily_portfolio_returns_ts = torch.zeros(num_days, dtype=torch.float64)
    positions_at_prev_eod = torch.zeros(num_tickers, dtype=torch.float64)
    tc_pct_tensor = torch.tensor(transaction_cost_pct, dtype=torch.float64)
    trade_thresh_up_tensor = torch.tensor(trade_threshold_up, dtype=torch.float64)
    trade_thresh_down_tensor = torch.tensor(trade_threshold_down, dtype=torch.float64)

    for day_idx in range(num_days):
        capital_at_start_of_day = portfolio_values_ts[day_idx]
        signals_today = predictions_1day_ahead[day_idx]
        std_devs_today = predicted_std_devs[day_idx]
        realized_1d_returns_today = actual_1d_returns[day_idx]

        long_signals_mask = signals_today > trade_thresh_up_tensor
        short_signals_mask = signals_today < -trade_thresh_down_tensor
        active_signals_mask = long_signals_mask | short_signals_mask
        
        target_positions_today = torch.zeros_like(positions_at_prev_eod)

        if torch.any(active_signals_mask):
            if allocation_strategy == 'risk_adjusted':
                # Calculate a weight for each active signal.
                # Here, we use signal strength / (std_dev^k)
                # A simple and effective weight is the predicted Sharpe Ratio: signal / std_dev
                # Add a small epsilon to std_dev to avoid division by zero
                risk_adjusted_weights = signals_today[active_signals_mask] / (std_devs_today[active_signals_mask]**risk_aversion + 1e-9)
                
                # We use the absolute value of these weights for allocation, but keep the sign for direction
                total_abs_weight = torch.sum(torch.abs(risk_adjusted_weights))

                if total_abs_weight > 1e-9:
                    # Proportions are based on the magnitude of the risk-adjusted signal
                    proportions = torch.abs(risk_adjusted_weights) / total_abs_weight
                    dollar_allocations = proportions * capital_at_start_of_day
                    
                    # Apply the correct sign (long/short) from the original signal
                    signs = torch.sign(signals_today[active_signals_mask])
                    target_positions_today[active_signals_mask] = dollar_allocations * signs

            elif allocation_strategy == 'equal':
                num_active_signals = active_signals_mask.sum()
                capital_per_trade = capital_at_start_of_day / num_active_signals
                target_positions_today[long_signals_mask] = capital_per_trade
                target_positions_today[short_signals_mask] = -capital_per_trade

            elif allocation_strategy == 'signal_strength':
                # Get the absolute strength of active signals for weighting
                active_signal_strengths = torch.abs(signals_today[active_signals_mask])
                total_weight = torch.sum(active_signal_strengths)

                if total_weight > 0:
                    # Allocate capital proportionally to signal strength
                    proportions = active_signal_strengths / total_weight
                    dollar_allocations = proportions * capital_at_start_of_day
                    
                    # Apply the correct sign (long/short)
                    # Get original signs of the active signals
                    signs = torch.sign(signals_today[active_signals_mask])
                    target_positions_today[active_signals_mask] = dollar_allocations * signs
            else:
                raise ValueError(f"Unknown allocation_strategy: '{allocation_strategy}'. "
                                 f"Choose from 'equal' or 'signal_strength' or 'risk_adjusted'.")

        # --- P&L and Turnover Calculation (same as before) ---
        pnl_today = torch.sum(target_positions_today * realized_1d_returns_today)
        trade_delta = target_positions_today - positions_at_prev_eod
        turnover = torch.sum(torch.abs(trade_delta))
        transaction_cost_today = turnover * tc_pct_tensor
        eod_portfolio_value = capital_at_start_of_day + pnl_today - transaction_cost_today
        portfolio_values_ts[day_idx + 1] = eod_portfolio_value
        
        if capital_at_start_of_day.abs().item() > 1e-9:
            daily_portfolio_returns_ts[day_idx] = (eod_portfolio_value / capital_at_start_of_day) - 1.0
        else:
            daily_portfolio_returns_ts[day_idx] = torch.tensor(0.0, dtype=torch.float64)

        positions_at_prev_eod = target_positions_today * (1 + realized_1d_returns_today)

    # --- Performance Statistics Calculation (same as before) ---
    portfolio_df = pd.DataFrame({
        "portfolio_value": portfolio_values_ts[1:].numpy(),
        "daily_return": daily_portfolio_returns_ts.numpy()
    })

    final_capital = portfolio_values_ts[-1]

    if daily_portfolio_returns_ts.numel() == 0:
        total_return = 0.0
        annualized_return = 0.0
        annualized_volatility = 0.0
        sharpe_ratio = 0.0
        max_drawdown = 0.0
        num_winning_days = 0
        num_losing_days = 0
        win_loss_ratio = 0.0
    else:
        # daily_returns_np = np.array(daily_portfolio_returns)
        daily_returns_np = daily_portfolio_returns_ts.numpy()
        total_return = (portfolio_values_ts[-1] / initial_capital) - 1.0
        
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
        "Final Capital": final_capital,
    }

    if verbose > 0:
        print(f"\n--- Strategy Summary ({signal_horizon_name}) ---")
        # (Full stats printing from previous answers)
        for k, v in stats_dict.items(): print(f"{k}: {v}")
    return portfolio_df, stats_dict

def calculate_probabilities(predictions_mean, predictions_std, threshold=0.0):
    """
    Calculates the probability of returns being above/below a threshold given a predicted normal distribution.

    Args:
        predictions_mean (torch.Tensor): The predicted mean returns (μ).
        predictions_std (torch.Tensor): The predicted standard deviations (σ).
        threshold (float): The profitability threshold (e.g., 0 for any positive return, or 0.0005 for transaction costs).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing (prob_up, prob_down).
    """
    # Ensure std is not zero to avoid division errors
    std_devs = predictions_std + 1e-9

    # Create a standard normal distribution object (mean=0, std=1) for CDF calculations
    standard_normal = torch.distributions.normal.Normal(0, 1)

    # Z-score for the UP threshold
    z_score_up = (threshold - predictions_mean) / std_devs
    # P(return > T) = 1 - P(return <= T) = 1 - CDF(z_score_up)
    prob_up = 1.0 - standard_normal.cdf(z_score_up)

    # Z-score for the DOWN threshold (e.g., P(return < -T))
    z_score_down = (-threshold - predictions_mean) / std_devs
    # P(return < -T) = CDF(z_score_down)
    prob_down = standard_normal.cdf(z_score_down)

    return prob_up, prob_down

def run_strategy_with_probability_sizing(
    predictions_1day_ahead: torch.Tensor,
    predicted_std_devs: torch.Tensor,
    actual_1d_returns: torch.Tensor,
    confidence_threshold: float = 0.55, # Trade if P > 55%
    initial_capital: float = 100000.0,
    transaction_cost_pct: float = 0.0005,
    signal_horizon_name: str = "1-day signal with probability sizing",
    verbose: int = 0
):
    """
    Simulates a trading strategy with position sizing based on the predicted probability of a profitable move.
    """
    # --- Initial Setup ---
    predictions_1day_ahead = predictions_1day_ahead.cpu().to(torch.float64)
    predicted_std_devs = predicted_std_devs.cpu().to(torch.float64)
    actual_1d_returns = actual_1d_returns.cpu().to(torch.float64)
    num_days, num_tickers = predictions_1day_ahead.shape
    portfolio_values_ts = torch.zeros(num_days + 1, dtype=torch.float64)
    portfolio_values_ts[0] = initial_capital
    daily_portfolio_returns_ts = torch.zeros(num_days, dtype=torch.float64)
    positions_at_prev_eod = torch.zeros(num_tickers, dtype=torch.float64)
    tc_pct_tensor = torch.tensor(transaction_cost_pct, dtype=torch.float64)

    for day_idx in range(num_days):
        capital_at_start_of_day = portfolio_values_ts[day_idx]
        
        # --- PROBABILITY CALCULATION ---
        # Calculate probabilities of crossing the transaction cost threshold
        prob_up, prob_down = calculate_probabilities(
            predictions_1day_ahead[day_idx],
            predicted_std_devs[day_idx],
            threshold=transaction_cost_pct
        )

        # --- SIGNAL GENERATION ---
        # We trade if the probability is above our confidence threshold
        long_signals_mask = prob_up > confidence_threshold
        short_signals_mask = prob_down > confidence_threshold
        active_signals_mask = long_signals_mask | short_signals_mask

        target_positions_today = torch.zeros_like(positions_at_prev_eod)

        if torch.any(active_signals_mask):
            # --- POSITION SIZING ---
            # Size is proportional to our "edge" or "conviction" (how far the probability is from 50%)
            # For long positions, conviction is (prob_up - 0.5)
            # For short positions, conviction is (prob_down - 0.5)
            long_conviction = (prob_up[long_signals_mask] - 0.5)
            short_conviction = (prob_down[short_signals_mask] - 0.5)
            
            # Combine all active convictions to get a total weight
            all_convictions = torch.cat([long_conviction, short_conviction])
            total_conviction_weight = torch.sum(all_convictions)
            
            if total_conviction_weight > 1e-9:
                # Allocate capital proportionally to conviction
                long_proportions = long_conviction / total_conviction_weight
                short_proportions = short_conviction / total_conviction_weight

                target_positions_today[long_signals_mask] = long_proportions * capital_at_start_of_day
                target_positions_today[short_signals_mask] = -short_proportions * capital_at_start_of_day

        # --- P&L and Turnover Calculation (same as before) ---
        pnl_today = torch.sum(target_positions_today * actual_1d_returns[day_idx])
        trade_delta = target_positions_today - positions_at_prev_eod
        turnover = torch.sum(torch.abs(trade_delta))
        transaction_cost_today = turnover * tc_pct_tensor
        eod_portfolio_value = capital_at_start_of_day + pnl_today - transaction_cost_today
        portfolio_values_ts[day_idx + 1] = eod_portfolio_value
        
        if capital_at_start_of_day.abs().item() > 1e-9:
            daily_portfolio_returns_ts[day_idx] = (eod_portfolio_value / capital_at_start_of_day) - 1.0
        else:
            daily_portfolio_returns_ts[day_idx] = torch.tensor(0.0, dtype=torch.float64)

        positions_at_prev_eod = target_positions_today * (1 + actual_1d_returns[day_idx])

    # --- Performance Statistics Calculation (omitted for brevity, same as other functions) ---
    portfolio_df = pd.DataFrame({
        "portfolio_value": portfolio_values_ts[1:].numpy(),
        "daily_return": daily_portfolio_returns_ts.numpy()
    })

    final_capital = portfolio_values_ts[-1]

    if daily_portfolio_returns_ts.numel() == 0: # Handles case where num_days = 0
        total_return = 0.0
        annualized_return = 0.0
        annualized_volatility = 0.0
        sharpe_ratio = 0.0
        max_drawdown = 0.0
        num_winning_days = 0
        num_losing_days = 0
        win_loss_ratio = 0.0
    else:
        # daily_returns_np = np.array(daily_portfolio_returns)
        daily_returns_np = daily_portfolio_returns_ts.numpy()
        total_return = (portfolio_values_ts[-1] / initial_capital) - 1.0
        
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
        "Confidence Threshold": confidence_threshold,
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Number of Trading Days": num_days,
        "Number of Winning Days": int(num_winning_days),
        "Number of Losing Days": int(num_losing_days),
        "Win/Loss Day Ratio": win_loss_ratio,
        "Final Capital": final_capital,
    }

    if verbose > 0:
        print(f"\n--- Strategy Summary ({signal_horizon_name}) ---")
        # (Full stats printing from previous answers)
        for k, v in stats_dict.items(): print(f"{k}: {v}")
    return portfolio_df, stats_dict

def run_strategy_with_sharpe_sizing(
    predictions_1day_ahead: torch.Tensor,
    predicted_std_devs: torch.Tensor,
    actual_1d_returns: torch.Tensor,
    sharpe_threshold_long: float = 0.2,   # Positive threshold to go long
    sharpe_threshold_short: float = -0.2, # Negative threshold to go short
    initial_capital: float = 100000.0,
    transaction_cost_pct: float = 0.0005,
    signal_horizon_name: str = "1-day signal with Sharpe sizing",
    verbose: int = 0
):
    """
    Simulates a trading strategy with position sizing based on the predicted Sharpe ratio (μ/σ).
    """
    # --- Initial Setup ---
    predictions_1day_ahead = predictions_1day_ahead.cpu().to(torch.float64)
    predicted_std_devs = predicted_std_devs.cpu().to(torch.float64)
    actual_1d_returns = actual_1d_returns.cpu().to(torch.float64)
    num_days, num_tickers = predictions_1day_ahead.shape
    portfolio_values_ts = torch.zeros(num_days + 1, dtype=torch.float64)
    portfolio_values_ts[0] = initial_capital
    daily_portfolio_returns_ts = torch.zeros(num_days, dtype=torch.float64)
    positions_at_prev_eod = torch.zeros(num_tickers, dtype=torch.float64)
    tc_pct_tensor = torch.tensor(transaction_cost_pct, dtype=torch.float64)

    # Validate thresholds to prevent logical errors
    if sharpe_threshold_long <= 0:
        raise ValueError("sharpe_threshold_long must be positive.")
    if sharpe_threshold_short >= 0:
        raise ValueError("sharpe_threshold_short must be negative.")

    for day_idx in range(num_days):
        capital_at_start_of_day = portfolio_values_ts[day_idx]
        
        # --- SHARPE RATIO CALCULATION ---
        # Calculate the predicted Sharpe ratio for today's signals
        predicted_sharpe_today = predictions_1day_ahead[day_idx] / (predicted_std_devs[day_idx] + 1e-9)

        # --- SIGNAL GENERATION ---
        # We trade if the predicted Sharpe ratio crosses our thresholds
        long_signals_mask = predicted_sharpe_today > sharpe_threshold_long
        short_signals_mask = predicted_sharpe_today < sharpe_threshold_short
        active_signals_mask = long_signals_mask | short_signals_mask

        target_positions_today = torch.zeros_like(positions_at_prev_eod)

        if torch.any(active_signals_mask):
            # --- POSITION SIZING ---
            # We will size positions based on the magnitude of the predicted Sharpe ratio.
            # This naturally allocates more capital to higher-conviction (higher Sharpe) trades.
            
            # Get the predicted Sharpe ratios for only the active signals
            active_sharpe_signals = predicted_sharpe_today[active_signals_mask]
            
            # The allocation weight is the absolute value of the Sharpe ratio
            abs_sharpe_weights = torch.abs(active_sharpe_signals)
            total_abs_weight = torch.sum(abs_sharpe_weights)
            
            if total_abs_weight > 1e-9:
                # Allocate capital proportionally to the absolute Sharpe ratio
                proportions = abs_sharpe_weights / total_abs_weight
                dollar_allocations = proportions * capital_at_start_of_day
                
                # Apply the correct sign (long/short). The sign of the Sharpe ratio is the sign of the trade.
                signs = torch.sign(active_sharpe_signals)
                target_positions_today[active_signals_mask] = dollar_allocations * signs

        # --- P&L and Turnover Calculation (same as before) ---
        pnl_today = torch.sum(target_positions_today * actual_1d_returns[day_idx])
        trade_delta = target_positions_today - positions_at_prev_eod
        turnover = torch.sum(torch.abs(trade_delta))
        transaction_cost_today = turnover * tc_pct_tensor
        eod_portfolio_value = capital_at_start_of_day + pnl_today - transaction_cost_today
        portfolio_values_ts[day_idx + 1] = eod_portfolio_value
        
        if capital_at_start_of_day.abs().item() > 1e-9:
            daily_portfolio_returns_ts[day_idx] = (eod_portfolio_value / capital_at_start_of_day) - 1.0
        else:
            daily_portfolio_returns_ts[day_idx] = torch.tensor(0.0, dtype=torch.float64)

        positions_at_prev_eod = target_positions_today * (1 + actual_1d_returns[day_idx])

    # --- Performance Statistics Calculation (omitted for brevity, same as other functions) ---
    portfolio_df = pd.DataFrame({
        "portfolio_value": portfolio_values_ts[1:].numpy(),
        "daily_return": daily_portfolio_returns_ts.numpy()
    })

    final_capital = portfolio_values_ts[-1]

    if daily_portfolio_returns_ts.numel() == 0: # Handles case where num_days = 0
        total_return = 0.0
        annualized_return = 0.0
        annualized_volatility = 0.0
        sharpe_ratio = 0.0
        max_drawdown = 0.0
        num_winning_days = 0
        num_losing_days = 0
        win_loss_ratio = 0.0
    else:
        # daily_returns_np = np.array(daily_portfolio_returns)
        daily_returns_np = daily_portfolio_returns_ts.numpy()
        total_return = (portfolio_values_ts[-1] / initial_capital) - 1.0
        
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
        "Sharpe Threshold Up": sharpe_threshold_long,
        "Sharpe Threshold Down": sharpe_threshold_short,
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Number of Trading Days": num_days,
        "Number of Winning Days": int(num_winning_days),
        "Number of Losing Days": int(num_losing_days),
        "Win/Loss Day Ratio": win_loss_ratio,
        "Final Capital": final_capital,
    }

    if verbose > 0:
        print(f"\n--- Strategy Summary ({signal_horizon_name}) ---")
        # (Full stats printing from previous answers)
        for k, v in stats_dict.items(): print(f"{k}: {v}")
    return portfolio_df, stats_dict


def objective_function_mtp(long_threshold_raw, short_threshold_raw, min_step=0.001, **ticker_include_params_raw):
    """Objective function for Bayesian Optimization for MTP (1-day signal)."""
    long_threshold = round(long_threshold_raw / min_step) * min_step
    short_threshold = round(short_threshold_raw / min_step) * min_step

    selected_ticker_indices = []
    for i in range(len(METADATA_ARGS.tickers)):
        param_name = f'include_ticker_{i}_raw'
        if ticker_include_params_raw.get(param_name, 0.0) > 0.5:
            selected_ticker_indices.append(i)
    
    if not selected_ticker_indices:
        # print("  BayesOpt: No tickers selected by parameters, defaulting to include all for this eval.")
        # Or penalize heavily:
        return -10.0 
    
    preds_for_selected_tickers = OPTIMIZATION_PREDS_1DAY[:, selected_ticker_indices]
    actual_returns_for_selected_tickers = OPTIMIZATION_ACTUAL_1D_RETURNS[:, selected_ticker_indices]


    _, strategy_stats = run_strategy_with_flexible_allocation(
        predictions_1day_ahead=preds_for_selected_tickers,
        actual_1d_returns=actual_returns_for_selected_tickers,
        trade_threshold_up=long_threshold,
        trade_threshold_down=short_threshold, 
        initial_capital=OPTIMIZATION_INITIAL_CAPITAL,
        transaction_cost_pct=OPTIMIZATION_TRANSACTION_COST,
        signal_horizon_name=OPTIMIZATION_SIGNAL_HORIZON_NAME,
        verbose=0, # Keep optimization quiet unless debugging
        # ticker_names=METADATA_ARGS.tickers,
        allocation_strategy="signal_strength"
    )
    sharpe = strategy_stats.get("Sharpe Ratio", 0.0)
    if sharpe < -5: # Penalize very bad Sharpe ratios
        return -10.0 
    return sharpe



# def get_mtp_predictions_for_backtest(model, all_mtp_input_features, args, nr_of_days_to_check, device):
#     """
#     Generates predictions for the backtest period using the MTP model.
#     This function now performs local z-normalization on each sequence before prediction.
#     Outputs 1-day ahead predictions for the 'Close' feature return.
#     """
#     seq_len = args.seq_len
#     num_pred_horizons_input = all_mtp_input_features.shape[1]

#     # --- Determine indices for local features dynamically from metadata ---
#     all_cols = args.columns
#     local_cols_start_index = -1
#     time_cols_start_index = -1
#     for i, col in enumerate(all_cols):
#         if 'local_' in col and local_cols_start_index == -1:
#             local_cols_start_index = i
#         if 'sin_day_of_week' in col and time_cols_start_index == -1: # First time feature
#             time_cols_start_index = i
    
#     if local_cols_start_index == -1 or time_cols_start_index == -1:
#         raise ValueError("Could not dynamically find start/end of local features in metadata columns.")

#     # Permute all_mtp_input_features for easier slicing
#     # Input shape: (features, horizon, time, tickers)
#     # Permuted shape: (horizon, time, tickers, features)
#     all_mtp_input_features_permuted = all_mtp_input_features.permute(1, 2, 3, 0).to(device)

#     model_predictions_1day_list = []

#     print(f"Generating predictions for {nr_of_days_to_check} days...")
#     for day_i in range(nr_of_days_to_check):
#         # Slice for the current window
#         # Shape: (horizon, seq_len, tickers, features)
#         current_window_globally_normed = all_mtp_input_features_permuted[:, day_i : day_i + seq_len, :, :]
        
#         # --- Perform Local Z-Normalization on the sliced window ---
#         # We normalize over the seq_len dimension (dim=1)
#         local_features_to_norm = current_window_globally_normed[:, :, :, local_cols_start_index:time_cols_start_index]
        
#         local_means = local_features_to_norm.mean(dim=1, keepdim=True)
#         local_stds = local_features_to_norm.std(dim=1, keepdim=True)
        
#         # Apply normalization
#         normalized_local_features = (local_features_to_norm - local_means) / (local_stds + 1e-8)
        
#         # Create the final model input tensor by replacing the local features part
#         final_model_input_window = current_window_globally_normed.clone()
#         final_model_input_window[:, :, :, local_cols_start_index:time_cols_start_index] = torch.nan_to_num(
#             normalized_local_features, nan=0.0
#         )
        
#         # Reshape for model: (batch_size=1, horizon, seq_len, tickers, features)
#         model_input_tensor = final_model_input_window.unsqueeze(0)
        
#         # Prepare separator and tickers for the model
#         # seperator_input = torch.zeros((1, num_pred_horizons_input, 1), dtype=torch.int, device=device)
#         tickers_input = torch.arange(len(args.tickers), device=device).unsqueeze(0).unsqueeze(0).repeat(1, num_pred_horizons_input, 1)

#         with torch.no_grad():
#             # --- MODEL FORWARD PASS & OUTPUT HANDLING (EXACTLY AS IN TRAINING) ---
#             # 1. Model returns a raw flattened tensor
#             raw_outputs = model(model_input_tensor, tickers_input)
            
#             # 2. Reshape the output exactly as in _shared_step
#             # Shape becomes: (batch, horizons, seq_len+1, tickers, 5_chlov)
#             outputs_viewed = raw_outputs.view(1, max(args.indices_to_predict), (seq_len+1), len(args.tickers), 5)
            
#             # 3. Permute the output exactly as in _shared_step
#             # Shape becomes: (batch, seq_len+1, 5_chlov, horizons, tickers)
#             outputs_permuted = outputs_viewed.permute(0, 2, 4, 1, 3)

#         # --- EXTRACT THE DESIRED PREDICTION FROM THE PERMUTED TENSOR ---
#         # We want the 1-day ahead prediction for the 'Close' return after the full sequence.
#         # - Dimension 0 (batch): index 0 (it's the only one)
#         # - Dimension 1 (seq_len+1): index -1 (the prediction for the step AFTER the last input)
#         # - Dimension 2 (5_chlov): index 0 ('Close' return)
#         # - Dimension 3 (horizons): index 0 (1-day prediction horizon)
#         # - Dimension 4 (tickers): index : (all tickers)
#         pred_1day_close_return_norm = outputs_permuted[0, -1, 0, 0, :] # Shape: (num_tickers)
#         model_predictions_1day_list.append(pred_1day_close_return_norm)

#     model_predictions_1day_tensor = torch.stack(model_predictions_1day_list, dim=0).to(device)

#     # --- De-normalize predictions using loaded stats ---
#     close_feature_original_index = 0 # 'close_returns' is the first feature
#     norm_means_for_close = args.normalization_means[close_feature_original_index, :].to(device)
#     norm_stds_for_close = args.normalization_stds[close_feature_original_index, :].to(device)

#     denormalized_predictions = model_predictions_1day_tensor * norm_stds_for_close.unsqueeze(0) + \
#                                norm_means_for_close.unsqueeze(0)
    
#     return denormalized_predictions

def get_mtp_predictions_for_backtest(model, all_mtp_input_features, args, nr_of_days_to_check, device):
    """
    Generates predictions for the backtest period using the MTP model.
    This function now performs local z-normalization on each sequence before prediction.
    ### CHANGED ###
    Outputs 1-day ahead predictions for 'Close' feature return AND its predicted standard deviation.
    """
    seq_len = args.seq_len
    num_pred_horizons_input = all_mtp_input_features.shape[1]

    # --- Determine indices for local features dynamically from metadata ---
    all_cols = args.columns
    local_cols_start_index = -1
    time_cols_start_index = -1
    for i, col in enumerate(all_cols):
        if 'local_' in col and local_cols_start_index == -1:
            local_cols_start_index = i
        if 'sin_day_of_week' in col and time_cols_start_index == -1: # First time feature
            time_cols_start_index = i
    
    if local_cols_start_index == -1 or time_cols_start_index == -1:
        raise ValueError("Could not dynamically find start/end of local features in metadata columns.")

    all_mtp_input_features_permuted = all_mtp_input_features.permute(1, 2, 3, 0).to(device)

    model_predictions_1day_list = []
    ### NEW ###
    model_std_devs_1day_list = []

    print(f"Generating predictions for {nr_of_days_to_check} days...")
    for day_i in range(nr_of_days_to_check):
        current_window_globally_normed = all_mtp_input_features_permuted[:, day_i : day_i + seq_len, :, :]
        
        local_features_to_norm = current_window_globally_normed[:, :, :, local_cols_start_index:time_cols_start_index]
        local_means = local_features_to_norm.mean(dim=1, keepdim=True)
        local_stds = local_features_to_norm.std(dim=1, keepdim=True)
        normalized_local_features = (local_features_to_norm - local_means) / (local_stds + 1e-8)
        
        final_model_input_window = current_window_globally_normed.clone()
        final_model_input_window[:, :, :, local_cols_start_index:time_cols_start_index] = torch.nan_to_num(
            normalized_local_features, nan=0.0
        )
        
        model_input_tensor = final_model_input_window.unsqueeze(0)
        
        tickers_input = torch.arange(len(args.tickers), device=device).unsqueeze(0).unsqueeze(0).repeat(1, num_pred_horizons_input, 1)

        with torch.no_grad():
            raw_outputs = model(model_input_tensor, tickers_input)
            
            ### CHANGED ###: The model now outputs 10 features (5 means, 5 log_stds)
            # The output shape is (batch, horizons, seq_len+1, tickers, 10)
            output_features = 10 if args.predict_gaussian else 5
            outputs_viewed = raw_outputs.view(1, max(args.indices_to_predict), (seq_len+1), len(args.tickers), output_features)
            
            # Shape becomes: (batch, seq_len+1, 10, horizons, tickers)
            outputs_permuted = outputs_viewed.permute(0, 2, 4, 1, 3)

            ### NEW ###: Split the output into mean and log_std, just like in the training script
            if args.predict_gaussian:
                mean_preds_norm, log_std_preds_norm = outputs_permuted.split([5, 5], dim=2)
            else:
                mean_preds_norm = outputs_permuted
                # Create a dummy tensor for std if not predicting gaussian for consistent flow
                log_std_preds_norm = torch.zeros_like(mean_preds_norm)


        # --- EXTRACT THE DESIRED PREDICTION FROM THE PERMUTED TENSOR ---
        # We want the 1-day ahead prediction for the 'Close' return after the full sequence.
        pred_1day_close_return_norm = mean_preds_norm[0, -1, 0, 0, :] # Shape: (num_tickers)
        model_predictions_1day_list.append(pred_1day_close_return_norm)

        ### NEW ###: Extract and store the corresponding standard deviation
        log_std_1day_close_return_norm = log_std_preds_norm[0, -1, 0, 0, :] # Shape: (num_tickers)
        std_dev_1day_close_return_norm = torch.exp(log_std_1day_close_return_norm)
        model_std_devs_1day_list.append(std_dev_1day_close_return_norm)


    model_predictions_1day_tensor = torch.stack(model_predictions_1day_list, dim=0).to(device)
    ### NEW ###
    model_std_devs_1day_tensor = torch.stack(model_std_devs_1day_list, dim=0).to(device)


    # --- De-normalize predictions using loaded stats ---
    close_feature_original_index = 0 # 'close_returns' is the first feature
    norm_means_for_close = args.normalization_means[close_feature_original_index, :].to(device)
    norm_stds_for_close = args.normalization_stds[close_feature_original_index, :].to(device)

    # De-normalize the mean (prediction)
    denormalized_predictions = model_predictions_1day_tensor * norm_stds_for_close.unsqueeze(0) + \
                               norm_means_for_close.unsqueeze(0)
    
    ### NEW ###: De-normalize the standard deviation
    # A standard deviation is a measure of scale, so we only multiply by the scaling factor. We don't add the mean.
    denormalized_std_devs = model_std_devs_1day_tensor * norm_stds_for_close.unsqueeze(0)
    
    ### CHANGED ###: Return both predictions and standard deviations
    return denormalized_predictions, denormalized_std_devs

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
    parser.add_argument("--model_path", type=str, default="good_models/MTP/3bee6_local_simple/name=0-epoch=30-val_loss=0.00-v57.ckpt", help="Path to the trained .pth model file.")
    parser.add_argument("--metadata_path", type=str, default="good_models/MTP/3bee6_local_simple/hparams.yaml", help="Path to the metadata.json file for the model.")
    parser.add_argument("--days_to_check", type=int, default=1300, help="Number of recent days to generate predictions for and backtest.")
    parser.add_argument("--start_date_data", type=str, default="2020-01-01", help="Start date for downloading historical data.")
    parser.add_argument("--end_date_data", type=str, default="2025-05-25", help="End date for downloading historical data (serves as backtest end).")
    parser.add_argument("--initial_capital", type=float, default=100000.0, help="Initial capital for backtesting.")
    parser.add_argument("--transaction_cost", type=float, default=0.0005, help="Transaction cost percentage.")
    parser.add_argument("--plot_equity", type=bool, default=True, help="Plot the equity curve of the optimized strategy.")
    parser.add_argument("--verbose_strategy", type=int, default=0, help="Verbosity level for strategy run printouts (0: silent, 1: summary).")
    parser.add_argument("--plot_individual_returns", type=bool, default=False, help="Plot predicted vs. actual returns for each ticker.")

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
    seq_len_for_model_input = args_from_metadata.seq_len
    max_feature_lookback = 100 # Estimate of max lookback used in any feature
    
    # The MTP input structure means we need `max(indices_to_predict)` extra history
    
    all_mtp_input_features, true_chlov_returns, columns = download_and_process_inference_data(
        args_from_metadata,
        cli_args.start_date_data,
        cli_args.end_date_data,
        seq_len_needed=seq_len_for_model_input # This 'seq_len' is for the model's direct input window
    )

    args_from_metadata.columns = columns

    # Ensure data is on the correct device
    all_mtp_input_features = all_mtp_input_features[:,:,-(cli_args.days_to_check+seq_len_for_model_input):].to(device)
    actual_close_returns = true_chlov_returns[0,:,:]
    
    if actual_close_returns.shape[0] < cli_args.days_to_check + 1:
        raise ValueError(f"Not enough true price data ({actual_close_returns.shape[0]} days) for {cli_args.days_to_check} days of backtesting. Need at least {cli_args.days_to_check + 1} price points.")

    actual_1d_returns_for_backtest_period = actual_close_returns[-cli_args.days_to_check:, :]

    # Generate model predictions for the same period
    # model_1day_predictions_denorm = get_mtp_predictions_for_backtest(
    #     model, all_mtp_input_features, args_from_metadata, cli_args.days_to_check, device
    # )
    model_1day_predictions_denorm, model_1day_std_devs_denorm = get_mtp_predictions_for_backtest(
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
    
    pre_optim_predictions = model_1day_predictions_denorm.cpu()
    pre_optim_actuals = actual_1d_returns_for_backtest_period.cpu()
    
    simple_trade_threshold = 0*cli_args.transaction_cost # Trade if signal is stronger than one-way cost

    if cli_args.plot_individual_returns:
        for i in range(pre_optim_predictions.shape[1]):
            print(f"\n--- Running PRE-OPTIMIZATION Backtest for {args_from_metadata.tickers[i]} ---")
            individual_portfolio_df, individual_stats = run_strategy_simple_with_turnover_costs(
                predictions_1day_ahead=pre_optim_predictions[:, i].unsqueeze(1),
                actual_1d_returns=pre_optim_actuals[:, i].unsqueeze(1),
                trade_threshold_up=simple_trade_threshold,
                trade_threshold_down=simple_trade_threshold,
                initial_capital=cli_args.initial_capital,
                transaction_cost_pct=cli_args.transaction_cost,
                signal_horizon_name="1-day (Pre-Opt signal strength)",
                verbose=cli_args.verbose_strategy
            )
            
            plt.figure(figsize=(14, 7)) # Create a new figure for this plot
            
            # Plot Pre-Optimization Strategy
            individual_equity_curve = (1 + pd.Series(individual_portfolio_df['daily_return'])).cumprod()
            plt.plot(individual_equity_curve.index, individual_equity_curve.values, label=f"Pre-Opt Strategy (|Pred| > TxCost)")
            
            # Add Buy and Hold for comparison (copied from your existing plot logic)
            buy_and_hold_returns_single_ticker = actual_1d_returns_for_backtest_period.cpu()[:,i]
            buy_and_hold_equity_curve_single = (1 + buy_and_hold_returns_single_ticker).cumprod(dim=0)
            plt.plot(individual_equity_curve.index, buy_and_hold_equity_curve_single.numpy(), label=f"Buy & Hold {args_from_metadata.tickers[i]}", linestyle=":")

            print(f"Annualized return: {individual_stats['Annualized Return']:.4f}")
            print(f"Sharpe ratio: {individual_stats['Sharpe Ratio']:.4f}")
            converted_t_value = (individual_stats['Sharpe Ratio'] * math.sqrt(len(individual_portfolio_df)))/(252**0.5)
            print(f"Converted t-value: {converted_t_value:.4f}")
            print(f"Win/Loss day ratio: {individual_stats['Win/Loss Day Ratio']:.4f}")

            plt.title(f"Pre-Optimization Strategy vs. Buy & Hold (Tickers: {', '.join(args_from_metadata.tickers)})")
            plt.xlabel("Trading Day in Backtest Period")
            plt.ylabel("Cumulative Return (Normalized to 1)")
            plt.legend()
            plt.grid(True)
            plt.show()
    
    tickers_to_include = [0,1,2,3,4,5]
    print("\n--- Running PRE-OPTIMIZATION Backtest (Trade if |PredReturn| > TxCost) ---")
    pre_optim_portfolio_df, pre_optim_stats = run_strategy_with_flexible_allocation(
        predictions_1day_ahead=pre_optim_predictions[:,tickers_to_include],
        actual_1d_returns=pre_optim_actuals[:,tickers_to_include],
        trade_threshold_up=simple_trade_threshold,
        trade_threshold_down=simple_trade_threshold,
        initial_capital=cli_args.initial_capital,
        transaction_cost_pct=cli_args.transaction_cost,
        signal_horizon_name="1-day (Pre-Opt signal strength)",
        verbose=cli_args.verbose_strategy,
        allocation_strategy="equal" # "signal_strength"
        # ticker_names=args_from_metadata.tickers, # Pass ticker names
        # capital_base_for_allocation="portfolio_value"
    )

    pre_optim_portfolio_df_simple, pre_optim_stats_simple = run_trading_strategy_1day_signal_simple(
        predictions_1day_ahead=pre_optim_predictions[:, tickers_to_include],
        actual_1d_returns=pre_optim_actuals[:, tickers_to_include],
        trade_threshold_up=simple_trade_threshold,
        trade_threshold_down=simple_trade_threshold,
        initial_capital=cli_args.initial_capital,
        transaction_cost_pct=cli_args.transaction_cost,
        signal_horizon_name="1-day (Pre-Opt Signal Strength)",
        verbose=cli_args.verbose_strategy        
    )

    pre_optim_equity_curve = (1 + pd.Series(pre_optim_portfolio_df['daily_return'])).cumprod()
    # print(f"new: \n{pre_optim_equity_curve.values}")
    pre_optim_equity_curve_simple = (1 + pd.Series(pre_optim_portfolio_df_simple['daily_return'])).cumprod()
    # print(f"simple: {pre_optim_equity_curve_simple.values}")
    print(f"diff: \n{pre_optim_equity_curve.values[-1] - pre_optim_equity_curve_simple.values[-1]}")
    # print(f"diff: \n{pre_optim_equity_curve.values - pre_optim_equity_curve_simple.values}")

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
    METADATA_ARGS = args_from_metadata
    OPTIMIZATION_PREDS_1DAY = model_1day_predictions_denorm[:optim_period_len].cpu()
    OPTIMIZATION_ACTUAL_1D_RETURNS = actual_1d_returns_for_backtest_period[:optim_period_len].cpu()
    OPTIMIZATION_INITIAL_CAPITAL = cli_args.initial_capital
    OPTIMIZATION_TRANSACTION_COST = cli_args.transaction_cost
    OPTIMIZATION_SIGNAL_HORIZON_NAME = "1-day (Optimized)"
    # OPTIMIZATION_CLOSE_THRESHOLD_PCT = 0.001
    # OPTIMIZATION_CAPITAL_BASE = "portfolio_value"

    pbounds_asymmetric = {
        'long_threshold_raw': (0.000, 0.1), # Search range for long thresh
        'short_threshold_raw': (0.000, 0.1) # Search range for absolute short thresh
    }
    min_step_thresh = 0.00001 # Discretization step for thresholds

    for i in range(len(METADATA_ARGS.tickers)):
        pbounds_asymmetric[f'include_ticker_{i}_raw'] = (0.0, 1.0)

    optimizer = BayesianOptimization(
        f=objective_function_mtp,
        pbounds=pbounds_asymmetric,
        random_state=1,
        verbose=2 # 0 (silent), 1 (steps), 2 (all)
    )
    optimizer.maximize(init_points=300, n_iter=100) # Adjust init_points and n_iter as needed

    best_params = optimizer.max['params']
    best_sharpe = optimizer.max['target']
    
    optimal_long_thresh = round(best_params['long_threshold_raw'] / min_step_thresh) * min_step_thresh
    optimal_short_thresh_abs = round(best_params['short_threshold_raw'] / min_step_thresh) * min_step_thresh

    optimal_included_ticker_indices = []
    included_tickers = []
    for i in range(len(METADATA_ARGS.tickers)):
        if best_params[f'include_ticker_{i}_raw'] > 0.5:
            included_tickers.append(METADATA_ARGS.tickers[i])
            optimal_included_ticker_indices.append(i)

    print("\n--- Optimal Asymmetric Thresholds Found ---")
    print(f"Optimal Long Threshold: {optimal_long_thresh:.4f}")
    print(f"Optimal Short Threshold (absolute): {optimal_short_thresh_abs:.4f}")
    print(f"Achieved Sharpe Ratio on optimization period: {best_sharpe:.4f}")
    print(f"Included Tickers: {included_tickers}")

    # 5. Run Final Backtest with Optimized Thresholds on the Full Period
    print("\n--- Running Final Backtest with Optimized Thresholds (Full Period) ---")
    final_portfolio_df, final_stats = run_strategy_with_flexible_allocation(
        model_1day_predictions_denorm[:, optimal_included_ticker_indices].cpu(),
        actual_1d_returns_for_backtest_period[:, optimal_included_ticker_indices].cpu(),
        trade_threshold_up=optimal_long_thresh,
        trade_threshold_down=optimal_short_thresh_abs,
        initial_capital=cli_args.initial_capital,
        transaction_cost_pct=cli_args.transaction_cost,
        signal_horizon_name="1-day (Final Optimized)",
        verbose=cli_args.verbose_strategy,
        allocation_strategy="signal_strength"
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