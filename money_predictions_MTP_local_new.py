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
from training.data_loaders.test_feats_stocks_time_series_2_MTP_new import (
    align_financial_dataframes,
    feature_time_data,
    download_with_retry,
    calculate_features,
    data_fix_ffill
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
    
    if type(model) == type(checkpoint):
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

    # time_data, time_columns = feature_time_data(indexes, target_dates, tickers)
    # data = torch.cat((data, time_data), dim=0)
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
    local_features_start_idx = data.shape[0] - num_of_non_global_norm_feats
    
    if (type(args.normalization_means) != torch.Tensor) or (type(args.normalization_stds) != torch.Tensor):
        args.normalization_means = torch.tensor(args.normalization_means).clone().detach().to(data.device)
        args.normalization_stds = torch.tensor(args.normalization_stds).clone().detach().to(data.device)

    means = args.normalization_means.view(data.shape[0] - num_of_non_global_norm_feats, 1, 1, -1).to(data.device)
    stds = args.normalization_stds.view(data.shape[0] - num_of_non_global_norm_feats, 1, 1, -1).to(data.device)
    data[:local_features_start_idx] = (data[:local_features_start_idx] - means) / (stds + 1e-8)
    
    data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    # local norm for non-global features
    data = data.unfold(2, args.seq_len, 1).contiguous() # shape (features, max_pred_horizon, time_steps, num_tickers, seq_len_needed)
    data = data.permute(0, 1, 2, 4, 3) # (features, max_pred_horizon, time_steps, seq_len_needed, num_tickers)

    if local_features_start_idx != data.shape[0] - len(time_columns):
        local_means = data[local_features_start_idx:-len(time_columns)].mean(dim=3, keepdim=True)
        local_stds = data[local_features_start_idx:-len(time_columns)].std(dim=3, keepdim=True)
        data[local_features_start_idx:-len(time_columns)] = (data[local_features_start_idx:-len(time_columns)] - local_means) / (local_stds + 1e-8)

        data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data, MTP_full_returns

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
            downside_returns = daily_returns_np[daily_returns_np < 0]
            downside_volatility = np.std(downside_returns) * np.sqrt(252.0)
            sortino_ratio = (annualized_return / downside_volatility) if downside_volatility > 1e-9 else 0.0
        else:
            annualized_return = 0.0
            annualized_volatility = 0.0
            sharpe_ratio = 0.0
            sortino_ratio = 0.0

        cumulative_returns_pd = (1 + pd.Series(daily_returns_np)).cumprod()
        # Ensure cumulative_returns_pd is not empty before calling .expanding or .min
        if not cumulative_returns_pd.empty:
            peak = cumulative_returns_pd.expanding(min_periods=1).max()
            drawdown = (cumulative_returns_pd / peak) - 1.0
            max_drawdown = drawdown.min()
            calmar_ratio = (annualized_return / -max_drawdown) if max_drawdown < -0.04 else (annualized_return / 0.04)
        else:
            max_drawdown = 0.0
            calmar_ratio = 0.0
        
        num_winning_days = (daily_returns_np > 0).sum()
        num_losing_days = (daily_returns_np < 0).sum()
        win_loss_ratio = num_winning_days / num_losing_days if num_losing_days > 0 else float("inf")


    stats_dict = {
        "Signal Horizon Used": signal_horizon_name,
        "Sharpe Threshold Up": trade_threshold_up,
        "Sharpe Threshold Down": trade_threshold_down,
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Calmar Ratio": calmar_ratio,
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
    verbose: int = 0,
    prediction_type: str = "regression", # 'regression' or 'classification'
    decision_type: str = "threshold", # 'argmax' or 'threshold'
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
    num_days = predictions_1day_ahead.shape[0]
    num_tickers = predictions_1day_ahead.shape[1]
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

        if prediction_type == "regression":
            long_signals_mask = signals_today > trade_thresh_up_tensor
            short_signals_mask = signals_today < -trade_thresh_down_tensor

        elif prediction_type == "classification":
            if decision_type == "argmax":
                long_signals_mask = signals_today.argmax(dim=-1) == 2
                short_signals_mask = signals_today.argmax(dim=-1) == 0
            elif decision_type == "threshold":
                long_signals_mask = signals_today[:, 2] > trade_thresh_up_tensor
                short_signals_mask = signals_today[:, 0] < -trade_thresh_down_tensor
        active_signals_mask = long_signals_mask | short_signals_mask
        
        target_positions_today = torch.zeros_like(positions_at_prev_eod)

        if torch.any(active_signals_mask):
            # --- NEW: ALLOCATION LOGIC BLOCK ---
            if allocation_strategy == 'equal':
                num_active_signals = active_signals_mask.sum()
                capital_per_trade = capital_at_start_of_day / num_active_signals
                target_positions_today[long_signals_mask] = capital_per_trade
                target_positions_today[short_signals_mask] = -capital_per_trade

            elif allocation_strategy == 'signal_strength': # TODO add for classification
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
            downside_returns = daily_returns_np[daily_returns_np < 0]
            if downside_returns.size > 1:
                downside_volatility = np.std(downside_returns) * np.sqrt(252.0)
                sortino_ratio = (annualized_return / downside_volatility) if downside_volatility > 1e-9 else 0.0
            else:
                sortino_ratio = 0.0
        else:
            annualized_return = 0.0
            annualized_volatility = 0.0
            sharpe_ratio = 0.0
            sortino_ratio = 0.0

        cumulative_returns_pd = (1 + pd.Series(daily_returns_np)).cumprod()
        if not cumulative_returns_pd.empty:
            peak = cumulative_returns_pd.expanding(min_periods=1).max()
            drawdown = (cumulative_returns_pd / peak) - 1.0
            max_drawdown = drawdown.min()
            calmar_ratio = (annualized_return / -max_drawdown) if max_drawdown < -0.04 else (annualized_return / 0.04)
        else:
            max_drawdown = 0.0
            calmar_ratio = 0.0
        
        num_winning_days = (daily_returns_np > 0).sum()
        num_losing_days = (daily_returns_np < 0).sum()
        win_loss_ratio = num_winning_days / num_losing_days if num_losing_days > 0 else float("inf")


    stats_dict = {
        "Signal Horizon Used": signal_horizon_name,
        "Sharpe Threshold Up": trade_threshold_up,
        "Sharpe Threshold Down": trade_threshold_down,
        "Total Return": total_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Calmar Ratio": calmar_ratio,
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
        allocation_strategy="equal",#"equal",
        signal_horizon_name=OPTIMIZATION_SIGNAL_HORIZON_NAME,
        verbose=0,
        prediction_type=OPTIMIZATION_PREDICTION_TYPE
    )
    sharpe = strategy_stats.get("Sharpe Ratio", 0.0)
    # sharpe = strategy_stats.get("Annualized Return", 0.0)
    # sharpe = strategy_stats.get("Sortino Ratio", 0.0)
    # sharpe = strategy_stats.get("Calmar Ratio", 0.0)
    return sharpe


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
                        outputs_viewed = raw_outputs.view(
                            model_input_batch.shape[0], max(args.indices_to_predict), (seq_len+1), len(args.tickers), 5 * args.num_classes
                        )

            outputs_permuted = outputs_viewed.permute(0, 2, 4, 1, 3)  # Shape: (batch, seq_len+1, 5_chlov, horizons, tickers)

            match args.prediction_type:
                case "regression":
                    outputs_permuted = outputs_permuted[:, -1, 0, :, :]  # Shape: (batch, horizons, num_tickers)
                    model_predictions_1day_list.append(outputs_permuted)
                case "classification":
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

def trading_metrics(actuals, predictions, args):
    metrics = {}
    portfolio_df, stats = run_strategy_with_flexible_allocation(
        predictions_1day_ahead=predictions,
        actual_1d_returns=actuals,
        trade_threshold_up=0.5,
        trade_threshold_down=0.5,
        initial_capital=1000,
        transaction_cost_pct=0.0,
        allocation_strategy="equal",
        signal_horizon_name="training",
        prediction_type=args.prediction_type,
        decision_type="argmax"
    )
    metrics["Annualized Return"] = stats["Annualized Return"]
    metrics["Sharpe Ratio"] = stats["Sharpe Ratio"]
    metrics["Sortino Ratio"] = stats["Sortino Ratio"]
    metrics["Calmar Ratio"] = stats["Calmar Ratio"]
    metrics["Max Drawdown"] = stats["Max Drawdown"]
    metrics["Win/Loss Day Ratio"] = stats["Win/Loss Day Ratio"]

    return metrics



# def calculate_metrics(actuals, predictions, args, ticker_names: list):
    # """
    # Calculates predictive quality metrics.
    # Returns two dictionaries: one for the portfolio and one for individual stocks.
    # """
    # portfolio_metrics = {}
    # individual_metrics = {}
    # num_tickers = predictions.shape[1]

    # # --- Overall Portfolio Metrics ---
    # if args.prediction_type == "regression":
    #     portfolio_metrics["MAE_Loss"] = torch.mean(torch.abs(predictions - actuals)).item()
    #     portfolio_metrics["Total_Accuracy"] = torch.mean((predictions.sign() == actuals.sign()).float()).item()
    
    # elif args.prediction_type == "classification":
    #     pred_classes_all = predictions.argmax(dim=-1)
    #     actual_classes_all = torch.ones_like(actuals, dtype=torch.long)
    #     actual_classes_all[actuals < -args.classification_threshold] = 0
    #     actual_classes_all[actuals > args.classification_threshold] = 2
    #     portfolio_metrics["Total_Accuracy"] = torch.mean((pred_classes_all == actual_classes_all).float()).item()

    # # --- Per-Stock Metrics ---
    # for i in range(num_tickers):
    #     ticker = ticker_names[i]
    #     preds_stock = predictions[:, i]
    #     actuals_stock = actuals[:, i]

    #     if args.prediction_type == "regression":
    #         individual_metrics[f"{ticker}_MAE_Loss"] = torch.mean(torch.abs(preds_stock - actuals_stock)).item()
    #         individual_metrics[f"{ticker}_Accuracy"] = torch.mean((preds_stock.sign() == actuals_stock.sign()).float()).item()

    #     elif args.prediction_type == "classification":
    #         pred_classes = preds_stock.argmax(dim=-1)
    #         actual_classes = torch.ones_like(actuals_stock, dtype=torch.long)
    #         actual_classes[actuals_stock < -args.classification_threshold] = 0
    #         actual_classes[actuals_stock > args.classification_threshold] = 2

    #         individual_metrics[f"{ticker}_Accuracy"] = torch.mean((pred_classes == actual_classes).float()).item()
            
    #         if (pred_classes == 2).sum().item() > 0:
    #             individual_metrics[f"{ticker}_Conf_Up"] = ((preds_stock[:, 2] * (pred_classes == 2)).sum() / (pred_classes == 2).sum()).item()
    #         if (pred_classes == 0).sum().item() > 0:
    #              individual_metrics[f"{ticker}_Conf_Down"] = ((preds_stock[:, 0] * (pred_classes == 0)).sum() / (pred_classes == 0).sum()).item()

    # return portfolio_metrics, individual_metrics


# def trading_metrics(actuals, predictions, args, ticker_names: list):
    # """
    # Calculates trading strategy metrics.
    # Returns two dictionaries: one for the portfolio and one for individual stocks.
    # """
    # portfolio_metrics = {}
    # individual_metrics = {}
    
    # # --- 1. Run Backtest on the Full Portfolio ---
    # _, portfolio_stats = run_strategy_with_flexible_allocation(
    #     predictions_1day_ahead=predictions,
    #     actual_1d_returns=actuals,
    #     trade_threshold_up=0.5 if args.prediction_type == "classification" else 0.0,
    #     trade_threshold_down=0.5 if args.prediction_type == "classification" else 0.0,
    #     initial_capital=1000,
    #     transaction_cost_pct=0.0,
    #     allocation_strategy="equal",
    #     signal_horizon_name="validation_portfolio",
    #     prediction_type=args.prediction_type,
    #     decision_type="argmax"
    # )

    # # Populate the portfolio metrics dictionary
    # for stat_name, stat_value in portfolio_stats.items():
    #     if not any(x in stat_name for x in ["Threshold", "Signal Horizon", "Final", "Total"]):
    #         portfolio_metrics[stat_name] = stat_value

    # # --- 2. Run Individual Backtest for Each Stock ---
    # num_tickers = predictions.shape[1]
    # for i in range(num_tickers):
    #     ticker = ticker_names[i]
        
    #     preds_stock = predictions[:, i].unsqueeze(1)
    #     actuals_stock = actuals[:, i].unsqueeze(1)

    #     _, stock_stats = run_strategy_with_flexible_allocation(
    #         predictions_1day_ahead=preds_stock,
    #         actual_1d_returns=actuals_stock,
    #         trade_threshold_up=0.5 if args.prediction_type == "classification" else 0.0,
    #         trade_threshold_down=0.5 if args.prediction_type == "classification" else 0.0,
    #         initial_capital=1000,
    #         transaction_cost_pct=0.0,
    #         allocation_strategy="equal",
    #         signal_horizon_name=f"validation_{ticker}",
    #         prediction_type=args.prediction_type,
    #         decision_type="argmax"
    #     )
        
    #     # Populate the individual metrics dictionary with prefixed keys
    #     for stat_name, stat_value in stock_stats.items():
    #         if not any(x in stat_name for x in ["Threshold", "Signal Horizon", "Final", "Total"]):
    #             individual_metrics[f"{ticker}_{stat_name}"] = stat_value

    # return portfolio_metrics, individual_metrics


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Make predictions with a trained MTP stock model.")
    # parser.add_argument("--model_path", type=str, default="good_models/MTP/goated/6tick_class/name=0-epoch=63-val_loss=0.00-v4.ckpt", help="Path to the trained .pth model file.")
    # parser.add_argument("--metadata_path", type=str, default="good_models/MTP/goated/6tick_class/hparams.yaml", help="Path to the metadata.json file for the model.")
    parser.add_argument("--model_path", type=str, default="good_models/MTP/bee6_local/curr_test/name=0-epoch=40-val_loss=0.00-v10.ckpt", help="Path to the trained .pth model file.")
    parser.add_argument("--metadata_path", type=str, default="good_models/MTP/bee6_local/curr_test/hparams.yaml", help="Path to the metadata.json file for the model.")
    parser.add_argument("--days_to_check", type=int, default=1300, help="Number of recent days to generate predictions for and backtest.")
    parser.add_argument("--start_date_data", type=str, default="2020-01-01", help="Start date for downloading historical data.")
    parser.add_argument("--end_date_data", type=str, default="2025-05-25", help="End date for downloading historical data (serves as backtest end).")
    parser.add_argument("--initial_capital", type=float, default=100000.0, help="Initial capital for backtesting.")
    parser.add_argument("--transaction_cost", type=float, default=0.0, help="Transaction cost percentage.")
    parser.add_argument("--plot_equity", type=bool, default=True, help="Plot the equity curve of the optimized strategy.")
    parser.add_argument("--verbose_strategy", type=int, default=0, help="Verbosity level for strategy run printouts (0: silent, 1: summary).")
    parser.add_argument("--plot_individual_returns", type=bool, default=True, help="Plot predicted vs. actual returns for each ticker.")
    parser.add_argument("--pred_day", type=int, default=1, help="MTP block prediction to use.")
    parser.add_argument("--average_predictions", type=bool, default=True, help="Average predictions across all timesteps of MTP.") # aligned of course
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for model inference.")

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
        cli_args.pred_day = max(args_from_metadata.indices_to_predict) # Use the last prediction day for averaging (just because of how i am implementing the predictions with padding)

    model = load_mtp_model(cli_args.model_path, args_from_metadata).to(device)

    # 2. Download and Process Data
    
    all_mtp_input_features, true_chlov_returns = download_and_process_inference_data(
        args_from_metadata,
        cli_args.start_date_data,
        cli_args.end_date_data,
        cli_args=cli_args
    )

    # Ensure data is on the correct device
    all_mtp_input_features = all_mtp_input_features[:,:,-cli_args.days_to_check:].to(device)
    actual_close_returns = true_chlov_returns[0,:,:]
    
    if actual_close_returns.shape[0] < cli_args.days_to_check + 1:
        raise ValueError(f"Not enough true price data ({actual_close_returns.shape[0]} days) for {cli_args.days_to_check} days of backtesting. Need at least {cli_args.days_to_check + 1} price points.")

    actual_1d_returns_for_backtest_period = actual_close_returns[-cli_args.days_to_check:, :]

    # Generate model predictions for the same period
    print(f"--- Generating model predictions for backtest period ({cli_args.days_to_check} days) ---")
    model_1day_predictions_denorm = get_mtp_predictions_for_backtest(
        model, all_mtp_input_features, args_from_metadata, cli_args.days_to_check, device, cli_args
    ) 
    # Ensure predictions and actuals have the same length for the backtest
    if model_1day_predictions_denorm.shape[0] != actual_1d_returns_for_backtest_period.shape[0]:
        raise ValueError(f"Mismatch in length between predictions ({model_1day_predictions_denorm.shape[0]}) and actuals ({actual_1d_returns_for_backtest_period.shape[0]}) for backtest.")


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
        for i in range(pre_optim_predictions.shape[1]):
            print(f"\n--- Running PRE-OPTIMIZATION Backtest for {args_from_metadata.tickers[i]} ---")
            individual_portfolio_df, individual_stats = run_strategy_with_flexible_allocation(
                predictions_1day_ahead=pre_optim_predictions[:, i].unsqueeze(1),
                actual_1d_returns=pre_optim_actuals[:, i].unsqueeze(1),
                trade_threshold_up=simple_trade_threshold,
                trade_threshold_down=simple_trade_threshold,
                initial_capital=cli_args.initial_capital,
                transaction_cost_pct=cli_args.transaction_cost,
                signal_horizon_name="1-day (Pre-Opt signal strength)",
                verbose=cli_args.verbose_strategy,
                allocation_strategy="equal",
                prediction_type=args_from_metadata.prediction_type,
                decision_type="argmax",
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
        allocation_strategy="equal", # "signal_strength"
        prediction_type=args_from_metadata.prediction_type,
        decision_type="argmax"
    )

    if args_from_metadata.prediction_type == "regression":
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
    OPTIMIZATION_PREDICTION_TYPE = args_from_metadata.prediction_type

    pbounds_asymmetric = {
        'long_threshold_raw': (0.0, 0.9999), # Search range for long thresh
        'short_threshold_raw': (0.0, 0.9999) # Search range for absolute short thresh
    }
    min_step_thresh = 0.0001 # Discretization step for thresholds

    for i in range(len(METADATA_ARGS.tickers)):
        pbounds_asymmetric[f'include_ticker_{i}_raw'] = (0.0, 1.0)

    optimizer = BayesianOptimization(
        f=objective_function_mtp,
        pbounds=pbounds_asymmetric,
        random_state=1,
        verbose=2 # 0 (silent), 1 (steps), 2 (all)
    )
    optimizer.maximize(init_points=100, n_iter=100) # Adjust as needed

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
        allocation_strategy="equal",
        prediction_type=args_from_metadata.prediction_type
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
                    buy_and_hold_returns_avg_tickers = actual_1d_returns_for_backtest_period[:, optimal_included_ticker_indices].cpu().mean(dim=1)
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