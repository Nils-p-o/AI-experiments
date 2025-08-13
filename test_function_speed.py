import torch
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt


# actual correct logic function
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
    return portfolio_values_ts, daily_portfolio_returns_ts


# Helper function to pre-compute allocation weights vectorially
def _get_allocation_weights(
    predictions: torch.Tensor,
    prediction_type: str = "regression",
    decision_type: str = "threshold",
    allocation_strategy: str = "equal",
    trade_threshold_up: float = 0.01,
    trade_threshold_down: float = 0.01,
) -> torch.Tensor:
    # This function is fully vectorized, no loops

    # 1. Get target positions (-1, 0, 1)
    if prediction_type == "classification":
        if decision_type == "argmax":
            predicted_classes = torch.argmax(predictions, dim=-1)
            longs = (predicted_classes == 2).double()
            shorts = (predicted_classes == 0).double() * -1
            target_positions = longs + shorts
        else:  # threshold
            longs = (predictions[..., 2] > trade_threshold_up).double()
            shorts = (predictions[..., 0] > trade_threshold_down).double() * -1
            target_positions = longs + shorts
    else:  # regression
        if decision_type == "threshold":
            longs = (predictions > trade_threshold_up).double()
            shorts = (predictions < -trade_threshold_down).double() * -1
            target_positions = longs + shorts
        else:  # sign
            target_positions = torch.sign(predictions)

    # 2. Convert to allocation weights
    if allocation_strategy == "equal":
        num_trades = torch.sum(target_positions != 0, dim=1, keepdim=True)
        # Avoid division by zero where there are no trades
        num_trades = torch.where(num_trades == 0, 1.0, num_trades)
        allocation_weights = target_positions / num_trades
    else:  # signal_strength
        # This is harder to fully vectorize without a loop if signals are mixed per day.
        # But can be done for cases where daily predictions are all positive or negative.
        # For simplicity, we'll note JIT is better for this complex case.
        # We will assume a simplified signal strength for this example
        abs_predictions = torch.abs(predictions)
        signal_strength_sum = torch.sum(
            abs_predictions * (target_positions != 0), dim=1, keepdim=True
        )
        signal_strength_sum = torch.where(
            signal_strength_sum == 0, 1.0, signal_strength_sum
        )
        allocation_weights = (target_positions * abs_predictions) / signal_strength_sum

    return allocation_weights


# @torch.jit.script
def ground_truth_strategy_trade_jit(
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

    allocation_weights = _get_allocation_weights(
        predictions,
        prediction_type,
        decision_type,
        allocation_strategy,
        trade_threshold_up,
        trade_threshold_down,
    )  # You would call the helper here

    for trade in range(potential_trades):
        allocations = allocation_weights[trade]  # Use pre-computed allocations

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
    return portfolio_values_ts, daily_portfolio_returns_ts


# Add the JIT decorator for maximum performance on the remaining loop
@torch.jit.script
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

    return portfolio_values_ts, daily_portfolio_returns_ts



def ground_truth_strategy(
    predictions: torch.Tensor,
    actual_movements: torch.Tensor,
    trade_threshold_up: float = 0.01,
    trade_threshold_down: float = 0.01,
    initial_capital: float = 100000.0,
    transaction_cost_pct: float = 0.0005,
    prediction_type: str = "regression",  # 'regression' or 'classification'
    decision_type: str = "threshold",  # 'argmax' or 'threshold'
    allocation_strategy: str = "equal",  # 'equal' or 'signal_strength'
):
    # predictions: [days, tickers, classes] or [days, tickers]
    # actual_movements: [days, tickers]

    predictions = predictions.cpu().to(torch.float64)
    actual_movements = actual_movements.cpu().to(torch.float64)
    num_days = actual_movements.shape[0]
    num_tickers = actual_movements.shape[1]
    portfolio_values_ts = torch.zeros(num_days + 1, dtype=torch.float64)
    portfolio_values_ts[0] = initial_capital
    daily_portfolio_returns_ts = torch.zeros(num_days, dtype=torch.float64)
    current_capital = initial_capital
    current_positions = torch.zeros(num_tickers, dtype=torch.float64)

    for day in range(num_days):
        # Determine the target positions for the current day based on predictions.
        target_positions = torch.zeros(num_tickers, dtype=torch.float64)
        daily_predictions = predictions[day]

        if prediction_type == "classification":
            if decision_type == "argmax":
                # Assumes 3 classes: 0 for hold, 1 for long, 2 for short
                predicted_classes = torch.argmax(daily_predictions, dim=-1)
                target_positions[predicted_classes == 2] = 1.0
                target_positions[predicted_classes == 0] = -1.0
            elif decision_type == "threshold":
                # Assumes probabilities for long (class 1) and short (class 2)
                # probs = torch.softmax(daily_predictions, dim=-1)
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

        # Calculate transaction costs only on the change in positions.
        position_change = allocations - current_positions
        transaction_costs = (
            torch.sum(torch.abs(position_change))
            * transaction_cost_pct
            * current_capital
        )
        current_capital -= transaction_costs

        # Calculate the portfolio's return for the day.
        daily_returns = (
            torch.sum(current_positions * actual_movements[day]) * current_capital
        )

        # Update the portfolio's capital.
        current_capital += daily_returns
        portfolio_values_ts[day + 1] = current_capital

        if portfolio_values_ts[day] > 0:
            daily_portfolio_returns_ts[day] = (
                current_capital / portfolio_values_ts[day]
            ) - 1

        # Update the positions for the next trading day.
        current_positions = allocations

    return portfolio_values_ts, daily_portfolio_returns_ts


def run_strategy_with_flexible_allocation(
    predictions_1day_ahead: torch.Tensor,
    actual_1d_returns: torch.Tensor,
    trade_threshold_up: float = 0.01,
    trade_threshold_down: float = 0.01,
    initial_capital: float = 100000.0,
    transaction_cost_pct: float = 0.0005,
    allocation_strategy: str = "equal",  # 'equal' or 'signal_strength'
    signal_horizon_name: str = "1-day signal with turnover costs",
    verbose: int = 0,
    prediction_type: str = "regression",  # 'regression' or 'classification'
    decision_type: str = "threshold",  # 'argmax' or 'threshold'
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
        print(
            f"Initial Capital: ${initial_capital:,.2f}, T.Cost (on turnover): {transaction_cost_pct*100:.4f}%"
        )

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
            if allocation_strategy == "equal":
                num_active_signals = active_signals_mask.sum()
                capital_per_trade = capital_at_start_of_day / num_active_signals
                target_positions_today[long_signals_mask] = capital_per_trade
                target_positions_today[short_signals_mask] = -capital_per_trade

            elif (
                allocation_strategy == "signal_strength"
            ):  # TODO add for classification
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
                    target_positions_today[active_signals_mask] = (
                        dollar_allocations * signs
                    )

            else:
                raise ValueError(
                    f"Unknown allocation_strategy: '{allocation_strategy}'. "
                    f"Choose from 'equal' or 'signal_strength'."
                )

        # --- P&L and Turnover Calculation (same as before) ---
        pnl_today = torch.sum(target_positions_today * realized_1d_returns_today)
        trade_delta = target_positions_today - positions_at_prev_eod
        turnover = torch.sum(torch.abs(trade_delta))
        transaction_cost_today = turnover * tc_pct_tensor
        eod_portfolio_value = (
            capital_at_start_of_day + pnl_today - transaction_cost_today
        )
        portfolio_values_ts[day_idx + 1] = eod_portfolio_value

        if capital_at_start_of_day.abs().item() > 1e-9:
            daily_portfolio_returns_ts[day_idx] = (
                eod_portfolio_value / capital_at_start_of_day
            ) - 1.0
        else:
            daily_portfolio_returns_ts[day_idx] = torch.tensor(0.0, dtype=torch.float64)

        positions_at_prev_eod = target_positions_today * (1 + realized_1d_returns_today)

    # --- Performance Statistics Calculation (same as before) ---
    portfolio_df = pd.DataFrame(
        {
            "portfolio_value": portfolio_values_ts[1:].numpy(),
            "daily_return": daily_portfolio_returns_ts.numpy(),
        }
    )

    final_capital = portfolio_values_ts[-1]

    if daily_portfolio_returns_ts.numel() == 0:  # Handles case where num_days = 0
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
            sharpe_ratio = (
                (annualized_return / annualized_volatility)
                if annualized_volatility > 1e-9
                else 0.0
            )
            downside_returns = daily_returns_np[daily_returns_np < 0]
            if downside_returns.size > 1:
                downside_volatility = np.std(downside_returns) * np.sqrt(252.0)
                sortino_ratio = (
                    (annualized_return / downside_volatility)
                    if downside_volatility > 1e-9
                    else 0.0
                )
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
            calmar_ratio = (
                (annualized_return / -max_drawdown)
                if max_drawdown < -0.04
                else (annualized_return / 0.04)
            )
        else:
            max_drawdown = 0.0
            calmar_ratio = 0.0

        num_winning_days = (daily_returns_np > 0).sum()
        num_losing_days = (daily_returns_np < 0).sum()
        win_loss_ratio = (
            num_winning_days / num_losing_days if num_losing_days > 0 else float("inf")
        )

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
        for k, v in stats_dict.items():
            print(f"{k}: {v}")
    return portfolio_df, stats_dict


def _vectorized_calculate_target_weights(
    predictions: torch.Tensor,
    trade_threshold_up: float,
    trade_threshold_down: float,
    allocation_strategy: str,
    prediction_type: str,
    decision_type: str,
) -> torch.Tensor:
    """
    Vectorized calculation of target portfolio weights for all days.
    This is performed once, before the main simulation loop.
    """
    if prediction_type == "regression":
        long_signals_mask = predictions > trade_threshold_up
        short_signals_mask = predictions < -trade_threshold_down
    elif prediction_type == "classification":
        if decision_type == "argmax":
            # Assuming predictions shape is [days, tickers, classes]
            class_predictions = predictions.argmax(dim=-1)
            long_signals_mask = class_predictions == 2
            short_signals_mask = class_predictions == 0
        elif decision_type == "threshold":
            # Assuming class order is [short, neutral, long]
            long_signals_mask = predictions[:, :, 2] > trade_threshold_up
            short_signals_mask = (
                predictions[:, :, 0] > trade_threshold_down
            )  # Note: using > for threshold on prob
    else:
        raise ValueError(f"Unknown prediction_type: '{prediction_type}'")

    active_signals_mask = long_signals_mask | short_signals_mask
    target_weights = torch.zeros(predictions.shape[:2], dtype=torch.float64)

    if allocation_strategy == "equal":
        num_active_signals_per_day = active_signals_mask.sum(
            dim=1, keepdim=True, dtype=torch.float64
        )
        # Avoid division by zero on days with no signals
        inv_num_active = torch.where(
            num_active_signals_per_day > 0, 1.0 / num_active_signals_per_day, 0.0
        )

        target_weights[long_signals_mask] = inv_num_active.expand_as(target_weights)[
            long_signals_mask
        ]
        target_weights[short_signals_mask] = -inv_num_active.expand_as(target_weights)[
            short_signals_mask
        ]

    elif allocation_strategy == "signal_strength":
        if prediction_type == "classification":
            raise NotImplementedError(
                "Signal strength allocation is not implemented for classification yet."
            )

        # Use absolute signal strength for weighting
        strengths = torch.abs(predictions) * active_signals_mask
        total_strength_per_day = strengths.sum(dim=1, keepdim=True)

        # Avoid division by zero
        inv_total_strength = torch.where(
            total_strength_per_day > 1e-12, 1.0 / total_strength_per_day, 0.0
        )

        proportions = strengths * inv_total_strength
        signs = torch.sign(predictions)
        target_weights = proportions * signs

    else:
        raise ValueError(f"Unknown allocation_strategy: '{allocation_strategy}'")

    return target_weights


def run_strategy_with_flexible_allocation_optimized(
    predictions_1day_ahead: torch.Tensor,
    actual_1d_returns: torch.Tensor,
    trade_threshold_up: float = 0.01,
    trade_threshold_down: float = 0.01,
    initial_capital: float = 100000.0,
    transaction_cost_pct: float = 0.0005,
    allocation_strategy: str = "equal",
    prediction_type: str = "regression",
    decision_type: str = "threshold",
    track_ticker_performance: bool = False,  # NEW: Flag to enable ticker tracking
    ticker_names: list = None,  # NEW: Optional list of ticker names for labeling
):
    """
    Optimized day-trading strategy simulation with optional per-ticker performance tracking.
    """
    # --- Initial Setup ---
    predictions_1day_ahead = predictions_1day_ahead.cpu().to(torch.float64)
    actual_1d_returns = actual_1d_returns.cpu().to(torch.float64)
    num_days, num_tickers = predictions_1day_ahead.shape[:2]

    portfolio_values_ts = torch.zeros(num_days + 1, dtype=torch.float64)
    portfolio_values_ts[0] = initial_capital

    positions_at_prev_eod = torch.zeros(num_tickers, dtype=torch.float64)
    tc_pct_tensor = torch.tensor(transaction_cost_pct, dtype=torch.float64)

    # --- NEW: Initialization for ticker tracking ---
    if track_ticker_performance:
        daily_pnl_per_ticker = torch.zeros(num_days, num_tickers, dtype=torch.float64)
        daily_tcost_per_ticker = torch.zeros(num_days, num_tickers, dtype=torch.float64)

    # --- Pre-calculate all target weights ---
    target_weights = _vectorized_calculate_target_weights(
        predictions=predictions_1day_ahead,
        trade_threshold_up=trade_threshold_up,
        trade_threshold_down=trade_threshold_down,
        allocation_strategy=allocation_strategy,
        prediction_type=prediction_type,
        decision_type=decision_type,
    )

    # --- SIMPLIFIED: Main simulation loop ---
    for day_idx in range(num_days):
        pnl_vector_today = positions_at_prev_eod * actual_1d_returns[day_idx]
        # capital at start of day + pnl today
        capital_after_pnl = portfolio_values_ts[day_idx] + torch.sum(pnl_vector_today)

        if capital_after_pnl <= 1e-9:  # Bankruptcy check
            portfolio_values_ts[day_idx + 1 :] = 0
            break

        target_positions_today = capital_after_pnl * target_weights[day_idx]
        # target of today - values of positions from yesterday (post price movement)
        trade_delta_vector = target_positions_today - (
            positions_at_prev_eod * (1 + actual_1d_returns[day_idx])
        )
        turnover = torch.sum(torch.abs(trade_delta_vector))
        transaction_cost_today = turnover * tc_pct_tensor

        portfolio_values_ts[day_idx + 1] = capital_after_pnl - transaction_cost_today

        positions_at_prev_eod = target_positions_today

        # --- NEW: Store and attribute per-ticker data ---
        if track_ticker_performance:
            daily_pnl_per_ticker[day_idx] = pnl_vector_today

            # Attribute transaction costs based on contribution to turnover
            if turnover > 1e-9:
                turnover_proportions = torch.abs(trade_delta_vector) / turnover
                tcost_vector_today = transaction_cost_today * turnover_proportions
                daily_tcost_per_ticker[day_idx] = tcost_vector_today

    # --- Performance Statistics Calculation (remains the same) ---
    # For brevity, this block is omitted, but you should copy it from your function
    stats_dict = {"Info": "Stats calculation omitted for brevity"}
    portfolio_df = pd.DataFrame({"portfolio_value": portfolio_values_ts[1:].numpy()})

    # --- NEW: Package and return per-ticker performance ---
    ticker_performance_dict = None
    if track_ticker_performance:
        if ticker_names is None:
            ticker_names = [f"Ticker_{i}" for i in range(num_tickers)]

        net_pnl_per_ticker = daily_pnl_per_ticker - daily_tcost_per_ticker

        ticker_performance_dict = {
            "gross_pnl": pd.DataFrame(
                daily_pnl_per_ticker.numpy(), columns=ticker_names
            ),
            "t_costs": pd.DataFrame(
                daily_tcost_per_ticker.numpy(), columns=ticker_names
            ),
            "net_pnl": pd.DataFrame(net_pnl_per_ticker.numpy(), columns=ticker_names),
        }

    return portfolio_df, stats_dict, ticker_performance_dict


@torch.jit.script
def _simulation_loop_jit(
    num_days: int,
    initial_capital: float,
    target_weights: torch.Tensor,
    actual_1d_returns: torch.Tensor,
    tc_pct: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    A JIT-compiled function to run the core path-dependent simulation.
    Note: JIT functions require type hints and cannot contain non-torch operations like pandas.
    """
    num_tickers = actual_1d_returns.shape[1]

    # Initialize tensors for tracking portfolio history
    portfolio_values_ts = torch.zeros(num_days + 1, dtype=torch.float64)
    portfolio_values_ts[0] = initial_capital
    daily_portfolio_returns_ts = torch.zeros(num_days, dtype=torch.float64)

    # Holds the dollar value of positions from the end of the previous day
    positions_eod_prev = torch.zeros(num_tickers, dtype=torch.float64)

    for day_idx in range(num_days):
        capital_at_start_of_day = portfolio_values_ts[day_idx]
        returns_today = actual_1d_returns[day_idx]

        # 1. Account for overnight P&L
        # The value of our holdings changes based on today's returns
        pnl_today = torch.sum(positions_eod_prev * returns_today)
        capital_after_pnl = capital_at_start_of_day + pnl_today

        # This is the current market value of the positions we started the day with
        sod_positions_value = positions_eod_prev * (1 + returns_today)

        # 2. Determine new target positions
        # Use capital *after* P&L for more accurate compounding
        target_positions_today = capital_after_pnl * target_weights[day_idx]

        # 3. Calculate turnover and transaction costs
        # Delta is based on the difference from the *current market value* of old positions
        trade_delta = target_positions_today - sod_positions_value
        turnover = torch.sum(torch.abs(trade_delta))
        transaction_cost_today = turnover * tc_pct

        # 4. Calculate End-of-Day (EOD) portfolio value
        eod_portfolio_value = capital_after_pnl - transaction_cost_today
        portfolio_values_ts[day_idx + 1] = eod_portfolio_value

        # Calculate daily return based on start-of-day capital
        if capital_at_start_of_day > 1e-9:
            daily_portfolio_returns_ts[day_idx] = (
                eod_portfolio_value / capital_at_start_of_day
            ) - 1.0
        else:
            daily_portfolio_returns_ts[day_idx] = 0.0

        # 5. Update state for the next day
        positions_eod_prev = target_positions_today

    return portfolio_values_ts, daily_portfolio_returns_ts


def run_strategy_with_flexible_allocation_jit(
    predictions_1day_ahead: torch.Tensor,
    actual_1d_returns: torch.Tensor,
    trade_threshold_up: float = 0.01,
    trade_threshold_down: float = 0.01,
    initial_capital: float = 100000.0,
    transaction_cost_pct: float = 0.0005,
    allocation_strategy: str = "equal",
    signal_horizon_name: str = "1-day signal with turnover costs",
    verbose: int = 0,
    prediction_type: str = "regression",
    decision_type: str = "threshold",
):
    """
    JIT-Optimized day-trading strategy simulation.
    """
    # --- Initial Setup ---
    predictions_1day_ahead = predictions_1day_ahead.cpu().to(torch.float64)
    actual_1d_returns = actual_1d_returns.cpu().to(torch.float64)
    num_days = predictions_1day_ahead.shape[0]

    if verbose > 0:
        print(
            f"\n--- Running JIT-Compiled Trading Strategy ({signal_horizon_name}) ---"
        )
        # ... (verbose printing) ...

    # --- 1. Pre-calculate weights (already optimized) ---
    target_weights = _vectorized_calculate_target_weights(
        predictions=predictions_1day_ahead,
        trade_threshold_up=trade_threshold_up,
        trade_threshold_down=trade_threshold_down,
        allocation_strategy=allocation_strategy,
        prediction_type=prediction_type,
        decision_type=decision_type,
    )

    # --- 2. Run the JIT-compiled simulation loop ---
    tc_pct_tensor = torch.tensor(transaction_cost_pct, dtype=torch.float64)
    portfolio_values_ts, daily_portfolio_returns_ts = _simulation_loop_jit(
        num_days=num_days,
        initial_capital=initial_capital,
        target_weights=target_weights,
        actual_1d_returns=actual_1d_returns,
        tc_pct=tc_pct_tensor,
    )

    # --- 3. Performance Statistics Calculation (unchanged) ---
    # --- Performance Statistics Calculation (same as before) ---
    portfolio_dict = {
        "portfolio_value": portfolio_values_ts[1:],
        "daily_return": daily_portfolio_returns_ts,
    }

    final_capital = portfolio_values_ts[-1]

    if daily_portfolio_returns_ts.numel() == 0:  # Handles case where num_days = 0
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
            sharpe_ratio = (
                (annualized_return / annualized_volatility)
                if annualized_volatility > 1e-9
                else 0.0
            )
            downside_returns = daily_returns_np[daily_returns_np < 0]
            if downside_returns.size > 1:
                downside_volatility = np.std(downside_returns) * np.sqrt(252.0)
                sortino_ratio = (
                    (annualized_return / downside_volatility)
                    if downside_volatility > 1e-9
                    else 0.0
                )
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
            calmar_ratio = (
                (annualized_return / -max_drawdown)
                if max_drawdown < -0.04
                else (annualized_return / 0.04)
            )
        else:
            max_drawdown = 0.0
            calmar_ratio = 0.0

        num_winning_days = (daily_returns_np > 0).sum()
        num_losing_days = (daily_returns_np < 0).sum()
        win_loss_ratio = (
            num_winning_days / num_losing_days if num_losing_days > 0 else float("inf")
        )

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
        for k, v in stats_dict.items():
            print(f"{k}: {v}")

    return portfolio_dict, stats_dict


# Simulate some predictions and actual returns
predictions = torch.randn(1000, 20, 3)  # 1000 days, 10 tickers, 3 classes
actual_returns = torch.randn(1000, 20) / 10  # 1000 days, 10 tickers

num_runs = 20
print(f"Running {num_runs} iterations of the ground truth strategy...")
start_time = time.time_ns()
for i in range(num_runs):
    base_values, daily_returns = ground_truth_strategy_trade(
        predictions=predictions,
        actual_movements=actual_returns,
        prediction_type="classification",
        decision_type="argmax",
        allocation_strategy="equal",
    )

print(
    f"Ground truth took {(time.time_ns() - start_time)/(num_runs*1e+6)} ms on average for {num_runs} runs."
)


print(f"Running {num_runs} iterations of the jit ground truth strategy...")
start_time = time.time_ns()
for i in range(num_runs):
    jit_values, daily_returns = ground_truth_strategy_trade_jit(
        predictions=predictions,
        actual_movements=actual_returns,
        prediction_type="classification",
        decision_type="argmax",
        allocation_strategy="equal",
    )

print(
    f"JIT ground truth took {(time.time_ns() - start_time)/(num_runs*1e+6)} ms on average for {num_runs} runs."
)

print(f"Running {num_runs} iterations of the per-stock strategy...")
start_time = time.time_ns()
for i in range(num_runs):
    per_stock_values, daily_returns = vectorized_per_stock_backtest(
        predictions=predictions,
        actual_movements=actual_returns,
        prediction_type="classification",
        decision_type="argmax",
    )

print(
    f"Per stock strategy took {(time.time_ns() - start_time)/(num_runs*1e+6)} ms on average for {num_runs} runs."
)


plt.plot(base_values, label="Ground Truth Portfolio Value")
plt.plot(jit_values, label="JIT Ground Truth Portfolio Value")
plt.xlabel("Days")
plt.ylabel("Portfolio Value")
plt.legend()
plt.show()

print(
    f"MSE between ground truth and JIT: {torch.mean((base_values - jit_values) ** 2).item()}"
)
