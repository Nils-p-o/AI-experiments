import numpy as np
import scipy.stats as stats
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator



shorthand_to_full = {
    "unseen_loss": "Losses_seen_unseen/val_loss_unseen",
    "backtest_loss": "Backtest_metrics/val_CE_Loss",
    "backtest_acc": "Backtest_metrics/val_Total_Accuracy",
    "f1": "Backtest_metrics/val_Total_F1",
    "calmar_ratio": "Trading_strategy_metrics/val_Calmar Ratio",
    "max_drawdown": "Trading_strategy_metrics/val_Max Drawdown",
    "annualized_return": "Trading_strategy_metrics/val_Annualized Return",
    "win_loss_ratio": "Trading_strategy_metrics/val_Win/Loss Day Ratio",
    "days_traded": "Trading_strategy_metrics/val_Days Traded",
    "aapl_acc": "Close_Sequence/val_unseen_AAPL_accuracy",
    "amzn_acc": "Close_Sequence/val_unseen_AMZN_accuracy",
    "intc_acc": "Close_Sequence/val_unseen_INTC_accuracy",
    "msft_acc": "Close_Sequence/val_unseen_MSFT_accuracy",
    "nvda_acc": "Close_Sequence/val_unseen_NVDA_accuracy",
    "gspc_acc": "Close_Sequence/val_unseen_^GSPC_accuracy",
    "calmar_ratio_costs": "Trading_strategy_metrics_with_costs/val_Calmar Ratio",
    "max_drawdown_costs": "Trading_strategy_metrics_with_costs/val_Max Drawdown",
    "annualized_return_costs": "Trading_strategy_metrics_with_costs/val_Annualized Return",
    "win_loss_ratio_costs": "Trading_strategy_metrics_with_costs/val_Win/Loss Day Ratio",
    "days_traded_costs": "Trading_strategy_metrics_with_costs/val_Days Traded",
}

higher_better = {
    "unseen_loss": 0,
    "backtest_loss": 0,
    "backtest_acc": 1,
    "f1": 1,
    "calmar_ratio": 1,
    "max_drawdown": 1,
    "annualized_return": 1,
    "win_loss_ratio": 1,
    "days_traded": 1,
    "aapl_acc": 1,
    "amzn_acc": 1,
    "intc_acc": 1,
    "msft_acc": 1,
    "nvda_acc": 1,
    "gspc_acc": 1,
    "calmar_ratio_costs": 1,
    "max_drawdown_costs": 1,
    "annualized_return_costs": 1,
    "win_loss_ratio_costs": 1,
    "days_traded_costs": 1,
}


def get_best_metrics(
    run_name: str,
    shorthand_metrics_list: list[str],
    version: int = 0,
    log_dir_base: str = "Money_logs",
):
    """
    Extracts the best value for a given list of metrics from a TensorBoard log file.

    Args:
        run_name (str): The name of the run, corresponding to the folder in the log directory.
                        e.g., "Money/MTP_classification/Money_former_MLA_DINT_cog_attn_MTP_128_256_1024_4_8_32_final"
        metrics_list (list[str]): A list of metric tags to extract from the logs.
                                  e.g., ["Loss/val_loss", "Trading_strategy_metrics/val_Calmar Ratio"]
        version (int, optional): The version number of the run. Defaults to 0.
        log_dir_base (str, optional): The base directory where logs are stored. Defaults to "Money_logs".

    Returns:
        dict: A dictionary where keys are metric names and values are another dictionary
              containing the 'best_value' and the 'step' at which it occurred.
              Returns None if the log directory is not found.
    """
    log_dir = os.path.join(log_dir_base, run_name, f"version_{version}")

    metrics_list = [
        shorthand_to_full[shorthand] for shorthand in shorthand_metrics_list
    ]

    if not os.path.exists(log_dir):
        print(f"Warning: Log directory not found at {log_dir}")
        return None

    # Find the tfevents file in the directory
    event_file = None
    try:
        event_file = [
            os.path.join(log_dir, f) for f in os.listdir(log_dir) if "tfevents" in f
        ][0]
    except IndexError:
        print(f"Warning: No tfevents file found in {log_dir}")
        return None

    # Initialize the EventAccumulator
    # Set size_guidance to 0 to load all data.
    accumulator = EventAccumulator(event_file, size_guidance={"scalars": 0})
    accumulator.Reload()

    available_tags = accumulator.Tags()["scalars"]
    results = {}

    for i in range(len(metrics_list)):
        if metrics_list[i] not in available_tags:
            continue

        events = accumulator.Scalars(metrics_list[i])
        if not events:
            continue

        # Extract steps and values, filtering out non-finite numbers
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])

        finite_mask = np.isfinite(values)
        if not np.any(finite_mask):
            continue  # Skip if all values are NaN or Inf

        steps = steps[finite_mask]
        values = values[finite_mask]

        # Determine if lower is better (for loss) or higher is better (for everything else)

        if not higher_better[shorthand_metrics_list[i]]:
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)

        best_value = values[best_idx]
        best_step = steps[best_idx]

        results[shorthand_metrics_list[i]] = { #round(float(best_value), 4)
        'best_value': round(float(best_value), 4),
        'step': int(best_step)
        }

    return results


def print_metrics(run_name: str, min_ver: int, max_ver: int):
    results = []
    full_results = {}
    for i in range(lowest_ver, max_ver + 1):
        results.append(get_best_metrics(current_name, list_of_losses, i))

        for metric_name, result in results[-1].items():
            if metric_name not in full_results:
                full_results[metric_name] = []
            full_results[metric_name].append(result)

    print(full_results)




# TODO mann-whitney u and wilcoxon signed rank tests

# list_of_losses = [
#     "backtest_loss",
#     "backtest_acc",
#     "f1",
#     "calmar_ratio",
#     "max_drawdown",
#     "annualized_return",
#     "win_loss_ratio",
#     "days_traded",
#     # "aapl_acc",
#     # "amzn_acc",
#     # "intc_acc",
#     # "msft_acc",
#     # "nvda_acc",
#     # "gspc_acc",
# ]

list_of_losses = [
    "backtest_loss",
    "backtest_acc",
    "f1",
    "calmar_ratio_costs",
    "max_drawdown_costs",
    "annualized_return_costs",
    "win_loss_ratio_costs",
    "days_traded_costs",
]

lowest_ver = 0
max_ver = 4
current_name = "Money/testing/fixed_mixing/testing_imbalance_weighted_stable_custom_linear_costs/Money_former_MLA_DINT_cog_attn_MTP_3_64_64_4_2_32"
print_metrics(current_name, lowest_ver, max_ver)

# diff models and their achieved metrics (max, or min depending on the metric)


# individual metrics
# best calmar by ticker
# TODO automate extracting best checkpoint and ver

# TODO check if combining multiple good models makes it better, or worse
# (combining logits, or probabilities)
# TODO ensemble

# but for now
# AAPL - ver_9 3199, 2.522
# alternatively - ver_11 3799, 2.462

# AMZN - ver_2 2299, 1.183
# 
# INTC - ver_28 2399, 1.406

# MSFT - ver_2 2299, 3.253
# ver_30 1199, 3.025
# ver_34 3599, 3.269
# ver_9 3199, 3.275
# ver_54 4099, 3.622
# ver_54 4199, 3.473

# NVDA - ver_50 99, 2.349
# ver_2 1899, 2.445
# ver_2 2299, 2.281
# 

# ^GSPC - ver_3 1399, 1.933
# ver_23 799, 1.676
# 