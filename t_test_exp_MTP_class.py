import numpy as np
import scipy.stats as stats
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def welch_t_test(a, b):
    t_value, p_value = stats.ttest_ind(a, b, equal_var=False)
    print(f"Welch's T statistic: {t_value}")
    print(f"Welch's T statistic p-value: {p_value}")
    return

shorthand_to_full = {
    "unseen_loss": "Losses_seen_unseen/val_loss_unseen",
    "backtest_loss": "Backtest_metrics/val_CE_loss",
    "calmar_ratio": "Trading_strategy_metrics/val_Calmar Ratio"

}

def get_best_metrics(run_name: str, shorthand_metrics_list: list[str], version: int=0, log_dir_base: str = "Money_logs"):
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

    metrics_list = [shorthand_to_full[shorthand] for shorthand in shorthand_metrics_list]

    if not os.path.exists(log_dir):
        print(f"Warning: Log directory not found at {log_dir}")
        return None

    # Find the tfevents file in the directory
    event_file = None
    try:
        event_file = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if "tfevents" in f][0]
    except IndexError:
        print(f"Warning: No tfevents file found in {log_dir}")
        return None

    # Initialize the EventAccumulator
    # Set size_guidance to 0 to load all data.
    accumulator = EventAccumulator(event_file, size_guidance={'scalars': 0})
    accumulator.Reload()

    available_tags = accumulator.Tags()['scalars']
    results = {}

    for metric_name in metrics_list:
        if metric_name not in available_tags:
            continue

        events = accumulator.Scalars(metric_name)
        if not events:
            continue

        # Extract steps and values, filtering out non-finite numbers
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])
        
        finite_mask = np.isfinite(values)
        if not np.any(finite_mask):
            continue # Skip if all values are NaN or Inf

        steps = steps[finite_mask]
        values = values[finite_mask]


        # Determine if lower is better (for loss) or higher is better (for everything else)
        lower_is_better_keywords = ['loss', 'mae', 'mse']
        is_lower_better = any(keyword in metric_name.lower() for keyword in lower_is_better_keywords)

        if is_lower_better:
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)

        best_value = values[best_idx]
        best_step = steps[best_idx]

        results[metric_name] = {
            # 'best_value': float(best_value),
            # 'step': int(best_step)
            float(best_value)
        }

    return results


# TODO paired by seeds for paired tests
# TODO mann-whitney u and wilcoxon signed rank tests

# list_of_losses = ["unseen_loss", "calmar_ratio"]

# lowest_ver = 1
# max_ver = 3
# current_name = "Money/testing/class_seq_2_1_no_ortho/Money_former_MLA_DINT_cog_attn_MTP_2_64_64_4_4_32"
# results = []
# full_results = {}
# for i in range(lowest_ver, max_ver+1):
#     results.append(get_best_metrics(current_name, list_of_losses, i))

#     for metric_name, result in results[-1].items():
#         if metric_name not in full_results:
#             full_results[metric_name] = []
#         full_results[metric_name].append(result)

# print(full_results)

# diff models and their achieved metrics (max, or min depending on the metric) 

#models with 4 layers, 64 dims, no ortho, full 3bee feats, 6 tickers 
seq_2_hor_1 = {
    "unseen_loss": np.array([0.825, 0.829, 0.8282]),
    "calmar_ratio": np.array([1.549, 0.786, 1.302]),
}
seq_2_hor_2 = { # seems like the best one? because the other extra horizons dont have better loss really
    "unseen_loss": np.array([0.8228, 0.8196]), # better loss?
    "calmar_ratio": np.array([1.249, 1.316]),
}
seq_2_hor_3 = {
    "unseen_loss": np.array([0.8215, 0.8210, 0.8216]),
    "calmar_ratio": np.array([1.566, 3.044, 0.804]),
}
seq_2_hor_4 = {
    "unseen_loss": np.array([0.8201, 0.8209, 0.8228]),
    "calmar_ratio": np.array([1.132, 1.337, 1.433]),
}
seq_2_hor_5 = {
    "unseen_loss": np.array([0.8207, 0.8230, 0.8211]),
    "calmar_ratio": np.array([0.882, 1.275, 1.190]),
}

seq_2_hor_10 = {
    "unseen_loss": np.array([]),
    "calmar_ratio": np.array([]),
}

print(f"seq_2_hor_1 unseen_loss: {np.mean(seq_2_hor_1['unseen_loss'])}")
print(f"seq_2_hor_1 calmar_ratio: {np.mean(seq_2_hor_1['calmar_ratio'])}")

print(f"seq_2_hor_2 unseen_loss: {np.mean(seq_2_hor_2['unseen_loss'])}")
print(f"seq_2_hor_2 calmar_ratio: {np.mean(seq_2_hor_2['calmar_ratio'])}")
print("\n--- seq_2_hor_1 vs seq_2_hor_2 ---")
welch_t_test(seq_2_hor_1['unseen_loss'], seq_2_hor_2['unseen_loss']) # maybe on this? (0.08 better)
welch_t_test(seq_2_hor_1['calmar_ratio'], seq_2_hor_2['calmar_ratio'])

print(f"seq_2_hor_3 unseen_loss: {np.mean(seq_2_hor_3['unseen_loss'])}")
print(f"seq_2_hor_3 calmar_ratio: {np.mean(seq_2_hor_3['calmar_ratio'])}")
print("\n--- seq_2_hor_2 vs seq_2_hor_3 ---")
welch_t_test(seq_2_hor_2['unseen_loss'], seq_2_hor_3['unseen_loss'])
welch_t_test(seq_2_hor_2['calmar_ratio'], seq_2_hor_3['calmar_ratio'])

print(f"seq_2_hor_4 unseen_loss: {np.mean(seq_2_hor_4['unseen_loss'])}")
print(f"seq_2_hor_4 calmar_ratio: {np.mean(seq_2_hor_4['calmar_ratio'])}")
print("\n--- seq_2_hor_2 vs seq_2_hor_4 ---")
welch_t_test(seq_2_hor_2['unseen_loss'], seq_2_hor_4['unseen_loss'])
welch_t_test(seq_2_hor_2['calmar_ratio'], seq_2_hor_4['calmar_ratio'])

print(f"seq_2_hor_5 unseen_loss: {np.mean(seq_2_hor_5['unseen_loss'])}")
print(f"seq_2_hor_5 calmar_ratio: {np.mean(seq_2_hor_5['calmar_ratio'])}")
print("\n--- seq_2_hor_2 vs seq_2_hor_5 ---")
welch_t_test(seq_2_hor_2['unseen_loss'], seq_2_hor_5['unseen_loss'])
welch_t_test(seq_2_hor_2['calmar_ratio'], seq_2_hor_5['calmar_ratio'])