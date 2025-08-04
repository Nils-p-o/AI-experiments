import numpy as np
import scipy.stats as stats
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def welch_t_test(a, b):
    t_value, p_value = stats.ttest_ind(a, b, equal_var=False)
    print(f"Welch's T statistic: {t_value:.6f}")
    print(f"Welch's T statistic p-value: {p_value:.6f}")
    return


def paired_t_test(a, b):
    t_value, p_value = stats.ttest_rel(a, b)
    print(f"Paired T statistic: {t_value:.6f}")
    print(f"Paired T statistic p-value: {p_value:.6f}")
    return


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
        lower_is_better_keywords = ["loss", "mae", "mse"]
        is_lower_better = any(
            keyword in metrics_list[i].lower() for keyword in lower_is_better_keywords
        )

        if is_lower_better:
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)

        best_value = values[best_idx]
        best_step = steps[best_idx]

        results[shorthand_metrics_list[i]] = round(float(best_value), 4)  # {
        # 'best_value': float(best_value),
        # 'step': int(best_step)
        # }

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


def compare(base_model, compare_model):
    for metric, results in base_model.items():
        if metric not in compare_model:
            continue
        print(f"{metric}:")
        if higher_better[metric]:
            welch_t_test(compare_model[metric], results)
        else:
            welch_t_test(results, compare_model[metric])
        print(f"diff: {np.mean(compare_model[metric]) - np.mean(results):.6f}")


def paired_compare(base_model, compare_model):
    for metric, results in base_model.items():
        if metric not in compare_model:
            continue
        print(f"{metric}:")
        if higher_better[metric]:
            paired_t_test(compare_model[metric], results)
        else:
            paired_t_test(results, compare_model[metric])
        print(f"diff: {np.mean(compare_model[metric]) - np.mean(results):.6f}")


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
current_name = "Money/testing/fixed_mixing/testing_weighted_cce_15_costs_paired/Money_former_MLA_DINT_cog_attn_MTP_3_64_64_4_2_32"
print_metrics(current_name, lowest_ver, max_ver)

# diff models and their achieved metrics (max, or min depending on the metric)

# pre mixing fix
# models with 4 layers, 64 dims, no ortho, full 3bee feats, 6 tickers
# seq_2_hor_1 = {
#     "unseen_loss": np.array([0.825, 0.829, 0.8282]),
#     "calmar_ratio": np.array([1.549, 0.786, 1.302]),
# }
seq_2_hor_2 = {  # seems like the best one? because the other extra horizons dont have better loss really
    "unseen_loss": np.array([0.8228, 0.8196, 0.8249]),  # better loss?
    "calmar_ratio": np.array([1.249, 1.316, 1.004]),
}
# seq_2_hor_3 = {
#     "unseen_loss": np.array([0.8215, 0.8210, 0.8216]),
#     "calmar_ratio": np.array([1.566, 3.044, 0.804]),
# }
# seq_2_hor_4 = {
#     "unseen_loss": np.array([0.8201, 0.8209, 0.8228]),
#     "calmar_ratio": np.array([1.132, 1.337, 1.433]),
# }
# seq_2_hor_5 = {
#     "unseen_loss": np.array([0.8207, 0.8230, 0.8211]),
#     "calmar_ratio": np.array([0.882, 1.275, 1.190]),
# }


# # fixed mixing
# # dim 192, 4 layers
# no_mix_seq_2_hor_2 = {
#     "unseen_loss": np.array([0]),
#     "calmar_ratio": np.array([]),
# }

# no_mix_seq_2_hor_5 = {
#     "unseen_loss": np.array([0.8779, 0.8689, 0.8748, 0.8583]),
#     "calmar_ratio": np.array([1.459, 1.316, 0.782, 1.067]),
# }

# current working baseline (6 tickers)
# dim 64 (model ff), 4 layers, g sep, "full" attn, no noise, 0.25 dropout, normed inputs, 0.8 opt feats, no gsa
# 3 seq_len, 2 horizons, 5000 t steps, 1e-3 lr, 32 batch
baseline = {
    "unseen_loss": np.array([0.8349, 0.8288, 0.8343]),
    "calmar_ratio": np.array([1.013, 1.578, 0.562]),
    "f1": np.array([0.3709, 0.3708, 0.3725]),
}

opt_feats_05 = {
    "unseen_loss": np.array([0.8344, 0.8276, 0.8314]),
    "calmar_ratio": np.array([0.9767, 1.1032, 0.7734]),
    "f1": np.array([0.3636, 0.3791, 0.3779]),
}

opt_feats_02 = (
    {  # for now baseline, as it has fewer inputs but still ~ same performance
        "unseen_loss": np.array([0.8344, 0.835, 0.8343, 0.833, 0.8347]),
        "calmar_ratio": np.array([0.9552, 1.1871, 0.9709, 1.1946, 0.7311]),
        "f1": np.array([0.3786, 0.386, 0.375, 0.3664, 0.3821]),
    }
)

opt_feats_02_no_norm = {  # no norm in attn outputs
    "unseen_loss": np.array([0.8349, 0.8376, 0.8345]),
    "calmar_ratio": np.array([1.374, 0.9949, 1.0676]),
    "f1": np.array([0.3756, 0.3747, 0.3509]),
}

# need more for this baseline
# same, but faster, so keeping this
opt_feats_02_no_bias = {
    "unseen_loss": [0.8324, 0.8354, 0.831, 0.8341, 0.8305],
    "calmar_ratio": [0.824, 0.6619, 0.4695, 0.965, 0.6466],
    "f1": [0.3711, 0.3832, 0.3606, 0.3737, 0.3827],
    "aapl_acc": [0.5366, 0.537, 0.5408, 0.5434, 0.5381],
    "amzn_acc": [0.7085, 0.7073, 0.7077, 0.7066, 0.7077],
    "intc_acc": [0.5396, 0.5404, 0.5385, 0.5358, 0.5389],
    "msft_acc": [0.4264, 0.4245, 0.4242, 0.4268, 0.4242],
    "nvda_acc": [0.3982, 0.3944, 0.4066, 0.4051, 0.404],
    "gspc_acc": [0.4668, 0.4623, 0.4688, 0.4665, 0.4695],
}

opt_feats_02_no_bias_emb_proj = {  # not worth
    "unseen_loss": np.array([0.8411, 0.8343, 0.8336]),
    "calmar_ratio": np.array([0.5876, 0.5343, 0.5613]),
    "f1": np.array([0.3674, 0.355, 0.3715]),
}

opt_feats_02_no_b_2_h_64_hdim = {  # just worse, but statistically
    "unseen_loss": np.array([0.839, 0.8397]),
    "calmar_ratio": np.array([0.6571, 0.5074]),
    "f1": np.array([0.3643, 0.3537]),
}

opt_feats_02_no_bias_4_h_32_hdim = {  # maybe? not super stat significant
    "unseen_loss": np.array([0.83, 0.8297, 0.8341]),
    "calmar_ratio": np.array([0.9939, 1.2007, 1.4946]),
    "f1": np.array([0.3749, 0.3708, 0.3794]),
}

opt_feats_02_no_bias_8_h_16_hdim = {  # nothing
    "unseen_loss": np.array([0.8346, 0.8341, 0.8325]),
    "calmar_ratio": np.array([0.4698, 0.9273, 0.6634]),
    "f1": np.array([0.3749, 0.3602, 0.3761]),
}

opt_feats_02_no_b_128_ff = {  # nothing
    "unseen_loss": np.array([0.8351, 0.8359, 0.8334]),
    "calmar_ratio": np.array([0.8832, 0.5473, 0.9565]),
    "f1": np.array([0.3668, 0.3672, 0.3773]),
}

opt_feats_02_no_bias_c_only = {  # kinda, not really
    "unseen_loss": np.array(
        [0.9932, 0.9944, 0.9963]
    ),  # loss is only c, so not really comparable
    "calmar_ratio": np.array([0.7651, 1.0021, 0.416]),
    "f1": np.array([0.3278, 0.3544, 0.3422]),
}

opt_feats_02_no_b_8_seq = {  # for the same lr, worse
    "unseen_loss": np.array([0.8764, 0.8797, 0.8597]),
    "calmar_ratio": np.array([0.6727, 0.6624, 0.9431]),
    "f1": np.array([0.3719, 0.3645, 0.3606]),
}

# 10x relative c loss
# better calmar, but worse f1 both stat signif (0.05, 0.01) plus better behaved metrics during training, so...
# loss not really comparable, cuz different objective, kinda
opt_feats_02_no_b_weighted_feats_loss = {
    "unseen_loss": np.array([0.8707, 0.8698, 0.8697]),
    "calmar_ratio": np.array([1.2551, 1.4029, 1.976]),
    "f1": np.array([0.3541, 0.3557, 0.3418]),
    "aapl_acc": np.array([0.5377, 0.5362, 0.5385]),
    "amzn_acc": np.array([0.7062, 0.7092, 0.7043]),
    "intc_acc": np.array([0.5354, 0.5396, 0.54]),
    "msft_acc": np.array([0.4295, 0.4291, 0.4257]),
    "nvda_acc": np.array([0.396, 0.3963, 0.3899]),
    "gspc_acc": np.array([0.4649, 0.46, 0.4638]),
}

# ##################################################################################### trying with different optimiser

# same-ish, but should work better for longer/bigger runs, so worth keeping in mind
opt_feats_02_no_b_weighted_feats_loss_adamw = {
    "unseen_loss": [0.8716, 0.8654, 0.8742],
    "calmar_ratio": [1.0912, 2.2633, 0.7635],
    "f1": [0.3472, 0.3432, 0.3565],
    "aapl_acc": [0.5343, 0.5366, 0.537],
    "amzn_acc": [0.46, 0.4623, 0.4588],
    "intc_acc": [0.4234, 0.4261, 0.4318],
    "msft_acc": [0.5373, 0.5339, 0.5381],
    "nvda_acc": [0.3914, 0.4055, 0.3895],
    "gspc_acc": [0.7066, 0.7077, 0.7054],
}


# deffinitely better loss, dunno if that necessarily means that model itself is better though
opt_feats_02_no_b_weighted_feats_loss_muon = {
    "unseen_loss": np.array([0.8497, 0.8511, 0.8505]),
    "calmar_ratio": np.array([1.3401, 0.6783, 0.6589]),
    "f1": np.array([0.3641, 0.353, 0.3454]),
    "aapl_acc": np.array([0.5354, 0.54, 0.5324]),
    "amzn_acc": np.array([0.463, 0.4581, 0.4607]),
    "intc_acc": np.array([0.4314, 0.4272, 0.4272]),
    "msft_acc": np.array([0.5393, 0.5362, 0.5343]),
    "nvda_acc": np.array([0.4082, 0.396, 0.4028]),
    "gspc_acc": np.array([0.7077, 0.7069, 0.7092]),
}

# opt_feats_02_no_b_weighted_feats_weighted_tickers_loss_aapl = {
#     "unseen_loss": np.array([0.8871, 0.8944, 0.89]),
#     "calmar_ratio": np.array([1.9643, 1.3764, 1.0254]),
#     "f1": np.array([0.3469, 0.3455, 0.3593]),
#     "aapl_acc": np.array([0.5335, 0.5385, 0.5328]),
#     "amzn_acc": np.array([0.7054, 0.705, 0.7027]),
#     "intc_acc": np.array([0.5358, 0.5343, 0.5366]),
#     "msft_acc": np.array([0.4318, 0.423, 0.4249]),
#     "nvda_acc": np.array([0.388, 0.3857, 0.3777]),
#     "gspc_acc": np.array([0.4642, 0.4604, 0.4554]),
# }

# opt_feats_02_no_b_weighted_feats_weighted_tickers_loss_amzn = {
#     "unseen_loss": [0.8879, 0.8906, 0.8987],
#     "calmar_ratio": [0.401, 0.2199, -0.0033],
#     "f1": [0.2974, 0.3028, 0.3048],
#     "aapl_acc": [0.529, 0.5297, 0.5293],
#     "amzn_acc": [0.7073, 0.7054, 0.7027],
#     "intc_acc": [0.5312, 0.5385, 0.5316],
#     "msft_acc": [0.4253, 0.423, 0.4257],
#     "nvda_acc": [0.3716, 0.3655, 0.3594],
#     "gspc_acc": [0.4581, 0.4581, 0.4588],
# }

# opt_feats_02_no_b_weighted_feats_weighted_tickers_loss_nvda = {
#     "unseen_loss": [0.8835, 0.8823, 0.8886],
#     "calmar_ratio": [0.8211, 0.597, 0.5808],
#     "f1": [0.3633, 0.3571, 0.3717],
#     "aapl_acc": [0.5324, 0.5232, 0.5297],
#     "amzn_acc": [0.7024, 0.7066, 0.7046],
#     "intc_acc": [0.5354, 0.5328, 0.5347],
#     "msft_acc": [0.4249, 0.4223, 0.4261],
#     "nvda_acc": [0.3971, 0.3994, 0.3986],
#     "gspc_acc": [0.4634, 0.4604, 0.463],
# }


##########################################################################
# normal baseline
# handpicked feats, no weights for loss
#
hand_feats_nb_weight_loss_1 = {
    "backtest_loss": [0.9884, 0.9899, 0.9929],
    "backtest_acc": [0.5103, 0.5092, 0.5109],
    "f1": [0.3579, 0.356, 0.3579],
    "calmar_ratio": [1.1611, 1.1646, 1.2098],
    "max_drawdown": [-0.2857, -0.2288, -0.2487],
    "annualized_return": [0.3318, 0.2695, 0.509],
    "win_loss_ratio": [0.7021, 0.7684, 0.8535],
    "days_traded": [634.0, 630.0, 758.0],
}

hand_feats_nb_weight_loss_5 = {
    "backtest_loss": [0.9895, 0.9872, 0.9915, 0.9904, 0.991],
    "backtest_acc": [0.5103, 0.5103, 0.5113, 0.5107, 0.512],
    "f1": [0.3373, 0.3387, 0.3519, 0.3577, 0.3507],
    "calmar_ratio": [1.3194, 1.0223, 1.5986, 0.9386, 1.4063],
    "max_drawdown": [-0.2867, -0.2945, -0.2975, -0.2896, -0.1992],
    "annualized_return": [0.4063, 0.3322, 0.5338, 0.376, 0.343],
    "win_loss_ratio": [0.7872, 0.8547, 0.8034, 0.8381, 0.8302],
    "days_traded": [661.0, 667.0, 662.0, 733.0, 641.0],
}

hand_feats_nb_normal_paired = {
    "backtest_loss": [0.9875, 0.9896, 0.9904, 0.9895, 0.9888],
    "backtest_acc": [0.5106, 0.5103, 0.5127, 0.5121, 0.5132],
    "f1": [0.3642, 0.3504, 0.3498, 0.3643, 0.3648],
    "calmar_ratio": [0.6723, 0.7954, 1.1027, 0.842, 0.9468],
    "max_drawdown": [-0.1852, -0.329, -0.2041, -0.3098, -0.3506],
    "annualized_return": [0.414, 0.4118, 0.3984, 0.2888, 0.3319],
    "win_loss_ratio": [0.8784, 0.8496, 0.9298, 0.9286, 0.8433],
    "days_traded": [974.0, 661.0, 783.0, 634.0, 677.0],
}

# statistically worse acc, f1, days_traded
hand_feats_nb_weight_loss_5_paired = {
    "backtest_loss": [0.9882, 0.9911, 0.9889, 0.9929, 0.9898],
    "backtest_acc": [0.5083, 0.5085, 0.5123, 0.5099, 0.5088],
    "f1": [0.3587, 0.3396, 0.3481, 0.3458, 0.3573],
    "calmar_ratio": [0.9439, 1.0223, 1.3234, 2.1736, 0.3883],
    "max_drawdown": [-0.2716, -0.2906, -0.207, -0.2075, -0.4053],
    "annualized_return": [0.2753, 0.297, 0.6081, 0.4509, 0.1642],
    "win_loss_ratio": [0.8114, 0.792, 0.8991, 0.8462, 0.8956],
    "days_traded": [810.0, 569.0, 664.0, 526.0, 714.0],
}

# lower win_loss_ratio 0.0007, but all else is ~the same
# considering the potential benefits, i will keep it for bigger model consideration
hand_feats_nb_lora_ff = {
    "backtest_loss": [0.9894, 0.9872, 0.9887, 0.9911, 0.9873],
    "backtest_acc": [0.51, 0.5121, 0.5089, 0.5121, 0.5107],
    "f1": [0.3531, 0.3804, 0.3731, 0.3606, 0.3533],
    "calmar_ratio": [1.3683, 1.5083, 0.9266, 1.1275, 1.0291],
    "max_drawdown": [-0.272, -0.2834, -0.3567, -0.3187, -0.2957],
    "annualized_return": [0.4424, 0.4653, 0.379, 0.4263, 0.3142],
    "win_loss_ratio": [0.8182, 0.7642, 0.8245, 0.8261, 0.7746],
    "days_traded": [664.0, 714.0, 835.0, 646.0, 651.0],
}

# higher loss, lower win_loss ratio, maybe higher drawdown
# idk man
hand_feats_nb_local_simple = {
    "backtest_loss": [0.9954, 0.9921, 0.9909, 0.9912, 0.9947],
    "backtest_acc": [0.5108, 0.5125, 0.512, 0.5117, 0.5076],
    "f1": [0.3547, 0.3452, 0.355, 0.3651, 0.3469],
    "calmar_ratio": [0.6043, 1.6189, 0.8691, 1.5839, 0.6145],
    "max_drawdown": [-0.4443, -0.3426, -0.3639, -0.345, -0.3186],
    "annualized_return": [0.272, 0.5547, 0.416, 0.5469, 0.2862],
    "win_loss_ratio": [0.7353, 0.8073, 0.8356, 0.8241, 0.8017],
    "days_traded": [711.0, 627.0, 763.0, 816.0, 611.0],
}

# maybe higher loss, lower win_loss ratio
# i guess i stick w this?
hand_feats_nb_global = {
    "backtest_loss": [0.9909, 0.9948, 0.9892, 0.9929, 0.9927],
    "backtest_acc": [0.5092, 0.5121, 0.5128, 0.51, 0.5094],
    "f1": [0.3581, 0.3499, 0.3603, 0.3561, 0.3625],
    "calmar_ratio": [0.9572, 1.4625, 0.9253, 0.8334, 0.6655],
    "max_drawdown": [-0.3355, -0.3272, -0.322, -0.3137, -0.3159],
    "annualized_return": [0.3419, 0.62, 0.3798, 0.4015, 0.3002],
    "win_loss_ratio": [0.8652, 0.7579, 0.7941, 0.8121, 0.78],
    "days_traded": [755.0, 686.0, 650.0, 738.0, 808.0],
}

# loss higher, acc lower, f1 higher, calmar higher, drawdown lower, return higher, wlr higher, higher traded days
# 2x weights
hand_feats_nb_weighted_cce = {
    "backtest_loss": [1.0032, 1.0037, 1.0008, 1.013, 1.0095],
    "backtest_acc": [0.4793, 0.4839, 0.4882, 0.4739, 0.4808],
    "f1": [0.4261, 0.4285, 0.4224, 0.4309, 0.4289],
    "calmar_ratio": [1.5693, 1.7939, 2.061, 1.6628, 1.5736],
    "max_drawdown": [-0.3084, -0.199, -0.2715, -0.2654, -0.301],
    "annualized_return": [0.5584, 0.5022, 0.64, 0.5382, 0.5487],
    "win_loss_ratio": [0.8843, 0.9372, 0.9245, 1.0366, 0.9732],
    "days_traded": [1261.0, 1253.0, 1260.0, 1234.0, 1241.0],
}

# 1.5 weights
# comparing to 2x weight:
# better loss, acc, worse f1, fewer traded days
# somewhat worse wlr, lower drawdown, higher return
hand_feats_nb_weighted_cce_15 = {
    "backtest_loss": [0.988, 0.9878, 0.9916, 0.9918, 0.9889],
    "backtest_acc": [0.5085, 0.5041, 0.5111, 0.5022, 0.5038],
    "f1": [0.407, 0.4184, 0.4145, 0.4122, 0.4184],
    "calmar_ratio": [1.0318, 1.6044, 2.2066, 2.2995, 1.1897],
    "max_drawdown": [-0.3786, -0.2485, -0.2618, -0.3016, -0.2702],
    "annualized_return": [0.5565, 0.526, 0.83, 0.8373, 0.422],
    "win_loss_ratio": [0.8776, 0.9323, 0.8988, 0.9369, 0.9358],
    "days_traded": [1182.0, 1094.0, 1232.0, 1089.0, 1112.0],
}

# 2.5 weights
# comparing to 2x weight:
# worse loss, acc, f1, return, better drawdown, higher traded days
# betterish wlr
hand_feats_nb_weighted_cce_25 = {
    "backtest_loss": [1.0351, 1.0274, 1.0347, 1.0363, 1.0313],
    "backtest_acc": [0.4568, 0.4566, 0.4448, 0.4474, 0.4517],
    "f1": [0.4282, 0.4228, 0.4175, 0.423, 0.4251],
    "calmar_ratio": [1.3678, 1.2912, 1.6128, 2.6314, 1.9084],
    "max_drawdown": [-0.2276, -0.2264, -0.1939, -0.2054, -0.2121],
    "annualized_return": [0.4702, 0.4232, 0.4041, 0.5404, 0.4245],
    "win_loss_ratio": [0.9589, 1.0032, 0.9811, 1.0282, 0.9706],
    "days_traded": [1264.0, 1263.0, 1264.0, 1260.0, 1255.0],
}

# weight 3.0
# worse loss, acc, f1, return

hand_feats_nb_weighted_cce_30 = {
    "backtest_loss": [1.051, 1.0628, 1.0493, 1.062, 1.0613],
    "backtest_acc": [0.4357, 0.4223, 0.4336, 0.4217, 0.4251],
    "f1": [0.416, 0.4056, 0.4074, 0.405, 0.4067],
    "calmar_ratio": [2.0055, 1.9342, 2.0923, 1.4961, 1.8603],
    "max_drawdown": [-0.1762, -0.215, -0.1937, -0.2728, -0.1493],
    "annualized_return": [0.4321, 0.4302, 0.4851, 0.433, 0.4619],
    "win_loss_ratio": [0.9825, 0.9984, 0.9865, 0.9763, 0.9729],
    "days_traded": [1264.0, 1264.0, 1263.0, 1264.0, 1262.0],
}

# calmar higher, return higher, wlr lower, higher traded days
hand_feats_nb_aux_8 = {
    "backtest_loss": [0.9898, 0.991, 0.999, 0.9876, 0.9887],
    "backtest_acc": [0.5135, 0.5156, 0.5116, 0.5121, 0.5137],
    "f1": [0.3714, 0.3527, 0.3456, 0.37, 0.362],
    "calmar_ratio": [1.1511, 1.3998, 1.6962, 1.2433, 2.1664],
    "max_drawdown": [-0.3316, -0.1954, -0.277, -0.3883, -0.2853],
    "annualized_return": [0.4949, 0.7165, 0.5184, 0.5258, 0.767],
    "win_loss_ratio": [0.7442, 0.7706, 0.7576, 0.8401, 0.7686],
    "days_traded": [967.0, 811.0, 894.0, 828.0, 933.0],
}

# just no, horrible
# hand_feats_nb_anti_weighted_cce = {
#     "backtest_loss": [1.05, 1.0775, 1.0737, 1.0767, 1.0547],
#     "backtest_acc": [0.4936, 0.49, 0.4926, 0.4922, 0.4957],
#     "f1": [0.2509, 0.2541, 0.2394, 0.2464, 0.2578],
#     "calmar_ratio": [0.153, 0.1942, 1.0453, 0.5591, 0.6985],
#     "max_drawdown": [-0.1993, -0.1633, -0.1365, -0.1547, -0.1484],
#     "annualized_return": [0.0426, 0.0423, 0.1599, 0.1229, 0.1122],
#     "win_loss_ratio": [0.4324, 0.2, 0.619, 0.7143, 0.375],
#     "days_traded": [209.0, 121.0, 69.0, 119.0, 101.0],
# }

# lower loss, higher calmar
hand_feats_nb_custom_loss = {
    "backtest_loss": [0.987, 0.9866, 0.9844, 0.9873, 0.9878],
    "backtest_acc": [0.5078, 0.5127, 0.5127, 0.5122, 0.513],
    "f1": [0.3418, 0.3475, 0.3587, 0.358, 0.3639],
    "calmar_ratio": [0.8854, 1.1817, 1.4024, 1.2814, 1.2773],
    "max_drawdown": [-0.3279, -0.2768, -0.2237, -0.1752, -0.3358],
    "annualized_return": [0.3716, 0.4719, 0.3725, 0.4719, 0.4289],
    "win_loss_ratio": [0.7869, 0.8895, 0.9771, 0.9348, 0.7063],
    "days_traded": [682.0, 584.0, 681.0, 597.0, 723.0],
}

################################################################
# tests with costs
################################################################

hand_feats_baseline_costs = {
    "backtest_loss": [0.9875, 0.9896, 0.9904, 0.9895, 0.9888],
    "backtest_acc": [0.5106, 0.5103, 0.5127, 0.5121, 0.5132],
    "f1": [0.3642, 0.3504, 0.3498, 0.3643, 0.3648],
    "calmar_ratio_costs": [0.4842, 0.6172, 0.6609, 0.5128, 0.686],
    "max_drawdown_costs": [-0.2146, -0.3926, -0.242, -0.335, -0.3902],
    "annualized_return_costs": [0.3172, 0.3399, 0.3157, 0.2057, 0.2677],
    "win_loss_ratio_costs": [0.5361, 0.5625, 0.5581, 0.6, 0.5495],
    "days_traded_costs": [1095.0, 772.0, 926.0, 754.0, 836.0],
}


# weight 1.5
# better f1, calmar, return, wlr
# worse acc
# higher traded days
hand_feats_weighted_cce_15_costs = { # def keep
    "backtest_loss": [0.988, 0.9878, 0.9916, 0.9918, 0.9889],
    "backtest_acc": [0.5085, 0.5041, 0.5111, 0.5022, 0.5038],
    "f1": [0.407, 0.4184, 0.4145, 0.4122, 0.4184],
    "calmar_ratio_costs": [0.7022, 1.1433, 1.7142, 1.582, 0.6935],
    "max_drawdown_costs": [-0.4633, -0.3039, -0.3541, -0.3279, -0.3415],
    "annualized_return_costs": [0.4132, 0.3906, 0.6857, 0.6778, 0.2991],
    "win_loss_ratio_costs": [0.6706, 0.6723, 0.6537, 0.6559, 0.6706],
    "days_traded_costs": [1259.0, 1191.0, 1291.0, 1182.0, 1200.0],
}


opt_feats_02_placeholder = {
    "unseen_loss": np.array([]),
    "calmar_ratio": np.array([]),
    "f1": np.array([]),
}

print(f"baseline unseen_loss: {np.mean(baseline['unseen_loss'])}")
print(f"baseline calmar_ratio: {np.mean(baseline['calmar_ratio'])}")
print(f"baseline f1: {np.mean(baseline['f1'])}")

# print(f"\n--- hand_feats_nb_weight_loss_1 vs hand_feats_nb_weight_loss_5  ---")
# compare(hand_feats_nb_weight_loss_1, hand_feats_nb_weight_loss_5)

# print(f"\n--- hand_feats_nb_weight_normal vs hand_feats_nb_weight_loss_5 (paired)  ---")
# paired_compare(hand_feats_nb_normal_paired, hand_feats_nb_weight_loss_5_paired)

# print(f"\n--- hand_feats_nb_weight_normal vs hand_feats_nb_lora_ff (paired)  ---") # maybe
# paired_compare(hand_feats_nb_normal_paired, hand_feats_nb_lora_ff)

# print(f"\n--- hand_feats_nb_weight_normal vs hand_feats_nb_local_simple (paired)  ---")
# paired_compare(hand_feats_nb_normal_paired, hand_feats_nb_local_simple)

# print(f"\n--- hand_feats_nb_weight_normal vs hand_feats_nb_global (paired)  ---") # maybe
# paired_compare(hand_feats_nb_normal_paired, hand_feats_nb_global)

# print(f"\n--- hand_feats_nb_weight_normal vs hand_feats_nb_weighted_cce (paired)  ---") # yes
# paired_compare(hand_feats_nb_normal_paired, hand_feats_nb_weighted_cce)

# print(f"\n--- hand_feats_nb_weight_normal vs hand_feats_nb_aux_8 (paired)  ---") # yes
# paired_compare(hand_feats_nb_normal_paired, hand_feats_nb_aux_8)

# print(f"\n--- hand_feats_nb_weight_normal vs hand_feats_nb_anti_weighted_cce (paired)  ---")
# paired_compare(hand_feats_nb_normal_paired, hand_feats_nb_anti_weighted_cce)

print(
    f"\n--- hand_feats_nb_weight_normal vs hand_feats_nb_custom_loss (paired)  ---"
)  # kinda yes, if it works together with cce weighing
paired_compare(hand_feats_nb_normal_paired, hand_feats_nb_custom_loss)

# print(
#     f"\n--- hand_feats_nb_weighted_cce (20) vs hand_feats_nb_weighted_cce_15 (paired)  ---"
# )
# paired_compare(hand_feats_nb_weighted_cce, hand_feats_nb_weighted_cce_15)

# print(
#     f"\n--- hand_feats_nb_weighted_cce (20) vs hand_feats_nb_weighted_cce_25 (paired)  ---"
# )
# paired_compare(hand_feats_nb_weighted_cce, hand_feats_nb_weighted_cce_25)

# print(
#     f"\n--- hand_feats_nb_weighted_cce (20) vs hand_feats_nb_weighted_cce_30 (paired)  ---"
# )
# paired_compare(hand_feats_nb_weighted_cce, hand_feats_nb_weighted_cce_30)


#################################################################
# comparisons with costs
#################################################################

print(f"\n--- hand_feats_nb_weight_normal vs hand_feats_nb_weighted_cce (paired)  ---")
paired_compare(hand_feats_baseline_costs, hand_feats_weighted_cce_15_costs)
