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
    "aapl_acc": "Close_Sequence/val_unseen_AAPL_accuracy",
    "amzn_acc": "Close_Sequence/val_unseen_AMZN_accuracy",
    "intc_acc": "Close_Sequence/val_unseen_INTC_accuracy",
    "msft_acc": "Close_Sequence/val_unseen_MSFT_accuracy",
    "nvda_acc": "Close_Sequence/val_unseen_NVDA_accuracy",
    "gspc_acc": "Close_Sequence/val_unseen_^GSPC_accuracy",
    "calmar_ratio": "Strategy_metrics/val_Calmar Ratio",
    "max_drawdown": "Strategy_metrics/val_Max Drawdown",
    "annualized_return": "Strategy_metrics/val_Annualized Return",
    "win_loss_ratio": "Strategy_metrics/val_WLR",
    "days_traded": "Strategy_metrics/val_Days Traded",
    "gspc_calmar_ratio": "Strategy_metrics_individual/val_Calmar Ratio_^GSPC",
    "gspc_annualized_return": "Strategy_metrics_individual/val_Annualized Return_^GSPC",
    "gspc_max_drawdown": "Strategy_metrics_individual/val_Max Drawdown_^GSPC",
    "gspc_win_loss_ratio": "Strategy_metrics_individual/val_WLR_^GSPC",
}

higher_better = {
    "unseen_loss": 0,
    "backtest_loss": 0,
    "backtest_acc": 1,
    "f1": 1,
    "calmar_ratio": 1,
    "max_drawdown": 0,
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
    "gspc_calmar_ratio": 1,
    "gspc_annualized_return": 1,
    "gspc_max_drawdown": 0,
    "gspc_win_loss_ratio": 1,
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
    "calmar_ratio",
    "max_drawdown",
    "annualized_return",
    "win_loss_ratio",
    "days_traded",
    # "gspc_calmar_ratio",
    # "gspc_max_drawdown",
    # "gspc_annualized_return",
    # "gspc_win_loss_ratio",
]

lowest_ver = 0
max_ver = 4
# current_name = "Money/testing/fixed_mixing/testing_imbalance_weighted_stable_custom_linear_costs/Money_former_MLA_DINT_cog_attn_MTP_3_64_64_4_2_32"
# current_name = "Money/opts/class/rng_class_weighted_15/Money_former_MLA_DINT_cog_attn_MTP_3_64_64_4_2_32"
current_name = "Money/testing/class/class_imb_opt_exp_add_cost_adj_data/Money_former_MLA_DINT_cog_attn_MTP_3_64_64_4_2_32"
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
hand_feats_weighted_cce_15_costs = {  # def keep
    "backtest_loss": [0.988, 0.9878, 0.9916, 0.9918, 0.9889],
    "backtest_acc": [0.5085, 0.5041, 0.5111, 0.5022, 0.5038],
    "f1": [0.407, 0.4184, 0.4145, 0.4122, 0.4184],
    "calmar_ratio_costs": [0.7022, 1.1433, 1.7142, 1.582, 0.6935],
    "max_drawdown_costs": [-0.4633, -0.3039, -0.3541, -0.3279, -0.3415],
    "annualized_return_costs": [0.4132, 0.3906, 0.6857, 0.6778, 0.2991],
    "win_loss_ratio_costs": [0.6706, 0.6723, 0.6537, 0.6559, 0.6706],
    "days_traded_costs": [1259.0, 1191.0, 1291.0, 1182.0, 1200.0],
}

# better loss, calmar
hand_feats_custom_loss_exp_costs = {
    "backtest_loss": [0.987, 0.9866, 0.9844, 0.9873, 0.9878],
    "backtest_acc": [0.5078, 0.5127, 0.5127, 0.5122, 0.513],
    "f1": [0.3418, 0.3475, 0.3587, 0.358, 0.3639],
    "calmar_ratio_costs": [0.572, 0.919, 1.0214, 0.9374, 1.0415],
    "max_drawdown_costs": [-0.3475, -0.2964, -0.2397, -0.1835, -0.3505],
    "annualized_return_costs": [0.269, 0.3701, 0.2769, 0.3725, 0.3651],
    "win_loss_ratio_costs": [0.4571, 0.5318, 0.583, 0.5806, 0.4684],
    "days_traded_costs": [842.0, 739.0, 807.0, 739.0, 864.0],
}

# same weights as exp, so maybe not a fair comparison, but (still comparing with baseline)

# NOTE wlr till now was being gotten wrong, it was getting the lowest, not the highest, so...

# ~ the same as exp version
hand_feats_custom_loss_linear_costs = {
    "backtest_loss": [0.9875, 0.989, 0.986, 0.9862, 0.9864],
    "backtest_acc": [0.5127, 0.5113, 0.514, 0.51, 0.5141],
    "f1": [0.3643, 0.3428, 0.3501, 0.3471, 0.3637],
    "calmar_ratio_costs": [0.6821, 0.9826, 0.9402, 0.5966, 0.9138],
    "max_drawdown_costs": [-0.3687, -0.2951, -0.233, -0.2767, -0.2757],
    "annualized_return_costs": [0.4016, 0.3459, 0.3901, 0.1651, 0.2724],
    "win_loss_ratio_costs": [0.9206, 0.7579, 0.8586, 0.7553, 0.8474],
    "days_traded_costs": [1016.0, 699.0, 855.0, 592.0, 725.0],
}

# better calmar, return
# more traded days
hand_feats_aux_8_costs = {
    "backtest_loss": [0.9898, 0.991, 0.999, 0.9876, 0.9887],
    "backtest_acc": [0.5135, 0.5156, 0.5116, 0.5121, 0.5137],
    "f1": [0.3714, 0.3527, 0.3456, 0.37, 0.362],
    "calmar_ratio_costs": [0.9006, 1.1504, 1.3282, 0.972, 1.9402],
    "max_drawdown_costs": [-0.3511, -0.2178, -0.2885, -0.4312, -0.3076],
    "annualized_return_costs": [0.4027, 0.6217, 0.4348, 0.4381, 0.7018],
    "win_loss_ratio_costs": [0.8936, 0.8761, 0.9215, 0.8373, 0.9942],
    "days_traded_costs": [1096.0, 928.0, 1003.0, 942.0, 1029.0],
}


# kinda worse loss, acc
# pretty much the same as baseline, at least, statistically
# maybe try this for model scale up, because its more regularized?
hand_feats_causal_costs = {
    "backtest_loss": [0.993, 0.9932, 0.9928, 0.9889, 0.9881],
    "backtest_acc": [0.5121, 0.5076, 0.5086, 0.5086, 0.509],
    "f1": [0.3661, 0.3569, 0.364, 0.3477, 0.3629],
    "calmar_ratio_costs": [0.7449, 0.3188, 0.4678, 0.414, 0.5887],
    "max_drawdown_costs": [-0.2026, -0.3737, -0.3494, -0.3598, -0.3438],
    "annualized_return_costs": [0.3685, 0.1507, 0.2188, 0.1941, 0.2953],
    "win_loss_ratio_costs": [0.8302, 0.8071, 0.8116, 0.8204, 0.7962],
    "days_traded_costs": [946.0, 862.0, 888.0, 822.0, 952.0],
}

# higher loss,
#
#
hand_global_only_costs = {
    "backtest_loss": [0.9909, 0.9948, 0.9892, 0.9929, 0.9927],
    "backtest_acc": [0.5092, 0.5121, 0.5128, 0.51, 0.5094],
    "f1": [0.3581, 0.3499, 0.3603, 0.3561, 0.3625],
    "calmar_ratio_costs": [0.7423, 1.104, 0.6353, 0.5971, 0.4001],
    "max_drawdown_costs": [-0.3612, -0.3489, -0.3437, -0.354, -0.3741],
    "annualized_return_costs": [0.2751, 0.516, 0.2795, 0.3062, 0.2008],
    "win_loss_ratio_costs": [0.7383, 0.7371, 0.7678, 0.7563, 0.7354],
    "days_traded_costs": [930.0, 856.0, 817.0, 908.0, 966.0],
}

# camparing w weighted_15
# better acc,
# worse f1,
# lower traded days
# kinda worse calmar, return
# comparing w custom loss exp
# better
# worse loss, drawdown
# higher traded days
hand_feats_custom_loss_exp_weighted_15_costs = {
    "backtest_loss": [0.9909, 0.9948, 0.9892, 0.9929, 0.9927],
    "backtest_acc": [0.5092, 0.5121, 0.5128, 0.51, 0.5094],
    "f1": [0.3581, 0.3499, 0.3603, 0.3561, 0.3625],
    "calmar_ratio_costs": [0.7423, 1.104, 0.6353, 0.5971, 0.4001],
    "max_drawdown_costs": [-0.3612, -0.3489, -0.3437, -0.354, -0.3741],
    "annualized_return_costs": [0.2751, 0.516, 0.2795, 0.3062, 0.2008],
    "win_loss_ratio_costs": [0.7383, 0.7371, 0.7678, 0.7563, 0.7354],
    "days_traded_costs": [930.0, 856.0, 817.0, 908.0, 966.0],
}

# comparing w aux_8
# better f1,
# worse acc, drawdown, return
# kinda worse calmar, higher traded days
# comparing weighted_cce_15
# -
# -
# -
hand_feats_aux_8_weighted_15_costs = {
    "backtest_loss": [0.989, 0.9912, 0.9917, 0.9889, 0.9956],
    "backtest_acc": [0.5117, 0.5062, 0.505, 0.5075, 0.499],
    "f1": [0.4175, 0.4148, 0.4031, 0.4138, 0.4235],
    "calmar_ratio_costs": [0.8537, 1.2093, 0.5382, 0.6994, 0.8053],
    "max_drawdown_costs": [-0.4241, -0.3736, -0.5626, -0.4123, -0.3875],
    "annualized_return_costs": [0.4281, 0.4518, 0.3028, 0.3619, 0.4045],
    "win_loss_ratio_costs": [1.0545, 1.0525, 1.0235, 1.0233, 1.0583],
    "days_traded_costs": [1245.0, 1268.0, 1245.0, 1224.0, 1231.0],
}

# comparing w normal baseline
# better f1,
# worse loss, acc, return
# higher traded days
hand_feats_imbalance_weighted = {
    "backtest_loss": [1.0258, 1.0254, 1.0267, 1.027, 1.0231],
    "backtest_acc": [0.4783, 0.4772, 0.4739, 0.48, 0.4724],
    "f1": [0.4184, 0.4263, 0.4107, 0.4177, 0.4189],
    "calmar_ratio_costs": [0.9843, 0.4097, 0.2957, 0.4753, 0.3513],
    "max_drawdown_costs": [-0.2529, -0.2889, -0.2435, -0.3543, -0.2791],
    "annualized_return_costs": [0.249, 0.1277, 0.1163, 0.1981, 0.098],
    "win_loss_ratio_costs": [0.907, 0.9766, 0.812, 0.7847, 0.829],
    "days_traded_costs": [1125.0, 1184.0, 1089.0, 1028.0, 1087.0],
}

# w normal baseline
# better f1
# worse loss, acc, return
# higher traded days
# w unstable ver
# -
# worse acc
# -
hand_feats_imbalance_weighted_stable = {
    "backtest_loss": [1.026, 1.0219, 1.0298, 1.0264, 1.0298],
    "backtest_acc": [0.4739, 0.4775, 0.4701, 0.4744, 0.4699],
    "f1": [0.4153, 0.4204, 0.4148, 0.4194, 0.4178],
    "calmar_ratio_costs": [0.7181, 0.4503, 0.4344, 0.0532, 0.8789],
    "max_drawdown_costs": [-0.2303, -0.313, -0.3198, -0.2965, -0.2827],
    "annualized_return_costs": [0.169, 0.1409, 0.1462, 0.0178, 0.2485],
    "win_loss_ratio_costs": [0.9123, 0.9167, 0.8692, 0.7912, 0.8026],
    "days_traded_costs": [1184.0, 1127.0, 1170.0, 1055.0, 1113.0],
}

# comparing w normal baseline
# better drawdown,
# worse loss, acc, f1, return
# fewer tradeddays
hand_feats_imbalance_weighted_custom_linear_add = {
    "backtest_loss": [1.0047, 1.004, 1.0098, 1.0065, 1.0009],
    "backtest_acc": [0.4999, 0.5025, 0.4971, 0.4969, 0.5015],
    "f1": [0.3506, 0.3279, 0.324, 0.3265, 0.333],
    "calmar_ratio_costs": [1.5741, 0.343, 0.8302, 0.4702, 0.3701],
    "max_drawdown_costs": [-0.1393, -0.216, -0.1719, -0.2088, -0.2911],
    "annualized_return_costs": [0.2652, 0.1086, 0.1577, 0.0982, 0.1077],
    "win_loss_ratio_costs": [0.8079, 0.7284, 0.761, 0.6856, 0.6813],
    "days_traded_costs": [553.0, 536.0, 405.0, 415.0, 566.0],
}


# better ...
# worse loss,
# fewer ish traded days?
# seems ~ the same, no real impact
hand_feats_no_mask_2 = {
    "backtest_loss": [0.9901, 0.9913, 0.9912, 0.9894, 0.99],
    "backtest_acc": [0.5127, 0.5126, 0.5109, 0.5121, 0.5084],
    "f1": [0.3495, 0.3535, 0.3701, 0.3459, 0.3511],
    "calmar_ratio_costs": [0.6591, 0.7513, 0.6464, 0.5093, 0.251],
    "max_drawdown_costs": [-0.3642, -0.3376, -0.2289, -0.3406, -0.3172],
    "annualized_return_costs": [0.309, 0.3113, 0.3757, 0.1792, 0.112],
    "win_loss_ratio_costs": [0.872, 0.8337, 0.8068, 0.762, 0.7052],
    "days_traded_costs": [750.0, 751.0, 890.0, 792.0, 717.0],
}

# better
# worse
# ~ the same, maybe worse loss, fewer traded days?
hand_feats_scaled_swiglu = {
    "backtest_loss": [0.9893, 0.9918, 0.99, 0.989, 0.9929],
    "backtest_acc": [0.512, 0.5122, 0.5118, 0.5141, 0.5118],
    "f1": [0.3599, 0.348, 0.3623, 0.3676, 0.3483],
    "calmar_ratio_costs": [0.8643, 0.3075, 0.804, 0.6053, 1.0179],
    "max_drawdown_costs": [-0.1895, -0.3508, -0.368, -0.2932, -0.2803],
    "annualized_return_costs": [0.2669, 0.1675, 0.434, 0.2229, 0.2853],
    "win_loss_ratio_costs": [0.8446, 0.7457, 0.7725, 0.761, 0.7627],
    "days_traded_costs": [905.0, 717.0, 865.0, 894.0, 737.0],
}

##############################################################################################
# NOTE post ensemble
hand_feats_ensemble_weighted_15 = {
    "backtest_loss": [0.9892, 0.9938, 0.9888, 0.9896, 0.9862, 0.9877],
    "backtest_acc": [0.508, 0.5085, 0.5076, 0.5004, 0.514, 0.5024],
    "f1": [0.4226, 0.4176, 0.4187, 0.4169, 0.4229, 0.416],
    "calmar_ratio": [1.1368, 1.2622, 1.8974, 1.0151, 1.1435, 1.0314],
    "max_drawdown": [0.3399, 0.3138, 0.333, 0.3327, 0.1358, 0.3107],
    "annualized_return": [0.4721, 0.5544, 0.6412, 0.4641, 0.5529, 0.3808],
    "win_loss_ratio": [0.9343, 1.097, 1.0344, 1.0654, 1.028, 1.0096],
    "days_traded": [1194.0, 1275.0, 1262.0, 1270.0, 1188.0, 1263.0],
    "gspc_calmar_ratio": [0.4227, 1.0727, 1.115, 1.9333, 0.823, 1.2895],
    "gspc_max_drawdown": [0.1691, 0.0968, 0.097, 0.01, 0.096, 0.098],
    "gspc_annualized_return": [0.0715, 0.137, 0.1126, 0.1895, 0.0867, 0.1264],
    "gspc_win_loss_ratio": [0.6812, 0.78, 0.8611, 1.0, 0.9773, 0.8657],
}

# gspc is upweighted 5x compared to other tickers
#
# worse f1
# maybe better gspc (calmar, drawdown, return)
# success?
hand_feats_ensemble_weighted_15_gspc = {
    "backtest_loss": [0.9866, 0.9893, 0.9906, 0.9917, 0.9885, 0.9893],
    "backtest_acc": [0.5081, 0.5097, 0.5038, 0.5022, 0.5103, 0.5042],
    "f1": [0.4146, 0.4184, 0.4161, 0.4083, 0.4172, 0.4145],
    "calmar_ratio": [1.1375, 0.9277, 0.7507, 1.2114, 1.4431, 0.7969],
    "max_drawdown": [0.3336, 0.2487, 0.2212, 0.3319, 0.2843, 0.3136],
    "annualized_return": [0.4843, 0.4886, 0.4264, 0.5631, 0.5541, 0.4721],
    "win_loss_ratio": [1.0697, 1.1, 1.0146, 1.0356, 1.069, 1.0532],
    "days_traded": [1308.0, 1281.0, 1244.0, 1258.0, 1273.0, 1273.0],
    "gspc_calmar_ratio": [1.4637, 2.1088, 1.3147, 1.5248, 1.4824, 1.422],
    "gspc_max_drawdown": [0.0, 0.0005, 0.097, 0.0, 0.0, 0.096],
    "gspc_annualized_return": [0.1405, 0.1695, 0.1402, 0.1653, 0.1423, 0.1442],
    "gspc_win_loss_ratio": [0.8621, 1.0, 1.04, 0.8267, 1.0, 0.7381],
}

#
# worse f1, return, wlr,
#
hand_feats_ensemble_weighted_15_unique_outputs = {
    "backtest_loss": [0.9859, 0.9898, 0.9901, 0.989, 0.9866, 0.9896],
    "backtest_acc": [0.5112, 0.5046, 0.5047, 0.505, 0.5076, 0.5029],
    "f1": [0.4156, 0.412, 0.4096, 0.4123, 0.4056, 0.4043],
    "calmar_ratio": [1.12, 0.6696, 0.9004, 1.1861, 0.9091, 0.8905],
    "max_drawdown": [0.278, 0.2218, 0.2265, 0.0652, 0.2523, 0.2944],
    "annualized_return": [0.5613, 0.3632, 0.4201, 0.3551, 0.3296, 0.3268],
    "win_loss_ratio": [0.9082, 0.9108, 1.0, 0.9604, 0.9128, 0.9448],
    "days_traded": [1143.0, 1295.0, 1192.0, 1183.0, 1281.0, 1279.0],
    "gspc_calmar_ratio": [1.0426, 0.4304, 0.8412, 0.8216, 1.0859, 0.6104],
    "gspc_max_drawdown": [0.096, 0.0, 0.096, 0.0, 0.0251, 0.0],
    "gspc_annualized_return": [0.1004, 0.0728, 0.1008, 0.0958, 0.1042, 0.0831],
    "gspc_win_loss_ratio": [0.7941, 0.6481, 0.9412, 0.92, 0.8889, 0.6389],
}

####### seeds 12345 again ########
hand_feats_ensemble_imbalance_base = {
    "backtest_loss": [1.026, 1.0219, 1.0298, 1.0264, 1.0298],
    "backtest_acc": [0.4739, 0.4775, 0.4701, 0.4744, 0.4699],
    "f1": [0.4153, 0.4204, 0.4148, 0.4194, 0.4178],
    "calmar_ratio": [0.7179, 0.4502, 0.4344, 0.0531, 0.8788],
    "max_drawdown": [0.2302, 0.3128, 0.3197, 0.2962, 0.2826],
    "annualized_return": [0.1689, 0.1408, 0.1461, 0.0177, 0.2484],
    "win_loss_ratio": [0.9123, 0.9214, 0.8692, 0.7912, 0.8026],
    "days_traded": [1184.0, 1124.0, 1170.0, 1055.0, 1113.0],
}

# imbalance global only - now worse apparently?
# compared w imbalance weight
#
# worse drawdown (maybe calmar, wlr)
#
hand_feats_ensemble_imbalance_global_only = {
    "backtest_loss": [1.0313, 1.0293, 1.0178, 1.0184, 1.0308],
    "backtest_acc": [0.4762, 0.4677, 0.4837, 0.4828, 0.4697],
    "f1": [0.4219, 0.4186, 0.4141, 0.4137, 0.416],
    "calmar_ratio": [0.2427, 0.1641, 0.1175, 0.3742, 0.16],
    "max_drawdown": [0.4397, 0.3272, 0.4542, 0.4004, 0.5144],
    "annualized_return": [0.1067, 0.0537, 0.0563, 0.168, 0.0831],
    "win_loss_ratio": [0.8247, 0.8079, 0.8121, 0.826, 0.7923],
    "days_traded": [1099.0, 1118.0, 1136.0, 1081.0, 1184.0],
    "gspc_calmar_ratio": [0.6299, 0.2613, 0.4327, 1.8713, 0.7646],
    "gspc_max_drawdown": [0.1994, 0.2426, 0.186, 0.1036, 0.2167],
    "gspc_annualized_return": [0.1256, 0.0906, 0.0805, 0.1938, 0.1657],
    "gspc_win_loss_ratio": [0.8581, 0.7414, 0.7723, 0.811, 0.8656],
}

# imbalance hlo on prev c global only
# compared w imbalance_weight
# better (maybe wlr?)
# worse loss, acc, f1 (maybe everything)
#
hand_feats_ensemble_imbalance_global_only_hlo_on_c = {
    "backtest_loss": [1.0485, 1.0466, 1.0408, 1.0408, 1.0469],
    "backtest_acc": [0.4718, 0.4678, 0.4639, 0.4701, 0.4662],
    "f1": [0.4142, 0.409, 0.4056, 0.4108, 0.4103],
    "calmar_ratio": [0.3351, 0.1394, 0.5149, 0.2023, 0.6421],
    "max_drawdown": [0.3156, 0.4258, 0.2528, 0.4062, 0.2464],
    "annualized_return": [0.1057, 0.0625, 0.1971, 0.0878, 0.1641],
    "win_loss_ratio": [0.9178, 0.8977, 0.9023, 0.8957, 0.9468],
    "days_traded": [1120.0, 1089.0, 1209.0, 1003.0, 1112.0],
}

# unique outputs
# comp w imb base
# better acc
# worse f1, calmar, drawdown, return, wlr
# fewer traded days
hand_feats_ensemble_imbalance_unique_outputs = {
    "backtest_loss": [1.0311, 1.0293, 1.0318, 1.0287, 1.0284],
    "backtest_acc": [0.477, 0.4847, 0.4737, 0.4844, 0.4733],
    "f1": [0.4084, 0.4103, 0.4104, 0.4123, 0.4097],
    "calmar_ratio": [0.1886, 0.1299, 0.1784, 0.204, 0.0491],
    "max_drawdown": [0.3447, 0.3074, 0.3568, 0.3085, 0.4263],
    "annualized_return": [0.0671, 0.0412, 0.0714, 0.0629, 0.0223],
    "win_loss_ratio": [0.8303, 0.7456, 0.7978, 0.7706, 0.782],
    "days_traded": [1085.0, 1031.0, 1045.0, 1031.0, 1007.0],
}


# multiple seperators, full attn - ~no difference
#
# worse (maybe f1)
#
hf_ens_imb_multiple_seps = {
    "backtest_loss": [1.0309, 1.0181, 1.0294, 1.0306, 1.0146],
    "backtest_acc": [0.4654, 0.4774, 0.4727, 0.4762, 0.4799],
    "f1": [0.4142, 0.4063, 0.4089, 0.4165, 0.4166],
    "calmar_ratio": [0.465, 0.8372, 0.4653, 0.8462, 0.8418],
    "max_drawdown": [0.4079, 0.2708, 0.257, 0.1918, 0.2401],
    "annualized_return": [0.1991, 0.2267, 0.2003, 0.2161, 0.2021],
    "win_loss_ratio": [0.9404, 0.8933, 0.8564, 0.8078, 0.8003],
    "days_traded": [1205.0, 1047.0, 1099.0, 1073.0, 1073.0],
}

# exp_add from cost optimizations - deff keep
# better loss, acc, f1 (maybe calmar, return, wlr)
# worse -
# higher days traded
hf_ens_imb_exp_add_cost = {
    "backtest_loss": [1.0089, 0.9965, 1.0108, 1.0072, 1.0143],
    "backtest_acc": [0.4835, 0.4935, 0.4811, 0.4867, 0.4827],
    "f1": [0.4331, 0.4292, 0.4293, 0.4305, 0.4267],
    "calmar_ratio": [0.8728, 1.0621, 0.4786, 1.1256, 0.6204],
    "max_drawdown": [0.372, 0.2303, 0.2788, 0.2489, 0.2843],
    "annualized_return": [0.3487, 0.305, 0.2389, 0.4073, 0.1764],
    "win_loss_ratio": [1.0017, 1.0177, 1.0032, 0.9439, 0.9741],
    "days_traded": [1238.0, 1255.0, 1269.0, 1211.0, 1224.0],
}

# compared w exp add - just worse
#
# worse loss, f1, calmar, drawdown, return, wlr (maybe acc)
# fewer traded days
hf_ens_imb_lin_add_cost = {
    "backtest_loss": [1.0197, 1.0187, 1.0309, 1.0237, 1.0324],
    "backtest_acc": [0.485, 0.4853, 0.4772, 0.4853, 0.4809],
    "f1": [0.4262, 0.4246, 0.428, 0.4251, 0.4204],
    "calmar_ratio": [0.1119, 0.3943, 0.5313, 0.2201, 0.1809],
    "max_drawdown": [0.5233, 0.2536, 0.4428, 0.3578, 0.2742],
    "annualized_return": [0.0586, 0.1161, 0.2352, 0.0908, 0.0496],
    "win_loss_ratio": [0.8789, 0.87, 0.9414, 0.8569, 0.8315],
    "days_traded": [1181.0, 1179.0, 1236.0, 1148.0, 1179.0],
}


# same as exp_add, but using auto adjust from yfinance - no real diff
# -
# -
#
hf_ens_imb_exp_add_cost_adj_data = {
    "backtest_loss": [1.0004, 1.001, 1.0099, 1.0066, 1.015],
    "backtest_acc": [0.4875, 0.4851, 0.4808, 0.4849, 0.485],
    "f1": [0.4283, 0.4284, 0.4242, 0.428, 0.4349],
    "calmar_ratio": [0.5338, 0.7477, 0.7627, 0.6787, 1.1308],
    "max_drawdown": [0.4041, 0.2765, 0.2975, 0.2836, 0.2239],
    "annualized_return": [0.2314, 0.2067, 0.2269, 0.1924, 0.2531],
    "win_loss_ratio": [0.9732, 0.9671, 1.0175, 0.8861, 1.0169],
    "days_traded": [1251.0, 1293.0, 1271.0, 1223.0, 1215.0],
}


print(f"baseline unseen_loss: {np.mean(baseline['unseen_loss'])}")
print(f"baseline calmar_ratio: {np.mean(baseline['calmar_ratio'])}")
print(f"baseline f1: {np.mean(baseline['f1'])}")


#################################################################
# comparisons with costs
#################################################################

# print(f"\n--- hand_feats_nb_weight_normal vs hand_feats_nb_weighted_cce (paired)  ---")
# paired_compare(hand_feats_baseline_costs, hand_feats_weighted_cce_15_costs)

# print(f"\n--- hand_feats_nb_weight_normal vs hand_feats_custom_loss_exp (paired)  ---")
# paired_compare(hand_feats_baseline_costs, hand_feats_custom_loss_exp_costs)

# print(f"\n--- hand_feats_custom_loss_exp vs hand_feats_custom_loss_linear (paired)  ---")
# paired_compare(hand_feats_custom_loss_exp_costs, hand_feats_custom_loss_linear_costs)

print(f"\n--- hand_feats_nb_weight_normal vs hand_feats_aux_8_costs (paired)  ---")
paired_compare(hand_feats_baseline_costs, hand_feats_aux_8_costs)

# print(f"\n--- hand_feats_nb_weight_normal vs hand_feats_causal (paired)  ---")
# paired_compare(hand_feats_baseline_costs, hand_feats_causal_costs)

# print(f"\n--- hand_feats_nb_weight_normal vs hand_global_only_costs (paired)  ---")
# paired_compare(hand_feats_baseline_costs, hand_global_only_costs)

# print(
#     f"\n--- hand_feats_nb_weighted_cce vs hand_feats_custom_loss_exp_weighted_15_costs (paired)  ---"
# )
# paired_compare(
#     hand_feats_weighted_cce_15_costs, hand_feats_custom_loss_exp_weighted_15_costs
# )

# print(
#     f"\n--- hand_feats_custom_loss_exp vs hand_feats_custom_loss_exp_weighted_15_costs (paired)  ---"
# )
# paired_compare(
#     hand_feats_custom_loss_exp_costs, hand_feats_custom_loss_exp_weighted_15_costs
# )

# print(f"\n--- hand_feats_aux_8 vs hand_feats_aux_8_weighted_15_costs (paired)  ---")
# paired_compare(hand_feats_aux_8_costs, hand_feats_aux_8_weighted_15_costs)

# print(f"\n--- hand_feats_weighted_cce_15_costs vs hand_feats_aux_8_weighted_15_costs (paired)  ---")
# paired_compare(hand_feats_weighted_cce_15_costs, hand_feats_aux_8_weighted_15_costs)

# print(f"\n--- hand_feats_nb_weight_normal vs hand_feats_imbalance_weighted (paired)  ---")
# paired_compare(hand_feats_baseline_costs, hand_feats_imbalance_weighted)

# print(f"\n--- hand_feats_nb_weight_normal vs hand_feats_no_mask_2 (paired)  ---")
# paired_compare(hand_feats_baseline_costs, hand_feats_no_mask_2)

# print(f"\n--- hand_feats_nb_weight_normal vs hand_feats_scaled_swiglu (paired)  ---")
# paired_compare(hand_feats_baseline_costs, hand_feats_scaled_swiglu)

# print(
#     f"\n--- hand_feats_nb_weight_normal vs hand_feats_imbalance_weighted_stable (paired)  ---"
# )
# paired_compare(hand_feats_baseline_costs, hand_feats_imbalance_weighted_stable)

# print(
#     f"\n--- hand_feats_imbalance_weighted vs hand_feats_imbalance_weighted_stable (paired)  ---"
# )
# paired_compare(hand_feats_imbalance_weighted, hand_feats_imbalance_weighted_stable)

# print(
#     f"\n--- hand_feats_nb_weight_normal vs hand_feats_imbalance_weighted_custom_linear_add (paired)  ---"
# )
# paired_compare(
#     hand_feats_baseline_costs, hand_feats_imbalance_weighted_custom_linear_add
# )

######################################################
# post ensemble
######################################################
# print(
#     "\n--- hand_feats_ensemble_weighted_15 vs hand_feats_ensemble_weighted_15_gspc (paired)  ---"
# )
# paired_compare(hand_feats_ensemble_weighted_15, hand_feats_ensemble_weighted_15_gspc)


# print(
#     "\n--- hand_feats_ensemble_weighted_15 vs hand_feats_ensemble_weighted_15_unique_outputs (paired)  ---"
# )
# paired_compare(
#     hand_feats_ensemble_weighted_15, hand_feats_ensemble_weighted_15_unique_outputs
# )

# print("\n--- hand_feats_ensemble_imbalance_base vs hand_feats_ensemble_imbalance_global_only (paired)  ---")
# paired_compare(hand_feats_ensemble_imbalance_base, hand_feats_ensemble_imbalance_global_only)

# print("\n--- hand_feats_ensemble_imbalance_base vs hand_feats_ensemble_imbalance_global_only_hlo_on_c (paired)  ---")
# paired_compare(hand_feats_ensemble_imbalance_base, hand_feats_ensemble_imbalance_global_only_hlo_on_c)

# print("\n--- hand_feats_ensemble_imbalance_base vs hand_feats_ensemble_imbalance_unique_outputs (paired)  ---")
# paired_compare(hand_feats_ensemble_imbalance_base, hand_feats_ensemble_imbalance_unique_outputs)

# print("\n--- hand_feats_ensemble_imbalance_base vs hf_ens_imb_multiple_seps (paired)  ---")
# paired_compare(hand_feats_ensemble_imbalance_base, hf_ens_imb_multiple_seps)

print(
    "\n--- hand_feats_ensemble_imbalance_base vs hf_ens_imb_exp_add_cost (paired) ---"
)
paired_compare(hand_feats_ensemble_imbalance_base, hf_ens_imb_exp_add_cost)

print("\n--- hf_ens_imb_exp_add_cost vs hf_ens_imb_exp_add_cost_adj_data (paired)  ---")
paired_compare(hf_ens_imb_exp_add_cost, hf_ens_imb_exp_add_cost_adj_data)
