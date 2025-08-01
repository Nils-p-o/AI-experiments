# TODO add more features, add more indicators (quarterly reports, EPS, etc.)
# TODO test putting all input features as part of sequence?

# TODO bayesian optimization (inference + training)
# TODO add more things as part of config (weights for loss, features, etc.)
# TODO rewrite dataloader to take in a list of features

# TODO noMachine on hypkos computer

# TODO make timing be per epoch
# TODO maybe try optimising some parts of the code by using c?

# TODO set up old pc for training

# TODO test nGPT

# different scaling in attn
# TODO redo some tests (global vs local, etc. groupnorm)


import json
import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from training.utils import (
    count_parameters,
)
# from training.data_loaders.stocks_time_series_2_MTP import (
#     FinancialNumericalDataModule,
#     download_numerical_financial_data,
# )

from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.nn.attention import SDPBackend

from transformer_arch.money.money_former_2 import Money_former
from transformer_arch.money.money_former_DINT_2 import Money_former_DINT
from transformer_arch.money.money_former_MLA_DINT_2 import Money_former_MLA_DINT
from transformer_arch.money.money_former_DINT_cog_attn_2 import Money_former_DINT_cog_attn
from transformer_arch.money.money_former_MLA_2 import Money_former_MLA
from transformer_arch.money.money_former_nGPT_2 import Money_former_nGPT, normalize_weights_and_enforce_positive_eigenvalues
from transformer_arch.money.money_former_MLA_DINT_cog_attn_2 import Money_former_MLA_DINT_cog_attn
# from transformer_arch.money.money_former_MLA_DINT_cog_attn_2_MTP_diff_attn_dims import Money_former_MLA_DINT_cog_attn_MTP

# from transformer_arch.money.money_former_MLA_DINT_cog_attn_2_MTP_muP import Money_former_MLA_DINT_cog_attn_MTP
from transformer_arch.money.money_former_MLA_DINT_cog_attn_2_MTP import Money_former_MLA_DINT_cog_attn_MTP
from training.data_loaders.test_feats_stocks_time_series_2_MTP_new import (
    FinancialNumericalDataModule,
    download_numerical_financial_data,
)
from training.money_experiment_2_MTP import MoneyExperiment

import torch.distributed as dist
from functools import wraps

def apply_distributed_patch():
    """
    Applies a comprehensive monkey-patch to torch.distributed functions to prevent
    crashes in single-device environments. This patches get_world_size, get_rank,
    and all_gather.
    """
    # --- Patch get_world_size ---
    if not hasattr(dist, '_original_get_world_size'):
        dist._original_get_world_size = dist.get_world_size

        @wraps(dist._original_get_world_size)
        def patched_get_world_size(*args, **kwargs):
            if not dist.is_available() or not dist.is_initialized():
                return 1
            return dist._original_get_world_size(*args, **kwargs)

        dist.get_world_size = patched_get_world_size

    # --- Patch get_rank ---
    if not hasattr(dist, '_original_get_rank'):
        dist._original_get_rank = dist.get_rank

        @wraps(dist._original_get_rank)
        def patched_get_rank(*args, **kwargs):
            if not dist.is_available() or not dist.is_initialized():
                return 0
            return dist._original_get_rank(*args, **kwargs)

        dist.get_rank = patched_get_rank

    # --- Patch all_gather ---
    if not hasattr(dist, '_original_all_gather'):
        dist._original_all_gather = dist.all_gather

        @wraps(dist._original_all_gather)
        def patched_all_gather(tensor_list, tensor, *args, **kwargs):
            """Patched version that performs a local copy if distributed is not initialized."""
            if not dist.is_available() or not dist.is_initialized():
                # In a single-process world, all_gather is a no-op
                # The tensor_list (output) should be filled with the tensor (input).
                # With a world size of 1, the list has 1 element.
                if len(tensor_list) != 1:
                     # This should not happen with our world_size patch, but as a safeguard:
                     raise ValueError("Patched all_gather expects a tensor_list of length 1.")
                tensor_list[0].copy_(tensor)
                return None
            return dist._original_all_gather(tensor_list, tensor, *args, **kwargs)

        dist.all_gather = patched_all_gather

    print("--- Applied comprehensive torch.distributed patches for get_world_size, get_rank, and all_gather ---")

# from transformer_arch.money.money_former_MLA_DINT_cog_attn_2_MTP_c_only import Money_former_MLA_DINT_cog_attn_MTP
# from training.data_loaders.test_feats_stocks_time_series_2_MTP_new_c_only import (
#     FinancialNumericalDataModule,
#     download_numerical_financial_data,
# )
# from training.money_experiment_2_MTP_c_only import MoneyExperiment


def proceed(args: argparse.Namespace):
    apply_distributed_patch()

    architecture = args.architecture
    seq_len = args.seq_len
    batch_size = args.batch_size
    d_model = args.d_model
    nhead = args.nhead
    num_layers = args.num_layers
    d_ff = args.d_ff
    dropout = args.dropout
    lr = args.lr
    t_total = args.t_total
    warmup_steps = args.warmup_steps
    t_0 = args.t_0 + warmup_steps
    t_mult = args.t_mult
    lr_mult = args.lr_mult
    pred_indices = (
        args.indices_to_predict
    )  # how many datapoints in the future to predict (workdays, not regular days, because market no work weekend)
    args.tickers = sorted(args.tickers)

    match args.dtype:
        case "fp32":
            trainer_precision = "32-true"
        case "fp16":
            trainer_precision = "16-mixed"
        case "bf16":
            trainer_precision = "bf16-mixed"
        case _:
            args.dtype = "fp32"
            trainer_precision = "32-true"
    seed = torch.seed()
    seed = seed % (2**32)
    args.seed = seed
    pl.seed_everything(seed)
    print(
        f"LLaMa seq_len:{seq_len} d_model:{d_model} d_ff:{d_ff} num_layers:{num_layers} nhead:{nhead} dropout:{dropout} lr:{lr} t_total:{t_total} warmup_steps:{warmup_steps} t_0:{t_0} t_mult:{t_mult} lr_mult:{lr_mult} batch_size:{batch_size}"
    )

    name = f"{args.dataset}/{args.folder_name}/{architecture}_{seq_len}_{d_model}_{d_ff}_{num_layers}_{nhead}_{batch_size}"
    if args.extra_descriptor:
        name = name + "_" + args.extra_descriptor

    logger = TensorBoardLogger(
        "Money_logs",
        name=name,  # seq, d_model, d_ff mult, num_layers, nhead
    )  # Optional logging
    # --- Data Loading ---
    if args.dataset == "Money":  # yahoo finance stock data

        args.normalization_means, args.normalization_stds = download_numerical_financial_data(
            tickers=args.tickers,
            seq_len=seq_len,
            check_if_already_downloaded=False,  # TODO make this better/check which features are missing
            target_dates=pred_indices,
            config_args=args,
        )
        data_module = FinancialNumericalDataModule(
            train_file="time_series_data/train.pt",
            train_targets_file="time_series_data/train_MTP_targets.pt",
            val_file="time_series_data/val.pt",
            val_targets_file="time_series_data/val_MTP_targets.pt",
            test_file="time_series_data/test.pt",
            test_targets_file="time_series_data/test_MTP_targets.pt",
            metadata_file="time_series_data/metadata.json",
            seq_len=seq_len,
            batch_size=batch_size,
        )

    data_module.setup()  # Very important to setup the data
    # vocab_size = data_module.get_vocab_size()
    args.input_features = len(data_module._metadata["columns"])
    # --- Model Definition ---
    match architecture:  # TODO auto format qk_rope_dim for non MLA (currently all of them)
        case "Money_former":
            model = Money_former(args=args)
        case "Money_former_DINT":
            model = Money_former_DINT(args=args)
        case "Money_former_MLA_DINT":
            model = Money_former_MLA_DINT(args=args)
        case "Money_former_DINT_cog_attn":
            model = Money_former_DINT_cog_attn(args=args)
        case "Money_former_MLA":
            model = Money_former_MLA(args=args)
        case "Money_former_nGPT":
            model = Money_former_nGPT(args=args)
        case "Money_former_MLA_DINT_cog_attn":
            model = Money_former_MLA_DINT_cog_attn(args=args)
        case "Money_former_MLA_DINT_cog_attn_MTP":
            model = Money_former_MLA_DINT_cog_attn_MTP(args=args)
        case _:
            raise ValueError(f"Architecture {architecture} not supported")
    # Print parameter count:
    num_params = count_parameters(model)
    print(
        f"The model has {num_params:,} trainable parameters. Parameter dtype: {args.dtype}"
    )
    args.num_params = num_params

    # torch compile
    # model = torch.compile(model)

    # --- Training Setup ---
    if model.__class__.__name__ == "Money_former_nGPT":
        normalize_weights_and_enforce_positive_eigenvalues(model)

    experiment = MoneyExperiment(
        model,
        learning_rate=lr,
        batch_size=batch_size,
        warmup_steps=warmup_steps,
        t_0=t_0,
        t_mult=t_mult,
        lr_mult=lr_mult,
        args=args,
        # dtype=torch_dtype_for_params
    )  # Use vocab_size

    # Checkpointing
    checkpoint_callback = ModelCheckpoint(
        # dirpath="checkpoints/",
        filename="{name}-{epoch}-{Trading_strategy_metrics/val_Calmar Ratio:.2f}",
        save_top_k=3,
        monitor="Trading_strategy_metrics/val_Calmar Ratio",
        mode="max",
        # monitor="Losses_seen_unseen/val_loss_unseen",
        # monitor="Loss/val_loss",
        # mode="min",
    )

    # Early Stopping
    early_stopping_callback = EarlyStopping(
        monitor="Losses_seen_unseen/val_loss_unseen", patience=1000, verbose=True, mode="min"
        # monitor="Loss/val_loss", patience=1000, verbose=True, mode="min"
    )

    trainer = pl.Trainer(
        max_steps=t_total,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback, early_stopping_callback],
        # limit_train_batches=1000,
        limit_val_batches=50,
        logger=logger,
        log_every_n_steps=100,
        val_check_interval=300,
        precision=trainer_precision,
        check_val_every_n_epoch=None,
    )

    trainer.fit(experiment, datamodule=data_module)

    # model_dir = f"models"
    # if not os.path.exists(model_dir):
    #     os.makedirs(model_dir)
    # torch.save(
    #     experiment.model, f"{model_dir}/{args.architecture}_{name.split('/')[-1]}.pth"
    # )  # TODO make this more specific
    # print("Model saved.")
    return


def run_experiment(args: argparse.Namespace):
    torch.set_float32_matmul_precision("medium")
    if torch.cuda.is_available():
        with torch.nn.attention.sdpa_kernel(
            [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
                SDPBackend.CUDNN_ATTENTION,
            ],
            set_priority=True,
        ):
            proceed(args)
            return
    else:
        proceed(args)
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Transformer model.")

    parser.add_argument(
        "--config",
        type=str,
        # default="./experiment_configs/MTP_triplicate.json",
        # default="./experiment_configs/diff_head_dims_MTP.json",
        # default="./experiment_configs/profile.json",
        # default="./experiment_configs/MTP_experiment_trip.json",
        default="./experiment_configs/MTP_classification_exp.json",
        help="Path to config file.",
    )
    if parser.parse_known_args()[0].config != "":
        with open(parser.parse_known_args()[0].config, "r") as f:
            args = json.load(f)
        for k, v in args.items():
            parser.set_defaults(**{k: v})
            # parser.add_argument(f"--{k}", type=type(v), default=v)
    else:
        # Model architecture arguments (same as before)
        parser.add_argument(
            "--architecture",
            type=str,
            default="Money_former",  # "DINT",
            help="Model architecture (LLaMa, ...)",
        )
        parser.add_argument(
            "--d_model", type=int, default=128, help="Embedding dimension."
        )
        parser.add_argument(
            "--nhead", type=int, default=8, help="Number of attention heads."
        )
        parser.add_argument(
            "--num_layers", type=int, default=4, help="Number of layers."
        )
        parser.add_argument("--d_ff", type=int, default=512, help="dimension in d_ff")

        parser.add_argument(
            "--dropout", type=float, default=0.1, help="Dropout probability."
        )
        parser.add_argument(
            "--type",
            type=str,
            default="baseline",
            help="Experiment type (for logging).",
        )

        # Training arguments (same as before)
        parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
        parser.add_argument(
            "--warmup_steps", type=int, default=2000, help="Warmup steps."
        )
        parser.add_argument(
            "--t_total", type=int, default=100000, help="Total training steps."
        )
        parser.add_argument(
            "--t_0", type=int, default=5000, help="Initial period for cosine annealing."
        )
        parser.add_argument(
            "--t_mult", type=float, default=1.5, help="Multiplier for period."
        )
        parser.add_argument(
            "--lr_mult", type=float, default=0.6, help="Multiplier for peak LR."
        )
        parser.add_argument("--seq_len", type=int, default=128, help="Sequence length.")
        parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")

        # parser.add_argument(
        #     "--seed", type=int, default=42, help="Seed for reproducibility."
        # )
        parser.add_argument(
            "--extra_descriptor",
            type=str,
            default="",
            help="Extra descriptor for logging.",
        )
        parser.add_argument(
            "--orthograd", type=bool, default=True, help="Use OrthoGrad."
        )
        parser.add_argument(
            "--dataset", type=str, default="Money", help="Dataset to use."
        )

    args = parser.parse_args()
    run_experiment(args)


# What to do now:

# Feature Engineering: The model defaulting to persistence suggests your current input features might not contain sufficient signal to predict changes effectively. Revisit your features:
# Are you using standard technical indicators (moving averages, RSI, MACD, Bollinger Bands)?
# Volume data?
# Volatility measures (e.g., ATR)? Historical Volatility, Implied Volatility, Future / Expected Volatility

# Model Complexity and Regularization:
# Is the model complex enough to capture non-linear patterns (if they exist)?
# Is it too complex and overfitting to noise, effectively cancelling out any real signal and defaulting to persistence? Try adjusting model size, dropout rates, weight decay.
# Using techniques designed for time series with distribution shifts (though this is advanced).


# Standard Deviation of IC:
# Measures the volatility or consistency of your ICs over time. A lower standard deviation is generally better, indicating a more stable signal.

# Information Ratio (IR):
# IR = Mean(IC) / StdDev(IC)
# This is a crucial metric, similar to a Sharpe ratio for a trading strategy. It tells you how much predictive power you get per unit of risk (volatility of the IC). Higher is better. A common rule of thumb is that an IR > 0.5 is considered good.
# IC Skewness and Kurtosis: Provides insights into the distribution of your ICs.
# Percentage of Positive ICs (Hit Rate):
# The proportion of periods where IC_t > 0.
# t-statistic for Mean IC:
# t_stat = Mean(IC) * sqrt(Number of Periods) / StdDev(IC)
# This tests the statistical significance of your mean IC (i.e., how likely is it that your mean IC is different from zero by chance). A t-stat > 2 (or < -2) is often considered statistically significant.
# Example Scenario:
# Predictions: At the end of Day D, your model predicts the next day's return for 100 stocks. You get a list of 100 predicted returns.
# Actuals: At the end of Day D+1, you record the actual returns for those 100 stocks that occurred during Day D+1. You get a list of 100 actual returns.
# Calculate IC_D: You compute the Spearman correlation between your list of 100 predictions and the list of 100 actual returns. This gives you one IC value for Day D.
# Repeat: You do this every day for a year. You now have ~252 daily IC values.
# Analyze: Calculate the mean, std dev, IR, and t-stat of these ~252 ICs.
# What do IC values mean (heuristics for daily stock return prediction):
# |IC| > 0.02 - 0.03: Might be interesting if very consistent.
# |IC| > 0.05: Generally considered good.
# |IC| > 0.10: Very good.
# |IC| > 0.15: Excellent (rarely sustained).
# Important Considerations:
# Forward-Looking Nature: Ensure your actual outcomes are strictly after your predictions are made. No look-ahead bias.
# Alignment: Predictions and actuals must be perfectly aligned by asset and by the prediction/outcome period.
# Universe Consistency: The set of assets used for IC calculation should be consistent or handled carefully if it changes (e.g., stocks entering/leaving an index).
# Transaction Costs: IC measures raw predictive power. It doesn't account for transaction costs, liquidity, or portfolio construction constraints that would affect a real trading strategy.
# Decay: Signals can decay. The IC might be stronger for 1-day forward returns than for 5-day forward returns using the same prediction.
