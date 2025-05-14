# TODO add more features, add more indicators (quarterly reports, EPS, etc.)
# TODO upgrade model architecture to be what i envisioned
# plus, add more metrics/engineered features

import json
import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from training.money_experiment import MoneyExperiment

from training.utils import (
    count_parameters,
)
from training.data_loaders.stocks_time_series import (
    FinancialNumericalDataModule,
    download_numerical_financial_data
)

from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.nn.attention import SDPBackend

from transformer_arch.money_former import Money_former


def proceed(args: argparse.Namespace):
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
    type = args.type
    # TODO add loss functions
    # seed = args.seed
    extra_descriptor = args.extra_descriptor
    pred_indices = args.indices_to_predict # how many datapoints in the future to predict (workdays, not regular days, because market no work weekend)

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

    # pl.seed_everything(seed)
    print(
        f"type: {type} LLaMa seq_len:{seq_len} d_model:{d_model} d_ff:{d_ff} num_layers:{num_layers} nhead:{nhead} dropout:{dropout} lr:{lr} t_total:{t_total} warmup_steps:{warmup_steps} t_0:{t_0} t_mult:{t_mult} lr_mult:{lr_mult} batch_size:{batch_size}"
    )

    name = f"{args.dataset}/{type}_LLaMa_{seq_len}_{d_model}_{d_ff}_{num_layers}_{nhead}_{batch_size}"
    if extra_descriptor != "":
        name = name + "_" + extra_descriptor
    

    logger = TensorBoardLogger(
        "lightning_logs",
        name=name,  # seq, d_model, d_ff mult, num_layers, nhead
    )  # Optional logging
    # --- Data Loading ---
    if args.dataset == "Money": # yahoo finance stock data
        download_numerical_financial_data(
            tickers=args.tickers,
            seq_len=seq_len,
            check_if_already_downloaded=False
        )
        data_module = FinancialNumericalDataModule(
            train_file="time_series_data/train.pt",
            val_file="time_series_data/val.pt",
            test_file="time_series_data/test.pt",
            metadata_file="time_series_data/metadata.json",
            seq_len=seq_len + max(pred_indices)-1,
            batch_size=batch_size
        )

    data_module.setup()  # Very important to setup the data
    # vocab_size = data_module.get_vocab_size()

    # --- Model Definition ---
    match architecture:
        case "Money_former":
            model = Money_former(
                args=args
                # vocab_size=vocab_size
            )
        case _:
            raise ValueError(f"Architecture {architecture} not supported")
    # Print parameter count:
    num_params = count_parameters(model)
    print(f"The model has {num_params:,} trainable parameters. Parameter dtype: {args.dtype}")
    args.num_params = num_params

    # --- Training Setup ---

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
        dirpath="checkpoints/",
        filename="{name}-{epoch}-{val_loss:.2f}-{val_perplexity:.2f}",
        save_top_k=3,
        monitor="Loss/val_loss",
        mode="min",
    )

    # Early Stopping
    early_stopping_callback = EarlyStopping(
        monitor="Loss/val_loss", patience=1000, verbose=True, mode="min"
    )

    trainer = pl.Trainer(
        max_steps=t_total,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback, early_stopping_callback],
        # limit_train_batches=1000,
        limit_val_batches=50,
        logger=logger,
        log_every_n_steps=100, # 100
        val_check_interval=400,
        precision=trainer_precision
    )

    trainer.fit(experiment, datamodule=data_module)

    model_dir = f"models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(experiment.model, f"{model_dir}/money_{args.architecture}_{args.dataset}_{name.split('/')[-1]}.pth") # TODO make this more specific
    print("Model saved.")
    return


def run_experiment(args: argparse.Namespace):
    torch.set_float32_matmul_precision(
        "medium"
    )
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
        "--config", type=str, default="./experiment_configs/Money_test.json", help="Path to config file."
    )
    if parser.parse_known_args()[0].config != "":
        with open(parser.parse_known_args()[0].config, "r") as f:
            args = json.load(f)
        for k, v in args.items():
            parser.set_defaults(**{k: v})
    else:
        # Model architecture arguments (same as before)
        parser.add_argument(
            "--architecture",
            type=str,
            default="Money_former",#"DINT",
            help="Model architecture (LLaMa, ...)",
        )
        parser.add_argument("--d_model", type=int, default=128, help="Embedding dimension.")
        parser.add_argument(
            "--nhead", type=int, default=8, help="Number of attention heads."
        )
        parser.add_argument("--num_layers", type=int, default=4, help="Number of layers.")
        parser.add_argument("--d_ff", type=int, default=512, help="dimension in d_ff")

        parser.add_argument(
            "--dropout", type=float, default=0.1, help="Dropout probability."
        )
        parser.add_argument(
            "--type", type=str, default="baseline", help="Experiment type (for logging)."
        )

        # Training arguments (same as before)
        parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
        parser.add_argument("--warmup_steps", type=int, default=2000, help="Warmup steps.")
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
            "--extra_descriptor", type=str, default="", help="Extra descriptor for logging."
        )
        parser.add_argument("--orthograd", type=bool, default=True, help="Use OrthoGrad.")
        parser.add_argument(
            "--dataset", type=str, default="Money", help="Dataset to use."
        )


    args = parser.parse_args()
    run_experiment(args)


# What to do now:
# Your focus needs to shift entirely away from minimizing MSE and towards demonstrating value above and beyond the persistence model.
# Switch Primary Evaluation Metrics: Stop optimizing for MSE. Start focusing on:
# Directional Accuracy: How often does your model correctly predict whether the price will go UP or DOWN compared to the previous day?. The persistence model often has a directional accuracy near 50%. Can your model significantly beat this (e.g., >55% or 60% consistently)?
# Your goal is a MASE consistently less than 1.

# (Optional) Information Coefficient (IC) / Correlation: Calculate the correlation between your predicted returns (e.g., predicted[t+1] / predicted[t] - 1) and the actual returns (actual[t+1] / actual[t] - 1). A consistently positive IC shows some predictive alignment.
# Try Predicting Returns Instead of Prices: 
# Price series are strongly persistent (non-stationary). Daily returns (price[t]/price[t-1] - 1) or log returns (log(price[t]/price[t-1])) are generally closer to stationary, making them potentially easier to model meaningfully.

# Feature Engineering: The model defaulting to persistence suggests your current input features might not contain sufficient signal to predict changes effectively. Revisit your features:
# Are you using standard technical indicators (moving averages, RSI, MACD, Bollinger Bands)?
# Volume data?
# Volatility measures (e.g., ATR)?
# Time-based features (day of week, month)?
# Model Complexity and Regularization:
# Is the model complex enough to capture non-linear patterns (if they exist)?
# Is it too complex and overfitting to noise, effectively cancelling out any real signal and defaulting to persistence? Try adjusting model size, dropout rates, weight decay.
# Address the Time Gap: That 1980-2013 vs 2020-2024 gap is still a major hurdle. Patterns learned pre-2014 might be entirely irrelevant. The model might be implicitly learning this irrelevance and correctly defaulting to persistence as the most robust strategy across the gap. Consider:
# Training on more recent data (e.g., 2010-2019) to validate on 2020-2024.
# Using techniques designed for time series with distribution shifts (though this is advanced).


# The IC is essentially a correlation coefficient (typically Pearson or Spearman) calculated between your model's predictions for a set of assets at a given time and the actual subsequent realized outcomes for those assets.
# Here's a breakdown of how to calculate and interpret it:
# 1. Gather Your Data:
# Model Predictions (Signals or Alphas):
# For each time step t (e.g., end of day), your model generates a prediction for each asset i in your universe (e.g., all stocks in the S&P 500).
# This prediction P_i,t is for a future outcome, say at time t+k (e.g., prediction for next day's return, next week's return).
# So, at each time t, you'll have a vector of predictions: Predictions_t = [P_1,t, P_2,t, ..., P_N,t] for N assets.
# Actual Realized Outcomes:
# For each asset i and prediction made at time t for period t+k, you need the actual outcome A_i,t+k that occurred.
# If predicting returns, this would be the actual realized forward return for each asset over the period k (e.g., next day's open-to-close return, or close-to-close return).
# So, corresponding to Predictions_t, you'll have a vector of actuals: Actuals_t+k = [A_1,t+k, A_2,t+k, ..., A_N,t+k].
# 2. Calculate the IC for Each Period (Cross-Sectional IC):
# At each time step t, you calculate the correlation between the vector of predictions Predictions_t made at that time and the vector of corresponding actual future outcomes Actuals_t+k.
# IC_t = Correlation(Predictions_t, Actuals_t+k)
# Types of Correlation:
# Pearson IC: Measures linear correlation. Use scipy.stats.pearsonr or numpy.corrcoef.
# Spearman Rank IC: Measures the correlation between the ranks of predictions and the ranks of actuals. This is often preferred because it's less sensitive to outliers and doesn't assume a linear relationship (only monotonic). Use scipy.stats.spearmanr.
# 3. Analyze the Time Series of ICs:
# You will now have a time series of IC values (IC_1, IC_2, ..., IC_T). Analyzing this series tells you about the quality and consistency of your predictive signal:
# Mean IC (Average IC):
# The average of all your calculated IC_t values.
# A positive mean IC suggests your model has, on average, predictive power in the desired direction (higher predictions correspond to higher actual outcomes).
# A negative mean IC suggests your model has predictive power but in the opposite direction (your signal might need to be inverted).
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