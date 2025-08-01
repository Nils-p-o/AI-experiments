import json
import os
import argparse
import optuna
from optuna.pruners import SuccessiveHalvingPruner
from optuna.integration import PyTorchLightningPruningCallback
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.nn.attention import SDPBackend

from transformer_arch.money.money_former_MLA_DINT_cog_attn_2_MTP_muP import Money_former_MLA_DINT_cog_attn_MTP
from training.data_loaders.test_feats_stocks_time_series_2_MTP_new import (
    FinancialNumericalDataModule,
    download_numerical_financial_data,
)
from training.money_experiment_2_MTP import MoneyExperiment
from training.utils import count_parameters

from money_train_2_MTP_exp import apply_distributed_patch

def objective(trial: optuna.trial.Trial, base_config_path: str, data_module: FinancialNumericalDataModule):
    """
    The objective function for Optuna. Each call to this function is one "trial".
    """
    # --- 1. Load Base Configuration ---
    with open(base_config_path, "r") as f:
        config = json.load(f)
    args = argparse.Namespace(**config)

    # --- 2. Suggest Hyperparameters ---
    # Optuna will suggest values for these hyperparameters.
    args.lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    args.muon_lr = trial.suggest_float('muon_lr', 5e-4, 5e-2, log=True)
    args.dropout = trial.suggest_float('dropout', 0.1, 0.5)
    # We also need to add weight_decay, which requires modifying MoneyExperiment
    # Let's add it to the args that MoneyExperiment will use.
    args.weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

    args.num_layers = trial.suggest_int('num_layers', 2, 8)
    args.scheduler_type = trial.suggest_categorical(
        'scheduler_type', ['cosine_restarts', 'linear_decay', 'constant']
    )

    # --- 3. Set up Model and Experiment ---
    pl.seed_everything(42) # Use a fixed seed for HPO for comparability

    # Use the muP-compliant model architecture
    model = Money_former_MLA_DINT_cog_attn_MTP(args=args)
    
    # IMPORTANT: We need a version of MoneyExperiment that can accept weight_decay.
    # We will pass the full 'args' namespace to it.
    experiment = MoneyExperiment(
        model,
        learning_rate=args.lr, # Main LR for aux optimizer
        args=args
    )

    # --- 4. Set up Callbacks and Logger ---
    # Pruning callback to stop unpromising trials early
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="Losses_seen_unseen/val_loss_unseen")

    # Unique logger for each trial
    logger = TensorBoardLogger("HPO_Logs", name=f"trial_{trial.number}")
    
    # Early stopping can also be used
    early_stopping_callback = EarlyStopping(
        monitor="Losses_seen_unseen/val_loss_unseen", patience=5, mode="min"
    )

    # --- 5. Set up and Run Trainer ---
    # We train for a fixed number of steps, long enough to see a trend but short enough for speed.
    # The pruner will handle early termination of bad trials.
    trainer = pl.Trainer(
        max_steps=args.t_total, # Use total steps from config
        accelerator="auto",
        devices="auto",
        callbacks=[pruning_callback, early_stopping_callback],
        logger=logger,
        log_every_n_steps=50,
        val_check_interval=150, # Check validation loss periodically for pruning
        enable_progress_bar=False, # Disable progress bar for cleaner HPO logs
        precision="32"#"bf16-mixed" if torch.cuda.is_available() else "32-true"
    )

    try:
        trainer.fit(experiment, datamodule=data_module)
    except Exception as e:
        print(f"Trial {trial.number} failed with exception: {e}")
        # Return a large value to indicate failure
        return float('inf')

    # --- 6. Return the Metric to Optimize ---
    # We want to minimize the validation loss
    return trainer.callback_metrics.get("Losses_seen_unseen/val_loss_unseen", float('inf')).item()


def run_hpo():
    apply_distributed_patch()
    
    # --- A. Define Base Configuration ---
    # This JSON should define your BASE model size (the small, fast one)
    base_config_path = './experiment_configs/MTP_classification_exp.json'
    with open(base_config_path, "r") as f:
        args = argparse.Namespace(**json.load(f))

    # --- B. Pre-download and Prepare Data (DO THIS ONCE!) ---
    print("--- Preparing data module once before starting HPO study ---")
    args.tickers = sorted(args.tickers)
    args.normalization_means, args.normalization_stds = download_numerical_financial_data(
        tickers=args.tickers,
        seq_len=args.seq_len,
        target_dates=args.indices_to_predict,
        config_args=args,
        check_if_already_downloaded=False # Set to False if you need to re-download
    )
    data_module = FinancialNumericalDataModule(
        train_file="time_series_data/train.pt",
        train_targets_file="time_series_data/train_MTP_targets.pt",
        val_file="time_series_data/val.pt",
        val_targets_file="time_series_data/val_MTP_targets.pt",
        test_file="time_series_data/test.pt",
        test_targets_file="time_series_data/test_MTP_targets.pt",
        metadata_file="time_series_data/metadata.json",
        seq_len=args.seq_len,
        batch_size=args.batch_size,
    )
    print("--- Data module prepared ---")
    
    # --- C. Create and Run the Optuna Study ---
    # Use the SuccessiveHalvingPruner to implement ASHA-like behavior
    pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=4)

    study = optuna.create_study(
        study_name="money-former-hpo",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(), # Bayesian optimization
        pruner=pruner
    )

    known_good_params = {
        'lr': 1e-3,
        'muon_lr': 0.005,
        'dropout': 0.25,
        'weight_decay': 0.0, # Use a reasonable default or known value
        'num_layers': 4,
        'scheduler_type': 'cosine_restarts'
    }
    
    # Enqueue this trial. It will be the VERY FIRST one Optuna runs.
    study.enqueue_trial(known_good_params)
    print(f"--- Enqueued known good trial: {known_good_params} ---")

    # Use a lambda to pass the static data_module and config path to the objective
    objective_fn = lambda trial: objective(trial, base_config_path, data_module)

    print(f"--- Starting HPO with {args.hpo_trials} trials ---")
    study.optimize(objective_fn, n_trials=getattr(args, 'hpo_trials', 100))

    # --- D. Print Results ---
    print("\n--- HPO Study Complete ---")
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best validation loss: {study.best_trial.value:.6f}")
    print("Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    # This allows you to add command line arguments if needed, e.g., --hpo_trials 200
    parser = argparse.ArgumentParser(description="Run Hyperparameter Optimization.")
    parser.add_argument(
        "--config", type=str, default="./experiment_configs/MTP_classification_exp.json"
    )
    # Add an argument for number of trials
    parser.add_argument("--hpo_trials", type=int, default=100, help="Number of HPO trials to run.")
    
    # Load config and add hpo_trials to it
    if os.path.exists(parser.parse_known_args()[0].config):
        with open(parser.parse_known_args()[0].config, 'r') as f:
            config = json.load(f)
        for k, v in config.items():
            parser.set_defaults(**{k: v})

    args = parser.parse_args()
    
    # This ensures hpo_trials is available in the args namespace
    setattr(args, 'hpo_trials', args.hpo_trials)
    
    # Set torch matmul precision
    torch.set_float32_matmul_precision("medium")

    # Run the HPO process
    run_hpo()