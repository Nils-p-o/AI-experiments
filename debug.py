# replicates full training data flow, but instead of taining i just call validation dataloader
# and run validation for specfic model
# hopefully that is consistent with my metrics


import argparse
import os
import torch
import pytorch_lightning as pl
import yaml

# Import the exact same components used in your training and ensemble scripts
from training.money_experiment_2_MTP import MoneyExperiment
from transformer_arch.money.money_former_MLA_DINT_cog_attn_2_MTP import Money_former_MLA_DINT_cog_attn_MTP
from training.data_loaders.test_feats_stocks_time_series_2_MTP_new import FinancialNumericalDataModule

# This script assumes it is in the same directory level as the original scripts
# to resolve the imports correctly.

def load_config_and_args_from_metadata(metadata_path):
    """
    Loads experiment configuration from a YAML metadata file.
    (Copied from ensemble_script.py for consistency)
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, "r") as f:
        try:
            full_config_data = yaml.unsafe_load(f)
        except yaml.YAMLError as exc:
            print(f"Error parsing YAML file: {exc}")
            raise

    if not isinstance(full_config_data, dict) or 'args' not in full_config_data:
        raise ValueError("YAML file does not have the expected top-level 'args' key.")

    args_namespace_obj = full_config_data['args']

    if not isinstance(args_namespace_obj, argparse.Namespace):
        raise ValueError("The 'args' key in YAML did not resolve to an argparse.Namespace object.")

    # Convert back to a dictionary and then to a new Namespace
    args_dict = vars(args_namespace_obj)
    final_args_namespace = argparse.Namespace(**args_dict)

    # --- Load and convert normalization_means and normalization_stds ---
    if hasattr(final_args_namespace, 'normalization_means') and \
       hasattr(final_args_namespace, 'normalization_stds'):
        norm_means_list = final_args_namespace.normalization_means
        norm_stds_list = final_args_namespace.normalization_stds
        final_args_namespace.normalization_means = torch.tensor(norm_means_list, dtype=torch.float32)
        final_args_namespace.normalization_stds = torch.tensor(norm_stds_list, dtype=torch.float32)
    else:
        # This is expected since the training script calculates this, but the experiment
        # module itself might not need it if the model doesn't use it directly.
        print("Warning: normalization_means and/or normalization_stds not found in metadata.")


    return final_args_namespace

def run_validation():
    """
    Main function to load a model and run a validation epoch.
    """
    parser = argparse.ArgumentParser(
        description="Run a validation epoch on a pre-trained MTP model to replicate training results."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="ensemble/models/AAPL/ver_9_3199.ckpt",
        help="Path to the trained .ckpt model file from PyTorch Lightning.",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="ensemble/models/hparams.yaml",
        help="Path to the hparams.yaml file saved with the model.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="time_series_data",
        help="Directory where the train.pt, val.pt, etc., files are stored."
    )

    cli_args = parser.parse_args()

    # 1. Setup device and seed for reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the exact configuration the model was trained with
    print(f"Loading configuration from: {cli_args.metadata_path}")
    args_from_metadata = load_config_and_args_from_metadata(cli_args.metadata_path)
    
    # Use the same seed to ensure any minor random operations are consistent
    pl.seed_everything(args_from_metadata.seed)
    torch.set_float32_matmul_precision("medium")


    # 2. Instantiate the DataModule
    # This uses the exact same data files and batch size as the training run
    print("Setting up the FinancialNumericalDataModule...")
    data_module = FinancialNumericalDataModule(
        train_file=os.path.join(cli_args.data_dir, "train.pt"),
        train_targets_file=os.path.join(cli_args.data_dir, "train_MTP_targets.pt"),
        val_file=os.path.join(cli_args.data_dir, "val.pt"),
        val_targets_file=os.path.join(cli_args.data_dir, "val_MTP_targets.pt"),
        test_file=os.path.join(cli_args.data_dir, "test.pt"),
        test_targets_file=os.path.join(cli_args.data_dir, "test_MTP_targets.pt"),
        metadata_file=os.path.join(cli_args.data_dir, "metadata.json"),
        batch_size=args_from_metadata.batch_size,
        seq_len=args_from_metadata.seq_len,
    )
    # Important: setup() will load the metadata from json and prepare datasets
    data_module.setup(stage='fit') 
    print("DataModule setup complete.")

    # 3. Load the model using the Lightning Experiment module
    # MoneyExperiment.load_from_checkpoint will restore the model state AND the hyperparameters
    print(f"Loading model and experiment state from checkpoint: {cli_args.model_path}")
    experiment = MoneyExperiment.load_from_checkpoint(
        cli_args.model_path,
        map_location=device,
        # We pass the loaded model architecture to the experiment,
        # which it will use internally.
        model=Money_former_MLA_DINT_cog_attn_MTP(args_from_metadata)
    )
    experiment.to(device)
    experiment.eval() # Set model to evaluation mode
    print("Model loaded successfully.")

    # 4. Initialize the PyTorch Lightning Trainer
    # No logger or callbacks are needed for validation
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed" if args_from_metadata.dtype == "bf16" else "32-true",
        logger=False, # Disable logging for this validation run
    )

    # 5. Run the validation loop
    print("\n--- Starting Validation ---")
    print("This will run the validation data through the exact same logic as in the training script.")
    print("The metrics printed below should be identical to the validation metrics from the training logs.")
    
    # The `validate` method runs the validation_step and on_validation_epoch_end hooks
    validation_results = trainer.validate(model=experiment, datamodule=data_module, verbose=True)
    
    print("\n--- Validation Complete ---")
    print("Final validation results dictionary:")
    print(validation_results)
    
    print("\nCompare the metrics above (e.g., 'val_loss', 'val_Sharpe Ratio') with the logs from your original training run.")
    print("If they match, your validation process is deterministic. The discrepancy is likely in 'ensemble_script.py'.")
    print("If they do NOT match, there might be a source of non-determinism in your validation logic itself.")


if __name__ == "__main__":
    run_validation()