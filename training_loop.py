
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from training.experiment import TransformerExperiment
from transformer_arch.components import ClassicTransformer
from transformer_arch.LLaMa import LLaMa
from transformer_arch.nGPT import nGPT
from training.utils import (
    count_parameters,
    ShakespeareDataModule,
    download_and_split_shakespeare,
)
from pytorch_lightning.loggers import TensorBoardLogger
import torch

# some problem with nGPT, need to fix it
# maybe the sWiGLu is the problem

# modules to try:
# Attention
# SwiGLU
# nGPT_block
# weight norm
# higher lr

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    seq_len = 128  # Or whatever sequence length you want
    batch_size = 32  # Choose an appropriate batch size
    d_model = 64 # 512
    nhead = 4 # 8
    num_layers = 5 # 9
    d_ff_mult = 4



    logger = TensorBoardLogger(
        "lightning_logs", name=f"baseline_LLaMa_transformer_{seq_len}_{d_model}_{d_ff_mult}_{num_layers}_{nhead}" # seq, d_model, d_ff mult, num_layers, nhead
    )  # Optional logging
    # --- Data Loading ---
    download_and_split_shakespeare()  # Download and prepare data if needed


    data_module = ShakespeareDataModule(
        train_file="train.txt",
        val_file="val.txt",
        test_file="test.txt",
        seq_len=seq_len,
        batch_size=batch_size,
    )
    data_module.setup()  # Very important to setup the data
    vocab_size = data_module.get_vocab_size()

    # --- Model Definition ---
    model = LLaMa(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=0.1,
        vocab_size=vocab_size,
        seq_len=seq_len,
    )
    # Print parameter count:
    num_params = count_parameters(model)
    print(f"The model has {num_params:,} trainable parameters.")

    # --- Training Setup ---
    experiment = TransformerExperiment(
        model, learning_rate=2e-4, vocab_size=vocab_size
    )  # Use vocab_size

    # Checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="{epoch}-{val_loss:.2f}-{val_perplexity:.2f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )

    # Early Stopping
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=20, verbose=True, mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=5,  # Or use max_steps for finer control with LR scheduling
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback, early_stopping_callback],
        limit_train_batches=1000,
        limit_val_batches=25,
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=0.2,
    )

    trainer.fit(experiment, datamodule=data_module)
