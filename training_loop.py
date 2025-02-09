
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from training.experiment import TransformerExperiment
from transformer_arch.components import ClassicTransformer
from transformer_arch.LLaMa import LLaMa
from transformer_arch.nGPT import nGPT, normalize_weights_and_enforce_positive_eigenvalues
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

# TODO from nGPT implementation
# def _init_weights(self, module):
#     if isinstance(module, nn.Linear):
#         torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.base_scale)
#         if module.bias is not None:
#             torch.nn.init.zeros_(module.bias)
#     elif isinstance(module, nn.Embedding):
#         torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.base_scale)

# # init all weights
# self.apply(self._init_weights)
# # apply special scaled init to the residual projections, per GPT-2 paper
# for pn, p in self.named_parameters():
#     if pn.endswith('c_proj.weight'):
#         torch.nn.init.normal_(p, mean=0.0, std=config.base_scale/math.sqrt(2 * config.n_layer))


def run_experiment():
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    seq_len = 128
    batch_size = 32
    d_model = 512
    nhead = 8
    num_layers = 9
    d_ff_mult = 4
    group_size = 4
    dropout = 0.1
    lr = 7.5e-4
    warmup_steps=100
    t_0=5000
    t_mult=1.5
    lr_mult=0.5


    logger = TensorBoardLogger(
        "lightning_logs", name=f"baseline_nGPT_transformer_{seq_len}_{d_model}_{d_ff_mult}_{num_layers}_{nhead}" # seq, d_model, d_ff mult, num_layers, nhead
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
    model = nGPT(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
        vocab_size=vocab_size,
        seq_len=seq_len,
    )
    # Print parameter count:
    num_params = count_parameters(model)
    print(f"The model has {num_params:,} trainable parameters.")

    # --- Training Setup ---
    if model.__class__.__name__ == "nGPT":
        normalize_weights_and_enforce_positive_eigenvalues(model)
    
    experiment = TransformerExperiment(
        model, learning_rate=lr, vocab_size=vocab_size
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
        max_steps=5000,  # Or use max_steps for finer control with LR scheduling
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback, early_stopping_callback],
        limit_val_batches=25,
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=200,
    )

    trainer.fit(experiment, datamodule=data_module)

    return

if __name__ == "__main__":
    run_experiment()