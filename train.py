import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from training.experiment import TransformerExperiment
from transformer_arch.components import ClassicTransformer
from transformer_arch.LLaMa import LLaMa
# from transformer_arch.nGPT import nGPT, normalize_weights_and_enforce_positive_eigenvalues
# from transformer_arch.DIFF_transformer import DiffTransformer
from training.utils import (
    count_parameters,
    ShakespeareDataModule,
    download_and_split_shakespeare,
)
from pytorch_lightning.loggers import TensorBoardLogger
import torch

# dataleakage, ignore all results pre ~nGPT architecture, as this is when i found out

# TODO implement flashattention
# combine swiglu into one module
# combine uv into one linear layer (linear_in and linear_gate)

# change how args get passed to model, should use args instead
# update older code (maybe, idk)

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


def run_experiment(args):
    torch.set_float32_matmul_precision("medium") # turns out this is not exclusive to gpu(~20% faster), cpu(~0% faster, maybe even slower)
    # add flashattn to speed things up for gpu and cpu too
    # torch.bfloat16 # extra speed up??

    architecture = args.architecture
    seq_len = args.seq_len
    batch_size = args.batch_size
    d_model = args.d_model
    nhead = args.nhead
    num_layers = args.num_layers
    d_ff_mult = args.d_ff_mult
    groups = args.groups
    dropout = args.dropout
    lr = args.lr
    t_total=args.t_total
    warmup_steps=args.warmup_steps
    t_0=args.t_0
    t_mult=args.t_mult
    lr_mult=args.lr_mult
    type = args.type


    logger = TensorBoardLogger(
        "lightning_logs", name=f"{type}_{architecture}_transformer_{seq_len}_{d_model}_{d_ff_mult}_{num_layers}_{nhead}" # seq, d_model, d_ff mult, num_layers, nhead
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
    match architecture:
        case "Classic":
            model = ClassicTransformer(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dropout=dropout,
                vocab_size=vocab_size,
                seq_len=seq_len,
            )
        case "LLaMa":
            model = LLaMa(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                d_ff=d_model * d_ff_mult,
                dropout=dropout,
                vocab_size=vocab_size,
                seq_len=seq_len,
            )
        case "nGPT":
            model = nGPT(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dropout=dropout,
                vocab_size=vocab_size,
                seq_len=seq_len,
            )
        case "Diff":
            model = DiffTransformer(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                d_ff=d_model * d_ff_mult,
                dropout=dropout,
                vocab_size=vocab_size,
                seq_len=seq_len,
                groups=groups,
            )
        case _:
            raise ValueError(f"Architecture {architecture} not supported")
    # Print parameter count:
    num_params = count_parameters(model)
    print(f"The model has {num_params:,} trainable parameters.")

    # --- Training Setup ---
    # if model.__class__.__name__ == "nGPT":
    #     normalize_weights_and_enforce_positive_eigenvalues(model)
    
    experiment = TransformerExperiment(
        model, learning_rate=lr, batch_size=batch_size, vocab_size=vocab_size, warmup_steps=warmup_steps, t_0=t_0, t_mult=t_mult, lr_mult=lr_mult
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
        monitor="val_loss", patience=25, verbose=True, mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=t_total//1000,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback, early_stopping_callback],
        limit_train_batches=1000
        limit_val_batches=25,
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=200,
    )

    trainer.fit(experiment, datamodule=data_module)

    return

if __name__ == "__main__":
     parser = argparse.ArgumentParser(description="Train a Transformer model.")

     # Model architecture arguments (same as before)
     parser.add_argument("--architecture", type=str, default="LLaMa", help="Model architecture (LLaMa, ...)")
     parser.add_argument("--d_model", type=int, default=512, help="Embedding dimension.")
     parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads.")
     parser.add_argument("--num_layers", type=int, default=9, help="Number of layers.")
     parser.add_argument("--d_ff_mult", type=int, default=4, help="Multiplier for d_ff")
     parser.add_argument("--groups", type=int, default=4, help="Number of groups for GQA.")
     parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability.")
     parser.add_argument("--type", type=str, default="baseline", help="Experiment type (for logging).")

     # Training arguments (same as before)
     parser.add_argument("--lr", type=float, default=4e-4, help="Learning rate.")
     parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps.")
     parser.add_argument("--t_total", type=int, default=100000, help="Total training steps.")
     parser.add_argument("--t_0", type=int, default=5000, help="Initial period for cosine annealing.")
     parser.add_argument("--t_mult", type=float, default=1.5, help="Multiplier for period.")
     parser.add_argument("--lr_mult", type=float, default=0.5, help="Multiplier for peak LR.")
     parser.add_argument("--seq_len", type=int, default=128, help="Sequence length.")
     parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")

     args = parser.parse_args()
     run_experiment(args)