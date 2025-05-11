# TODO use NLL for std dev in v2 loss



import json
import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from training.money_experiment import TransformerExperiment

from training.utils import (
    count_parameters,
)
from training.data_loaders.stocks_time_series import (
    FinancialNumericalDataModule,
    download_and_process_numerical_financial_data
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
    cce_fn = args.custom_cross_entropy
    # seed = args.seed
    extra_descriptor = args.extra_descriptor

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
        f"type: {type} {architecture}_transformer seq_len:{seq_len} d_model:{d_model} d_ff:{d_ff} num_layers:{num_layers} nhead:{nhead} dropout:{dropout} lr:{lr} t_total:{t_total} warmup_steps:{warmup_steps} t_0:{t_0} t_mult:{t_mult} lr_mult:{lr_mult} batch_size:{batch_size} cce_fn:{cce_fn}"
    )

    name = f"money_{type}_{architecture}_transformer_{seq_len}_{d_model}_{d_ff}_{num_layers}_{nhead}_{batch_size}"
    if cce_fn == "stablemax" or cce_fn == "taylor_softmax" or cce_fn == "softmax":
        name = name + "_" + cce_fn
    if extra_descriptor != "":
        name = name + "_" + extra_descriptor
    
    if args.use_character_encoding:
        name = args.dataset + "_char/" + name
    else:
        name = args.dataset + "/" + name

    logger = TensorBoardLogger(
        "lightning_logs",
        name=name,  # seq, d_model, d_ff mult, num_layers, nhead
    )  # Optional logging
    # --- Data Loading ---
    if args.dataset == "stocks_yf": # yahoo finance stock data
        download_and_process_numerical_financial_data(...) # TODO
        data_module = FinancialNumericalDataModule(
            ...
        )

    data_module.setup()  # Very important to setup the data
    vocab_size = data_module.get_vocab_size()

    # --- Model Definition ---
    match architecture:
        case "Money_former":
            model = Money_former( # TODO
                args=args,
                vocab_size=vocab_size
            )
        case _:
            raise ValueError(f"Architecture {architecture} not supported")
    # Print parameter count:
    num_params = count_parameters(model)
    print(f"The model has {num_params:,} trainable parameters. Parameter dtype: {args.dtype}")
    args.num_params = num_params

    # --- Training Setup ---

    experiment = TransformerExperiment(
        model,
        learning_rate=lr,
        batch_size=batch_size,
        vocab_size=vocab_size,
        warmup_steps=warmup_steps,
        t_0=t_0,
        t_mult=t_mult,
        lr_mult=lr_mult,
        cce_fn=cce_fn,
        args=args,
        # dtype=torch_dtype_for_params
    )  # Use vocab_size

    # Checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="{name}-{epoch}-{val_loss:.2f}-{val_perplexity:.2f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )

    # Early Stopping
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=1000, verbose=True, mode="min"
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
        val_check_interval=500,
        precision=trainer_precision
    )

    trainer.fit(experiment, datamodule=data_module)

    model_dir = f"models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(experiment.model, f"{model_dir}/money_{args.architecture}_{args.dataset}.pth") # TODO make this more specific
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
        "--config", type=str, default="./experiment_configs/MLA_test.json", help="Path to config file."
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
            default="DINT_nGPT",#"DINT",
            help="Model architecture (LLaMa, ...)",
        )
        parser.add_argument("--d_model", type=int, default=128, help="Embedding dimension.")
        parser.add_argument(
            "--nhead", type=int, default=8, help="Number of attention heads."
        )
        parser.add_argument("--num_layers", type=int, default=4, help="Number of layers.")
        parser.add_argument("--d_ff_mult", type=int, default=4, help="Multiplier for d_ff")
        parser.add_argument(
            "--groups", type=int, default=4, help="Number of groups for GQA."
        )
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

        # for custor_cross_entropy
        parser.add_argument(
            "--custom_cross_entropy",
            type=str,
            default="false",
            help="Use custom cross entropy.",
        )  # stablemax, taylor_softmax

        parser.add_argument(
            "--seed", type=int, default=42, help="Seed for reproducibility."
        )
        parser.add_argument(
            "--extra_descriptor", type=str, default="", help="Extra descriptor for logging."
        )
        parser.add_argument("--orthograd", type=bool, default=True, help="Use OrthoGrad.")
        parser.add_argument(
            "--v1", type=bool, default=False, help="Use V1. (currently only Dint)"
        )
        parser.add_argument(
            "--dataset", type=str, default="tiny_shakespeare", help="Dataset to use."
        )
        parser.add_argument(
            "--use_character_encoding",
            type=bool,
            default=False,
            help="Use character-level encoding instead of the tokenizer.",
        )

    args = parser.parse_args()
    run_experiment(args)
