# dataleakage, ignore all results pre ~nGPT architecture, as this is when i found out
# TODO implement flashattention (doesn't work, compile fails)
# TODO better RoPE (not very efficient rn, maybe)

# TODO add weight sharing for embeddings, holy moly 100M parameters just for embeddings!!!!
# TODO make encoding an argument
import json
import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from training.experiment import TransformerExperiment
from transformer_arch.components import ClassicTransformer
from transformer_arch.LLaMa import LLaMa
from transformer_arch.nGPT import (
    nGPT,
    normalize_weights_and_enforce_positive_eigenvalues,
)
from transformer_arch.DIFF import DiffTransformer
from transformer_arch.DINT import DintTransformer
from training.utils import (
    count_parameters,
)
from training.data_loaders.wikitext import (
    WikitextDataModule,
    download_and_split_wikitext,
)
from training.data_loaders.tiny_shakespeare import (
    ShakespeareDataModule,
    download_and_split_shakespeare,
)
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.nn.attention import SDPBackend


from transformer_arch.DINT_nGPT import DINT_nGPT

from transformer_arch.LLaMa_MLA import LLaMa_MLA

# TODO figure out group norm for DINT and DIFF

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

    # pl.seed_everything(seed)

    print(
        f"type: {type} {architecture}_transformer seq_len:{seq_len} d_model:{d_model} d_ff_mult:{d_ff} num_layers:{num_layers} nhead:{nhead} dropout:{dropout} lr:{lr} t_total:{t_total} warmup_steps:{warmup_steps} t_0:{t_0} t_mult:{t_mult} lr_mult:{lr_mult} batch_size:{batch_size} cce_fn:{cce_fn}"
    )

    name = f"{type}_{architecture}_transformer_{seq_len}_{d_model}_{d_ff}_{num_layers}_{nhead}_{batch_size}"
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
    if args.dataset == "tiny_shakespeare":
        download_and_split_shakespeare()  # Download and prepare data if needed
        data_module = ShakespeareDataModule(
            train_file="tiny_shakespeare/train.txt",
            val_file="tiny_shakespeare/val.txt",
            test_file="tiny_shakespeare/test.txt",
            seq_len=seq_len,
            batch_size=batch_size,
            use_character_encoding=args.use_character_encoding,
        )
    elif args.dataset == "wikitext2":
        download_and_split_wikitext()  # Download and prepare data if needed
        data_module = WikitextDataModule(
            train_file="wikitext_data/wiki.train.tokens",
            val_file="wikitext_data/wiki.valid.tokens",
            test_file="wikitext_data/wiki.test.tokens",
            seq_len=seq_len,
            batch_size=batch_size,
            use_character_encoding=args.use_character_encoding,
            encoding_name="r50k_base",
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

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
                groups=groups,
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
        case "DIFF":
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
        case "DINT":
            model = DintTransformer(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                d_ff=d_model * d_ff_mult,
                dropout=dropout,
                vocab_size=vocab_size,
                seq_len=seq_len,
                groups=groups,
                v1=args.v1,
            )
        case "DINT_nGPT":
            model = DINT_nGPT(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                d_ff=d_model * d_ff_mult,
                dropout=dropout,
                vocab_size=vocab_size,
                seq_len=seq_len,
                groups=groups,
            )
        case "MLA":
            model = LLaMa_MLA(
                args=args,
                vocab_size=vocab_size
            )
        case _:
            raise ValueError(f"Architecture {architecture} not supported")
    # Print parameter count:
    num_params = count_parameters(model)
    print(f"The model has {num_params:,} trainable parameters.")

    # --- Training Setup ---
    if model.__class__.__name__ == "nGPT" or model.__class__.__name__ == "DINT_nGPT":
        normalize_weights_and_enforce_positive_eigenvalues(model)

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
        log_every_n_steps=30,
        val_check_interval=500,
    )

    trainer.fit(experiment, datamodule=data_module)

    model_dir = f"models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(experiment.model, f"{model_dir}/{args.architecture}_{args.dataset}.pth") # TODO make this more specific
    print("Model saved.")
    return


def run_experiment(args: argparse.Namespace):
    torch.set_float32_matmul_precision(
        "medium"
    )  # turns out this is not exclusive to gpu(~20% faster), cpu(~0% faster, maybe even slower)
    # add flashattn to speed things up for gpu and cpu too
    # torch.bfloat16 # extra speed up??
    if torch.cuda.is_available():
        # if torch.backends.cuda.is_flash_attention_available():
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
        # else:
        #     proceed(args)
        #     return
    else:
        proceed(args)
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Transformer model.")

    parser.add_argument(
        "--config", type=str, default="", help="Path to config file."
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
