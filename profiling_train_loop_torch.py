# TODO add more features, add more indicators (quarterly reports, EPS, etc.)
# TODO test putting all input features as part of sequence?

# TODO add r squared to loss metrics


# TODO bayesian optimization (inference + training)
# TODO add more things as part of config (weights for loss, features, etc.)
# TODO rewrite dataloader to take in a list of features

# TODO noMachine on hypkos computer

# TODO make timing be per epoch
# TODO maybe try optimising some parts of the code by using c?

# TODO set up old pc for training

# TODO test nGPT

# TODO check stat for noise

# different scaling in attn
# TODO redo some tests (global vs local, etc. groupnorm)


import json
import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from training.money_experiment_2_MTP import MoneyExperiment

from training.utils import (
    count_parameters,
)
# from training.data_loaders.stocks_time_series_2_MTP import (
#     FinancialNumericalDataModule,
#     download_numerical_financial_data,
# )
from training.data_loaders.test_feats_stocks_time_series_2_MTP_new import (
    FinancialNumericalDataModule,
    download_numerical_financial_data,
)

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
# from transformer_arch.money.money_former_MLA_DINT_cog_attn_2_MTP_embed_proj import Money_former_MLA_DINT_cog_attn_MTP
from transformer_arch.money.money_former_MLA_DINT_cog_attn_2_MTP import Money_former_MLA_DINT_cog_attn_MTP


# for profiling
from torch.profiler import profile, record_function, ProfilerActivity
import time

# Before compiling run this line (to set up c++ compiling things)
# & "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64


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
    # TODO add loss functions
    # seed = args.seed
    pred_indices = (
        args.indices_to_predict
    )  # how many datapoints in the future to predict (workdays, not regular days, because market no work weekend)

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
        f"LLaMa seq_len:{seq_len} d_model:{d_model} d_ff:{d_ff} num_layers:{num_layers} nhead:{nhead} dropout:{dropout} lr:{lr} t_total:{t_total} warmup_steps:{warmup_steps} t_0:{t_0} t_mult:{t_mult} lr_mult:{lr_mult} batch_size:{batch_size}"
    )

    name = f"profiling/{args.dataset}/{args.folder_name}/{architecture}_{seq_len}_{d_model}_{d_ff}_{num_layers}_{nhead}_{batch_size}"
    if args.extra_descriptor:
        name = name + "_" + args.extra_descriptor

    logger = TensorBoardLogger(
        "Money_logs",
        name=name,  # seq, d_model, d_ff mult, num_layers, nhead
    )  # Optional logging
    # --- Data Loading ---

    # NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() > 1 else 0
    NUM_WORKERS = 0

    if args.dataset == "Money":  # yahoo finance stock data

        args.normalization_means, args.normalization_stds = download_numerical_financial_data(
            tickers=args.tickers,
            seq_len=seq_len,
            check_if_already_downloaded=False,  # TODO make this better/check which features are missing
            target_dates=pred_indices,
            prediction_type=args.prediction_type,
            classification_threshold=args.classification_threshold
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
            num_workers=NUM_WORKERS
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


    model = experiment.model
    optimizer = experiment.configure_optimizers()['optimizer']
    # checking torch.compile()
    model = torch.compile(model)

    train_loader = data_module.train_dataloader()
    data_iter = iter(train_loader)

    compile_start_time = time.time()
    print("Warming up...")
    for _ in range(2):
        try:
            batch = next(data_iter)
            # ... (simplified training step logic) ...
            inputs, labels = batch
            inputs = inputs.permute(0, 2, 3, 4, 1)
            tickers = torch.arange(experiment.num_sequences, device=inputs.device).unsqueeze(0).unsqueeze(0).repeat(inputs.shape[0], inputs.shape[1], 1)
            outputs = model(inputs, tickers)
            loss = torch.mean(outputs.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        except StopIteration:
            print("Dataloader too small for warmup.")
            break # Exit warmup if dataloader is exhausted
    
    compile_time = time.time() - compile_start_time
    print(f"Warmed and/or compiled model in {compile_time:.2f} seconds")
    
    # --- The Main Profiling Block (e.g., 5 steps) ---
    start_time = time.time()
    print("Starting profiled steps...")
    PROFILED_STEPS = 20
    with profile(
        activities=[ProfilerActivity.CPU], # Change to [ProfilerActivity.CPU, ProfilerActivity.CUDA] on GPU
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=18, repeat=1), # A schedule to ignore the first step
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'), # For detailed analysis
        record_shapes=False,
        with_stack=True,
        with_modules=True
    ) as prof:
        for step in range(PROFILED_STEPS):
            try:
                batch = next(data_iter)
            except StopIteration:
                print("Dataloader ran out of batches during profiling.")
                break # Exit loop if dataloader is exhausted

            # with record_function(f"train_step"):
                # The full training step logic is inside the profiler's scope
            inputs, labels = batch
            inputs = inputs.permute(0, 2, 3, 4, 1)
            tickers = torch.arange(experiment.num_sequences, device=inputs.device).unsqueeze(0).unsqueeze(0).repeat(inputs.shape[0], inputs.shape[1], 1)
            outputs = model(inputs, tickers)
            loss = torch.mean(outputs.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            # This tells the profiler that a step is complete. Important for the schedule.
            prof.step()


    # --- Print the results ---
    print("\n--- Profiler Results (Averaged over profiled steps) ---")
    # Sort by total CUDA time if on GPU, otherwise by total CPU time
    sort_key = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=30))

    print(f"Total time: {time.time() - start_time:.2f} seconds")

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
