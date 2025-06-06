import argparse
import json
import os, io
from money_train_2 import run_experiment
import cProfile,pstats

# based on results, rewrite the download and prepare data to be more efficient for sma and ema calculations
# on my laptop, run dataloader w/o workers and no persistent workers (much faster than default)


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    parser = argparse.ArgumentParser(description="Train a Transformer model.")

    parser.add_argument(
        "--config",
        type=str,
        default="./experiment_configs/profile.json",
        # default="./experiment_configs/experiment.json",
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

    profiler.disable()
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    # ps.print_stats()
    # print(s.getvalue())

    ps.dump_stats("profile.prof")

    with open("profile.txt", "w") as f:
        f.write(s.getvalue())