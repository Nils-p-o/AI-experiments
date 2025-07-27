import pytorch_lightning as pl
import torch.optim as optim
import torch
import torch.nn as nn
import math
from torch.optim.lr_scheduler import LambdaLR
from .utils import OrthoGrad, custom_cross_entropy, stablemax, taylor_softmax
import argparse
import numpy as np
from scipy import stats
from transformer_arch.money.money_former_nGPT_2 import (
    normalize_weights_and_enforce_positive_eigenvalues,
)
from money_predictions_MTP_local_new import (
    download_and_process_inference_data,
    get_mtp_predictions_for_backtest,
    calculate_metrics,
    trading_metrics
)

# for debug / timimg for speedups
import time

cummulative_times = {"preprocessing": 0, "model": 0, "loss": 0, "metrics": 0}


def get_direction_accuracy_returns(actual_returns, predicted_returns, thresholds=0.01):
    batch, seq_len, features, targets, sequences = actual_returns.shape
    actual_movements = torch.where(
        actual_returns > thresholds, 1, torch.where(actual_returns < -thresholds, -1, 0)
    )
    predicted_movements = torch.where(
        predicted_returns > thresholds,
        1,
        torch.where(predicted_returns < -thresholds, -1, 0),
    )

    acc_up = ((actual_movements == 1) * (predicted_movements == 1)).sum() / (
        actual_movements == 1
    ).sum()
    acc_down = ((actual_movements == -1) * (predicted_movements == -1)).sum() / (
        actual_movements == -1
    ).sum()
    if actual_movements.sum() == 0:
        acc_flat = 0
    else:
        acc_flat = ((actual_movements == 0) * (predicted_movements == 0)).sum() / (
            actual_movements == 0
        ).sum()

    total_acc = (
        actual_movements == predicted_movements
    ).sum() / actual_movements.numel()

    # per target
    actual_up = (actual_movements == 1).sum(dim=(0, 1, 2, 4)).unsqueeze(0)
    actual_down = (actual_movements == -1).sum(dim=(0, 1, 2, 4)).unsqueeze(0)
    actual_flat = (actual_movements == 0).sum(dim=(0, 1, 2, 4)).unsqueeze(0)

    expected_acc = torch.max(
        torch.cat([actual_up, actual_down, actual_flat], dim=0), dim=0
    ).values / (batch * seq_len * sequences * features)
    expected_strat = (
        torch.cat([actual_up, actual_down, actual_flat], dim=0).square().sum(dim=0)
        / (batch * seq_len * sequences * features) ** 2
    )
    expected_max_acc_target = torch.max(
        torch.cat([expected_acc.unsqueeze(0), expected_strat.unsqueeze(0)], dim=0),
        dim=0,
    ).values
    acc_per_target = torch.sum(
        actual_movements == predicted_movements, dim=(0, 1, 2, 4)
    ) / (batch * seq_len * sequences * features)

    # per sequence
    # actual_up = (actual_movements == 1).sum(dim=(0, 1, 2)).unsqueeze(0)
    # actual_down = (actual_movements == -1).sum(dim=(0, 1, 2)).unsqueeze(0)
    # actual_flat = (actual_movements == 0).sum(dim=(0, 1, 2)).unsqueeze(0)
    # expected_acc = torch.max(
    #     torch.cat([actual_up, actual_down, actual_flat], dim=0), dim=0
    # ).values / (batch * seq_len * targets)
    # expected_strat = (
    #     torch.cat([actual_up, actual_down, actual_flat], dim=0).square().sum(dim=0)
    #     / (batch * seq_len * targets) ** 2
    # )
    # expected_max_acc_sequence = torch.max(
    #     torch.cat([expected_acc.unsqueeze(0), expected_strat.unsqueeze(0)], dim=0),
    #     dim=0,
    # ).values
    # acc_per_sequence = torch.sum(
    #     actual_movements == predicted_movements, dim=(0, 1, 2)
    # ) / (batch * seq_len * targets)

    return (
        total_acc,
        acc_up,
        acc_down,
        acc_flat,
        expected_max_acc_target,
        # expected_max_acc_sequence,
        acc_per_target,
        # acc_per_sequence,
    )


def get_spearmanr_correlations_pytorch( # TODO adapt for MTP
    actual_values: torch.Tensor, predicted_values: torch.Tensor, epsilon: float = 1e-12
) -> torch.Tensor:
    """
    Calculates Spearman rank correlation coefficients using PyTorch operations.
    Correlations are computed along the last dimension.

    Args:
        actual_values (torch.Tensor): Tensor of actual values. Shape (..., N_assets)
        predicted_values (torch.Tensor): Tensor of predicted values. Shape (..., N_assets)
        epsilon (float): Small value to prevent division by zero.

    Returns:
        torch.Tensor: Spearman correlations. Shape (...)
    """
    batch_size, seq_len, num_targets, sequences = actual_values.shape

    if sequences < 2:
        # For consistency, create a tensor of NaNs with the correct leading dimensions
        correlations = torch.full(
            (batch_size, seq_len, num_targets),
            float("nan"),
            device=actual_values.device,
            dtype=actual_values.dtype,
        )
        return correlations

    # --- Step 1: Get ranks ---
    # PyTorch doesn't have a direct rankdata equivalent that handles ties like 'average' easily
    # for multidimensional tensors along a specific axis in one go.
    # We need to sort and then deal with ties.
    # A common way to get ranks is to sort, then argsort the sort.

    # Sort actual_values to get indices
    actual_sorted_indices = torch.argsort(actual_values, dim=-1)
    # Create ranks based on these sorted indices.
    # Add 1 because ranks are typically 1-based, though for correlation math it doesn't strictly matter
    # as long as it's consistent. Using 0-based is also fine.
    # To handle ties like scipy's 'average', it's more complex.
    # For a simpler PyTorch version (which might differ slightly from scipy with ties):
    ranked_actual = torch.empty_like(actual_values)
    ranked_actual.scatter_(
        dim=-1,
        index=actual_sorted_indices,
        src=torch.arange(
            sequences, device=actual_values.device, dtype=actual_values.dtype
        ).repeat(batch_size, seq_len, num_targets, 1),
    )
    # This simple scatter approach gives ranks for unique values, but for ties, it assigns
    # the rank of the first occurrence.
    # For a more robust ranking like 'average', we'd need to identify ties and adjust.
    # However, let's proceed with this simpler ranking first and see performance.
    # If exact tie handling is critical and different, a custom loop or more complex ops are needed.
    # For many financial use cases, slight differences in tie handling in ranks might be acceptable.

    predicted_sorted_indices = torch.argsort(predicted_values, dim=-1)
    ranked_predicted = torch.empty_like(predicted_values)
    ranked_predicted.scatter_(
        dim=-1,
        index=predicted_sorted_indices,
        src=torch.arange(
            sequences, device=predicted_values.device, dtype=predicted_values.dtype
        ).repeat(batch_size, seq_len, num_targets, 1),
    )

    # The ranks from argsort().argsort() would be 0 to N-1. If we want average for ties:
    # This is a bit more involved to do efficiently in pure PyTorch without loops.
    # For now, let's use the ranks from above (0 to N-1) and proceed to Pearson on ranks.
    # Note: The exact values will differ from scipy's 'average' if there are many ties.
    # If this is a major concern, a more complex ranking function is needed.
    # A common workaround for simpler PyTorch ranking when direct 'average' tie method isn't available:
    # Add small random noise to break ties before argsort if an exact match to scipy is not paramount
    # but you want to avoid the bias of first-occurrence ranking.
    # actual_values_noised = actual_values + torch.rand_like(actual_values) * 1e-9
    # predicted_values_noised = predicted_values + torch.rand_like(predicted_values) * 1e-9
    # ranked_actual = torch.argsort(torch.argsort(actual_values_noised, dim=-1), dim=-1).to(actual_values.dtype)
    # ranked_predicted = torch.argsort(torch.argsort(predicted_values_noised, dim=-1), dim=-1).to(predicted_values.dtype)

    # Using the argsort().argsort() approach which gives ranks from 0 to N-1
    # This handles ties by assigning the same rank to all tied elements if they were truly identical
    # before sorting, but gives unique ranks if values are distinct.
    # This is a common way to get ranks in PyTorch.
    ranked_actual = torch.argsort(torch.argsort(actual_values, dim=-1), dim=-1).to(
        actual_values.dtype
    )
    ranked_predicted = torch.argsort(
        torch.argsort(predicted_values, dim=-1), dim=-1
    ).to(predicted_values.dtype)

    # --- Step 2: Calculate Pearson correlation on the ranks ---
    # Demean the ranks along the last axis
    mean_ranked_actual = ranked_actual.mean(dim=-1, keepdim=True)
    mean_ranked_predicted = ranked_predicted.mean(dim=-1, keepdim=True)

    ranked_actual_demeaned = ranked_actual - mean_ranked_actual
    ranked_predicted_demeaned = ranked_predicted - mean_ranked_predicted

    # Numerator of Pearson correlation
    numerator = (ranked_actual_demeaned * ranked_predicted_demeaned).sum(dim=-1)

    # Denominator of Pearson correlation
    sum_sq_demeaned_actual = (ranked_actual_demeaned**2).sum(dim=-1)
    sum_sq_demeaned_predicted = (ranked_predicted_demeaned**2).sum(dim=-1)

    denominator = torch.sqrt(sum_sq_demeaned_actual * sum_sq_demeaned_predicted)

    # Calculate correlations
    correlations = torch.full_like(denominator, float("nan"))  # Initialize with NaN

    # Create a mask for valid denominators (non-zero)
    valid_mask = denominator > epsilon

    correlations[valid_mask] = numerator[valid_mask] / denominator[valid_mask]

    # Clip to ensure values are within [-1, 1] due to potential floating point inaccuracies
    correlations = torch.clamp(correlations, -1.0, 1.0)

    return correlations


class MoneyExperiment(pl.LightningModule):
    def __init__(
        self,
        model,
        learning_rate=2e-5,
        batch_size=32,
        # vocab_size=2,
        warmup_steps=100,
        t_0=5000,
        t_mult=1.5,
        lr_mult=0.5,  # maybe higher?
        args: argparse.Namespace = None,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.t_0 = t_0
        self.t_mult = t_mult
        self.lr_mult = lr_mult
        self.batch_size = batch_size
        match args.prediction_type:
            case "gaussian":
                self.loss_fn = nn.GaussianNLLLoss(reduction="mean")
            case "regression":
                self.loss_fn = nn.L1Loss()
            case "classification":
                self.loss_fn = nn.CrossEntropyLoss()
        self.MAE = nn.L1Loss()
        self.threshold = 0.003
        self.pred_indices = args.indices_to_predict
        self.save_hyperparameters(ignore=["model"])
        self.args = args
        self.num_sequences = len(args.tickers)
        self.tickers = args.tickers

        self.cli_args = argparse.Namespace()
        self.cli_args.start_date_data = "2020-01-01"
        self.cli_args.end_date_data = "2025-05-25"
        self.cli_args.days_to_check = 1300
        self.cli_args.average_predictions = True
        self.cli_args.batch_size = batch_size
        self.cli_args.pred_day = 1
        if self.cli_args.average_predictions:
            self.cli_args.pred_day = max(self.pred_indices)
        self.validation_step_outputs = []
        # self.backtest_input_data, self.backtest_target_data = download_and_process_inference_data(
        #     args, self.cli_args.start_date_data, self.cli_args.end_date_data, self.cli_args)
        # self.backtest_input_data = self.backtest_input_data[:, :, -self.cli_args.days_to_check:, :, :]
        # self.backtest_target_data = self.backtest_target_data[0, -self.cli_args.days_to_check:, :]


    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def _shared_step(self, batch, batch_idx, stage="train"):
        time_preprocessing_start = time.time_ns()
        inputs, labels = batch
        # inputs (batch_size, features, targets, seq_len, num_sequences)
        # labels (batch_size, features (chlov), targets, seq_len, num_sequences)
        batch_size, _, _, seq_len, num_sequences = inputs.shape

        # inputs = inputs.permute(0, 2, 3, 1)  # (batch_size, seq_len, num_sequences, features)
        inputs = inputs.permute(0, 2, 3, 4, 1) # (batch_size, targets, seq_len, num_sequences, features)

        targets = labels  # (batch_size, features, targets, seq_len, num_sequences)
        targets = targets.permute(0, 3, 1, 2, 4)  # (batch_size, seq_len, features, targets, num_sequences)

        # seperator = torch.zeros(
        #     (batch_size, len(self.pred_indices), 1), dtype=torch.int, device=inputs.device
        # )  # (batch_size, targets, 1)
        tickers = torch.arange(self.num_sequences, device=inputs.device)
        tickers = tickers.unsqueeze(0).unsqueeze(0).repeat(
            batch_size, len(self.pred_indices), 1
        )  # (batch_size, targets, num_sequences)

        if stage == "train":
            cummulative_times["preprocessing"] += (
                time.time_ns() - time_preprocessing_start
            ) / 1e6
        time_model_start = time.time_ns()
        outputs = self(inputs, tickers)
        if self.args.use_global_seperator:
            outputs = outputs[:, :, 1:, :]
        match self.args.prediction_type:
            case "gaussian":
                outputs = outputs.view(
                    batch_size, max(self.pred_indices), (seq_len+1-int(self.args.use_global_seperator)), self.num_sequences, 10
                ).permute(0, 2, 4, 1, 3)  # (batch_size, seq_len, features (chlov), targets, num_sequences)
            case "regression":
                outputs = outputs.view(
                    batch_size, max(self.pred_indices), (seq_len+1-int(self.args.use_global_seperator)), self.num_sequences, 5
                ).permute(0, 2, 4, 1, 3)  # (batch_size, seq_len, features (chlov), targets, num_sequences)
            case "classification":
                outputs = outputs.view(
                    batch_size, max(self.pred_indices), (seq_len+1-int(self.args.use_global_seperator)), self.num_sequences, 5 * self.args.num_classes
                ).permute(0, 2, 4, 1, 3)  # (batch_size, seq_len, features (chlov) * classes, targets, num_sequences)
        if stage == "train":
            cummulative_times["model"] += (time.time_ns() - time_model_start) / 1e6
        if not (self.args.use_global_seperator or self.args.include_sep_in_loss):
            outputs = outputs[:, 1:, :, :, :]

        # for debugging reasons
        if torch.isnan(outputs).any():
            print("NAN IN OUTPUTS")

        if isinstance(outputs, torch.Tensor):
            raw_logits = outputs
        else:
            raw_logits = outputs.logits


        time_loss_calc_start = time.time_ns()
        unseen_losses = []
        seen_losses = []
        match self.args.prediction_type:
            case "gaussian":
                raw_logits, raw_pred_std = raw_logits.split([5, 5], dim=2) # mu and log_std (5, 5)
                preds = raw_logits
                pred_std = torch.exp(raw_pred_std)
                pred_variance = pred_std ** 2

                loss = 0
                target_weights = [1.0 for _ in range(max(self.pred_indices))]
                for i in range(max(self.pred_indices)):
                    seen_unseen_weights = [1.0, 1.0]
                    seen_current_preds = preds[:, :-1, :, i, :]
                    unseen_current_preds = preds[:, -1:, :, i, :]
                    seen_current_targets = targets[:, :-1, :, i, :]
                    unseen_current_targets = targets[:, -1:, :, i, :]
                    seen_current_pred_variance = pred_variance[:, :-1, :, i, :]
                    unseen_current_pred_variance = pred_variance[:, -1:, :, i, :]

                    loss += (
                        target_weights[i]
                        * (seen_unseen_weights[0] * 2 / sum(seen_unseen_weights))
                        * self.loss_fn(seen_current_preds, seen_current_targets, seen_current_pred_variance)
                        * ((seq_len-1)/seq_len)
                    )
                    loss += (
                        target_weights[i]
                        * (seen_unseen_weights[1] * 2 / sum(seen_unseen_weights))
                        * self.loss_fn(unseen_current_preds, unseen_current_targets, unseen_current_pred_variance)
                        * (1/seq_len)
                    )

                    seen_losses.append(
                        self.MAE(seen_current_preds, seen_current_targets)
                    )
                    unseen_losses.append(
                        self.MAE(unseen_current_preds, unseen_current_targets)
                    )

                loss = loss / sum(target_weights)
            case "regression":
                preds = raw_logits
                loss = 0
                target_weights = [1.0 for _ in range(max(self.pred_indices))]
                for i in range(max(self.pred_indices)):
                    # seen_unseen_weights = [
                    #     self.pred_indices[i],
                    #     seq_len - self.pred_indices[i],
                    # ]
                    seen_unseen_weights = [1.0, 1.0]
                    seen_current_preds = preds[:, :-1, :, i, :]
                    unseen_current_preds = preds[:, -1:, :, i, :]
                    seen_current_targets = targets[:, :-1, :, i, :]
                    unseen_current_targets = targets[:, -1:, :, i, :]
                    loss += (
                        target_weights[i]
                        * (seen_unseen_weights[0] * 2 / sum(seen_unseen_weights))
                        * self.loss_fn(seen_current_preds, seen_current_targets)
                        * ((seq_len-1)/seq_len)
                    )
                    loss += (
                        target_weights[i]
                        * (seen_unseen_weights[1] * 2 / sum(seen_unseen_weights))
                        * self.loss_fn(unseen_current_preds, unseen_current_targets)
                        * (1/seq_len)
                    )

                    seen_losses.append(
                        self.MAE(seen_current_preds, seen_current_targets)
                    )
                    unseen_losses.append(
                        self.MAE(unseen_current_preds, unseen_current_targets)
                    )
                loss = loss / sum(target_weights)
            case "classification":
                targets_classes = torch.full_like(targets, 1)
                targets_classes[targets < -self.args.classification_threshold] = 0
                targets_classes[targets > self.args.classification_threshold] = 2
                targets_classes = targets_classes.long()
                if self.args.include_sep_in_loss:
                    view_shape = (batch_size, seq_len+1, 5, self.args.num_classes, max(self.pred_indices), self.num_sequences)
                else:
                    view_shape = (batch_size, seq_len, 5, self.args.num_classes, max(self.pred_indices), self.num_sequences)
                logits = raw_logits.view(view_shape).permute(0,3,1,2,4,5)  # (batch_size, num_classes, seq_len, features, targets, num_sequences)
                loss = 0
                target_weights = [1.0 for _ in range(max(self.pred_indices))]
                for i in range(max(self.pred_indices)):
                    seen_unseen_weights = [1.0, 1.0]
                    seen_current_logits = logits[:, :, :-1, :, i, :]
                    unseen_current_logits = logits[:, :, -1:, :, i, :]
                    seen_current_targets = targets_classes[:, :-1, :, i, :]
                    unseen_current_targets = targets_classes[:, -1:, :, i, :]
                    loss += (
                        target_weights[i]
                        * (seen_unseen_weights[0] * 2 / sum(seen_unseen_weights))
                        * self.loss_fn(seen_current_logits, seen_current_targets)
                        * ((seq_len-1)/seq_len if not self.args.include_sep_in_loss else (seq_len/(seq_len+1)))
                    )
                    loss += (
                        target_weights[i]
                        * (seen_unseen_weights[1] * 2 / sum(seen_unseen_weights))
                        * self.loss_fn(unseen_current_logits, unseen_current_targets)
                        * (1/seq_len if not self.args.include_sep_in_loss else (1/(seq_len+1)))
                    )
                    seen_losses.append(
                        self.loss_fn(
                            seen_current_logits, seen_current_targets
                        )
                    )
                    unseen_losses.append(
                        self.loss_fn(
                            unseen_current_logits, unseen_current_targets
                        )
                    )
                loss = loss / sum(target_weights)

                

        if stage == "train":
            cummulative_times["loss"] += (time.time_ns() - time_loss_calc_start) / 1e6

        if torch.isnan(loss).any():
            print("NAN IN LOSS")

        if self.args.prediction_type == "gaussian" or self.args.prediction_type == "classification":
            self.log(
                f"Loss/{stage}_loss",
                loss,
                prog_bar=True,
                on_step=(stage == "train"),
                on_epoch=(stage != "train"),
                logger=True,
                sync_dist=True,
            )
        elif self.args.prediction_type == "regression":
            naive_loss = self.MAE(torch.zeros_like(targets), targets)
            self.log(
                f"Loss/{stage}_loss",
                loss / (naive_loss + 1e-6),
                prog_bar=True,
                on_step=(stage == "train"),
                on_epoch=(stage != "train"),
                logger=True,
                sync_dist=True,
            )

        calculate_detailed_metrics = False
        if stage == "train":
            if (self.global_step + 1) % self.trainer.log_every_n_steps == 0:
                if self.trainer.is_global_zero:
                    calculate_detailed_metrics = True
        elif stage in ["val", "test"]:
            if self.trainer.is_global_zero:
                calculate_detailed_metrics = True

        if calculate_detailed_metrics:
            time_metrics_start = time.time_ns()
            log_opts_step = {
                "on_step": True,
                "on_epoch": False,
                "logger": True,
                "sync_dist": True,
            }
            log_opts_epoch = {
                "on_step": False,
                "on_epoch": True,
                "logger": True,
                "sync_dist": True,
            }
            current_log_opts = log_opts_step if stage == "train" else log_opts_epoch

            self.log(
                f"Losses_seen_unseen/{stage}_loss_seen",
                torch.mean(torch.stack(seen_losses)),
                **current_log_opts,
            )
            self.log(
                f"Losses_seen_unseen/{stage}_loss_unseen",
                torch.mean(torch.stack(unseen_losses)),
                **current_log_opts,
            )
            for i in range(len(self.pred_indices)): # always MAE to compare
                self.log(
                    f"Losses_seen_unseen/{stage}_loss_seen_{self.pred_indices[i]}",
                    seen_losses[i],
                    **current_log_opts,
                )
                self.log(
                    f"Losses_seen_unseen/{stage}_loss_unseen_{self.pred_indices[i]}",
                    unseen_losses[i],
                    **current_log_opts,
                )

            if self.args.prediction_type != "classification":
                (
                    direction_total_acc, _, _, _, expected_acc_target,
                    # expected_acc_sequence,
                    acc_target,
                    # acc_sequence,
                ) = get_direction_accuracy_returns(targets, preds, thresholds=self.threshold)
                self.log(
                    f"Accuracy/{stage}_accuracy", direction_total_acc, **current_log_opts
                )

                if stage == "val":
                    for i in range(len(self.pred_indices)):
                        self.log(
                            f"Accuracy_finer/{stage}_expected_max_acc_{self.pred_indices[i]}",
                            expected_acc_target[i],
                            **current_log_opts,
                        )
                    # for i in range(self.num_sequences):
                    #     self.log(
                    #         f"Accuracy_finer/{stage}_expected_max_acc_{self.tickers[i]}",
                    #         expected_acc_sequence[i],
                    #         **current_log_opts,
                    #     )
                if stage == "val":
                    avg_unseen_target_acc = []
                    for i in range(len(self.pred_indices)):
                        seen_current_preds = preds[:, :-1, :, i, :]
                        seen_current_targets = targets[:, :-1, :, i, :]
                        seen_MAE = self.MAE(seen_current_preds, seen_current_targets)

                        unseen_current_preds = preds[:, -1:, :, i, :]
                        unseen_current_targets = targets[:, -1:, :, i, :]
                        unseen_MAE = self.MAE(unseen_current_preds, unseen_current_targets)

                        relative_MAE = unseen_MAE / (seen_MAE + 1e-6)

                        self.log(
                            f"Split_Error/{stage}_relative_MAE_{self.pred_indices[i]}",
                            relative_MAE,
                            **current_log_opts,
                        )

                        # TODO maybe add split directional accuracy
                        seen_target_movements = torch.where(
                            seen_current_targets > self.threshold,
                            1,
                            torch.where(seen_current_targets < -self.threshold, -1, 0),
                        )
                        seen_predicted_movements = torch.where(
                            seen_current_preds > self.threshold,
                            1,
                            torch.where(seen_current_preds < -self.threshold, -1, 0),
                        )

                        seen_direction_acc = (
                            seen_target_movements == seen_predicted_movements
                        ).sum() / seen_target_movements.numel()
                        self.log(
                            f"Split_Error/{stage}_seen_accuracy_{self.pred_indices[i]}",
                            seen_direction_acc,
                            **current_log_opts,
                        )

                        unseen_target_movements = torch.where(
                            unseen_current_targets > self.threshold,
                            1,
                            torch.where(unseen_current_targets < -self.threshold, -1, 0),
                        )
                        unseen_predicted_movements = torch.where(
                            unseen_current_preds > self.threshold,
                            1,
                            torch.where(unseen_current_preds < -self.threshold, -1, 0),
                        )

                        unseen_direction_acc = (
                            unseen_target_movements == unseen_predicted_movements
                        ).sum() / unseen_target_movements.numel()
                        avg_unseen_target_acc.append(unseen_direction_acc)
                        self.log(
                            f"Split_Error/{stage}_unseen_accuracy_{self.pred_indices[i]}",
                            unseen_direction_acc,
                            **current_log_opts,
                        )
                        self.log(
                            f"Split_Error/{stage}_relative_accuracy_{self.pred_indices[i]}",
                            unseen_direction_acc / (seen_direction_acc + 1e-6),
                            **current_log_opts,
                        )

                    avg_unseen_target_acc = torch.mean(torch.stack(avg_unseen_target_acc))
                    self.log(
                        f"Split_Error/{stage}_unseen_accuracy_avg",
                        avg_unseen_target_acc,
                        **current_log_opts,
                    )

                # can only be done per time step
                # ICs = get_spearmanr_correlations_pytorch(
                #     targets.clone().detach(), preds.clone().detach()
                # )
                for i in range(self.num_sequences):
                    seen_current_preds = preds[:, :-1, :, :, i]
                    seen_current_targets = targets[:, :-1, :, :, i]
                    seen_MAE = self.MAE(seen_current_preds, seen_current_targets)

                    unseen_current_preds = preds[:, -1:, :, :, i]
                    unseen_current_targets = targets[:, -1:, :, :, i]
                    unseen_MAE = self.MAE(unseen_current_preds, unseen_current_targets)

                    self.log(
                        f"Split_Error/{stage}_seen_{self.tickers[i]}_MAE",
                        seen_MAE,
                        **current_log_opts,
                    )
                    self.log(
                        f"Split_Error/{stage}_unseen_{self.tickers[i]}_MAE",
                        unseen_MAE,
                        **current_log_opts,
                    )

                if self.args.prediction_type == "gaussian":
                    self.log(
                        f"Std_dev/{stage}",
                        torch.mean(pred_std),
                        **current_log_opts,
                    )
                    for i in range(self.num_sequences):
                        self.log(
                            f"Std_dev/{stage}_{self.tickers[i]}",
                            torch.mean(pred_std[:, :, :, :, i]),
                            **current_log_opts,
                        )
                    for i in range(len(self.pred_indices)):
                        seen_current_pred_std = pred_std[:, :-1, :, :, i]
                        unseen_current_pred_std = pred_std[:, -1:, :, :, i]

                        self.log(
                            f"Std_dev/{stage}_seen_{self.pred_indices[i]}",
                            torch.mean(seen_current_pred_std),
                            **current_log_opts,
                        )
                        self.log(
                            f"Std_dev/{stage}_unseen_{self.pred_indices[i]}",
                            torch.mean(unseen_current_pred_std),
                            **current_log_opts,
                        )
                

                # close specific losses + accs (unseen, val specifically because these are the ones i actually care about)
                for i in range(len(self.pred_indices)): # close by target period
                    unseen_current_preds_close = preds[:, -1:, 0, i, :]
                    unseen_current_targets_close = targets[:, -1:, 0, i, :]
                    unseen_MAE_close = self.MAE(
                        unseen_current_preds_close, unseen_current_targets_close
                    )
                    self.log(
                        f"Close_Target/{stage}_unseen_{self.pred_indices[i]}_MAE",
                        unseen_MAE_close,
                        **current_log_opts,
                    )

                    unseen_target_movements_close = torch.where(
                        unseen_current_targets_close > self.threshold,
                        1,
                        torch.where(
                            unseen_current_targets_close < -self.threshold, -1, 0
                        ),
                    )
                    unseen_predicted_movements_close = torch.where(
                        unseen_current_preds_close > self.threshold,
                        1,
                        torch.where(
                            unseen_current_preds_close < -self.threshold, -1, 0
                        ),
                    )
                    unseen_direction_acc_close = (
                        unseen_target_movements_close
                        == unseen_predicted_movements_close
                    ).sum() / unseen_target_movements_close.numel()

                    self.log(
                        f"Close_Target/{stage}_unseen_{self.pred_indices[i]}_accuracy",
                        unseen_direction_acc_close,
                        **current_log_opts,
                    )
                
                for i in range(self.num_sequences): # close by sequence
                    unseen_current_preds_close = preds[:, -1:, 0, :, i]
                    unseen_current_targets_close = targets[:, -1:, 0, :, i]
                    unseen_MAE_close = self.MAE(
                        unseen_current_preds_close, unseen_current_targets_close
                    )
                    self.log(
                        f"Close_Sequence/{stage}_unseen_{self.tickers[i]}_MAE",
                        unseen_MAE_close,
                        **current_log_opts,
                    )

                    unseen_target_movements_close = torch.where(
                        unseen_current_targets_close > self.threshold,
                        1,
                        torch.where(
                            unseen_current_targets_close < -self.threshold, -1, 0
                        ),
                    )
                    unseen_predicted_movements_close = torch.where(
                        unseen_current_preds_close > self.threshold,
                        1,
                        torch.where(
                            unseen_current_preds_close < -self.threshold, -1, 0
                        ),
                    )
                    unseen_direction_acc_close = (
                        unseen_target_movements_close
                        == unseen_predicted_movements_close
                    ).sum() / unseen_target_movements_close.numel()
                    self.log(
                        f"Close_Sequence/{stage}_unseen_{self.tickers[i]}_accuracy",
                        unseen_direction_acc_close,
                        **current_log_opts,
                    )
            else: # acc metrics for classification
                if stage == "val": # expected accuracy
                    for i in range(len(self.pred_indices)):
                        blind_guess_acc = []
                        for j in range(self.args.num_classes):
                            blind_guess_acc.append(
                                torch.sum(targets_classes[:, :, :, i, :] == j)
                                / targets_classes[:, :, :, i, :].numel()
                            )
                        expected_acc_target = torch.max(
                            torch.tensor(blind_guess_acc, device=targets_classes.device)
                        )
                        self.log(
                            f"Accuracy_finer/{stage}_expected_max_acc_{self.pred_indices[i]}",
                            expected_acc_target,
                            **current_log_opts,
                        )

                avg_unseen_target_acc = []
                for i in range(max(self.pred_indices)):
                    seen_current_logits = logits[:, :, :-1, :, i, :]
                    unseen_current_logits = logits[:, :, -1:, :, i, :]
                    seen_current_targets = targets_classes[:, :-1, :, i, :]
                    unseen_current_targets = targets_classes[:, -1:, :, i, :]

                    seen_direction_acc = (
                        torch.argmax(seen_current_logits, dim=1)
                        == seen_current_targets
                    ).sum() / seen_current_targets.numel()
                    unseen_direction_acc = (
                        torch.argmax(unseen_current_logits, dim=1)
                        == unseen_current_targets
                    ).sum() / unseen_current_targets.numel()

                    self.log(
                        f"Split_Error/{stage}_seen_accuracy_{self.pred_indices[i]}",
                        seen_direction_acc,
                        **current_log_opts,
                    )
                    self.log(
                        f"Split_Error/{stage}_unseen_accuracy_{self.pred_indices[i]}",
                        unseen_direction_acc,
                        **current_log_opts,
                    )
                    avg_unseen_target_acc.append(unseen_direction_acc)

                self.log(
                    f"Split_Error/{stage}_unseen_accuracy_avg",
                    torch.mean(torch.stack(avg_unseen_target_acc)),
                    **current_log_opts,
                )

                for i in range(max(self.pred_indices)):
                    unseen_current_logits_close = logits[:, :, -1:, 0, i, :]
                    unseen_current_targets_close = targets_classes[:, -1:, 0, i, :]
                    unseen_direction_acc_close = (
                        torch.argmax(unseen_current_logits_close, dim=1)
                        == unseen_current_targets_close
                    ).sum() / unseen_current_targets_close.numel()
                    self.log(
                        f"Close_Target/{stage}_unseen_{self.pred_indices[i]}_accuracy",
                        unseen_direction_acc_close,
                        **current_log_opts,
                    )


                for i in range(self.num_sequences):
                    unseen_current_logits_close = logits[:, :, -1:, 0, :, i]
                    unseen_current_targets_close = targets_classes[:, -1:, 0, :, i]
                    unseen_direction_acc_close = (
                        torch.argmax(unseen_current_logits_close, dim=1)
                        == unseen_current_targets_close
                    ).sum() / unseen_current_targets_close.numel()
                    self.log(
                        f"Close_Sequence/{stage}_unseen_{self.tickers[i]}_accuracy",
                        unseen_direction_acc_close,
                        **current_log_opts,
                    )
                
                
                unseen_logits_close = logits[:, :, -1:, 0, :, :]
                unseen_targets_close = targets_classes[:, -1:, 0, :, :]
                unseen_class_close = torch.argmax(unseen_logits_close, dim=1)

                # Calculate confusion matrix
                confusion_matrix = torch.zeros(
                    (self.args.num_classes, self.args.num_classes),
                    device=unseen_class_close.device,
                )
                for i in range(self.args.num_classes):
                    for j in range(self.args.num_classes):
                        confusion_matrix[i, j] = (
                            (unseen_class_close == i)
                            & (unseen_targets_close == j)
                        ).sum()
                
                unseen_accuracy = (
                    confusion_matrix.diag().sum() / confusion_matrix.sum()
                )
                self.log(
                    f"Close_Confusion/{stage}_unseen_accuracy",
                    unseen_accuracy,
                    **current_log_opts,
                )

                unseen_precisions = torch.zeros(
                    self.args.num_classes, device=unseen_class_close.device
                )
                unseen_recalls = torch.zeros(
                    self.args.num_classes, device=unseen_class_close.device
                )
                unseen_f1s = torch.zeros(
                    self.args.num_classes, device=unseen_class_close.device
                )
                for i in range(self.args.num_classes):
                    true_positives = confusion_matrix[i, i]
                    false_positives = confusion_matrix[:, i].sum() - true_positives
                    false_negatives = confusion_matrix[i, :].sum() - true_positives

                    if true_positives + false_positives > 0:
                        unseen_precisions[i] = true_positives / (
                            true_positives + false_positives
                        )
                    else:
                        unseen_precisions[i] = 0.0

                    if true_positives + false_negatives > 0:
                        unseen_recalls[i] = true_positives / (
                            true_positives + false_negatives
                        )
                    else:
                        unseen_recalls[i] = 0.0

                    if unseen_precisions[i] + unseen_recalls[i] > 0:
                        unseen_f1s[i] = (
                            2 * unseen_precisions[i] * unseen_recalls[i]
                        ) / (unseen_precisions[i] + unseen_recalls[i])
                    else:
                        unseen_f1s[i] = 0.0
                
                for i in range(self.args.num_classes):
                    self.log(
                        f"Close_Confusion/{stage}_unseen_precision_class_{i}",
                        unseen_precisions[i],
                        **current_log_opts,
                    )
                    self.log(
                        f"Close_Confusion/{stage}_unseen_recall_class_{i}",
                        unseen_recalls[i],
                        **current_log_opts,
                    )
                    self.log(
                        f"Close_Confusion/{stage}_unseen_f1_class_{i}",
                        unseen_f1s[i],
                        **current_log_opts,
                    )
                
                self.log(
                    f"Close_Confusion/{stage}_unseen_precision",
                    unseen_precisions.mean(),
                    **current_log_opts,
                )
                self.log(
                    f"Close_Confusion/{stage}_unseen_recall",
                    unseen_recalls.mean(),
                    **current_log_opts,
                )
                self.log(
                    f"Close_Confusion/{stage}_unseen_f1",
                    unseen_f1s.mean(),
                    **current_log_opts,
                )
                

                # confidence levels
                unseen_probs_close = torch.softmax(
                    unseen_logits_close, dim=1
                ).max(dim=1).values
                uccc_up = unseen_class_close == 2
                unseen_confidence_close_up = (unseen_probs_close * (uccc_up).float()).sum() / (
                    (uccc_up).float().sum() + 1e-6
                ) if uccc_up.sum() > 0 else torch.tensor(0.0, device=unseen_probs_close.device)
                uccc_down = unseen_class_close == 0
                unseen_confidence_close_down = (unseen_probs_close * (uccc_down).float()).sum() / (
                    (uccc_down).float().sum() + 1e-6
                ) if uccc_down.sum() > 0 else torch.tensor(0.0, device=unseen_probs_close.device)
                uccc_flat = unseen_class_close == 1
                unseen_confidence_close_flat = (unseen_probs_close * (uccc_flat).float()).sum() / (
                    (uccc_flat).float().sum() + 1e-6
                ) if uccc_flat.sum() > 0 else torch.tensor(0.0, device=unseen_probs_close.device)

                self.log(
                    f"Close_Conf/{stage}_unseen_confidence_up",
                    unseen_confidence_close_up,
                    **current_log_opts,
                )

                self.log(
                    f"Close_Conf/{stage}_unseen_confidence_down",
                    unseen_confidence_close_down,
                    **current_log_opts,
                )

                self.log(
                    f"Close_Conf/{stage}_unseen_confidence_flat",
                    unseen_confidence_close_flat,
                    **current_log_opts,
                )

            if stage == "train":
                cummulative_times["metrics"] += (
                    time.time_ns() - time_metrics_start
                ) / 1e6

                self.log(
                    f"{stage}_lr",
                    self.lr_schedulers().get_last_lr()[0],
                    on_step=True,
                    logger=True,
                )

        if stage == 'val':
            if self.args.prediction_type == 'regression':
                final_preds = preds
            elif self.args.prediction_type == 'classification':
                final_preds = logits

            self.validation_step_outputs.append({
                'preds': final_preds.detach().cpu(),
                'targets': targets.detach().cpu()
            })

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        if self.args.orthograd:
            optimizer = OrthoGrad(
                params=self.parameters(),
                base_optimizer_cls=optim.Adam,
                lr=self.learning_rate,
            )
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        def lr_lambda(current_step):
            min_lr = 1e-8 / self.learning_rate

            current_cycle_step = current_step
            cycle_nr = 0
            for _ in range(50):
                t_curr = self.t_0 * (self.t_mult**cycle_nr)
                if current_cycle_step > t_curr:
                    current_cycle_step -= t_curr
                    cycle_nr += 1
                else:
                    break

            current_peak_lr = self.lr_mult**cycle_nr

            if current_cycle_step < self.warmup_steps:  # Linear warmup
                return (
                    (current_peak_lr)
                    * float(current_cycle_step)
                    / float(max(1, self.warmup_steps))
                )
                # return (current_peak_lr - min_lr) * float(current_cycle_step) / float(
                #     max(1, self.warmup_steps)
                # ) + min_lr

            if current_cycle_step >= self.warmup_steps and current_cycle_step <= t_curr:
                progress = float(current_cycle_step - self.warmup_steps) / float(
                    max(1, t_curr - self.warmup_steps)
                )
                return (
                    current_peak_lr * 0.5 * (math.cos(math.pi * progress) + 1) + min_lr
                )

        scheduler = LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def on_validation_epoch_start(self):
        self.validation_step_outputs.clear()

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        all_preds = torch.cat([x['preds'] for x in outputs], dim=0) # (batch/time, seq_len, features, targets, num_sequences) or
        # (batch_size, num_classes, seq_len, features, targets, num_sequences)
        # depending on prediction type
        if self.cli_args.average_predictions:
            if self.args.prediction_type == 'regression':
                preds = all_preds[:, -1, 0, :, :]
                norm_means_for_close = self.args.normalization_means[0, :].to(device)
                norm_stds_for_close = self.args.normalization_stds[0, :].to(device)

                aligned_predictions = torch.empty((preds.shape[0]-(preds.shape[1]-1), preds.shape[1], preds.shape[2]), device=device)
                for i in range(preds.shape[1]):
                    start_day = preds.shape[1] - i - 1
                    aligned_predictions[:, i, :] = preds[start_day:preds.shape[0]-i, i, :]
                aligned_predictions = aligned_predictions.mean(dim=1)
                backtest_predictions = (aligned_predictions * norm_stds_for_close) + norm_means_for_close
            elif self.args.prediction_type == 'classification':
                preds = all_preds[:, :, -1, 0, :, :] # time, classes, targets, num_sequences
                preds = preds.permute(0, 2, 3, 1) # time, targets, num_sequences, classes
                aligned_predictions = torch.empty((preds.shape[0]-(preds.shape[1]-1), preds.shape[1], preds.shape[2], preds.shape[3]), device=device)
                for i in range(preds.shape[1]):
                    start_day = preds.shape[1] - i - 1
                    aligned_predictions[:, i, :, :] = preds[start_day:preds.shape[0]-i, i, :, :]
                aligned_predictions = aligned_predictions.mean(dim=1)
                backtest_predictions = torch.softmax(aligned_predictions, dim=-1)
        else:
            if self.args.prediction_type == 'regression':
                preds = all_preds[:, -1, 0, self.cli_args.pred_day-1, :]
                norm_means_for_close = self.args.normalization_means[0, :].to(device)
                norm_stds_for_close = self.args.normalization_stds[0, :].to(device)

                backtest_predictions = (aligned_predictions * norm_stds_for_close) + norm_means_for_close
            elif self.args.prediction_type == 'classification':
                preds = all_preds[:, :, -1, 0, self.cli_args.pred_day-1, :] # time, classes, num_sequences
                preds = preds.permute(0, 2, 1) # time, num_sequences, classes
                backtest_predictions = torch.softmax(preds, dim=-1)

        all_targets = torch.cat([x['targets'] for x in outputs], dim=0) # (batch/time, seq_len, features, targets, num_sequences)
        targets = all_targets[:, -1, 0, self.cli_args.pred_day-1, :] # time, num_sequences 
        if self.cli_args.average_predictions:
            if max(self.pred_indices) != 1:
                targets = targets[:-(preds.shape[1]-1)]


        # gets average time for each stage
        if self.trainer.is_global_zero:
            log_opts_epoch = {
                "on_step": False,
                "on_epoch": True,
                "logger": True,
                "sync_dist": False,
            }
            time_metrics_start = time.time_ns()
            # reliable backtest metrics
            # backtest_predictions = get_mtp_predictions_for_backtest(self.model, self.backtest_input_data, self.args, self.cli_args.days_to_check, device, self.cli_args)
            backtest_metrics = calculate_metrics(targets, backtest_predictions, self.args)
            for metric_name, metric_value in backtest_metrics.items():
                self.log(
                    f"Backtest_metrics/val_{metric_name}",
                    metric_value,
                    **log_opts_epoch,
                )
            trading_strategy_metrics = trading_metrics(targets, backtest_predictions, self.args)
            for metric_name, metric_value in trading_strategy_metrics.items():
                self.log(
                    f"Trading_strategy_metrics/val_{metric_name}",
                    metric_value,
                    **log_opts_epoch,
                )
            
            cummulative_times["metrics"] += (
                time.time_ns() - time_metrics_start
            ) / 1e6

            self.log(
                f"Times/preprocessing",
                cummulative_times["preprocessing"] / (self.global_step + 1),
                logger=True,
            )
            self.log(
                f"Times/forward_pass",
                cummulative_times["model"] / (self.global_step + 1),
                logger=True,
            )
            self.log(
                f"Times/loss",
                cummulative_times["loss"] / (self.global_step + 1),
                logger=True,
            )
            self.log(
                f"Times/metrics",
                cummulative_times["metrics"] / (self.global_step + 1),
                logger=True,
            )

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.model.__class__.__name__ == "Money_former_nGPT":
            normalize_weights_and_enforce_positive_eigenvalues(self.model)
        return
