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
    calculate_metrics,
    new_trading_metrics,
    new_trading_metrics_individual
)

from muon import MuonWithAuxAdam
from muon import Muon
from torch.optim import Optimizer
from functools import partial
from collections import ChainMap
import torch.distributed as dist

from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

# for debug / timimg for speedups
import time

cummulative_times = {"preprocessing": 0, "model": 0, "loss": 0, "metrics": 0}

# TODO re-add IC?

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


def calculate_expected_accuracy(targets_classes, num_classes):
    accuracy = 0.0
    for i in range(num_classes):
        accuracy = max(accuracy, (targets_classes == i).sum() / targets_classes.numel())
    return accuracy

class WeightedCustomLoss(nn.Module):
    def __init__(self, cost_matrix, weights, reduction='mean'):
        super(WeightedCustomLoss, self).__init__()
        if not isinstance(cost_matrix, torch.Tensor):
            cost_matrix = torch.tensor(cost_matrix, dtype=torch.float32)
        self.cost_matrix = cost_matrix

        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float32)
        self.weights = weights

        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Reduction must be one of 'mean', 'sum', or 'none', but got {reduction}")
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        self.cost_matrix = self.cost_matrix.to(y_pred.device)

        log_probs = torch.log_softmax(y_pred+1e-9, dim=1)

        one_hot_encoded = torch.nn.functional.one_hot(y_true, num_classes=log_probs.shape[1])
        dims = list(range(one_hot_encoded.dim()))
        one_hot_encoded = one_hot_encoded.permute(dims[0], dims[-1], *dims[1:-1])

        costs_for_batch = self.cost_matrix[y_true]
        dims = list(range(costs_for_batch.dim()))
        costs_for_batch = costs_for_batch.permute(dims[0], dims[-1], *dims[1:-1])

        weights_for_batch = self.weights[y_true].unsqueeze(1)

        loss = -torch.sum(one_hot_encoded*log_probs*weights_for_batch, dim=1) * torch.exp((torch.softmax(y_pred, dim=1)*costs_for_batch).sum(dim=1))

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class WeightedClassImbalanceCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights, reduction='mean'):
        super(WeightedClassImbalanceCrossEntropyLoss, self).__init__()
        if not isinstance(class_weights, torch.Tensor):
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.class_weights = class_weights.permute(2, 0, 1)  # (num_classes, features, num_sequences)
        self.class_weights = self.class_weights.unsqueeze(0).unsqueeze(2).unsqueeze(4)  # (1, num_classes, 1, features, 1, num_sequences)
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Reduction must be one of 'mean', 'sum', or 'none', but got {reduction}")
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        # y_pred: (batch_size, num_classes, seq_len, features, targets, num_sequences)
        self.class_weights = self.class_weights.to(y_pred.device)

        log_probs = torch.log_softmax(y_pred+1e-9, dim=1)

        one_hot_encoded = torch.nn.functional.one_hot(y_true, num_classes=log_probs.shape[1])
        dims = list(range(one_hot_encoded.dim()))
        one_hot_encoded = one_hot_encoded.permute(dims[0], dims[-1], *dims[1:-1])

        # class_weights (1, num_classes, 1, features, 1, num_sequences)
        loss = -torch.sum(one_hot_encoded * log_probs * self.class_weights, dim=1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # class_weights = [args.down_up_loss_weight, 1.0, args.down_up_loss_weight]
        # class_weights = torch.tensor([w*len(class_weights)/sum(class_weights) for w in class_weights])
        # cost_weights = torch.tensor([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])

        # self.custom_loss = True
        args.up_down_err_w = 1.0
        args.flat_p_err_w = 1.5
        args.flat_t_err_w = 2.0
        self.class_cost_matrix = torch.tensor([[0.0, args.flat_t_err_w, args.up_down_err_w], 
                                               [args.flat_p_err_w, 0.0, args.flat_p_err_w], 
                                               [args.up_down_err_w, args.flat_t_err_w, 0.0]]).to(device)

        self.class_weights = torch.tensor(args.class_weights).permute(2, 0, 1)  # (num_classes, features, num_sequences)
        self.class_weights = self.class_weights.unsqueeze(0).unsqueeze(2).unsqueeze(4).to(device)  # (1, num_classes, 1, features, 1, num_sequences)

        match args.prediction_type:
            case "gaussian":
                self.loss_fn = nn.GaussianNLLLoss(reduction="none")
            case "regression":
                self.loss_fn = nn.L1Loss(reduction="none")
            case "classification":
                self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        
        self.MAE = nn.L1Loss(reduction="none")
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


        feature_weights = [1, 1, 1, 1, 1]
        ticker_weights = [1 for _ in range(len(args.tickers))]
        seen_unseen_weights = [1 for _ in range(args.seq_len)]

        feature_weights = torch.tensor([w*len(feature_weights)/sum(feature_weights) for w in feature_weights])
        ticker_weights = torch.tensor([w*len(ticker_weights)/sum(ticker_weights) for w in ticker_weights])
        seen_unseen_weights = torch.tensor([w*len(seen_unseen_weights)/sum(seen_unseen_weights) for w in seen_unseen_weights])

        loss_weights = feature_weights.unsqueeze(-1) * ticker_weights.unsqueeze(0)
        loss_weights = seen_unseen_weights.unsqueeze(-1).unsqueeze(-1) * loss_weights.unsqueeze(0)
        self.loss_weights = loss_weights.unsqueeze(0).unsqueeze(3).to(device)

        self.num_base_tickers = 6


    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def _shared_step(self, batch, batch_idx, stage="train"):
        time_preprocessing_start = time.time_ns()
        inputs, labels = batch
        # inputs (batch_size, features, targets, seq_len, num_sequences)
        # labels (batch_size, features (chlov), targets, seq_len, num_sequences)
        batch_size, _, _, _, num_sequences = inputs.shape

        seq_len = self.args.seq_len

        # inputs = inputs.permute(0, 2, 3, 1)  # (batch_size, seq_len, num_sequences, features)
        inputs = inputs.permute(0, 2, 3, 4, 1) # (batch_size, targets, seq_len, num_sequences, features)

        targets = labels  # (batch_size, features, targets, seq_len, num_sequences)
        targets = targets.permute(0, 3, 1, 2, 4)  # (batch_size, seq_len, features, targets, num_sequences)

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

        match self.args.prediction_type:
            case "gaussian":
                raw_logits, raw_pred_std = raw_logits.split([5, 5], dim=2)
                preds = raw_logits
                pred_std = torch.exp(raw_pred_std)
                pred_variance = pred_std ** 2

                loss_tensor = self.loss_fn(preds, targets, pred_variance)
                loss = (loss_tensor * self.loss_weights).mean()

                seen_losses = loss_tensor[:, :-1, :, :, :].mean(dim=(0, 1, 2, 4))
                unseen_losses = loss_tensor[:, -1:, :, :, :].mean(dim=(0, 1, 2, 4))

            case "regression":
                preds = raw_logits

                loss_tensor = self.loss_fn(preds, targets)
                loss = (loss_tensor * self.loss_weights).mean()

                seen_losses = loss_tensor[:, :-1, :, :, :].mean(dim=(0, 1, 2, 4))
                unseen_losses = loss_tensor[:, -1:, :, :, :].mean(dim=(0, 1, 2, 4))

            case "classification":
                targets_classes = torch.full_like(targets, 1)
                targets_classes[targets < -self.args.classification_threshold] = 0
                targets_classes[targets > self.args.classification_threshold] = 2
                targets_classes = targets_classes.long()

                if self.args.include_sep_in_loss:
                    view_shape = (batch_size, seq_len + 1, 5, self.args.num_classes, max(self.pred_indices), self.num_sequences)
                else:
                    view_shape = (batch_size, seq_len, 5, self.args.num_classes, max(self.pred_indices), self.num_sequences)
                
                logits = raw_logits.view(view_shape).permute(0, 3, 1, 2, 4, 5)  # (batch_size, num_classes, seq_len, features, targets, num_sequences)

                loss_tensor = self.loss_fn(logits, targets_classes)

                # self.class_weights shape:(1, num_classes, 1, features, 1, num_sequences)
                class_weights = self.class_weights.expand(batch_size, -1, seq_len, -1, max(self.pred_indices), -1)
                class_weights = torch.gather(class_weights, dim=1, index=targets_classes.unsqueeze(1)).squeeze(1)

                # loss_tensor = loss_tensor * class_weights # class imbalance weighing (per feat, and per sequence)

                # convert weights and wrong probabilities to a cost multiplier, that scales the loss
                batch_cost_weights = self.class_cost_matrix[targets_classes]
                dims = list(range(batch_cost_weights.dim()))
                batch_cost_weights = batch_cost_weights.permute(dims[0], dims[-1], *dims[1:-1])
                
                batch_cost_weights = (torch.log_softmax(logits, dim=1).exp() * batch_cost_weights).sum(dim=1)
                batch_cost_weights = torch.exp(batch_cost_weights) - 1

                loss = (loss_tensor * self.loss_weights * class_weights + batch_cost_weights).mean()

                seen_losses = loss_tensor[:, :-1, :, :, :self.num_base_tickers].mean(dim=(0, 1, 2, 4))
                unseen_losses = loss_tensor[:, -1:, :, :, :self.num_base_tickers].mean(dim=(0, 1, 2, 4))

                preds = logits
                
        if stage == "train":
            cummulative_times["loss"] += (time.time_ns() - time_loss_calc_start) / 1e6

        if torch.isnan(loss).any():
            print("NAN IN LOSS")

        self.log(
            f"Loss/{stage}_loss",
            loss,
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
                torch.mean(seen_losses),
                **current_log_opts,
            )
            self.log(
                f"Losses_seen_unseen/{stage}_loss_unseen",
                torch.mean(unseen_losses),
                **current_log_opts,
            )
            for i in range(len(self.pred_indices)):
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
                targets_classes = torch.full_like(targets, 1)
                targets_classes[targets < -self.args.classification_threshold] = 0
                targets_classes[targets > self.args.classification_threshold] = 2
                targets_classes = targets_classes.long()

                preds_classes = torch.full_like(preds, 1)
                preds_classes[preds < -self.args.classification_threshold] = 0
                preds_classes[preds > self.args.classification_threshold] = 2
                preds_classes = preds_classes.long()
            else:
                preds_classes = preds.argmax(dim=1)
                preds_classes = preds_classes.long()

            if stage == "val":
                self.log(
                    f"Accuracy_finer/{stage}_expected_max_acc",
                    calculate_expected_accuracy(
                        targets_classes, self.args.num_classes
                    ),
                    **current_log_opts,
                )

            avg_unseen_target_acc = []
            for i in range(len(self.pred_indices)):
                seen_target_accuracy = (preds_classes == targets_classes)[:, :-1, :, i, :].sum() / targets_classes[:, :-1, :, i, :].numel()
                self.log(
                    f"Split_Error/{stage}_seen_accuracy_{self.pred_indices[i]}",
                    seen_target_accuracy,
                    **current_log_opts,
                )

                unseen_target_accuracy = (preds_classes == targets_classes)[:, -1:, :, i, :].sum() / targets_classes[:, -1:, :, i, :].numel()
                self.log(
                    f"Split_Error/{stage}_unseen_accuracy_{self.pred_indices[i]}",
                    unseen_target_accuracy,
                    **current_log_opts,
                )
                avg_unseen_target_acc.append(unseen_target_accuracy)

            avg_unseen_target_acc = torch.mean(torch.stack(avg_unseen_target_acc))
            self.log(
                f"Split_Error/{stage}_unseen_accuracy_avg",
                avg_unseen_target_acc,
                **current_log_opts,
            )

            if self.args.prediction_type != "classification":

                self.log(
                    f"Accuracy/{stage}_accuracy", (preds_classes == targets_classes).sum() / targets_classes.numel(), **current_log_opts
                )

                # can only be done per time step
                # ICs = get_spearmanr_correlations_pytorch(
                #     targets.clone().detach(), preds.clone().detach()
                # )
                mae_loss_tensor = self.MAE(preds, targets)

                for i in range(self.num_sequences):
                    self.log(
                        f"Split_Error/{stage}_seen_{self.tickers[i]}_MAE",
                        mae_loss_tensor[:, :-1, :, :, i].mean(),
                        **current_log_opts,
                    )
                    self.log(
                        f"Split_Error/{stage}_unseen_{self.tickers[i]}_MAE",
                        mae_loss_tensor[:, -1:, :, :, i].mean(),
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
                        self.log(
                            f"Std_dev/{stage}_seen_{self.pred_indices[i]}",
                            pred_std[:, :-1, :, :, i].mean(),
                            **current_log_opts,
                        )
                        self.log(
                            f"Std_dev/{stage}_unseen_{self.pred_indices[i]}",
                            pred_std[:, -1:, :, :, i].mean(),
                            **current_log_opts,
                        )
                
                # close specific losses + accs (unseen, val specifically because these are the ones i actually care about)
                for i in range(len(self.pred_indices)): # close by target period
                    self.log(
                        f"Close_Target/{stage}_unseen_{self.pred_indices[i]}_MAE",
                        mae_loss_tensor[:, -1:, 0, i, :].mean(),
                        **current_log_opts,
                    )

                    self.log(
                        f"Close_Target/{stage}_unseen_{self.pred_indices[i]}_accuracy",
                        (preds_classes == targets_classes)[:, -1:, 0, i, :].sum() / targets_classes[:, -1:, 0, i, :].numel(),
                        **current_log_opts,
                    )
                
                for i in range(self.num_sequences): # close by sequence
                    self.log(
                        f"Close_Sequence/{stage}_unseen_{self.tickers[i]}_MAE",
                        mae_loss_tensor[:, -1:, 0, :, i].mean(),
                        **current_log_opts,
                    )

                    self.log(
                        f"Close_Sequence/{stage}_unseen_{self.tickers[i]}_accuracy",
                        (preds_classes == targets_classes)[:, -1:, 0, :, i].sum() / targets_classes[:, -1:, 0, :, i].numel(),
                        **current_log_opts,
                    )
            else: # acc metrics for classification
                for i in range(max(self.pred_indices)):
                    self.log(
                        f"Close_Target/{stage}_unseen_{self.pred_indices[i]}_accuracy",
                        (preds_classes == targets_classes)[:, -1:, 0, i, :].sum() / targets_classes[:, -1:, 0, i, :].numel(),
                        **current_log_opts,
                    )

                for i in range(self.num_sequences):
                    self.log(
                        f"Close_Sequence/{stage}_unseen_{self.tickers[i]}_accuracy",
                        (preds_classes == targets_classes)[:, -1:, 0, :, i].sum() / targets_classes[:, -1:, 0, :, i].numel(),
                        **current_log_opts,
                    )
                
                
                unseen_logits_close = preds[:, :, -1:, 0, :, :]
                unseen_class_close = preds_classes[:, -1:, 0, :, :]
                unseen_targets_close = targets_classes[:, -1:, 0, :, :]

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

                    unseen_precisions[i] = (true_positives / (
                        true_positives + false_positives
                    )) if true_positives + false_positives > 0 else 0.0

                    unseen_recalls[i] = (true_positives / (
                        true_positives + false_negatives
                    )) if true_positives + false_negatives > 0 else 0.0
                    
                    unseen_f1s[i] = ((
                        2 * unseen_precisions[i] * unseen_recalls[i]
                    ) / (unseen_precisions[i] + unseen_recalls[i])) if unseen_precisions[i] + unseen_recalls[i] > 0 else 0.0
                
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
                for i in range(self.args.num_classes):
                    self.log(
                        f"Close_Conf/{stage}_unseen_confidence_class_{i}",
                        unseen_probs_close[unseen_class_close == i].mean() if unseen_class_close[unseen_class_close == i].numel() > 0 else torch.tensor(0.0),
                        **current_log_opts,
                    )

                cost_matrix = torch.tensor(
                    [[0, 1, 10], [1, 0, 1], [10, 1, 0]], device=self.device
                )
                
                for i in range(self.args.num_classes):
                    self.log(
                        f"Close_Conf/{stage}_unseen_cost_class_{i}",
                        (confusion_matrix[i] * cost_matrix[i]).sum(),
                        **current_log_opts,
                    )
                self.log(
                    f"Close_Conf/{stage}_unseen_cost",
                    (confusion_matrix * cost_matrix).sum(),
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

            self.validation_step_outputs.append({
                'preds': preds.detach().cpu(),
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
        weight_decay = getattr(self.args, 'weight_decay', 0.0)

        if self.args.optimizer == 'muon':
            print("Using Muon optimizer with FULLY refined parameter splitting for MLA.")

            muon_params = []
            aux_params = []

            # Define the full parameter names of layers that should NOT use Muon.
            non_muon_linear_weights = {
                'model.shared_value_input.weight',
                'model.out.weight'
            }
            if self.args.unique_inputs_ratio[0] > 0:
                for i in range(len(self.model.unique_value_input)):
                    non_muon_linear_weights.add(f'model.unique_value_input.{i}.weight')

            # The rest of the logic is the same and now works perfectly
            for name, p in self.named_parameters():
                if p.ndim == 2 and 'weight' in name:
                    if name in non_muon_linear_weights:
                        aux_params.append(p)
                    else:
                        muon_params.append(p)
                else:
                    aux_params.append(p)
            
            # 2. Define the parameter groups for the built-in optimizer
            param_groups = [
                dict(params=muon_params, use_muon=True, lr=self.args.muon_lr, weight_decay=weight_decay),
                # NOTE: MuonWithAuxAdam hardcodes the aux optimizer to Adam.
                # It will use the lr from this group.
                dict(params=aux_params, use_muon=False, lr=self.learning_rate, weight_decay=weight_decay),
            ]

            # 3. Instantiate the built-in optimizer directly
            optimizer = MuonWithAuxAdam(param_groups)

        elif self.args.optimizer == 'orthograd':
            optimizer = OrthoGrad(
                params=self.parameters(),
                base_optimizer_cls=optim.Adam,
                lr=self.learning_rate,
            )
        elif self.args.optimizer == 'adamw':
            optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        elif self.args.optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {self.args.optimizer}")

        scheduler_type = getattr(self.args, 'scheduler_type', 'cosine_restarts') # Default to your original
        print(f"Using learning rate scheduler: {scheduler_type}")

        if scheduler_type == 'cosine_restarts':
            def lr_lambda(current_step):
                min_lr = 1e-7 / self.learning_rate
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
                if current_cycle_step < self.warmup_steps:
                    return ((current_peak_lr) * float(current_cycle_step) / float(max(1, self.warmup_steps)))
                if current_cycle_step >= self.warmup_steps and current_cycle_step <= t_curr:
                    progress = float(current_cycle_step - self.warmup_steps) / float(max(1, t_curr - self.warmup_steps))
                    return (current_peak_lr * 0.5 * (math.cos(math.pi * progress) + 1) + min_lr)
                return min_lr
            
            scheduler = LambdaLR(optimizer, lr_lambda)

        elif scheduler_type == 'linear_decay':
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.args.t_total
            )
        
        elif scheduler_type == 'constant':
            scheduler = get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.warmup_steps
            )

        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

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

        all_preds = torch.cat([x['preds'] for x in outputs], dim=0) # (batch/time, seq_len, features, targets, num_sequences) or
        # (batch_size, num_classes, seq_len, features, targets, num_sequences)
        # depending on prediction type

        # TODO for comparison
        all_preds = all_preds[:,:,:,:,:,:self.num_base_tickers]

        if self.cli_args.average_predictions:
            if self.args.prediction_type == 'regression':
                preds = all_preds[:, -1, 0, :, :]
                norm_means_for_close = self.args.normalization_means[0, :]
                norm_stds_for_close = self.args.normalization_stds[0, :]

                aligned_predictions = torch.empty((preds.shape[0]-(preds.shape[1]-1), preds.shape[1], preds.shape[2]))
                for i in range(preds.shape[1]):
                    start_day = preds.shape[1] - i - 1
                    aligned_predictions[:, i, :] = preds[start_day:preds.shape[0]-i, i, :]
                aligned_predictions = aligned_predictions.mean(dim=1)
                backtest_predictions = (aligned_predictions * norm_stds_for_close) + norm_means_for_close
            elif self.args.prediction_type == 'classification':
                preds = all_preds[:, :, -1, 0, :, :] # time, classes, targets, num_sequences
                preds = preds.permute(0, 2, 3, 1) # time, targets, num_sequences, classes
                aligned_predictions = torch.empty((preds.shape[0]-(preds.shape[1]-1), preds.shape[1], preds.shape[2], preds.shape[3]))
                for i in range(preds.shape[1]):
                    start_day = preds.shape[1] - i - 1
                    aligned_predictions[:, i, :, :] = preds[start_day:preds.shape[0]-i, i, :, :]
                aligned_predictions = aligned_predictions.mean(dim=1)
                backtest_predictions = torch.softmax(aligned_predictions, dim=-1)
        else:
            if self.args.prediction_type == 'regression':
                preds = all_preds[:, -1, 0, self.cli_args.pred_day-1, :]
                norm_means_for_close = self.args.normalization_means[0, :]
                norm_stds_for_close = self.args.normalization_stds[0, :]

                backtest_predictions = (aligned_predictions * norm_stds_for_close) + norm_means_for_close
            elif self.args.prediction_type == 'classification':
                preds = all_preds[:, :, -1, 0, self.cli_args.pred_day-1, :] # time, classes, num_sequences
                preds = preds.permute(0, 2, 1) # time, num_sequences, classes
                backtest_predictions = torch.softmax(preds, dim=-1)

        all_targets = torch.cat([x['targets'] for x in outputs], dim=0) # (batch/time, seq_len, features, targets, num_sequences)

        # TODO for comparison
        all_targets = all_targets[:,:,:,:,:self.num_base_tickers]

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

            correct_backtest_metrics = new_trading_metrics(targets, backtest_predictions, self.args)
            for metric_name, metric_value in correct_backtest_metrics.items():
                self.log(
                    f"Strategy_metrics/val_{metric_name}",
                    metric_value,
                    **log_opts_epoch,
                )
            correct_individual_backtest_metrics = new_trading_metrics_individual(targets, backtest_predictions, self.args)
            for metric_name, metric_value in correct_individual_backtest_metrics.items():
                for i in range(targets.shape[-1]):
                    self.log(
                        f"Strategy_metrics_individual/val_{metric_name}_{self.tickers[i]}",
                        metric_value[i].float(),
                        **log_opts_epoch,
                    )

            cummulative_times["metrics"] += (
                time.time_ns() - time_metrics_start
            ) / 1e6

            self.log(
                f"Times/preprocessing",
                cummulative_times["preprocessing"] / (self.trainer.val_check_interval),
                logger=True,
            )
            self.log(
                f"Times/forward_pass",
                cummulative_times["model"] / (self.trainer.val_check_interval),
                logger=True,
            )
            self.log(
                f"Times/loss",
                cummulative_times["loss"] / (self.trainer.val_check_interval),
                logger=True,
            )
            self.log(
                f"Times/metrics",
                cummulative_times["metrics"] / (self.trainer.val_check_interval),
                logger=True,
            )

            cummulative_times["preprocessing"] = 0
            cummulative_times["model"] = 0
            cummulative_times["loss"] = 0
            cummulative_times["metrics"] = 0

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.model.__class__.__name__ == "Money_former_nGPT":
            normalize_weights_and_enforce_positive_eigenvalues(self.model)
        return

# class MuonWithCustomAux(Optimizer):
    # """
    # A final, robust custom optimizer that wraps the Muon optimizer for 2D weights
    # and a customizable auxiliary optimizer for all other parameters.
    # This version handles both single-device and distributed training environments.
    # """
    # def __init__(self, params, muon_optimizer_cls, aux_optimizer_cls, muon_params, aux_params):
    #     self.muon_optimizer = muon_optimizer_cls(muon_params)
    #     self.aux_optimizer = aux_optimizer_cls(aux_params)
    #     self.param_groups = self.muon_optimizer.param_groups + self.aux_optimizer.param_groups
    #     self.defaults = {}

    # @property
    # def state(self):
    #     return ChainMap(self.muon_optimizer.state, self.aux_optimizer.state)

    # # --- THIS IS THE CRITICAL FIX FOR THE NEW ERROR ---
    # def step(self, closure=None):
    #     # PyTorch Lightning's automatic optimization provides a closure that
    #     # does model_forward -> loss -> loss.backward().
    #     # We need to execute this once to get the gradients.
    #     if closure is not None:
    #         closure()

    #     # Now that gradients are populated, we can step both optimizers.
    #     self.muon_optimizer.step()
    #     self.aux_optimizer.step()

    # def zero_grad(self, set_to_none: bool = True):
    #     self.muon_optimizer.zero_grad(set_to_none=set_to_none)
    #     self.aux_optimizer.zero_grad(set_to_none=set_to_none)

    # def state_dict(self):
    #     return {
    #         "muon_optimizer": self.muon_optimizer.state_dict(),
    #         "aux_optimizer": self.aux_optimizer.state_dict(),
    #     }

    # def load_state_dict(self, state_dict):
    #     self.muon_optimizer.load_state_dict(state_dict["muon_optimizer"])
    #     self.aux_optimizer.load_state_dict(state_dict["aux_optimizer"])


