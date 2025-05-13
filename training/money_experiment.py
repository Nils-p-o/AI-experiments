import pytorch_lightning as pl
import torch.optim as optim
import torch
import torch.nn as nn
import math
from torch.optim.lr_scheduler import LambdaLR
from .utils import OrthoGrad, custom_cross_entropy, stablemax, taylor_softmax
from transformer_arch.nGPT import normalize_weights_and_enforce_positive_eigenvalues
import argparse
import numpy as np

def log_cosh_loss(y_pred, y_true):
    x = y_pred - y_true
    return torch.mean(torch.abs(x) + torch.log1p(torch.exp(-2. * torch.abs(x))) - np.log(2.0))

def get_direction_vectorized_torch(current_prices, previous_prices, threshold=0.01):
    """
    Determines the direction of price movement for batched, multi-target PyTorch tensors.

    Args:
        current_prices (torch.Tensor): Tensor of current prices (batch, seq_movements, num_targets).
        previous_prices (torch.Tensor): Tensor of previous prices (batch, seq_movements, num_targets).
        threshold (float, optional): Threshold to define "flat" movement.

    Returns:
        torch.Tensor: Tensor of directions (1 for UP, -1 for DOWN, 0 for FLAT)
                      with the same shape as input tensors.
    """
    differences = current_prices - previous_prices
    directions = torch.sign(differences)
    # Where the absolute difference is within the threshold, mark as flat (0)
    directions[torch.abs(differences) <= threshold] = 0
    return directions.to(torch.int8) # Store as int8 for efficiency

def calculate_directional_accuracy(actual_prices, predicted_values, movement_threshold=0.01):
    """
    Calculates directional accuracy for batched, multi-target sequential PyTorch tensors.

    Args:
        actual_prices (torch.Tensor): Tensor of actual historical prices.
                                      Shape: (batch, seq_len_actual, num_targets).
        predicted_values (torch.Tensor): Tensor of predicted values (model outputs).
                                         It's assumed that predicted_values[b, s, k] is the
                                         model's prediction for actual_prices[b, s+1, k].
                                         Shape: (batch, seq_len_actual - 1, num_targets).
        movement_threshold (float, optional): Threshold for "flat" movement.

    Returns:
        tuple: (overall_accuracy, accuracy_per_target)
               overall_accuracy (torch.Tensor): Scalar tensor with percentage of correct
                                                directions across all dimensions.
               accuracy_per_target (torch.Tensor): Tensor of shape (num_targets,) with
                                                   accuracy for each target.
               Returns (None, None) if insufficient data.
    """
    if not isinstance(actual_prices, torch.Tensor) or \
       not isinstance(predicted_values, torch.Tensor):
        raise TypeError("Inputs must be PyTorch Tensors.")

    if actual_prices.ndim != 3 or predicted_values.ndim != 3:
        raise ValueError("Inputs must be 3-dimensional (batch, sequence, num_targets).")

    batch_size_actual, seq_len_actual, num_targets_actual = actual_prices.shape
    batch_size_pred, seq_len_pred, num_targets_pred = predicted_values.shape

    if not (batch_size_actual == batch_size_pred and \
            num_targets_actual == num_targets_pred):
        raise ValueError("Batch size and number of targets must match between actuals and predictions.")

    if seq_len_actual != seq_len_pred + 1:
        raise ValueError(f"Sequence length of actual_prices ({seq_len_actual}) "
                         f"must be one greater than predicted_values ({seq_len_pred}).")

    if seq_len_actual < 2:
        print("Warning: Insufficient sequence length in actual_prices. "
              "Need at least 2 price points to determine movement.")
        return None, None

    # Ensure tensors are on the same device
    device = actual_prices.device
    predicted_values = predicted_values.to(device)

    # Actual movements:
    actual_prev = actual_prices[:, :-1, :]  # (batch, seq_len_actual-1, num_targets)
    actual_curr = actual_prices[:, 1:, :]   # (batch, seq_len_actual-1, num_targets)

    actual_directions = get_direction_vectorized_torch(actual_curr, actual_prev, movement_threshold)

    # Predicted movements:
    predicted_directions = get_direction_vectorized_torch(predicted_values, actual_prev, movement_threshold)

    # Compare directions
    correct_mask = (actual_directions == predicted_directions) # (batch, seq_len_actual-1, num_targets)

    num_comparisons_total = correct_mask.numel() # Total number of elements
    if num_comparisons_total == 0:
        print("Warning: Zero comparisons made.")
        return None, None

    # Overall accuracy
    total_correct_predictions = torch.sum(correct_mask)
    overall_accuracy = (total_correct_predictions.float() / num_comparisons_total)

    # Accuracy per target (averaged over batch and sequence dimensions)
    # Sum over batch (dim 0) and sequence (dim 1)
    correct_per_target = torch.sum(correct_mask, dim=(0, 1))
    comparisons_per_target_instance = batch_size_actual * (seq_len_actual - 1)

    if comparisons_per_target_instance > 0:
        accuracy_per_target = (correct_per_target.float() / comparisons_per_target_instance)
    else:
        accuracy_per_target = torch.zeros(num_targets_actual, device=device, dtype=torch.float)

    return overall_accuracy, accuracy_per_target

def get_movement_directions(actual_prices, predicted_values, movement_threshold=0.01):
    actual_prev = actual_prices[:,:-1,:]
    actual_curr = actual_prices[:,1:,:]

    actual_directions = get_direction_vectorized_torch(actual_curr, actual_prev, movement_threshold)
    predicted_directions = get_direction_vectorized_torch(predicted_values, actual_prev, movement_threshold)

    total_movements = actual_directions.numel()

    actual_up = torch.sum(actual_directions == 1)/total_movements
    actual_down = torch.sum(actual_directions == -1)/total_movements
    actual_flat = torch.sum(actual_directions == 0)/total_movements

    predicted_up = torch.sum(predicted_directions == 1)/total_movements
    predicted_down = torch.sum(predicted_directions == -1)/total_movements
    predicted_flat = torch.sum(predicted_directions == 0)/total_movements

    return actual_up, actual_down, actual_flat, predicted_up, predicted_down, predicted_flat

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
        lr_mult=0.5, # maybe higher?
        args: argparse.Namespace = None,
        normalized_threshold: float = 0.1
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.t_0 = t_0
        self.t_mult = t_mult
        self.lr_mult = lr_mult
        self.batch_size = batch_size
        self.loss_fn = nn.MSELoss() # MASE (NLL for distribution)
        self.MSE = nn.MSELoss()
        self.MAE = nn.L1Loss()
        # self.huber = nn.HuberLoss()
        self.normalized_threshold = normalized_threshold
        self.pred_indices = args.indices_to_predict
        self.save_hyperparameters(
            ignore=["model"]
        )
        self.args = args

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


    def _shared_step(self, batch, batch_idx, stage="train"):
        inputs, labels = batch
        # select price to predict from columns
        # currently just the first column (adj close)
        furthest_date = max(self.pred_indices)

        known_inputs = inputs[:, 0, :-(furthest_date-1)].unsqueeze(-1)
        labels = labels[:, 0, :].unsqueeze(-1)
        targets = labels[:, :-(furthest_date-1)]
        for i in range(1, len(self.pred_indices)):
            if self.pred_indices[i] == furthest_date:
                targets = torch.cat([targets, labels[:, self.pred_indices[i]-1:]], dim=-1)
            else:
                targets = torch.cat([targets, labels[:, self.pred_indices[i]-1:-(furthest_date-self.pred_indices[i])]], dim=-1)
    
        means = known_inputs.mean(dim=1).tile(1, known_inputs.shape[1]).unsqueeze(-1)
        stds = known_inputs.std(dim=1).tile(1, known_inputs.shape[1]).unsqueeze(-1)

        norm_inputs = (known_inputs - means) / stds
        norm_labels = (labels[:, (furthest_date-1):] - means) / stds # Important!!! notice how norm_labels is not the full vector of labels (because this works just fine)


        outputs = self(norm_inputs)#.transpose(-1, -2) # (batch_size, seq_len, vocab_size) 

        if isinstance(outputs, torch.Tensor):
            logits = outputs
        else:
            logits = outputs.logits

        # direction part of loss
        norm_full_seq_true_prices = torch.cat([norm_inputs, norm_labels[:,-(furthest_date+1):,:]], dim=1)
        norm_true_price_targets = norm_full_seq_true_prices[:, :-(furthest_date)]
        for i in range(1, len(self.pred_indices)):
            if self.pred_indices[i] == furthest_date:
                norm_true_price_targets = torch.cat([norm_true_price_targets, norm_full_seq_true_prices[:, self.pred_indices[i]:]], dim=-1)
            else:
                norm_true_price_targets = torch.cat([norm_true_price_targets, norm_full_seq_true_prices[:, self.pred_indices[i]:-(furthest_date-self.pred_indices[i])]], dim=-1)
        direction_accuracy, direction_accuracy_per_target = calculate_directional_accuracy(norm_true_price_targets, logits, movement_threshold=self.normalized_threshold)

        # just for seeing what the data looks like, and if there is some bias in predictions
        real_up, real_down, real_flat, pred_up, pred_down, pred_flat = get_movement_directions(
            norm_true_price_targets, logits, movement_threshold=self.normalized_threshold
        )

        logits = logits * stds.tile(1, 1, len(self.pred_indices)) + means.tile(1, 1, len(self.pred_indices))

        # reference loss of naive model
        naive_MSE = self.MSE(known_inputs.tile(1,1,len(self.pred_indices)), targets)
        naive_MAE = self.MAE(known_inputs.tile(1,1,len(self.pred_indices)), targets)

        loss = self.loss_fn(logits, targets)
        # adjusting loss to be scaled relative to persistance model
        loss = loss / (naive_MSE + 1e-6)


        loss = loss / (direction_accuracy + 1e-6)

        MSSE = self.MSE(logits, targets) / (naive_MSE + 1e-6)
        MASE = self.MAE(logits, targets) / (naive_MAE + 1e-6)


        # TODO try to get similar metrics into one graph (without a million logs)
        log_opts = {'on_step': (stage == 'train'), 'logger': True, 'sync_dist': True}

        # Log main loss and accuracy
        self.log(f"Loss/{stage}_loss", loss, prog_bar=True, **log_opts)
        self.log(f"Accuracy/{stage}_accuracy", direction_accuracy, **log_opts)

        # Log Naive Losses under a common group
        self.log(f"Naive_Losses/{stage}_MSE", naive_MSE, **log_opts)
        self.log(f"Naive_Losses/{stage}_MAE", naive_MAE, **log_opts)

        # Log Relative Losses under a common group
        self.log(f"Relative_Losses/{stage}_MSSE", MSSE, **log_opts)
        self.log(f"Relative_Losses/{stage}_MASE", MASE, **log_opts)


        self.log(f"Directions_Real/{stage}_Up", real_up, **log_opts)
        self.log(f"Directions_Real/{stage}_Down", real_down, **log_opts)
        self.log(f"Directions_Real/{stage}_Flat", real_flat, **log_opts)
        self.log(f"Directions_Pred/{stage}_Up", pred_up, **log_opts)
        self.log(f"Directions_Pred/{stage}_Down", pred_down, **log_opts)
        self.log(f"Directions_Pred/{stage}_Flat", pred_flat, **log_opts)

        # Log accuracy per target
        for i in range(len(self.pred_indices)):
            self.log(f"Target_Accuracy/{stage}_target_{self.pred_indices[i]}", direction_accuracy_per_target[i], **log_opts)

            # Log naive losses per target
            temp_naive_MSE = self.MSE(known_inputs, targets[:, :, i:i+1])
            temp_naive_MAE = self.MAE(known_inputs, targets[:, :, i:i+1])
            self.log(f"Naive_Losses_target/{stage}_target_{self.pred_indices[i]}_MSE", temp_naive_MSE, **log_opts)
            self.log(f"Naive_Losses_target/{stage}_target_{self.pred_indices[i]}_MAE", temp_naive_MAE, **log_opts)

            # log relative losses per target
            temp_MSSE = self.MSE(logits[:,:,i:i+1], targets[:, :, i:i+1]) / (temp_naive_MSE + 1e-6)
            temp_MASE = self.MAE(logits[:,:,i:i+1], targets[:, :, i:i+1]) / (temp_naive_MAE + 1e-6)
            self.log(f"Relative_Losses_target/{stage}_target_{self.pred_indices[i]}_MSSE", temp_MSSE, **log_opts)
            self.log(f"Relative_Losses_target/{stage}_target_{self.pred_indices[i]}_MASE", temp_MASE, **log_opts)
            # log loss by target
            temp_loss = self.loss_fn(logits[:, :, i:i+1], targets[:, :, i:i+1]) / (temp_naive_MSE + 1e-6)
            temp_loss = temp_loss / (direction_accuracy_per_target[i] + 1e-6)
            self.log(f"Loss_target/{stage}_target_{self.pred_indices[i]}", temp_loss, **log_opts)

        if stage == "train":
            self.log(f"{stage}_lr", self.lr_schedulers().get_last_lr()[0], on_step=True, logger=True)
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")
    
    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")


    def configure_optimizers(self):
        if self.args.orthograd:
            optimizer = OrthoGrad(params=self.parameters(), base_optimizer_cls=optim.Adam, lr=self.learning_rate)
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
                return (current_peak_lr) * float(current_cycle_step) / float(
                    max(1, self.warmup_steps)
                )
                # return (current_peak_lr - min_lr) * float(current_cycle_step) / float(
                #     max(1, self.warmup_steps)
                # ) + min_lr

            if current_cycle_step >= self.warmup_steps and current_cycle_step <= t_curr:
                progress = float(current_cycle_step - self.warmup_steps) / float(
                    max(1, t_curr - self.warmup_steps)
                )
                return current_peak_lr * 0.5 * (math.cos(math.pi * progress) + 1) + min_lr

        scheduler = LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
    
    # def on_train_batch_end(self, outputs, batch, batch_idx):
    #     if self.model.__class__.__name__ == "nGPT":
    #         normalize_weights_and_enforce_positive_eigenvalues(self.model)
    #     return
