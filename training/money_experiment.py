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
from scipy import stats

def log_cosh_loss(y_pred, y_true):
    x = y_pred - y_true
    return torch.mean(torch.abs(x) + torch.log1p(torch.exp(-2. * torch.abs(x))) - np.log(2.0))

# TODO rewrite direction to work with returns

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

def get_direction_accuracy_returns(actual_returns, predicted_returns, thresholds=0.01): # for one target?
    actual_up = actual_returns > thresholds
    actual_down = actual_returns < -thresholds
    actual_flat = torch.abs(actual_returns) < thresholds

    predicted_up = predicted_returns > thresholds
    predicted_down = predicted_returns < -thresholds
    predicted_flat = torch.abs(predicted_returns) < thresholds
    
    acc_up = (actual_up * predicted_up).sum() / actual_up.sum()
    acc_down = (actual_down * predicted_down).sum() / actual_down.sum()
    acc_flat = (actual_flat * predicted_flat).sum() / actual_flat.sum()

    total_acc = ((actual_up * predicted_up).sum() + (actual_down * predicted_down).sum() + (actual_flat * predicted_flat).sum())/ actual_up.numel()

    expected_acc = torch.max(torch.tensor([actual_up.sum(), actual_down.sum(), actual_flat.sum()])) / actual_up.numel()
    expected_strat = torch.tensor([actual_up.sum(), actual_down.sum(), actual_flat.sum()]).square().sum() / actual_up.numel()**2

    return total_acc, acc_up, acc_down, acc_flat, expected_acc, expected_strat

def z_normalize_additional_inputs(additional_inputs):
    for i in range(additional_inputs.shape[1]):
        additional_inputs[:, i, :] = (additional_inputs[:, i, :] - additional_inputs[:, i, :].mean(dim=-1, keepdim=True).tile(1, 1, additional_inputs.shape[2])) / additional_inputs[:, i, :].std(dim=-1, keepdim=True).tile(1, 1, additional_inputs.shape[2])
    return additional_inputs

def get_spearmanr_correlations(actual_prices, predicted_values):
    batch_size, seq_len, num_targets = actual_prices.shape
    ICs = torch.zeros((batch_size, num_targets))

    for i in range(batch_size):
        for j in range(seq_len):
            for k in range(num_targets):
                ICs[i,j,k] = stats.spearmanr(actual_prices[i,j,k], predicted_values[i,j,k]).correlation # only do when you have more than one data point per time/prediction

    return ICs
        

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
        furthest_date = max(self.pred_indices)

        if furthest_date != 1:
            time_inputs = inputs[:, :10, :-(furthest_date-1)].transpose(-1, -2)
        else:
            time_inputs = inputs[:, :10, :].transpose(-1, -2)
        inputs = inputs[:, 10:, :]
        labels = labels[:, 10:, :]
        # select price to predict from columns
        # currently just the first column (adj close)
        if furthest_date != 1:
            known_inputs = inputs[:, 0, :-(furthest_date-1)].unsqueeze(-1)
        else:
            known_inputs = inputs[:, 0, :].unsqueeze(-1)
        labels = labels[:, 0, :].unsqueeze(-1)
        if furthest_date != 1:
            targets = labels[:, :-(furthest_date-1)].unsqueeze(-1)
        else:
            targets = labels
        for i in range(1, len(self.pred_indices)):
            if self.pred_indices[i] == furthest_date:
                targets = torch.cat([targets, labels[:, self.pred_indices[i]-1:]], dim=-1)
            else:
                targets = torch.cat([targets, labels[:, self.pred_indices[i]-1:-(furthest_date-self.pred_indices[i])]], dim=-1)
    
        known_input_means = known_inputs.mean(dim=1).tile(1, known_inputs.shape[1]).unsqueeze(-1)
        known_input_stds = known_inputs.std(dim=1).tile(1, known_inputs.shape[1]).unsqueeze(-1)

        norm_inputs = (known_inputs - known_input_means) / known_input_stds
        norm_labels = (labels[:, (furthest_date-1):] - known_input_means) / known_input_stds # Important!!! notice how norm_labels is not the full vector of labels (because this works just fine)
        if furthest_date != 1:
            additional_inputs = inputs[:, 1:, :-(furthest_date-1)]
        else:
            additional_inputs = inputs[:, 1:, :]
        additional_inputs = additional_inputs.transpose(-1, -2)
        means_of_additonal_inputs = additional_inputs.mean(dim=1, keepdim=True).tile(1, additional_inputs.shape[1],1)
        stds_of_additonal_inputs = additional_inputs.std(dim=1, keepdim=True).tile(1, additional_inputs.shape[1],1)
        full_norm_inputs = torch.cat([norm_inputs, additional_inputs-means_of_additonal_inputs/stds_of_additonal_inputs], dim=-1)

        full_inputs = torch.cat([time_inputs, full_norm_inputs], dim=-1)
        outputs = self(full_inputs)#.transpose(-1, -2) # (batch_size, seq_len, vocab_size) 

        # for debugging reasons
        if torch.isnan(outputs).any():
            print("NAN IN OUTPUTS")

        if isinstance(outputs, torch.Tensor):
            raw_logits = outputs
        else:
            raw_logits = outputs.logits
        
        if self.args.predict_gaussian:
            raw_logits, raw_pred_variance = raw_logits.split([1,1], dim=-1)
            raw_logits = raw_logits.squeeze(-1)
            raw_pred_variance = raw_pred_variance.squeeze(-1)

        preds = raw_logits * known_input_stds.tile(1, 1, len(self.pred_indices)) + known_input_means.tile(1, 1, len(self.pred_indices))

        # direction part of loss
        # norm_full_seq_true_prices = torch.cat([norm_inputs, norm_labels[:,-(furthest_date+1):,:]], dim=1)
        # norm_true_price_targets = norm_full_seq_true_prices[:, :-(furthest_date)]
        # for i in range(1, len(self.pred_indices)):
        #     if self.pred_indices[i] == furthest_date:
        #         norm_true_price_targets = torch.cat([norm_true_price_targets, norm_full_seq_true_prices[:, self.pred_indices[i]:]], dim=-1)
        #     else:
        #         norm_true_price_targets = torch.cat([norm_true_price_targets, norm_full_seq_true_prices[:, self.pred_indices[i]:-(furthest_date-self.pred_indices[i])]], dim=-1)
        # direction_accuracy, direction_accuracy_per_target = calculate_directional_accuracy(norm_true_price_targets, raw_logits, movement_threshold=self.normalized_threshold)

        # # just for seeing what the data looks like, and if there is some bias in predictions
        # real_up, real_down, real_flat, pred_up, pred_down, pred_flat = get_movement_directions(
        #     norm_true_price_targets, raw_logits, movement_threshold=self.normalized_threshold
        # )
        # correct way to do directions
        thresholds = self.normalized_threshold * known_input_stds
        direction_total_acc, direction_acc_up, direction_acc_down, direction_acc_flat, expected_max_acc, expected_strat_acc = get_direction_accuracy_returns(targets, preds, thresholds=thresholds)

        # reference loss of naive model
        naive_MSE = self.MSE(torch.zeros_like(targets), targets) 
        naive_MAE = self.MAE(torch.zeros_like(targets), targets) # known_inputs.tile(1,1,len(self.pred_indices))


        if self.args.predict_gaussian:
            pred_variance = nn.functional.softplus(raw_pred_variance)
            loss = self.loss_fn(preds, targets, pred_variance)
        else:
            loss = self.loss_fn(preds, targets)
        
        # adjusting loss to be scaled relative to persistance model
        loss = loss / (naive_MSE + 1e-6)
        loss = loss / (direction_total_acc + 1e-6) 

        MSSE = self.MSE(preds, targets) / (naive_MSE + 1e-6)
        MASE = self.MAE(preds, targets) / (naive_MAE + 1e-6)


        # TODO try to get similar metrics into one graph (without a million logs)
        log_opts = {'on_step': (stage == 'train'), 'logger': True, 'sync_dist': True}

        # Log main loss and accuracy
        self.log(f"Loss/{stage}_loss", loss, prog_bar=True, **log_opts)
        self.log(f"Accuracy/{stage}_accuracy", direction_total_acc, **log_opts)
        self.log(f"Accuracy_finer/{stage}_up_accuracy", direction_acc_up, **log_opts)
        self.log(f"Accuracy_finer/{stage}_down_accuracy", direction_acc_down, **log_opts)
        self.log(f"Accuracy_finer/{stage}_flat_accuracy", direction_acc_flat, **log_opts)

        if stage=="val":
            self.log(f"Accuracy_finer/{stage}_expected_max", expected_max_acc, **log_opts)
            self.log(f"Accuracy_finer/{stage}_expected_strat", expected_strat_acc, **log_opts)

        # Log Naive Losses under a common group
        self.log(f"Naive_Losses/{stage}_MSE", naive_MSE, **log_opts)
        self.log(f"Naive_Losses/{stage}_MAE", naive_MAE, **log_opts)

        # Log Relative Losses under a common group
        self.log(f"Relative_Losses/{stage}_MSSE", MSSE, **log_opts)
        self.log(f"Relative_Losses/{stage}_MASE", MASE, **log_opts)


        # self.log(f"Directions_Real/{stage}_Up", real_up, **log_opts)
        # self.log(f"Directions_Real/{stage}_Down", real_down, **log_opts)
        # self.log(f"Directions_Real/{stage}_Flat", real_flat, **log_opts)
        # self.log(f"Directions_Pred/{stage}_Up", pred_up, **log_opts)
        # self.log(f"Directions_Pred/{stage}_Down", pred_down, **log_opts)
        # self.log(f"Directions_Pred/{stage}_Flat", pred_flat, **log_opts)

        # ICs = get_spearmanr_correlations(targets.clone().detach(), preds.clone().detach())

        # Log accuracy per target
        for i in range(len(self.pred_indices)):
            # temp_IR = ICs[:, i].mean()/(ICs[:,:, i].std() + 1e-6)
            # self.log(f"IR/{stage}_target_{self.pred_indices[i]}", temp_IR, **log_opts)
            # self.log(f"IC/{stage}_target_{self.pred_indices[i]}", ICs[:,:, i].mean(), **log_opts)
            # TODO?
            # self.log(f"Target_Accuracy/{stage}_target_{self.pred_indices[i]}", direction_accuracy_per_target[i], **log_opts)

            # Log naive losses per target
            temp_naive_MSE = self.MSE(known_inputs, targets[:, :, i:i+1])
            temp_naive_MAE = self.MAE(known_inputs, targets[:, :, i:i+1])
            self.log(f"Naive_Losses_target/{stage}_target_{self.pred_indices[i]}_MSE", temp_naive_MSE, **log_opts)
            self.log(f"Naive_Losses_target/{stage}_target_{self.pred_indices[i]}_MAE", temp_naive_MAE, **log_opts)

            # log relative losses per target
            temp_MSSE = self.MSE(preds[:,:,i:i+1], targets[:, :, i:i+1]) / (temp_naive_MSE + 1e-6)
            temp_MASE = self.MAE(preds[:,:,i:i+1], targets[:, :, i:i+1]) / (temp_naive_MAE + 1e-6)
            self.log(f"Relative_Losses_target/{stage}_target_{self.pred_indices[i]}_MSSE", temp_MSSE, **log_opts)
            self.log(f"Relative_Losses_target/{stage}_target_{self.pred_indices[i]}_MASE", temp_MASE, **log_opts)
            # log loss by target
            temp_loss = self.loss_fn(preds[:, :, i:i+1], targets[:, :, i:i+1])#, pred_variance[:, :, i:i+1])
            temp_loss = temp_loss / (temp_naive_MSE + 1e-6)
            # temp_loss = temp_loss / (direction_accuracy_per_target[i] + 1e-6)
            temp_loss = temp_loss / (direction_total_acc + 1e-6)
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
        if self.args.orthograd: # try sgd
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
