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

# for debug / timimg for speedups
import time
cummulative_times = {
    "preprocessing": 0,
    "model": 0,
    "loss": 0,
    "metrics": 0
}

def get_direction_accuracy_returns(actual_returns, predicted_returns, thresholds=0.01):
    batch,seq_len,targets, sequences = actual_returns.shape
    actual_movements = torch.where(actual_returns > thresholds, 1, torch.where(actual_returns < -thresholds, -1, 0))
    predicted_movements = torch.where(predicted_returns > thresholds, 1, torch.where(predicted_returns < -thresholds, -1, 0))

    acc_up = ((actual_movements==1) * (predicted_movements==1)).sum()/(actual_movements==1).sum()
    acc_down = ((actual_movements==-1) * (predicted_movements==-1)).sum()/(actual_movements==-1).sum()
    if actual_movements.sum() == 0:
        acc_flat = 0
    else:
        acc_flat = ((actual_movements==0) * (predicted_movements==0)).sum()/(actual_movements==0).sum()

    total_acc = (actual_movements == predicted_movements).sum() / actual_movements.numel()

    # per target
    actual_up = (actual_movements==1).sum(dim=(0,1,3)).unsqueeze(0)
    actual_down = (actual_movements==-1).sum(dim=(0,1,3)).unsqueeze(0)
    actual_flat = (actual_movements==0).sum(dim=(0,1,3)).unsqueeze(0)

    expected_acc = torch.max(torch.cat([actual_up, actual_down, actual_flat], dim=0),dim=0).values / (batch * seq_len * sequences)
    expected_strat = torch.cat([actual_up, actual_down, actual_flat], dim=0).square().sum(dim=0) / (batch * seq_len * sequences)**2
    expected_max_acc_target = torch.max(torch.cat([expected_acc.unsqueeze(0), expected_strat.unsqueeze(0)], dim=0),dim=0).values
    acc_per_target = torch.sum(actual_movements==predicted_movements, dim=(0,1,3)) / (batch * seq_len * sequences)

    # per sequence
    actual_up = (actual_movements==1).sum(dim=(0,1,2)).unsqueeze(0)
    actual_down = (actual_movements==-1).sum(dim=(0,1,2)).unsqueeze(0)
    actual_flat = (actual_movements==0).sum(dim=(0,1,2)).unsqueeze(0)
    expected_acc = torch.max(torch.cat([actual_up, actual_down, actual_flat], dim=0),dim=0).values / (batch * seq_len * targets)
    expected_strat = torch.cat([actual_up, actual_down, actual_flat], dim=0).square().sum(dim=0) / (batch * seq_len * targets)**2
    expected_max_acc_sequence = torch.max(torch.cat([expected_acc.unsqueeze(0), expected_strat.unsqueeze(0)], dim=0),dim=0).values
    acc_per_sequence = torch.sum(actual_movements==predicted_movements, dim=(0,1,2)) / (batch * seq_len * targets) 


    return total_acc, acc_up, acc_down, acc_flat, expected_max_acc_target, expected_max_acc_sequence, acc_per_target, acc_per_sequence

def z_normalize_additional_inputs(additional_inputs):
    for i in range(additional_inputs.shape[1]):
        additional_inputs[:, i, :] = (additional_inputs[:, i, :] - additional_inputs[:, i, :].mean(dim=-1, keepdim=True).tile(1, 1, additional_inputs.shape[2])) / additional_inputs[:, i, :].std(dim=-1, keepdim=True).tile(1, 1, additional_inputs.shape[2])
    return additional_inputs

def get_spearmanr_correlations(actual_prices, predicted_values):
    # batch_size, seq_len, num_targets, sequences = actual_prices.shape
    # ICs = torch.zeros((batch_size, seq_len, num_targets))

    # for i in range(batch_size):
    #     for j in range(seq_len):
    #         for k in range(num_targets):
    #             ICs[i,j,k] = stats.spearmanr(actual_prices[i,j,k], predicted_values[i,j,k]).correlation
    # TODO for later, rewrite in torch for speed
    actual_np = actual_prices.cpu().numpy()
    predicted_np = predicted_values.cpu().numpy()

    batch_size, seq_len, num_targets, sequences = actual_np.shape

    if sequences < 2:
        ICs_np = np.full((batch_size, seq_len, num_targets), np.nan)
        return torch.from_numpy(ICs_np).to(actual_prices.device)

    # Rank data along the last axis (axis=-1 is equivalent to axis=3 here)
    # method='average' handles ties in the standard way for Spearman.
    ranked_actual = stats.rankdata(actual_np, axis=-1, method='average')
    ranked_predicted = stats.rankdata(predicted_np, axis=-1, method='average')

    # Now calculate Pearson correlation on the ranks
    # Demean the ranks along the last axis
    ranked_actual_demeaned = ranked_actual - ranked_actual.mean(axis=-1, keepdims=True)
    ranked_predicted_demeaned = ranked_predicted - ranked_predicted.mean(axis=-1, keepdims=True)

    # Numerator of Pearson correlation
    numerator = (ranked_actual_demeaned * ranked_predicted_demeaned).sum(axis=-1)

    # Denominator of Pearson correlation
    # Sum of squares of demeaned ranks
    sum_sq_demeaned_actual = (ranked_actual_demeaned**2).sum(axis=-1)
    sum_sq_demeaned_predicted = (ranked_predicted_demeaned**2).sum(axis=-1)
    
    denominator = np.sqrt(sum_sq_demeaned_actual * sum_sq_demeaned_predicted)

    # Calculate correlations
    # Initialize with NaN to handle cases where denominator is zero (constant ranks)
    correlations_np = np.full(denominator.shape, np.nan) 
    
    # Create a mask for valid denominators (non-zero)
    # Use a small epsilon for floating point comparisons
    valid_mask = denominator > 1e-12 

    # Only calculate correlation where denominator is valid
    correlations_np[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
    
    # Clip to ensure values are within [-1, 1] due to potential floating point inaccuracies
    correlations_np = np.clip(correlations_np, -1.0, 1.0)

    # If one of the original (unranked) series was constant, its ranks will be constant,
    # leading to a standard deviation of ranks (and sum_sq_demeaned) being 0.
    # The `valid_mask` above handles this by leaving `correlations_np` as NaN.
    # This matches the behavior of `scipy.stats.spearmanr` which returns NaN for constant input.

    return torch.from_numpy(correlations_np).to(actual_prices.device)
        

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
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.t_0 = t_0
        self.t_mult = t_mult
        self.lr_mult = lr_mult
        self.batch_size = batch_size
        # self.loss_fn = nn.MSELoss() # MASE (NLL for distribution)
        self.loss_fn = nn.L1Loss()
        self.MSE = nn.MSELoss()
        self.MAE = nn.L1Loss()
        self.threshold = 0.01
        self.pred_indices = args.indices_to_predict
        self.save_hyperparameters(
            ignore=["model"]
        )
        self.args = args
        self.num_sequences = len(args.tickers)
        self.tickers = args.tickers

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


    def _shared_step(self, batch, batch_idx, stage="train"):
        time_preprocessing_start = time.time_ns()
        # recieve in shapes (batch_size, features, seq_len, (targets 1), num_sequences)
        inputs, labels = batch 
        batch_size, num_features, seq_len, num_sequences = inputs.shape
        
        known_inputs = inputs[:,0, :].unsqueeze(2) # (batch_size, seq_len, 1, num_sequences)

        targets = labels # (batch_size, (features 1), seq_len, targets, num_sequences)

        target_input_means = known_inputs.mean(dim=1, keepdim=True).tile(1, known_inputs.shape[1],1,1)
        target_input_stds = known_inputs.std(dim=1, keepdim=True).tile(1, known_inputs.shape[1],1,1)

        norm_inputs = (known_inputs - target_input_means) / target_input_stds

        additional_inputs = inputs[:, 1:, :, :]
        additional_inputs = additional_inputs.transpose(1, 2) # (batch_size, seq_len, features, num_sequences)
        means_of_additonal_inputs = additional_inputs.mean(dim=1, keepdim=True).tile(1, additional_inputs.shape[1],1,1)
        stds_of_additonal_inputs = additional_inputs.std(dim=1, keepdim=True).tile(1, additional_inputs.shape[1],1,1)
        full_inputs = torch.cat([norm_inputs, (additional_inputs-means_of_additonal_inputs)/stds_of_additonal_inputs], dim=2)
        # full_inputs = torch.cat([full_inputs, target_input_means, target_input_stds], dim=2)
        full_inputs = full_inputs.transpose(2, 3)

        seperator = torch.zeros((targets.shape[0], 1),dtype=torch.int, device=inputs.device) # (batch_size, 1)
        tickers = torch.arange(self.num_sequences, device=inputs.device)
        tickers = tickers.unsqueeze(0).repeat(targets.shape[0], 1) # (batch_size, num_sequences)

        if stage == "train":
            cummulative_times["preprocessing"] += (time.time_ns()-time_preprocessing_start)/1e6
        time_model_start = time.time_ns()
        outputs = self(full_inputs, seperator, tickers).view(targets.shape[0], -1, len(self.pred_indices), self.num_sequences) #.transpose(-1, -2) # (batch_size, seq_len, targets*num_sequences) 
        if stage == "train":
            cummulative_times["model"] += (time.time_ns()-time_model_start)/1e6
        outputs = outputs[:,1:,:,:]

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

        preds = raw_logits * target_input_stds.tile(1, 1, len(self.pred_indices),1) + target_input_means.tile(1, 1, len(self.pred_indices), 1)

        time_loss_calc_start = time.time_ns()
        if self.args.predict_gaussian:
            pred_variance = nn.functional.softplus(raw_pred_variance)
            loss = self.loss_fn(preds, targets, pred_variance)
        else:
            loss = 0
            target_weights = [2.0, 1.0, 0.5]
            seen_unseen_weights = [1.0, 1.0]
            # TODO seen/unseen weights
            for i in range(len(self.pred_indices)):
                seen_current_preds = preds[:, :-self.pred_indices[i], i, :]
                unseen_current_preds = preds[:, -self.pred_indices[i]:, i, :]
                seen_current_targets = targets[:, :-self.pred_indices[i], i, :]
                unseen_current_targets = targets[:, -self.pred_indices[i]:, i, :]
                loss += target_weights[i] * (seen_unseen_weights[0] * 2/sum(seen_unseen_weights)) * self.loss_fn(seen_current_preds, seen_current_targets) * ((seq_len-self.pred_indices[i])/seq_len)
                loss += target_weights[i] * (seen_unseen_weights[1] * 2/sum(seen_unseen_weights)) * self.loss_fn(unseen_current_preds, unseen_current_targets) * (self.pred_indices[i]/seq_len)
            loss = loss / sum(target_weights)
            # loss = self.loss_fn(preds, targets)
        
        naive_loss = self.loss_fn(torch.zeros_like(targets), targets) 

        # actual_movements_for_loss = torch.where(targets > self.threshold, 1, torch.where(targets < -self.threshold, -1, 0))
        # predicted_movements_for_loss = torch.where(preds > self.threshold, 1, torch.where(preds < -self.threshold, -1, 0))
        # direction_total_acc_for_loss = (actual_movements_for_loss == predicted_movements_for_loss).sum() / actual_movements_for_loss.numel()

        # adjusting loss to be scaled relative to persistance model and dicertional acc
        # loss = loss / (naive_loss + 1e-6) # testing for now
        # loss = loss / (direction_total_acc_for_loss + 1e-6)

        if stage == "train":
            cummulative_times["loss"] += (time.time_ns()-time_loss_calc_start)/1e6

        self.log(f"Loss/{stage}_loss", loss/(naive_loss + 1e-6), prog_bar=True, on_step=(stage == 'train'), on_epoch=(stage != 'train'), logger=True, sync_dist=True)
        # self.log(f"Accuracy/{stage}_accuracy", direction_total_acc_for_loss, on_step=(stage == 'train'), on_epoch=(stage != 'train'), logger=True, sync_dist=True)

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
            log_opts_step = {'on_step': True, 'on_epoch': False, 'logger': True, 'sync_dist': True}
            log_opts_epoch = {'on_step': False, 'on_epoch': True, 'logger': True, 'sync_dist': True}
            current_log_opts = log_opts_step if stage == "train" else log_opts_epoch

            direction_total_acc, _,_,_,expected_acc_target, expected_acc_sequence, acc_target, acc_sequence = get_direction_accuracy_returns(targets, preds, thresholds=self.threshold)
            self.log(f"Accuracy/{stage}_accuracy", direction_total_acc, **current_log_opts)

            if stage=="val":
                for i in range(len(self.pred_indices)):
                    self.log(f"Accuracy_finer/{stage}_expected_max_acc_{self.pred_indices[i]}", expected_acc_target[i], **current_log_opts)
                for i in range(self.num_sequences):
                    self.log(f"Accuracy_finer/{stage}_expected_max_acc_{self.tickers[i]}", expected_acc_sequence[i], **current_log_opts)
            if stage == "val":
                for i in range(len(self.pred_indices)):
                    seen_current_preds = preds[:,:-self.pred_indices[i],i]
                    seen_current_targets = targets[:,:-self.pred_indices[i],i]
                    seen_MSE = self.MSE(seen_current_preds, seen_current_targets)
                    seen_MAE = self.MAE(seen_current_preds, seen_current_targets)

                    unseen_current_preds = preds[:,-self.pred_indices[i]:,i]
                    unseen_current_targets = targets[:,-self.pred_indices[i]:,i]
                    unseen_MSE = self.MSE(unseen_current_preds, unseen_current_targets)
                    unseen_MAE = self.MAE(unseen_current_preds, unseen_current_targets)

                    relative_MSE = unseen_MSE/(seen_MSE + 1e-6)
                    relative_MAE = unseen_MAE/(seen_MAE + 1e-6)

                    self.log(f"Split_Error/{stage}_relative_MSE_{self.pred_indices[i]}", relative_MSE, **current_log_opts)
                    self.log(f"Split_Error/{stage}_relative_MAE_{self.pred_indices[i]}", relative_MAE, **current_log_opts)

                    # TODO maybe add split dicertional accuracy
                    seen_target_movements = torch.where(seen_current_targets > self.threshold, 1, torch.where(seen_current_targets < -self.threshold, -1, 0))
                    seen_predicted_movements = torch.where(seen_current_preds > self.threshold, 1, torch.where(seen_current_preds < -self.threshold, -1, 0))

                    seen_direction_acc = (seen_target_movements == seen_predicted_movements).sum() / seen_target_movements.numel()
                    self.log(f"Split_Error/{stage}_seen_accuracy_{self.pred_indices[i]}", seen_direction_acc, **current_log_opts)

                    unseen_target_movements = torch.where(unseen_current_targets > self.threshold, 1, torch.where(unseen_current_targets < -self.threshold, -1, 0))
                    unseen_predicted_movements = torch.where(unseen_current_preds > self.threshold, 1, torch.where(unseen_current_preds < -self.threshold, -1, 0))

                    unseen_direction_acc = (unseen_target_movements == unseen_predicted_movements).sum() / unseen_target_movements.numel()
                    self.log(f"Split_Error/{stage}_unseen_accuracy_{self.pred_indices[i]}", unseen_direction_acc, **current_log_opts)
                    self.log(f"Split_Error/{stage}_relative_accuracy_{self.pred_indices[i]}", unseen_direction_acc/(seen_direction_acc + 1e-6), **current_log_opts)

            naive_MSE = self.MSE(torch.zeros_like(targets), targets)
            naive_MAE = self.MAE(torch.zeros_like(targets), targets)
            MSSE = self.MSE(preds, targets) / (naive_MSE + 1e-6)
            MASE = self.MAE(preds, targets) / (naive_MAE + 1e-6)

            # Log Relative Losses under a common group
            self.log(f"Relative_Losses/{stage}_MSSE", MSSE, **current_log_opts)
            self.log(f"Relative_Losses/{stage}_MASE", MASE, **current_log_opts)

            # can only be done per time step
            ICs = get_spearmanr_correlations(targets.clone().detach(), preds.clone().detach())
            # Log accuracy per target
            for i in range(len(self.pred_indices)):
                valid_ics_target = ICs[:,:,i][~torch.isnan(ICs[:,:,i])]
                if len(valid_ics_target) > 1:
                    temp_IR = valid_ics_target.mean()/(valid_ics_target.std() + 1e-9)
                    mean_IC_target = valid_ics_target.mean()
                elif len(valid_ics_target) == 1:
                    temp_IR = torch.tensor(float('nan'), device=ICs.device)
                    mean_IC_target = valid_ics_target[0]
                else:
                    temp_IR = torch.tensor(float('nan'), device=ICs.device)
                    mean_IC_target = torch.tensor(float('nan'), device=ICs.device)

                self.log(f"IR/{stage}_target_{self.pred_indices[i]}", temp_IR, **current_log_opts)
                self.log(f"IC/{stage}_target_{self.pred_indices[i]}", mean_IC_target, **current_log_opts)
                self.log(f"Target_Accuracy/{stage}_target_{self.pred_indices[i]}", acc_target[i], **current_log_opts)


                # Log losses per target
                current_target_preds = preds[:,:,i:i+1,:]
                current_target_targets = targets[:, :, i:i+1, :]
                temp_MSE = self.MSE(current_target_preds, current_target_targets)
                temp_MAE = self.MAE(current_target_preds, current_target_targets)
                self.log(f"Losses_target/{stage}_target_{self.pred_indices[i]}_MSE", temp_MSE, **current_log_opts)
                self.log(f"Losses_target/{stage}_target_{self.pred_indices[i]}_MAE", temp_MAE, **current_log_opts)

                # log relative losses per target
                temp_naive_MSE = self.MSE(torch.zeros_like(current_target_targets), current_target_targets)
                temp_naive_MAE = self.MAE(torch.zeros_like(current_target_targets), current_target_targets)
                temp_MSSE = temp_MSE / (temp_naive_MSE + 1e-6)
                temp_MASE = temp_MAE / (temp_naive_MAE + 1e-6)
                self.log(f"Relative_Losses_target/{stage}_target_{self.pred_indices[i]}_MSSE", temp_MSSE, **current_log_opts)
                self.log(f"Relative_Losses_target/{stage}_target_{self.pred_indices[i]}_MASE", temp_MASE, **current_log_opts)
                # log loss by target
                temp_loss = self.loss_fn(current_target_preds, current_target_targets)
                temp_loss = temp_loss / (temp_naive_MSE + 1e-6) 
                temp_loss = temp_loss / (acc_target[i] + 1e-6)
                self.log(f"Losses_target/{stage}_target_{self.pred_indices[i]}", temp_loss, **current_log_opts)
            
            for i in range(self.num_sequences):
                self.log(f"Target_Accuracy/{stage}_sequence_{self.tickers[i]}", acc_sequence[i], **current_log_opts)
                # Log losses per sequence
                current_seq_preds = preds[:,:,:,i:i+1]
                current_seq_targets = targets[:, :, :, i:i+1]
                temp_MSE = self.MSE(current_seq_preds, current_seq_targets)
                temp_MAE = self.MAE(current_seq_preds, current_seq_targets)
                self.log(f"Losses_sequence/{stage}_sequence_{i}_MSE", temp_MSE, **current_log_opts)
                self.log(f"Losses_sequence/{stage}_sequence_{i}_MAE", temp_MAE, **current_log_opts)

                # log relative losses per sequence
                temp_naive_MSE = self.MSE(torch.zeros_like(current_seq_targets), current_seq_targets)
                temp_naive_MAE = self.MAE(torch.zeros_like(current_seq_targets), current_seq_targets)
                temp_MSSE = temp_MSE / (temp_naive_MSE + 1e-6)
                temp_MASE = temp_MAE / (temp_naive_MAE + 1e-6)
                self.log(f"Relative_Losses_sequence/{stage}_sequence_{self.tickers[i]}_MSSE", temp_MSSE, **current_log_opts)
                self.log(f"Relative_Losses_sequence/{stage}_sequence_{self.tickers[i]}_MASE", temp_MASE, **current_log_opts)
                # log loss by sequence
                temp_loss = self.loss_fn(current_seq_preds, current_seq_targets)
                temp_loss = temp_loss / (temp_naive_MSE + 1e-6) 
                temp_loss = temp_loss / (acc_sequence[i] + 1e-6)
                self.log(f"Losses_sequence/{stage}_sequence_{self.tickers[i]}", temp_loss, **current_log_opts)

            if stage == "train":
                cummulative_times["metrics"] += (time.time_ns() - time_metrics_start)/1e6
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
    
    def on_validation_epoch_end(self):
        # gets average time for each stage
        if self.trainer.is_global_zero:
            self.log(f"Times/preprocessing", cummulative_times["preprocessing"]/(self.global_step+1), logger=True)
            self.log(f"Times/forward_pass", cummulative_times["model"]/(self.global_step+1), logger=True)
            self.log(f"Times/loss", cummulative_times["loss"]/(self.global_step+1), logger=True)
            self.log(f"Times/metrics", cummulative_times["metrics"]/(self.global_step+1), logger=True)

    # def on_train_batch_end(self, outputs, batch, batch_idx):
    #     if self.model.__class__.__name__ == "nGPT":
    #         normalize_weights_and_enforce_positive_eigenvalues(self.model)
    #     return
