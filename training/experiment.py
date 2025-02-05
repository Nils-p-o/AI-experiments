import pytorch_lightning as pl
import torch.optim as optim
import torch
import torch.nn as nn
from torchmetrics import Accuracy
from torchmetrics.text import Perplexity
import math
from torch.optim.lr_scheduler import LambdaLR




class TransformerExperiment(pl.LightningModule):
    def __init__(self, model, learning_rate=2e-5, batch_size=32, vocab_size=2, warmup_steps=100, t_0=500, t_mult=1.5, lr_mult=0.5):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.t_0 = t_0
        self.t_mult = t_mult
        self.lr_mult = lr_mult
        self.batch_size = batch_size
        self.loss_fn = nn.CrossEntropyLoss()  # Example, adjust as needed
        self.accuracy = Accuracy(
            task="multiclass", num_classes=vocab_size
        )  # Adjust task and num_classes
        self.perplexity = Perplexity()
        self.save_hyperparameters(
            ignore=["model"]
        )  # Saves the hyperparameters to the checkpoint, except the model itself

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        # Adjust based on your model's input
        inputs, labels = batch  # labels -> (batch_size, seq_len) check this
        inputs = inputs[:, :-1]
        labels = labels[:, 1:]

        outputs = self(inputs).transpose(
            -1, -2
        )  # (batch_size, seq_len, vocab_size) -> (batch_size, vocab_size, seq_len)

        # Adjust according to your model's output and task
        if isinstance(outputs, torch.Tensor):
            logits = outputs  # If model directly returns logits
        else:
            logits = outputs.logits  # Or outputs[0] if it's a tuple/list

        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=-2)
        acc = self.accuracy(preds, labels)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        # Same input/output adjustments as in training_step
        inputs, labels = batch
        inputs = inputs[:, :-1]
        labels = labels[:, 1:]
        
        outputs = self(inputs).transpose(-1, -2)

        if isinstance(outputs, torch.Tensor):
            logits = outputs
        else:
            logits = outputs.logits

        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=-2)
        acc = self.accuracy(preds, labels)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    # def configure_optimizers(self):
    #     optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
    #     return optimizer
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        def lr_lambda(current_step):
            min_lr = 1e-6

            current_cycle_step = current_step
            cycle_nr = 0
            for _ in range(50):
                t_curr = self.t_0 * (self.t_mult ** cycle_nr)
                if current_cycle_step > t_curr:
                    current_cycle_step -= t_curr
                    cycle_nr += 1
                else:
                    break
            
            current_peak_lr = (self.lr_mult ** cycle_nr)
            

            if current_cycle_step < self.warmup_steps: # Linear warmup
                return (current_peak_lr - min_lr) * float(current_cycle_step) / float(max(1, self.warmup_steps)) + min_lr

            if current_cycle_step >= self.warmup_steps and current_cycle_step <= t_curr:
                progress = float(current_cycle_step - self.warmup_steps) / float(max(1, t_curr - self.warmup_steps))
                return current_peak_lr * 0.5 * (math.cos(math.pi * progress) + 1) + 1e-6


        scheduler = LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
