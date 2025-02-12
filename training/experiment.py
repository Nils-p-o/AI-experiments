import pytorch_lightning as pl
import torch.optim as optim
import torch
import torch.nn as nn
from torchmetrics import Accuracy
from torchmetrics.text import Perplexity
import math
from torch.optim.lr_scheduler import LambdaLR


class TransformerExperiment(pl.LightningModule):
    def __init__(
        self,
        model,
        learning_rate=2e-5,
        batch_size=32,
        vocab_size=2,
        warmup_steps=100,
        t_0=5000,
        t_mult=1.5,
        lr_mult=0.5, # maybe higher?
    ):
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
        )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


    def _shared_step(self, batch, batch_idx, stage="train"):
        inputs, labels = batch
        inputs = inputs[:, :-1] # shifted
        labels = labels[:, 1:]

        outputs = self(inputs).transpose(-1, -2) # (batch_size, seq_len, vocab_size) 

        if isinstance(outputs, torch.Tensor):
            logits = outputs
        else:
            logits = outputs.logits

        loss = self.loss_fn(logits, labels)
        perplexity = self.perplexity(logits.transpose(-1,-2), labels)
        preds = torch.argmax(logits, dim=-2)
        acc = self.accuracy(preds, labels)

        self.log(f"{stage}_loss", loss, on_step=(stage == "train"), logger=True)
        self.log(f"{stage}_acc", acc, on_step=(stage == "train"), logger=True)
        self.log(f"{stage}_perplexity", perplexity, on_step=(stage == 'train'), logger=True)
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
