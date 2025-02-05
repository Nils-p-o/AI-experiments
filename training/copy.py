import pytorch_lightning as pl
import torch.optim as optim
import torch
import torch.nn as nn
from torchmetrics import Accuracy
from torchmetrics import Perplexity  # Import Perplexity


class TransformerExperiment(pl.LightningModule):
    def __init__(self, model, learning_rate=2e-5, vocab_size=10000):  # Add vocab_size
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss()
        # Use vocab_size for num_classes
        self.accuracy = Accuracy(task="multiclass", num_classes=vocab_size)
        self.perplexity = Perplexity(ignore_index=-100) #ignore padding

        self.save_hyperparameters(ignore=["model"])

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def _shared_step(self, batch, batch_idx, stage="train"):
        inputs, labels = batch

        # 1. Shift inputs and targets
        shifted_inputs = inputs[:, :-1]  # All but the last token
        shifted_labels = labels[:, 1:]  # All but the first token

        # 2. Get model outputs (logits)
        outputs = self(shifted_inputs) # No need to get batch_size from outputs
        logits = outputs.transpose(-1, -2) # (batch_size, vocab_size, seq_len)

        # 3. Calculate loss
        loss = self.loss_fn(logits, shifted_labels)

        # 4. Flatten for accuracy calculation
        flattened_logits = logits.reshape(-1, logits.size(-1))  # (batch_size * seq_len, vocab_size)
        flattened_labels = shifted_labels.reshape(-1)  # (batch_size * seq_len)

        # 5. Calculate Accuracy (Optional, but useful for monitoring)
        preds = torch.argmax(flattened_logits, dim=-1)
        acc = self.accuracy(preds, flattened_labels)

        # 6. Calculate perplexity
        perplexity = self.perplexity(logits, shifted_labels)

        # Logging
        self.log(f"{stage}_loss", loss, on_step=(stage == "train"), on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{stage}_acc", acc, on_step=(stage == "train"), on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{stage}_perplexity", perplexity, on_step=(stage == 'train'), on_epoch=True, prog_bar=True, logger=True) # added perplexity

        return loss


    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")


    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx): # Added test_step
        return self._shared_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)  # Use AdamW
        return optimizer