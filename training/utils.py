from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch
import pytorch_lightning as pl


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def dummy_dataloader(
    batch_size=32,
    seq_length=128,
    num_samples=1000,
    model_type="TransformerEncoder",
    config=None,
):
    # Adjust dummy data creation based on model type
    if model_type == "EncoderDecoder" or (
        model_type == "MultiTransformer" and "decoder" in config
    ):
        encoder_input_ids = torch.randint(0, 30000, (num_samples, seq_length))
        encoder_attention_mask = torch.ones(num_samples, seq_length)
        decoder_input_ids = torch.randint(0, 30000, (num_samples, seq_length))
        decoder_attention_mask = torch.ones(num_samples, seq_length)
        labels = torch.randint(
            0, 10, (num_samples, seq_length)
        )  # Assuming sequence classification
        dataset = TensorDataset(
            (encoder_input_ids, encoder_attention_mask),
            (decoder_input_ids, decoder_attention_mask),
            labels,
        )
    elif model_type == "DualEncoder":
        # Create dummy data for two encoders
        input_ids1 = torch.randint(0, 30000, (num_samples, seq_length))
        attention_mask1 = torch.ones(num_samples, seq_length)
        input_ids2 = torch.randint(0, 30000, (num_samples, seq_length))
        attention_mask2 = torch.ones(num_samples, seq_length)
        labels = torch.randint(0, 2, (num_samples,))
        dataset = TensorDataset(
            (input_ids1, attention_mask1), (input_ids2, attention_mask2), labels
        )
    elif model_type == "MultiTransformer":
        # Create dummy data for multiple encoders
        encoders_data = []
        for _ in config["encoders"]:
            input_ids = torch.randint(0, 30000, (num_samples, seq_length))
            attention_mask = torch.ones(num_samples, seq_length)
            encoders_data.append((input_ids, attention_mask))

        # If there's a decoder, create decoder data
        if "decoder" in config:
            decoder_input_ids = torch.randint(0, 30000, (num_samples, seq_length))
            decoder_attention_mask = torch.ones(num_samples, seq_length)
            labels = torch.randint(
                0, 10, (num_samples, seq_length)
            )  # Example for sequence generation
            dataset = TensorDataset(
                encoders_data, (decoder_input_ids, decoder_attention_mask), labels
            )
        else:
            labels = torch.randint(0, 2, (num_samples,))  # Example for classification
            dataset = TensorDataset(encoders_data, labels)
    else:
        # Default case (e.g., single TransformerEncoder)
        input_ids = torch.randint(0, 30000, (num_samples, seq_length))
        attention_mask = torch.ones(num_samples, seq_length)
        labels = torch.randint(0, 2, (num_samples,))  # Binary classification
        dataset = TensorDataset(input_ids, attention_mask, labels)

    return DataLoader(dataset, batch_size=batch_size)


# def dataloader(batch_size, data, shuffle=True):
#     dataset = TensorDataset(*data)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class ShakespeareDataset(Dataset):
    def __init__(self, text_file, seq_len):
        """
        Args:
            text_file (string): Path to the text file.
            seq_len (int): Length of the input sequences.
        """
        self.seq_len = seq_len

        # Read the text file
        with open(text_file, "r") as f:
            self.text = f.read()

        # Create character-to-index and index-to-character mappings
        self.chars = sorted(list(set(self.text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

        # Encode the entire text
        self.encoded_text = [self.char_to_idx[ch] for ch in self.text]

    def __len__(self):
        # Number of sequences.  We subtract seq_len because the last tokens
        # won't have enough subsequent tokens to form a target sequence.
        return len(self.encoded_text) - self.seq_len

    def __getitem__(self, idx):
        # Create input sequence (all tokens except the last)
        input_sequence = self.encoded_text[idx : idx + self.seq_len]
        # Create target sequence (shifted by one position)
        target_sequence = self.encoded_text[idx + 1 : idx + self.seq_len + 1]
        return torch.tensor(input_sequence, dtype=torch.long), torch.tensor(
            target_sequence, dtype=torch.long
        )

    def get_vocab_size(self):
        return len(self.chars)

    def decode(self, token_ids):
        """Decodes a list of token IDs to a string."""
        return "".join([self.idx_to_char[i] for i in token_ids])

    def encode(self, text_string):
        """Encodes a string to a list of token IDs"""
        return [self.char_to_idx[ch] for ch in text_string]


class ShakespeareDataModule(pl.LightningDataModule):
    def __init__(self, train_file, val_file, test_file, seq_len, batch_size):
        super().__init__()
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.seq_len = seq_len
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = ShakespeareDataset(self.train_file, self.seq_len)
            self.val_dataset = ShakespeareDataset(self.val_file, self.seq_len)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = ShakespeareDataset(self.test_file, self.seq_len)

    def train_dataloader(self):  # increase workers
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )

    def get_vocab_size(self):  # Added for convenience
        #  We can get vocab size from any of the datasets, assuming they share the same vocab
        return self.train_dataset.get_vocab_size()


# --- Downloading and Splitting Tiny Shakespeare (if you don't have it) ---


def download_and_split_shakespeare(input_file_path="input.txt"):
    import requests
    import os

    if not os.path.exists(input_file_path):
        # Download the tiny shakespeare dataset
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        with open(input_file_path, "w") as f:
            f.write(response.text)

    with open(input_file_path, "r") as f:
        text = f.read()

    # Split the dataset (simple split for demonstration)
    n = len(text)
    train_data = text[: int(n * 0.8)]
    val_data = text[int(n * 0.8) : int(n * 0.9)]
    test_data = text[int(n * 0.9) :]

    # Write to files
    with open("train.txt", "w") as f:
        f.write(train_data)
    with open("val.txt", "w") as f:
        f.write(val_data)
    with open("test.txt", "w") as f:
        f.write(test_data)


def export_model(model, dummy_input, model_path):
    torch.onnx.export(
        model,
        dummy_input,
        model_path,
        verbose=True,
        input_names=["input"],
        output_names=["output"],
        opset_version=12,
    )


class OrthoGrad(torch.optim.Optimizer):
    def __init__(
        self, params, base_optimizer_cls=torch.optim.Adam, **base_optimizer_args
    ):
        """
        A wrapper optimizer that projects gradients to be orthogonal
        to the current parameters before performing an update.

        Args:
            params (iterable): Iterable of parameters to optimize.
            base_optimizer_cls (Optimizer class): The base optimizer class
                (e.g., torch.optim.SGD, torch.optim.AdamW).
            **base_optimizer_args: Arguments for the base optimizer.
                For example, lr=1e-3, weight_decay=1e-2, etc.
        """
        # Minimal defaults for OrthoGrad itself (nothing special needed).
        defaults = {}
        super().__init__(params, defaults)

        # Create the wrapped/base optimizer using *our* param_groups.
        self.base_optimizer = base_optimizer_cls(
            self.param_groups, **base_optimizer_args
        )

    @staticmethod
    def _orthogonalize_gradients(params):
        """
        Projects the gradient g to be orthogonal to the current weights w.

        g_orth = g - ( (w·g)/(w·w + eps) ) * w

        And then re-scales g_orth to have the same norm as g.
        """
        with torch.no_grad():
            for p in params:
                if p.grad is not None:
                    w = p.view(-1)
                    g = p.grad.view(-1)

                    w_norm_sq = torch.dot(w, w) + 1e-30
                    proj = torch.dot(w, g) / w_norm_sq
                    g_orth = g - proj * w

                    g_norm = g.norm(2)
                    g_orth_norm = g_orth.norm(2) + 1e-30
                    g_orth_scaled = g_orth * (g_norm / g_orth_norm)

                    p.grad.copy_(g_orth_scaled.view_as(p.grad))

    def step(self, closure=None):
        for group in self.param_groups:
            self._orthogonalize_gradients(group["params"])

        return self.base_optimizer.step(closure)
