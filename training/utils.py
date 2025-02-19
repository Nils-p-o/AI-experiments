from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch
import pytorch_lightning as pl

import re
from typing import List, Optional, Union
from sentencepiece import SentencePieceProcessor

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



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
            num_workers=4,
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


def stablemax(x, epsilon=1e-30, dim=-1):
    return torch.where(x < 0, 1 / (1 - x + epsilon), x + 1)


def log_stablemax(x, dim=-1):
    s_x = stablemax(x)
    return torch.log(s_x / torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, reduction="mean"):
    labels = labels.to(torch.int64)
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)
    prediction_logprobs = torch.gather(logprobs, index=labels[:, None], dim=-1).to(
        torch.float64
    )

    loss = (
        -torch.mean(prediction_logprobs)
        if reduction == "mean"
        else -prediction_logprobs
    )
    return loss


def taylor_softmax(x, dim=-1):  # TODO mess around with this
    x_prim = x - torch.min(x, dim=dim, keepdim=True).values
    y = 1 + x_prim + x_prim**2 / 2
    return y


def log_taylor_softmax(x, dim=-1):
    t_x = taylor_softmax(x, dim=dim)
    return torch.log(t_x / torch.sum(t_x, dim=dim, keepdim=True))


class custom_cross_entropy(torch.nn.Module):
    def __init__(self, reduction="mean", softmax_fn=torch.nn.functional.softmax):
        super(custom_cross_entropy, self).__init__()
        self.reduction = reduction
        self.softmax_fn = softmax_fn

    def forward(self, logits, labels):
        labels = labels.to(torch.int64) # [..., seq_len]
        soft_logits = self.softmax_fn(logits.to(torch.float64), dim=-2)  # [..., vocab_size, seq_len]
        logprobs = torch.log(
            soft_logits.to(torch.float64)
            / torch.sum(soft_logits, dim=-2, keepdim=True).to(torch.float64)
        )
        prediction_logprobs = torch.gather(logprobs, index=labels.unsqueeze(dim=-2), dim=-2).to(
            torch.float32
        )

        loss = (
            -torch.mean(prediction_logprobs)
            if self.reduction == "mean"
            else -prediction_logprobs
        )
        return loss


class WikitextDataset(Dataset):
    def __init__(
        self,
        text_file: str,
        seq_len: int,
        preload: bool = True,
        tokenizer_model_path: Optional[str] = None,
        use_character_encoding: bool = False,
    ):
        """
        Args:
            text_file: Path to the wikitext file.
            seq_len: Length of the input sequences.
            preload: Whether to load the entire file into memory.
            tokenizer_model_path: Path to a SentencePiece tokenizer model (e.g., LLaMA tokenizer.model).
            use_character_encoding: Use character-level encoding instead of the tokenizer.
        """
        self.seq_len = seq_len
        self.preload = preload
        self.use_character_encoding = use_character_encoding
        self.tokenizer = None

        if not use_character_encoding:
            if tokenizer_model_path is None:
                raise ValueError(
                    "tokenizer_model_path must be provided if use_character_encoding is False."
                )
            self.tokenizer = SentencePieceProcessor(model_file=tokenizer_model_path)
            self.vocab_size = self.tokenizer.vocab_size()  # Pre-calculate for efficiency.
            self.bos_id = self.tokenizer.bos_id()
            self.eos_id = self.tokenizer.eos_id()
            self.pad_id = self.tokenizer.pad_id()

        if preload:
            with open(text_file, "r", encoding="utf-8") as f:
                self.text = self.clean_wikitext(f.read())
            if use_character_encoding:
                self.chars = sorted(list(set(self.text)))
                self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
                self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
                self.encoded_text = [self.char_to_idx[ch] for ch in self.text]
            else:
                self.encoded_text = self.tokenizer.encode(self.text)  # Encode the whole text

        else:  # not preload
            self.text_file = text_file
            if use_character_encoding:
                self.chars = set()
                self.char_to_idx = {}
                self.idx_to_char = {}
                self._build_vocab()
                # Don't load self.encoded_text if not preloading
                self.encoded_text = None

            else:  # Tokenizer, no preload:  vocab is known from tokenizer.
               self.encoded_text = None

    def _build_vocab(self):
        """Builds the vocabulary from the text file (for character encoding, when not preloading)."""
        temp_chars = set()
        with open(self.text_file, "r", encoding="utf-8") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), ""):  # Read in 1MB chunks
                cleaned_chunk = self.clean_wikitext(chunk)
                temp_chars.update(cleaned_chunk)
        self.chars = sorted(list(temp_chars))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def clean_wikitext(self, text: str) -> str:
        """
        Cleans wikitext formatting.  Keeps some basic formatting.

        Args:
            text: The raw wikitext string.

        Returns:
            The cleaned string.
        """
        # Remove lines starting with = (headers) and empty lines
        text = re.sub(r"^=.+?=<span class="math-inline">\\n?", "", text, flags\=re\.MULTILINE\)
text \= re\.sub\(r"^\\s\*</span>", "", text, flags=re.MULTILINE)

        # Remove HTML comments
        text = re.sub(r"", "", text, flags=re.DOTALL)

        # Remove or simplify other common Wikitext elements
        text = re.sub(r"\[\[File:.*?\]\]", "", text)  # Remove file links
        text = re.sub(r"\[\[Image:.*?\]\]", "", text)  # Remove image links
        text = re.sub(r"&ndash;", "-", text)  # Replace en-dash
        text = re.sub(r"&mdash;", "--", text)  # Replace em-dash
        text = re.sub(r"&nbsp;", " ", text)  # Replace non breaking spaces

        # Basic handling of links (keeping the text, removing the link)
        text = re.sub(r"\[\[(.*?)\|(.*?)\]\]", r"\2", text)  # [[link|text]] -> text
        text = re.sub(r"\[\[(.*?)\]\]", r"\1", text)  # [[link]] -> link

        # Remove remaining brackets
        text = re.sub(r"[\[\]]", "", text)
        return text.strip()

    def __len__(self):
        if self.preload:
            return len(self.encoded_text) - self.seq_len
        else:
            # Approximate length - accurate enough for most training purposes.
            if self.use_character_encoding:
                approx_len = 0
                with open(self.text_file, "r", encoding="utf-8") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), ''):  #Read 1MB chunk at the time
                      cleaned_chunk = self.clean_wikitext(chunk)
                      approx_len += len(cleaned_chunk)
                return approx_len - self.seq_len

            else: # Tokenizer, no preload
                # Even more approximate, since tokenization changes length.
                approx_len = 0
                with open(self.text_file, "r", encoding="utf-8") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), ''):  #Read 1MB chunk at the time
                        cleaned_chunk = self.clean_wikitext(chunk)
                        approx_len += len(self.tokenizer.encode(cleaned_chunk))
                return approx_len- self.seq_len
    def __getitem__(self, idx):
      if self.preload:
        if self.use_character_encoding:
            input_sequence = self.encoded_text[idx: idx + self.seq_len]
            target_sequence = self.encoded_text[idx + 1: idx + self.seq_len + 1]
        else:
            input_sequence = self.encoded_text[idx: idx + self.seq_len]
            target_sequence = self.encoded_text[idx + 1: idx + self.seq_len + 1]
        return torch.tensor(input_sequence, dtype=torch.long), torch.tensor(target_sequence, dtype=torch.long)

      else:  # Not preloading
        if self.use_character_encoding:
          # Read only the necessary chunk from the file
          input_sequence = []
          target_sequence = []
          with open(self.text_file, 'r', encoding="utf-8") as f:
              f.seek(0) # Ensure we're at the beginning of the file.  Important!

              # Optimization: Read a larger chunk than just seq_len to minimize file reads
              chunk_size = self.seq_len * 10  # Adjust as needed
              current_pos = 0

              while current_pos <= idx:
                  chunk = f.read(chunk_size)
                  if not chunk:
                      break # End of file
                  cleaned_chunk = self.clean_wikitext(chunk)

                  start_index = max(0, idx-current_pos)
                  if len(cleaned_chunk) > start_index:
                      encoded_chunk = [self.char_to_idx.get(ch, 0) for ch in cleaned_chunk[start_index:]] #Use .get for unknown characters

                      needed = self.seq_len + 1  # +1 for the target
                      if len(input_sequence) < needed:
                          input_sequence.extend(encoded_chunk)
                  current_pos += len(chunk)


          input_sequence = input_sequence[:self.seq_len]
          target_sequence = input_sequence[1:self.seq_len+1] #Offset of 1
          #Pad if necessary
          padding_needed = self.seq_len - len(input_sequence)
          if padding_needed > 0:
            input_sequence.extend([0] * padding_needed)  # Pad with a valid index (e.g., 0)
            target_sequence.extend([0] * padding_needed)
          return torch.tensor(input_sequence,dtype=torch.long), torch.tensor(target_sequence,dtype=torch.long)

        else:  # Tokenizer, no preload -  CRUCIAL, corrected logic
            input_ids = []
            target_ids = []

            with open(self.text_file, "r", encoding="utf-8") as f:
                f.seek(0)  # Start from the beginning
                file_text = ""  # Accumulate cleaned text
                for chunk in iter(lambda: f.read(1024 * 1024), ""): #Read by chunks
                    file_text += self.clean_wikitext(chunk)
                    all_tokens = self.tokenizer.encode(file_text)

                    if len(all_tokens) > idx + self.seq_len:
                        input_ids = all_tokens[idx : idx + self.seq_len]
                        target_ids = all_tokens[idx + 1 : idx + self.seq_len + 1]
                        break #Exit when target and input are full

            # Padding (if necessary)
            padding_needed = self.seq_len - len(input_ids)
            if padding_needed > 0:
                input_ids.extend([self.pad_id] * padding_needed)
                target_ids.extend([self.pad_id] * padding_needed)

            return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)

    def get_vocab_size(self):
        if self.use_character_encoding:
            return len(self.chars)
        else:
            return self.vocab_size  # Use pre-calculated vocab size

    def decode(self, token_ids: List[int]) -> str:
        """Decodes a list of token IDs to a string."""
        if self.use_character_encoding:
            return "".join([self.idx_to_char.get(i, "") for i in token_ids])
        else:
            return self.tokenizer.decode(token_ids)

    def encode(self, text_string: str) -> List[int]:
        """Encodes a string to a list of token IDs"""
        if self.use_character_encoding:
            return [self.char_to_idx.get(ch, 0) for ch in text_string]
        else:
            return self.tokenizer.encode(text_string)



class WikitextDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_file: str,
        val_file: str,
        test_file: str,
        seq_len: int,
        batch_size: int,
        num_workers: int = 8,
        preload: bool = True,
        tokenizer_model_path: Optional[str] = None,
        use_character_encoding: bool = False,
    ):
        super().__init__()
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.preload = preload
        self.tokenizer_model_path = tokenizer_model_path
        self.use_character_encoding = use_character_encoding

    def setup(self, stage=None):
        # Assign train/val datasets
        if stage == "fit" or stage is None:
            self.train_dataset = WikitextDataset(
                self.train_file,
                self.seq_len,
                preload=self.preload,
                tokenizer_model_path=self.tokenizer_model_path,
                use_character_encoding=self.use_character_encoding,
            )
            self.val_dataset = WikitextDataset(
                self.val_file,
                self.seq_len,
                preload=self.preload,
                tokenizer_model_path=self.tokenizer_model_path,
                use_character_encoding=self.use_character_encoding,
            )

        # Assign test dataset
        if stage == "test" or stage is None:
            self.test_dataset = WikitextDataset(
                self.test_file,
                self.seq_len,
                preload=self.preload,
                tokenizer_model_path=self.tokenizer_model_path,
                use_character_encoding=self.use_character_encoding,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def get_vocab_size(self):
        return self.train_dataset.get_vocab_size()



def download_and_split_wikitext(output_dir="wikitext_data"):
    """Downloads and preprocesses a subset of Wikitext-2."""
    import requests
    import os
    import tarfile
    from io import BytesIO  # Import BytesIO

    url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(os.path.join(output_dir, "wikitext-2")):
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with BytesIO(response.content) as f, tarfile.open(fileobj=f, mode='r:gz') as tar:
            tar.extractall(path=output_dir)

    # Define train, validation, and test file paths
    train_file = os.path.join(output_dir, "wikitext-2", "wiki.train.tokens")
    val_file = os.path.join(output_dir, "wikitext-2", "wiki.valid.tokens")
    test_file = os.path.join(output_dir, "wikitext-2", "wiki.test.tokens")

    return train_file, val_file, test_file