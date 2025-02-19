import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import List
import tiktoken  # Install: pip install tiktoken
import requests
import os


class ShakespeareDataset(Dataset):
    def __init__(
        self,
        text_file: str,
        seq_len: int,
        preload: bool = True,
        encoding_name: str = "cl100k_base",  # Default to GPT-4 tokenizer
        use_character_encoding: bool = True,  # Default to character encoding
    ):
        """
        Args:
            text_file: Path to the text file.
            seq_len: Length of the input sequences.
            preload: Whether to load entire file into memory
            encoding_name: Tiktoken encoding name (e.g. "cl100k_base", "p50k_base").
            use_character_encoding: Whether to use character encoding.
        """
        self.seq_len = seq_len
        self.preload = preload
        self.use_character_encoding = use_character_encoding
        self.tokenizer = None

        if not use_character_encoding:
            try:
                self.tokenizer = tiktoken.get_encoding(encoding_name)
            except KeyError:
                print(f"Encoding '{encoding_name}' not found.  Falling back to 'cl100k_base'.")
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            self.vocab_size = self.tokenizer.n_vocab  # Get vocab size
            self.pad_id = 0  # Use 0 for padding.  Verify this is appropriate for your encoding!

        if preload:
            with open(text_file, "r", encoding="utf-8") as f: # Added encoding
                self.text = f.read()

            if use_character_encoding:
                self.chars = sorted(list(set(self.text)))
                self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
                self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
                self.encoded_text = [self.char_to_idx[ch] for ch in self.text]
            else:
                self.encoded_text = self.tokenizer.encode(self.text)

        else: # No preload
            self.text_file = text_file
            if use_character_encoding:
                self.chars = set()
                self.char_to_idx = {}
                self.idx_to_char = {}
                self._build_vocab()
                self.encoded_text = None # Not loaded yet
            else:
                self.encoded_text = None #Not loaded yet

    def _build_vocab(self):
      """Builds character vocab (for no preload, character encoding)."""
      temp_chars = set()
      with open(self.text_file, 'r', encoding="utf-8") as f:
          for chunk in iter(lambda: f.read(1024*1024), ''):
              temp_chars.update(chunk)
          self.chars = sorted(list(temp_chars))
          self.char_to_idx = {ch:i for i, ch in enumerate(self.chars)}
          self.idx_to_char = {i:ch for i, ch in enumerate(self.chars)}
    def __len__(self):
        if self.preload:
          return len(self.encoded_text) - self.seq_len
        else:
            if self.use_character_encoding:
              approx_len = 0
              with open(self.text_file, 'r', encoding="utf-8") as f:
                  for chunk in iter(lambda: f.read(1024*1024), ''):
                      approx_len += len(chunk)
              return approx_len - self.seq_len
            else:
              approx_len = 0
              with open(self.text_file, "r", encoding="utf-8") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), ""):
                    approx_len += len(self.tokenizer.encode(chunk))
              return approx_len - self.seq_len

    def __getitem__(self, idx):
        if self.preload:
            if self.use_character_encoding:
                input_sequence = self.encoded_text[idx : idx + self.seq_len]
                target_sequence = self.encoded_text[idx + 1 : idx + self.seq_len + 1]
            else:
                input_sequence = self.encoded_text[idx : idx + self.seq_len]
                target_sequence = self.encoded_text[idx + 1 : idx + self.seq_len + 1]

            return torch.tensor(input_sequence, dtype=torch.long), torch.tensor(target_sequence, dtype=torch.long)

        else: # No preload
            if self.use_character_encoding:
                input_sequence = []
                target_sequence = []
                with open(self.text_file, "r", encoding="utf-8") as f:
                    f.seek(0)  # IMPORTANT: Reset file pointer
                    current_pos = 0
                    chunk_size = self.seq_len * 10  # Read larger chunks

                    while current_pos <= idx:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break  # End of file
                        start_index = max(0, idx - current_pos)

                        if len(chunk) > start_index:
                            encoded_chunk = [self.char_to_idx.get(ch, 0) for ch in chunk[start_index:]] #Handle missing chars
                            needed = self.seq_len + 1 #+1 for the target sequence
                            if len(input_sequence) < needed:
                                input_sequence.extend(encoded_chunk)
                        current_pos += len(chunk)

                input_sequence = input_sequence[:self.seq_len]
                target_sequence = input_sequence[1 : self.seq_len+1] #Offset by one
                padding_needed = self.seq_len - len(input_sequence)

                if padding_needed > 0:
                  input_sequence.extend([0] * padding_needed) # Pad
                  target_sequence.extend([0] * padding_needed) # Pad
                return torch.tensor(input_sequence,dtype=torch.long), torch.tensor(target_sequence,dtype=torch.long)

            else:  # tiktoken, no preload.  Very similar to Wikitext.
                input_ids = []
                target_ids = []

                with open(self.text_file, "r", encoding="utf-8") as f:
                    f.seek(0)  # Start from the beginning
                    file_text = ""
                    for chunk in iter(lambda: f.read(1024 * 1024), ""):
                        file_text += chunk # No cleaning needed here
                        all_tokens = self.tokenizer.encode(file_text)

                        if len(all_tokens) > idx + self.seq_len:
                            input_ids = all_tokens[idx : idx + self.seq_len]
                            target_ids = all_tokens[idx + 1 : idx + self.seq_len + 1]
                            break

                padding_needed = self.seq_len - len(input_ids)
                if padding_needed > 0:
                    input_ids.extend([self.pad_id] * padding_needed)
                    target_ids.extend([self.pad_id] * padding_needed)
                return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)

    def get_vocab_size(self):
        if self.use_character_encoding:
            return len(self.chars)
        else:
            return self.vocab_size

    def decode(self, token_ids: List[int]) -> str:
        if self.use_character_encoding:
            return "".join([self.idx_to_char.get(i, "") for i in token_ids])  # Handle potential missing
        else:
            return self.tokenizer.decode(token_ids)

    def encode(self, text_string: str) -> List[int]:
        if self.use_character_encoding:
            return [self.char_to_idx.get(ch, 0) for ch in text_string]  # Handle missing chars
        else:
            return self.tokenizer.encode(text_string)



class ShakespeareDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_file: str,
        val_file: str,
        test_file: str,
        seq_len: int,
        batch_size: int,
        num_workers: int = 8,
        preload: bool = True,
        encoding_name: str = "cl100k_base",
        use_character_encoding: bool = True,
    ):
        super().__init__()
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.preload = preload
        self.encoding_name = encoding_name
        self.use_character_encoding = use_character_encoding

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = ShakespeareDataset(
                self.train_file,
                self.seq_len,
                preload=self.preload,
                encoding_name=self.encoding_name,
                use_character_encoding=self.use_character_encoding,
            )
            self.val_dataset = ShakespeareDataset(
                self.val_file,
                self.seq_len,
                preload=self.preload,
                encoding_name=self.encoding_name,
                use_character_encoding=self.use_character_encoding,
            )
        if stage == "test" or stage is None:
            self.test_dataset = ShakespeareDataset(
                self.test_file,
                self.seq_len,
                preload=self.preload,
                encoding_name=self.encoding_name,
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



def download_and_split_shakespeare(input_file_path="input.txt"):
    import requests
    import os

    if not os.path.exists(input_file_path):
        # Download the tiny shakespeare dataset
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        with open(input_file_path, "w", encoding="utf-8") as f:  # Added encoding
            f.write(response.text)

    with open(input_file_path, "r", encoding="utf-8") as f: # Added encoding
        text = f.read()

    # Split the dataset (simple split for demonstration)
    n = len(text)
    train_data = text[: int(n * 0.8)]
    val_data = text[int(n * 0.8) : int(n * 0.9)]
    test_data = text[int(n * 0.9) :]

    # Write to files 
    with open("tiny_shakespeare/train.txt", "w", encoding="utf-8") as f:  # Added encoding
        f.write(train_data)
    with open("tiny_shakespeare/val.txt", "w", encoding="utf-8") as f:    # Added encoding
        f.write(val_data)
    with open("tiny_shakespeare/test.txt", "w", encoding="utf-8") as f:   # Added encoding
        f.write(test_data)

    return "tiny_shakespeare/train.txt", "tiny_shakespeare/val.txt", "tiny_shakespeare/test.txt"




if __name__ == "__main__":
    train_file, val_file, test_file = download_and_split_shakespeare()

    # --- Character Encoding Examples ---
    print("-" * 20, "Character Encoding, Preload", "-" * 20)
    char_dm_preload = ShakespeareDataModule(
        train_file,
        val_file,
        test_file,
        seq_len=128,
        batch_size=32,
        use_character_encoding=True,
        preload=True,
    )
    char_dm_preload.setup()
    print("Vocab size:", char_dm_preload.get_vocab_size())
    for batch in char_dm_preload.train_dataloader():
        inputs, targets = batch
        print("Input shape:", inputs.shape)
        print("Decoded input:", char_dm_preload.train_dataset.decode(inputs[0].tolist()))
        break

    print("-" * 20, "Character Encoding, No Preload", "-" * 20)
    char_dm_no_preload = ShakespeareDataModule(
        train_file,
        val_file,
        test_file,
        seq_len=128,
        batch_size=32,
        use_character_encoding=True,
        preload=False,
    )
    char_dm_no_preload.setup()
    print("Vocab size:", char_dm_no_preload.get_vocab_size())
    for batch in char_dm_no_preload.train_dataloader():
        inputs, targets = batch
        print("Input shape:", inputs.shape)
        print("Decoded input:", char_dm_no_preload.train_dataset.decode(inputs[0].tolist()))
        break

    # --- Tiktoken Examples ---
    print("-" * 20, "Tiktoken (cl100k_base), Preload", "-" * 20)
    tiktoken_dm_preload = ShakespeareDataModule(
        train_file,
        val_file,
        test_file,
        seq_len=256,
        batch_size=64,
        encoding_name="cl100k_base",
        preload=True,
        use_character_encoding=False,  # Use tiktoken
    )
    tiktoken_dm_preload.setup()
    print("Vocab size:", tiktoken_dm_preload.get_vocab_size())
    for batch in tiktoken_dm_preload.train_dataloader():
        inputs, targets = batch
        print("Input shape:", inputs.shape)
        print("Decoded input:", tiktoken_dm_preload.train_dataset.decode(inputs[0].tolist()))
        break

    print("-" * 20, "Tiktoken (cl100k_base), No Preload", "-" * 20)
    tiktoken_dm_no_preload = ShakespeareDataModule(
        train_file,
        val_file,
        test_file,
        seq_len=256,
        batch_size=64,
        encoding_name="cl100k_base",
        preload=False,
        use_character_encoding=False,  # Use tiktoken
    )
    tiktoken_dm_no_preload.setup()
    print("Vocab size:", tiktoken_dm_no_preload.get_vocab_size())
    for batch in tiktoken_dm_no_preload.train_dataloader():
        inputs, targets = batch
        print("Input shape:", inputs.shape)
        print("Decoded input:", tiktoken_dm_no_preload.train_dataset.decode(inputs[0].tolist()))
        print("Decoded target:", tiktoken_dm_no_preload.train_dataset.decode(targets[0].tolist()))
        break