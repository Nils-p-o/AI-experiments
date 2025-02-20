from torch.utils.data import DataLoader, Dataset
import torch
import pytorch_lightning as pl

import re
from typing import List
from datasets import load_dataset
import os
import tiktoken


class WikitextDataset(Dataset):
    def __init__(
        self,
        text_file: str,
        seq_len: int,
        preload: bool = True,
        encoding_name: str = "cl100k_base",  # Default to GPT-4 tokenizer
        use_character_encoding: bool = False,
    ):
        """
        Args:
            text_file: Path to the wikitext file.
            seq_len: Length of the input sequences.
            preload: Whether to load the entire file into memory.
            encoding_name: Name of the tiktoken encoding to use (e.g., "cl100k_base", "p50k_base", "r50k_base").
                           See https://github.com/openai/tiktoken for a list.
            use_character_encoding: Use character-level encoding instead of the tokenizer.
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
            self.vocab_size = self.tokenizer.n_vocab  # Get vocab size directly
            #Tiktoken does not have pad, bos, or eos tokens.
            self.pad_id = 0  # Use 0 as a default padding token (ensure it's in your vocab)

        if preload:
            with open(text_file, "r", encoding="utf-8") as f:
                self.text = self.clean_wikitext(f.read())
            if use_character_encoding:
                self.chars = sorted(list(set(self.text)))
                self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
                self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
                self.encoded_text = [self.char_to_idx[ch] for ch in self.text]
            else:
                self.encoded_text = self.tokenizer.encode(self.text)

        else:  # not preload
            self.text_file = text_file
            if use_character_encoding:
                self.chars = set()
                self.char_to_idx = {}
                self.idx_to_char = {}
                self._build_vocab()
                self.encoded_text = None
            else:
                self.encoded_text = None

    def _build_vocab(self):
        """Builds vocab for char encoding (no preload)."""
        temp_chars = set()
        with open(self.text_file, "r", encoding="utf-8") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), ""):
                cleaned_chunk = self.clean_wikitext(chunk)
                temp_chars.update(cleaned_chunk)
        self.chars = sorted(list(temp_chars))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def clean_wikitext(self, text: str) -> str:
        """Cleans wikitext."""
        # Remove lines starting with = (headers) and empty lines
        text = re.sub(r"^=.+?=$\n?", "", text, flags=re.MULTILINE)  # Corrected regex
        text = re.sub(r"^\s*$", "", text, flags=re.MULTILINE)  # Corrected regex

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
            if self.use_character_encoding:
                approx_len = 0
                with open(self.text_file, "r", encoding="utf-8") as f:
                    for _ in iter(lambda: f.read(1024*1024), ''):
                        approx_len+= len(self.clean_wikitext(_))
                return approx_len - self.seq_len
            else:
                approx_len = 0
                with open(self.text_file, "r", encoding="utf-8") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), ""):
                        approx_len += len(self.tokenizer.encode(self.clean_wikitext(chunk)))
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

        else:  # No preload
            if self.use_character_encoding:
                input_sequence = []
                target_sequence = []
                with open(self.text_file, "r", encoding="utf-8") as f:
                    f.seek(0)
                    current_pos = 0
                    chunk_size = self.seq_len * 10

                    while current_pos <= idx:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        cleaned_chunk = self.clean_wikitext(chunk)
                        start_index = max(0, idx - current_pos)

                        if len(cleaned_chunk) > start_index:
                            encoded_chunk = [self.char_to_idx.get(ch, 0) for ch in cleaned_chunk[start_index:]]
                            needed = self.seq_len+1
                            if len(input_sequence) < needed:
                                input_sequence.extend(encoded_chunk)
                        current_pos += len(chunk)

                input_sequence = input_sequence[:self.seq_len]
                target_sequence = input_sequence[1:self.seq_len + 1]
                padding_needed = self.seq_len - len(input_sequence)

                if padding_needed > 0:
                    input_sequence.extend([0] * padding_needed)
                    target_sequence.extend([0] * padding_needed)
                return torch.tensor(input_sequence, dtype=torch.long), torch.tensor(target_sequence, dtype=torch.long)

            else: #Tiktoken, no preload
                input_ids = []
                target_ids = []
                with open(self.text_file, "r", encoding="utf-8") as f:
                    f.seek(0)
                    file_text = ""
                    for chunk in iter(lambda: f.read(1024 * 1024), ""):
                        file_text += self.clean_wikitext(chunk)
                        all_tokens = self.tokenizer.encode(file_text)
                        if len(all_tokens) > idx + self.seq_len:
                            input_ids = all_tokens[idx : idx + self.seq_len]
                            target_ids = all_tokens[idx+1 : idx + self.seq_len + 1]
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
            return "".join([self.idx_to_char.get(i, "") for i in token_ids])
        else:
            return self.tokenizer.decode(token_ids)

    def encode(self, text_string: str) -> List[int]:
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
        encoding_name: str = "cl100k_base",
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
        self.encoding_name = encoding_name
        self.use_character_encoding = use_character_encoding

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = WikitextDataset(
                self.train_file,
                self.seq_len,
                preload=self.preload,
                encoding_name=self.encoding_name,
                use_character_encoding=self.use_character_encoding,
            )
            self.val_dataset = WikitextDataset(
                self.val_file,
                self.seq_len,
                preload=self.preload,
                encoding_name=self.encoding_name,
                use_character_encoding=self.use_character_encoding,
            )
        if stage == "test" or stage is None:
            self.test_dataset = WikitextDataset(
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



def download_and_split_wikitext(output_dir="wikitext_data"): 
    """Downloads and preprocesses Wikitext-2 using Hugging Face Datasets."""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

        # Use Hugging Face Datasets
        dataset = load_dataset("wikitext", "wikitext-2-v1")

        train_file = os.path.join(output_dir, "wiki.train.tokens")
        val_file = os.path.join(output_dir, "wiki.valid.tokens")
        test_file = os.path.join(output_dir, "wiki.test.tokens")

        with open(train_file, "w", encoding="utf-8") as f:
            f.write("\n".join(dataset["train"]["text"]))
        with open(val_file, "w", encoding="utf-8") as f:
            f.write("\n".join(dataset["validation"]["text"]))
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("\n".join(dataset["test"]["text"]))

    else:
        train_file = os.path.join(output_dir, "wiki.train.tokens")
        val_file = os.path.join(output_dir, "wiki.valid.tokens")
        test_file = os.path.join(output_dir, "wiki.test.tokens")

    return train_file, val_file, test_file



if __name__ == "__main__":
    train_file, val_file, test_file = download_and_split_wikitext()

    # --- Character Encoding Examples ---
    print("-" * 20, "Character Encoding, Preload", "-" * 20)
    char_dm_preload = WikitextDataModule(
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
    char_dm_no_preload = WikitextDataModule(
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
    tiktoken_dm_preload = WikitextDataModule(
        train_file,
        val_file,
        test_file,
        seq_len=256,  # Increased seq_len for BPE
        batch_size=64, # And batch size
        encoding_name="cl100k_base",  # GPT-4 tokenizer
        preload=True,
    )
    tiktoken_dm_preload.setup()
    print("Vocab size:", tiktoken_dm_preload.get_vocab_size())
    for batch in tiktoken_dm_preload.train_dataloader():
        inputs, targets = batch
        print("Input shape:", inputs.shape)
        print("Decoded input:", tiktoken_dm_preload.train_dataset.decode(inputs[0].tolist()))
        break

    print("-" * 20, "Tiktoken (cl100k_base), No Preload", "-" * 20)
    tiktoken_dm_no_preload = WikitextDataModule(
        train_file,
        val_file,
        test_file,
        seq_len=256,  # Increased seq_len for BPE
        batch_size=64, # Increased batch size
        encoding_name="cl100k_base",
        preload=False,
    )
    tiktoken_dm_no_preload.setup()
    print("Vocab size:", tiktoken_dm_no_preload.get_vocab_size())
    for batch in tiktoken_dm_no_preload.train_dataloader():
        inputs, targets = batch
        print("Input shape:", inputs.shape)
        print("Decoded input:", tiktoken_dm_no_preload.train_dataset.decode(inputs[0].tolist()))
        print("Decoded target:", tiktoken_dm_no_preload.train_dataset.decode(targets[0].tolist()))
        break
