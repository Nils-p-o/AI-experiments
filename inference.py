import torch
import pytorch_lightning as pl  # We need this for the dummy classes
import numpy as np
from torch.nn import functional as F
import argparse # added argparse for easy use
from transformer_arch.components import ClassicTransformer
from transformer_arch.LLaMa import LLaMa
from transformer_arch.nGPT import nGPT
from transformer_arch.DIFF import DiffTransformer
from transformer_arch.DINT import DintTransformer

import tiktoken
from typing import List
import re
from torch.utils.data import Dataset


class WikitextDataset_inference(Dataset):
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



class TransformerExperiment(pl.LightningModule):
    #  Dummy version, just needs to exist
    def __init__(self, model, learning_rate, batch_size, vocab_size, warmup_steps, t_0, t_mult, lr_mult, cce_fn):
        super().__init__()
        self.model = model

# --- 2. Load the Model ---

def load_model(model_path, architecture, d_model, nhead, num_layers, dropout, vocab_size, seq_len, d_ff_mult, groups):
    # Instantiate the correct model class based on the architecture string
    if architecture == "Classic":
        model = ClassicTransformer(d_model, nhead, num_layers, dropout, vocab_size, seq_len)
    elif architecture == "LLaMa":
        model = LLaMa(d_model, nhead, num_layers, d_model * d_ff_mult, dropout, vocab_size, seq_len, groups)
    elif architecture == "nGPT":
        model = nGPT(d_model, nhead, num_layers, d_model * d_ff_mult, dropout, vocab_size, seq_len)
    elif architecture == "DIFF":
        model = DiffTransformer(d_model, nhead, num_layers, d_model * d_ff_mult, dropout, vocab_size, seq_len, groups)
    elif architecture == "DINT":
        model = DintTransformer(d_model, nhead, num_layers, d_model * d_ff_mult, dropout, vocab_size, seq_len, groups)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    # Load the state dict
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=False).state_dict())  # Load to CPU; add .to('cuda') if needed
    model.eval()  # Set to evaluation mode
    return model


# --- 3. Input Processing ---

def prepare_input(text, data_module):
  """Converts text to a model-ready tensor."""
  tensor = torch.tensor(data_module.encode(text)).to('cuda' if torch.cuda.is_available() else 'cpu')
  # Add batch dimension (batch_size=1)
  tensor = tensor.unsqueeze(0)
  return tensor

# --- 4. Inference and Output ---
def generate_text(model, data_module, start_text, max_length=200, temperature=1.0):
    """Generates text from the model."""

    input_tensor = prepare_input(start_text, data_module)
    generated_text = start_text

    with torch.no_grad():  # Disable gradient calculation
        for _ in range(max_length):
            output = model(input_tensor)
            # Get the logits for the last token in the sequence
            last_token_logits = output[0, -1, :] / temperature # added temperature

            # Apply softmax to get probabilities
            probabilities = F.softmax(last_token_logits, dim=-1)
            # Sample from the probabilities
            next_token_id = torch.multinomial(probabilities, num_samples=1).item()
            # Convert the token ID back to text
            next_token = data_module.decode([next_token_id])
            generated_text += next_token

            if len(generated_text) >= data_module.seq_len:
              # Prepare the next input (shift by one)
              input_tensor = prepare_input(generated_text[-data_module.seq_len:], data_module)
            else:
              input_tensor = prepare_input(generated_text, data_module) # if generated text is still short, dont crop it.

    return generated_text
# --- Main Function ---
def main(args):
    model_path = args.model_path
    start_text = args.prompt
    max_length = args.max_length
    d_model = args.d_model
    nhead = args.nhead
    num_layers = args.num_layers
    dropout = args.dropout
    seq_len = args.seq_len
    d_ff_mult = args.d_ff_mult
    groups = args.groups
    temperature = args.temperature

    architecture = model_path.split("/")[1].split("_")[0]

    # Create a dummy data module to get the vocab
    data_module = WikitextDataset_inference(
        text_file="shakespeare.txt",
        use_character_encoding=False,
        seq_len=seq_len,  # Use the sequence length from your model
        preload=False,
        encoding_name="r50k_base",
    )
    vocab_size = data_module.get_vocab_size()

    # Load the model
    model = load_model(model_path, architecture, d_model, nhead, num_layers, dropout, vocab_size, seq_len, d_ff_mult, groups)

    # Check for CUDA availability and move model to GPU if available
    if torch.cuda.is_available():
        model = model.to('cuda')
        # print("Model moved to CUDA.")

    # Generate text
    generated_text = generate_text(model, data_module, start_text, max_length, temperature)
    print(generated_text)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for Transformer model.")
    parser.add_argument("--model_path", type=str, help="Path to the saved .pth model file.", default="models/nGPT_wikitext2.pth")
    parser.add_argument("--prompt", type=str, default="The sharks ate", help="Starting text for generation.")
    parser.add_argument("--max_length", type=int, default=200, help="Maximum length of generated text.")
    # parser.add_argument("--architecture", type=str, required=True, help="Model architecture (Classic, LLaMa, nGPT, Diff, Dint).")
    parser.add_argument("--d_model", type=int, help="Embedding dimension.", default=512)
    parser.add_argument("--nhead", type=int, help="Number of attention heads.", default=8)
    parser.add_argument("--num_layers", type=int, help="Number of layers.", default=6)
    parser.add_argument("--dropout", type=float, help="Dropout probability.", default=0.1)
    parser.add_argument("--seq_len", type=int, help="Sequence length.", default=128)
    parser.add_argument("--d_ff_mult", type=int, default=4, help="Multiplier for d_ff (if applicable).")
    parser.add_argument("--groups", type=int, default=8, help="Number of groups for GQA (if applicable).")
    parser.add_argument("--temperature", type=float, default=1.2, help="Temperature for sampling.")


    args = parser.parse_args()
    main(args)