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

# TODO modify after syncing changes
# TODO rewrite to work with current code


class ShakespeareDataModule_inference(pl.LightningDataModule):
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
        model = nGPT(d_model, nhead, num_layers, dropout, vocab_size, seq_len)
    elif architecture == "DIFF":
        model = DiffTransformer(d_model, nhead, num_layers, d_model * d_ff_mult, dropout, vocab_size, seq_len, groups)
    elif architecture == "DINT":
        model = DintTransformer(d_model, nhead, num_layers, d_model * d_ff_mult, dropout, vocab_size, seq_len, groups)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    # Load the state dict
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Load to CPU; add .to('cuda') if needed
    model.eval()  # Set to evaluation mode
    return model


# --- 3. Input Processing ---

def prepare_input(text, data_module):
  """Converts text to a model-ready tensor."""
  tensor = data_module.encode(text)
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
            next_token = data_module.decode(torch.tensor([next_token_id]))
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
    architecture = args.architecture
    d_model = args.d_model
    nhead = args.nhead
    num_layers = args.num_layers
    dropout = args.dropout
    seq_len = args.seq_len
    d_ff_mult = args.d_ff_mult
    groups = args.groups
    temperature = args.temperature

    # Create a dummy data module to get the vocab
    data_module = ShakespeareDataModule_inference(
        text_file="shakespeare.txt",
        use_character_encoding=False,
        seq_len=seq_len,  # Use the sequence length from your model
        preload=False,
        encoding_name="r50k_base",
    )
    data_module.setup() #Very important, creates vocab
    vocab_size = data_module.get_vocab_size()

    # Load the model
    model = load_model(model_path, architecture, d_model, nhead, num_layers, dropout, vocab_size, seq_len, d_ff_mult, groups)

    # Check for CUDA availability and move model to GPU if available
    if torch.cuda.is_available():
        model = model.to('cuda')
        print("Model moved to CUDA.")

    # Generate text
    generated_text = generate_text(model, data_module, start_text, max_length, temperature)
    print(generated_text)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for Transformer model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved .pth model file.")
    parser.add_argument("--prompt", type=str, default="The ", help="Starting text for generation.")
    parser.add_argument("--max_length", type=int, default=200, help="Maximum length of generated text.")
    # parser.add_argument("--architecture", type=str, required=True, help="Model architecture (Classic, LLaMa, nGPT, Diff, Dint).")
    parser.add_argument("--d_model", type=int, required=True, help="Embedding dimension.")
    parser.add_argument("--nhead", type=int, required=True, help="Number of attention heads.")
    parser.add_argument("--num_layers", type=int, required=True, help="Number of layers.")
    parser.add_argument("--dropout", type=float, required=True, help="Dropout probability.")
    parser.add_argument("--seq_len", type=int, required=True, help="Sequence length.")
    parser.add_argument("--d_ff_mult", type=int, default=4, help="Multiplier for d_ff (if applicable).")
    parser.add_argument("--groups", type=int, default=4, help="Number of groups for GQA (if applicable).")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling.")


    args = parser.parse_args()
    main(args)