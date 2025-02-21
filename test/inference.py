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

# TODO modify after syncing changes
# TODO rewrite to work with current code


class ShakespeareDataModule_inference(pl.LightningDataModule):
    def __init__(self, train_file, val_file, test_file, seq_len, batch_size):
        super().__init__()
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.vocab = None  # Will be built during setup
        self.vocab_size = None

    def setup(self, stage=None):
        # This is a simplified version.  We just build the vocab.
        with open(self.train_file, 'r') as f:
            text = f.read()
        chars = sorted(list(set(text)))
        self.vocab = {ch: i for i, ch in enumerate(chars)}
        self.vocab_size = len(self.vocab)

    def get_vocab_size(self):
        if self.vocab_size is None:
            raise Exception("Must call setup() before get_vocab_size()")
        return self.vocab_size

    def text_to_tensor(self, text):
        # Convert text to a tensor of indices
        return torch.tensor([self.vocab[char] for char in text], dtype=torch.long)

    def tensor_to_text(self, tensor):
        # Convert a tensor of indices back to text
      return ''.join([list(self.vocab.keys())[list(self.vocab.values()).index(idx)] for idx in tensor.tolist()])

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
    elif architecture == "Diff":
        model = DiffTransformer(d_model, nhead, num_layers, d_model * d_ff_mult, dropout, vocab_size, seq_len, groups)
    elif architecture == "Dint":
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
  tensor = data_module.text_to_tensor(text)
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
            next_token = data_module.tensor_to_text(torch.tensor([next_token_id]))
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
        train_file="train.txt",  # Replace with your actual paths if needed
        val_file="val.txt",
        test_file="test.txt",
        seq_len=seq_len,  # Use the sequence length from your model
        batch_size=1,  # Batch size doesn't matter much for inference
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
    parser.add_argument("--architecture", type=str, required=True, help="Model architecture (Classic, LLaMa, nGPT, Diff, Dint).")
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