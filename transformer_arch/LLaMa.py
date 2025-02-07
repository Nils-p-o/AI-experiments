import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import pytorch_lightning as pl
from .components import ClassicTransformer, transformer_block, input_embedding, PositionalEncoding, feed_forward, mha

# Add RoPE and other modifications to base transformer
# establish baseline


class LLaMa(nn.Module):
    def __init__(self, d_model, nhead, num_layers, d_ff=None, dropout=0.1, vocab_size=10000, seq_len=128):
        super(LLaMa, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        if d_ff is None:
            d_ff = 4 * d_model
        else:
            self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                transformer_block(d_model, nhead, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )
        self.input_embedding = input_embedding(d_model, vocab_size)
        self.positional_encoding = PositionalEncoding(d_model, dropout, seq_len)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.input_embedding(x)  # (batch_size, seq_len, d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)  # (batch_size, seq_len, d_model)
        x = self.out(x)
        x = torch.softmax(x, -1)
        return x  # (batch_size, seq_len, vocab_size) logits
