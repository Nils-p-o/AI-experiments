from transformer_arch.components import ClassicTransformer
from training.utils import (
    export_model
)
import torch

vocab_size = 1000
batch_size = 1
seq_len = 128

model = ClassicTransformer(
    d_model=512,
    nhead=8,
    num_layers=9,
    dropout=0.1,
    vocab_size=vocab_size,
    seq_len=seq_len,
)
dummy_input = torch.randn(batch_size, seq_len).long()

model.eval()

export_model(model, dummy_input, "GPT/GPT.onnx")