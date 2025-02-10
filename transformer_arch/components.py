import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# decoder only model
class ClassicTransformer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        num_layers,
        d_ff=None,
        dropout=0.1,
        vocab_size=10000,
        seq_len=128,
    ):
        super(ClassicTransformer, self).__init__()
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


class feed_forward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(feed_forward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.l1 = nn.Linear(d_model, d_ff)  # (batch_size, seq_len, d_ff)
        self.l2 = nn.Linear(d_ff, d_model)  # (batch_size, seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.l2(F.gelu(self.l1(x))))


class transformer_block(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super(transformer_block, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_ff = d_ff
        self.mha = mha(d_model, nhead, dropout)
        self.ff = feed_forward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)


    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        mask = get_causal_mask(seq_len)
        mask = mask.repeat(batch_size, self.nhead, 1, 1)
        x = x + self.mha(self.norm1(x), mask)  # (batch_size, seq_len, d_model)
        x = x + self.ff(self.norm2(x))
        return x


class input_embedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(input_embedding, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)  # Register 'pe' as a buffer (not a parameter)

    def forward(self, x):
        # Add positional encoding to the input
        # x: (batch_size, seq_len, d_model)
        # pe: (1, max_len, d_model) -> (1, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class mha(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(mha, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Project and split x into multiple heads
        q = (
            self.q_linear(x)
            .view(batch_size, seq_len, self.nhead, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_linear(x)
            .view(batch_size, seq_len, self.nhead, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_linear(x)
            .view(batch_size, seq_len, self.nhead, self.head_dim)
            .transpose(1, 2)
        )  # (batch_size, nhead, seq_len, head_dim)

        # Calculate attention scores
        a = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply mask if provided
        if mask is not None:
            a = a.masked_fill(mask == 0, -1e9)

        # Calculate attention probabilities
        a = torch.softmax(a, dim=-1)
        a = self.dropout(a)

        # Apply attention to values
        attn_output = torch.matmul(a, v)

        # Concatenate heads and project
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )

        return self.dropout(self.o(attn_output))


def get_causal_mask(seq_len):  # ?????
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0).to(device="cuda" if torch.cuda.is_available() else "cpu")


class self_attention(nn.Module):
    def __init__(self, head_dim, model_dim, dropout=0.1):
        super(self_attention, self).__init__()
        self.head_dim = head_dim
        self.model_dim = model_dim
        self.k = torch.nn.Linear(model_dim, head_dim)
        self.v = torch.nn.Linear(model_dim, head_dim)
        self.q = torch.nn.Linear(model_dim, head_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        q = self.q(x)  # (batch_size, seq_len, head_dim)
        k = self.k(x)
        v = self.v(x)

        if mask is not None:
            a = torch.nn.Softmax(
                (
                    torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
                ).masked_fill(mask == 0, -1e9),
                -1,
            )  # (batch_size, seq_len, seq_len)
        else:
            a = torch.nn.Softmax(
                (torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)), -1
            )

        a = self.dropout(a)

        return torch.matmul(a, v)  # (batch_size, seq_len, head_dim)


class multi_head_attention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(multi_head_attention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        assert d_model % nhead == 0
        self.heads = nn.ModuleList(
            [self_attention(head_dim=self.d_head) for _ in range(nhead)]
        )
        self.o = torch.nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, is_causal=True):

        batch_size, seq_len, _ = x.size()

        if is_causal:
            mask = (
                torch.triu(torch.ones(seq_len, seq_len), 1)
                .unsqueeze(0)
                .repeat(batch_size, 1, 1)
            )
        else:
            mask = None

        x = torch.cat(
            [head(x, mask) for head in self.heads], -1
        )  # (batch_size, seq_len, d_model)

        return self.dropout(self.o(x))

