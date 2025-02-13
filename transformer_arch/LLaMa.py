import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from .components import (
    transformer_block,
    input_embedding,
    mha,
)

# added RoPE, GQA, swiGLU


class LLaMa(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        num_layers,
        d_ff=None,
        dropout=0.1,
        vocab_size=10000,
        seq_len=128,
        groups=4,
    ):
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
                LLaMa_block(d_model, nhead, d_ff, dropout, groups)
                for _ in range(num_layers)
            ]
        )
        self.input_embedding = input_embedding(d_model, vocab_size)
        self.norm = nn.RMSNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.input_embedding(x)  # (batch_size, seq_len, d_model)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)  # (batch_size, seq_len, d_model)
        x = self.out(x)
        # x = torch.softmax(x, -1) inference only, dumbass!!!
        return x  # (batch_size, seq_len, vocab_size) logits


class LLaMa_block(transformer_block):
    def __init__(self, d_model, nhead, d_ff=None, dropout=0.1, groups=4):
        super(LLaMa_block, self).__init__(d_model, nhead, d_ff, dropout)
        self.mha = GQA(d_model=d_model, nhead=nhead, dropout=dropout, groups=groups)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.ff = SwiGLU_feed_forward(d_model, d_ff, dropout)


class RoPE(nn.Module):

    def __init__(self, d: int, base: int = 10_000):

        super().__init__()
        self.base = base
        self.d = d
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, x: torch.Tensor):

        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
            return

        seq_len = x.shape[0]

        theta = 1.0 / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(
            x.device
        )  # THETA = 10,000^(-2*i/d) or 1/10,000^(2i/d)

        seq_idx = (
            torch.arange(seq_len, device=x.device).float().to(x.device)
        )  # Position Index -> [0,1,2...seq-1]

        idx_theta = torch.einsum(
            "n,d->nd", seq_idx, theta
        )  # Calculates m*(THETA) = [ [0, 0...], [THETA_1, THETA_2...THETA_d/2], ... [seq-1*(THETA_1), seq-1*(THETA_2)...] ]

        idx_theta2 = torch.cat(
            [idx_theta, idx_theta], dim=1
        )  # [THETA_1, THETA_2...THETA_d/2] -> [THETA_1, THETA_2...THETA_d]

        self.cos_cached = idx_theta2.cos()[
            :, None, None, :
        ]  # Cache [cosTHETA_1, cosTHETA_2...cosTHETA_d]
        self.sin_cached = idx_theta2.sin()[
            :, None, None, :
        ]  # cache [sinTHETA_1, sinTHETA_2...sinTHETA_d]

    def _neg_half(self, x: torch.Tensor):

        d_2 = self.d // 2  #

        return torch.cat(
            [-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1
        )  # [x_1, x_2,...x_d] -> [-x_d/2, ... -x_d, x_1, ... x_d/2]

    def forward(self, x: torch.Tensor):

        self._build_cache(x)

        neg_half_x = self._neg_half(x)

        x_rope = (x * self.cos_cached[: x.shape[0]]) + (
            neg_half_x * self.sin_cached[: x.shape[0]]
        )  # [x_1*cosTHETA_1 - x_d/2*sinTHETA_d/2, ....]

        return x_rope


class GQA(nn.Module):
    def __init__(self, d_model, nhead, groups=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.groups = groups
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model
        assert (
            nhead % groups == 0
        ), "Number of heads must be divisible by number of groups"
        self.heads_per_group = nhead // groups

        # Separate Q projections for each head
        self.q_linear = nn.Linear(d_model, d_model)
        # Shared K and V projections for each group
        self.k_linear = nn.Linear(d_model, self.head_dim * self.groups)
        self.v_linear = nn.Linear(d_model, self.head_dim * self.groups)

        self.o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.rotary_emb = RoPE(self.head_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # --- Q, K, V Projections ---
        q = (
            self.q_linear(x)
            .view(batch_size, seq_len, self.nhead, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_linear(x)
            .view(batch_size, seq_len, self.groups, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_linear(x)
            .view(batch_size, seq_len, self.groups, self.head_dim)
            .transpose(1, 2)
        )

        # --- RoPE ---
        q = self.rotary_emb(q)
        k = self.rotary_emb(k)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=0.1, enable_gqa=True
        )

        # --- Concatenate and Output Projection ---
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )
        output = self.dropout(self.o(attn_output))

        return output


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit (SwiGLU) activation function.

    This implements the SwiGLU activation as described in:
      "GLU Variants Improve Transformer"
      https://arxiv.org/abs/2002.05202
    and used in LLaMA models.

    In summary, it's a variant of Gated Linear Units (GLUs) that uses
    the Swish (SiLU) activation function:
        SwiGLU(x) = (x * W + b) ⊗ SiLU(x * V + c)
    where:
        x is the input
        W, V are learnable weight matrices
        b, c are learnable bias vectors
        ⊗ is the element-wise product
        SiLU(x) = x * sigmoid(x)  (also known as Swish-1)
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear_gate = nn.Linear(in_features, out_features, bias=bias)
        self.linear_out = nn.Linear(in_features, out_features, bias=bias)

        # Initialize weights (optional, but generally a good idea)
        self._init_weights()

    def _init_weights(self):
        # You can use different initialization schemes. This is just an example.
        nn.init.xavier_uniform_(self.linear_gate.weight)
        nn.init.xavier_uniform_(self.linear_out.weight)
        if self.linear_gate.bias is not None:
            nn.init.zeros_(self.linear_gate.bias)
        if self.linear_out.bias is not None:
            nn.init.zeros_(self.linear_out.bias)

    def forward(self, x):
        # (x * W + b)
        out = self.linear_out(x)
        # SiLU(x * V + c)
        gate = F.silu(self.linear_gate(x))  # Use SiLU (Swish) activation
        # Element-wise product: (xW + b) * SiLU(xV + c)
        return out * gate


class SwiGLU_feed_forward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # d_ff is now the "intermediate" dimension.  The SwiGLU layer
        # will project to d_ff, and then the output projection will go back to d_model.
        # self.swiglu = SwiGLU(d_model, d_ff)
        self.linear_in_gate = nn.Linear(d_model, d_ff*2)
        self.linear_out = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x):
        # x = self.swiglu(x) #

        u, v = self.linear_in_gate(x).chunk(
            2, dim=-1
        )  # * self.d_model ** 0.5)  # Apply SwiGLU with scaling
        x = u * F.silu(v)
        x = self.linear_out(x)
        x = self.dropout(x)  # Apply dropout *after* the output projection
        return x


class RoPE_mha(nn.Module):
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

        self.rotary_emb = RoPE(self.head_dim)

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

        q = self.rotary_emb(q, seq_dim=-2)
        k = self.rotary_emb(k, seq_dim=-2)

        a = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply mask if provided
        if mask is not None:
            a = a.masked_fill(mask == 1, -1e9)

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
