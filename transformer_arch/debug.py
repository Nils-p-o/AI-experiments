# for isolating problems in models

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from .components import (
    transformer_block,
    input_embedding,
    mha,
)

# checked:
# 1. swiglu combined



class debug(nn.Module):
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
        super(debug, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        if d_ff is None:
            d_ff = 4 * d_model
        else:
            self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [LLaMa_block(d_model, nhead, d_ff, dropout) for _ in range(num_layers)]
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
        x = torch.softmax(x, -1)
        return x  # (batch_size, seq_len, vocab_size) logits


class LLaMa_block(transformer_block):
    def __init__(self, d_model, nhead, d_ff=None, dropout=0.1):
        super(LLaMa_block, self).__init__(d_model, nhead, d_ff, dropout)
        self.mha = GQA(d_model=d_model, nhead=nhead, dropout=dropout)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.ff = SwiGLU_feed_forward(d_model, d_ff, dropout)


class RoPE(nn.Module):
    def __init__(self, dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        # self.register_buffer(
        #     "inv_freqs", self.inv_freq
        # )  # Make it a buffer, not a parameter

        # Create a cache for the cos and sin values.  This improves efficiency
        # as we can reuse these values for different batches with same seq_len.
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, x, seq_dim=1):
        """Builds the cos/sin cache if it doesn't exist or if seq_len changes."""
        seq_len = x.shape[seq_dim]  # batch, seq_len, dim
        if (
            self.cos_cached is not None and seq_len <= self.cos_cached.shape[seq_dim]
        ):  # Use cached values if appropriate
            return

        # contiguous is very important for performance.
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # Outer product
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :x.size(-1)//2].to(device=x.device)  # [1,1,seq_len, dim]
        self.sin_cached = emb.sin()[None, None, :, :x.size(-1)//2].to(device=x.device) # quick fix??? TODO last dim 2 times bigger than i need
        return

    def forward(self, x, seq_dim=1):  # added seq_dim
        """
        Args:
            x: Input tensor of shape (batch, seq_len, ... , dim)

        Returns:
            Tensor with RoPE applied, same shape as x.
        """
        self._build_cache(x, seq_dim=seq_dim)
        cos = self.cos_cached[
            :, :, : x.shape[seq_dim]
        ]  # Correctly slice cos/sin caches
        sin = self.sin_cached[:, :, : x.shape[seq_dim]]

        x_rope, x_pass = x.chunk(2, dim=-1)  # Split x into two parts

        # Apply the rotation to the first part of x
        x_rope_rotated = (x_rope * cos) + (self.rotate_half(x_rope) * sin)

        return torch.cat((x_rope_rotated, x_pass), dim=-1)  # Concatenate and return

    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)  # negative x2, positive x1


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

        q = self.rotary_emb(q, seq_dim=2)
        k = self.rotary_emb(k, seq_dim=2)

        a = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply mask if provided
        if mask is not None:
            a = a.masked_fill(mask == 0, -1e9)

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
        q = self.rotary_emb(q, seq_dim=2)
        k = self.rotary_emb(k, seq_dim=2)

        # --- Reshape K and V for Grouped Attention ---
        # Repeat K and V for each head within the group.
        k = k.repeat_interleave(
            self.heads_per_group, dim=1
        )  # (batch_size, nhead, seq_len, head_dim)
        v = v.repeat_interleave(
            self.heads_per_group, dim=1
        )  # (batch_size, nhead, seq_len, head_dim)

        # --- Attention Calculation ---
        a = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            a = a.masked_fill(mask == 0, -1e9)

        a = torch.softmax(a, dim=-1)
        a = self.dropout(a)
        attn_output = torch.matmul(a, v)  # (batch_size, nhead, seq_len, head_dim)

        # --- Concatenate and Output Projection ---
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )
        output = self.dropout(self.o(attn_output))

        return output

class SwiGLU_feed_forward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # d_ff is now the "intermediate" dimension.  The SwiGLU layer
        # will project to d_ff, and then the output projection will go back to d_model.
        self.d_model = d_model
        self.linear_in = nn.Linear(d_model, d_ff)
        self.linear_gate = nn.Linear(d_model, d_ff)
        self.linear_out = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        u = self.linear_in(x)
        v = self.linear_gate(x)
        x = u * F.silu(v)
        x = self.linear_out(x)
        x = self.dropout(x)  # Apply dropout *after* the output projection
        return x
