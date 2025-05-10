import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from .components import (
    input_embedding,
    mha,
    get_causal_mask,
)

# TODO rewrite input embedding and get causal mask (may be slightly faster)
# there is no need for the abstraction with input_embedding

# TODO efficient inference?? (probably not)


class LLaMa_MLA(nn.Module):
    def __init__(self, args, vocab_size=10000): #, dtype=torch.float32):
        super(LLaMa_MLA, self).__init__()
        # args.dtype = dtype
        self.d_model = args.d_model
        self.nhead = args.nhead
        self.num_layers = args.num_layers
        if args.d_ff is None:
            self.d_ff = 4 * args.d_model
        else:
            self.d_ff = args.d_ff
        self.dropout = nn.Dropout(args.dropout)
        self.layers = nn.ModuleList(
            [LLaMa_MLA_block(args) for _ in range(self.num_layers)]
        )
        self.input_embedding = input_embedding(
            self.d_model, vocab_size
        )
        self.norm = nn.RMSNorm(self.d_model)
        self.out = nn.Linear(self.d_model, vocab_size)

    def forward(self, x):
        x = self.input_embedding(x)  # (batch_size, seq_len, d_model)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)  # (batch_size, seq_len, d_model)
        x = self.out(x)
        return x  # (batch_size, seq_len, vocab_size) logits


class LLaMa_MLA_block(nn.Module):
    def __init__(self, args):
        super(LLaMa_MLA_block, self).__init__()
        self.nhead = args.nhead
        self.mha = MLA(args)
        self.norm1 = nn.RMSNorm(args.d_model)
        self.norm2 = nn.RMSNorm(args.d_model)
        self.ff = SwiGLU_feed_forward(args)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        mask = get_causal_mask(seq_len)
        mask = mask.repeat(batch_size, self.nhead, 1, 1)
        x = x + self.mha(self.norm1(x), mask)  # (batch_size, seq_len, d_model)
        x = x + self.ff(self.norm2(x))
        return x


class RoPE(nn.Module):
    def __init__(self, d: int, base: int = 10_000):
        super().__init__()
        self.base = base
        self.d = d
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, seq_len: int, x_device: torch.device):

        if self.cos_cached is not None and seq_len <= self.cos_cached.shape[0]:
            return

        theta = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.d, 2, device=x_device)
                / self.d
            )
        )  # THETA = 10,000^(-2*i/d) or 1/10,000^(2i/d)

        seq_idx = torch.arange(
            seq_len, device=x_device
        )  # Position Index -> [0,1,2...seq-1]

        idx_theta = torch.einsum(
            "n,d->nd", seq_idx, theta
        )  # Calculates m*(THETA) = [ [0, 0...], [THETA_1, THETA_2...THETA_d/2], ... [seq-1*(THETA_1), seq-1*(THETA_2)...] ]

        idx_theta2 = torch.cat(
            [idx_theta, idx_theta], dim=1
        )  # [THETA_1, THETA_2...THETA_d/2] -> [THETA_1, THETA_2...THETA_d]

        self.cos_cached = idx_theta2.cos()[
            :, None, None, :
        ].float()  # Cache [cosTHETA_1, cosTHETA_2...cosTHETA_d]
        self.sin_cached = idx_theta2.sin()[
            :, None, None, :
        ].float()  # cache [sinTHETA_1, sinTHETA_2...sinTHETA_d]

    def _neg_half(self, x: torch.Tensor):

        d_2 = self.d // 2  #

        return torch.cat(
            [-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1
        )  # [x_1, x_2,...x_d] -> [-x_d/2, ... -x_d, x_1, ... x_d/2]

    def forward(self, x: torch.Tensor):

        seq_len = x.shape[2]

        self._build_cache(seq_len, x.device)

        neg_half_x = self._neg_half(x)

        cos_ = self.cos_cached[: seq_len].view(1,1,seq_len,-1)
        sin_ = self.sin_cached[: seq_len].view(1,1,seq_len,-1)

        x_rope = (x * cos_) + (neg_half_x * sin_)  # [x_1*cosTHETA_1 - x_d/2*sinTHETA_d/2, ....]

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


class MLA_simple(
    nn.Module
):  # TODO make optimised version with "absorbed" weights + qkv norm
    def __init__(self, args):
        super().__init__()
        self.d_model = args.d_model
        self.nhead = args.nhead
        self.head_dim = self.d_model // self.nhead
        assert self.head_dim * self.nhead == self.d_model

        self.kv_compression_dim = (
            self.d_model // 4
        )  # TODO check how much this matters, should be a hyperparameter (roughly what R1 has)
        # probably stops mattering after some point
        self.q_compression_dim = (
            self.kv_compression_dim * 2
        )  # TODO check how much this matters, should be a hyperparameter (roughly what R1 has)

        self.rope_dim = args.qk_rope_dim  # TODO (i really have no clue my guy)

        # MLA part
        self.kv_down = nn.Linear(self.d_model, self.kv_compression_dim)
        self.q_down = nn.Linear(self.d_model, self.q_compression_dim)

        # Separate K and V projections for each head
        self.k_up = nn.Linear(self.kv_compression_dim, self.head_dim * self.nhead)
        self.v_up = nn.Linear(self.kv_compression_dim, self.head_dim * self.nhead)
        # Separate Q projections for each head
        self.q_up = nn.Linear(self.q_compression_dim, self.head_dim * self.nhead)

        self.o = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(args.dropout)
        self.rotary_emb = RoPE(self.rope_dim)

        self.k_up_rope = nn.Linear(self.d_model, self.rope_dim)
        self.q_up_rope = nn.Linear(self.q_compression_dim, self.rope_dim * self.nhead)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # --- Q, K, V Projections ---
        c_vk = self.kv_down(x)
        c_q = self.q_down(x)

        k = (
            self.k_up(c_vk)
            .view(batch_size, seq_len, self.nhead, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_up(c_vk)
            .view(batch_size, seq_len, self.nhead, self.head_dim)
            .transpose(1, 2)
        )
        q = (
            self.q_up(c_q)
            .view(batch_size, seq_len, self.nhead, self.head_dim)
            .transpose(1, 2)
        )

        k_rope = (
            self.k_up_rope(x)
            .view(batch_size, seq_len, 1, self.rope_dim)
            .tile(1, 1, self.nhead, 1)
            .transpose(1, 2)
        )
        q_rope = (
            self.q_up_rope(c_q)
            .view(batch_size, seq_len, self.nhead, self.rope_dim)
            .transpose(1, 2)
        )
        k_rope = self.rotary_emb(k_rope)
        q_rope = self.rotary_emb(q_rope)

        k = torch.concat([k, k_rope], dim=-1)
        q = torch.concat([q, q_rope], dim=-1)

        # attn_output = F.scaled_dot_product_attention(
        #     q, k, v, is_causal=True, dropout_p=0.1
        # )
        a = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            self.head_dim + self.rope_dim
        )

        # Apply mask if provided
        if mask is not None:
            a = a.masked_fill(mask == 1, -1e9)

        a = torch.softmax(a, dim=-1)
        attn_output = torch.matmul(a, v)

        # --- Concatenate and Output Projection ---
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )
        output = self.dropout(self.o(attn_output))

        return output


class MLA(nn.Module):  # done right and "optimised" TODO missing norms for q and k
    def __init__(self, args):
        super().__init__()
        self.d_model = args.d_model
        self.nhead = args.nhead
        self.head_dim = self.d_model // self.nhead
        assert self.head_dim * self.nhead == self.d_model

        self.kv_compression_dim = args.kv_compression_dim 
        self.q_compression_dim = args.q_compression_dim

        self.rope_dim = args.qk_rope_dim

        # MLA part
        self.kv_down = nn.Linear(
            self.d_model, self.kv_compression_dim + self.rope_dim
        )
        self.q_down = nn.Linear(self.d_model, self.q_compression_dim)

        self.kv_up = nn.Linear(
            self.kv_compression_dim, self.head_dim * self.nhead * 2
        )
        self.q_up = nn.Linear(
            self.q_compression_dim,
            (self.head_dim + self.rope_dim) * self.nhead
        )

        self.o = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(args.dropout)
        self.rotary_emb = RoPE(self.rope_dim)

        self.kv_norm = nn.RMSNorm(self.kv_compression_dim)
        self.q_norm = nn.RMSNorm(self.q_compression_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # --- Q, K, V Projections ---
        c_kv = self.kv_down(x)
        c_kv, k_rope = c_kv.split([self.kv_compression_dim, self.rope_dim], dim=-1)

        kv = (
            self.kv_up(self.kv_norm(c_kv))
            .view(batch_size, seq_len, self.nhead, self.head_dim * 2)
            .transpose(1, 2)
        )
        k, v = kv.split([self.head_dim, self.head_dim], dim=-1)

        q = (
            self.q_up(self.q_norm(self.q_down(x)))
            .view(batch_size, seq_len, self.nhead, (self.head_dim + self.rope_dim))
            .transpose(1, 2)
        )
        q, q_rope = q.split([self.head_dim, self.rope_dim], dim=-1)

        k_rope = (
            k_rope.view(batch_size, seq_len, 1, self.rope_dim)
            .tile(1, 1, self.nhead, 1)
            .transpose(1, 2)
        )

        k_rope = self.rotary_emb(k_rope)
        q_rope = self.rotary_emb(q_rope)

        k = torch.concat([k, k_rope], dim=-1)
        q = torch.concat([q, q_rope], dim=-1)

        # attn_output = F.scaled_dot_product_attention(
        #     q, k, v, is_causal=True, dropout_p=0.1
        # )
        a = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            self.head_dim + self.rope_dim
        )

        # Apply mask if provided
        if mask is not None:
            a = a.masked_fill(mask == 1, -float("inf"))

        a = torch.softmax(a, dim=-1)
        attn_output = torch.matmul(a, v)

        # --- Concatenate and Output Projection ---
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )
        output = self.dropout(self.o(attn_output))

        return output


class SwiGLU_feed_forward(nn.Module):
    def __init__(self, args):
        super().__init__()
        # d_ff is now the "intermediate" dimension.  The SwiGLU layer
        # will project to d_ff, and then the output projection will go back to d_model.
        # self.swiglu = SwiGLU(d_model, d_ff)
        self.linear_in_gate = nn.Linear(args.d_model, args.d_ff * 2)
        self.linear_out = nn.Linear(args.d_ff, args.d_model)
        self.dropout = nn.Dropout(args.dropout)
        self.d_model = args.d_model

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
