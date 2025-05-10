import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from .components import (
    mha,
    get_causal_mask,
)

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
        self.input = nn.Embedding(vocab_size, self.d_model) # input embedding
        self.norm = nn.RMSNorm(self.d_model)
        self.out = nn.Linear(self.d_model, vocab_size)

        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

    def forward(self, x):
        x = self.input(x) / math.sqrt(self.d_model)  # (batch_size, seq_len, d_model)
        x = self.dropout(x)
        seq_len = x.size(1)
        freqs_cis = self.freqs_cis[0:0+seq_len]
        for layer in self.layers:
            x = layer(x, freqs_cis)
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

    def forward(self, x, freqs_cis):
        batch_size, seq_len, _ = x.size()
        mask = get_causal_mask(seq_len)
        mask = mask.repeat(batch_size, self.nhead, 1, 1)
        x = x + self.mha(self.norm1(x), mask, freqs_cis)  # (batch_size, seq_len, d_model)
        x = x + self.ff(self.norm2(x))
        return x



def precompute_freqs_cis(args) -> torch.Tensor:
    # original values in deepseek V3 inference (https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py#L766)
    # seq_len: int = 4096
    # rope_theta: float = 10000.0
    # rope_factor: float = 40
    # beta_fast: int = 32
    # beta_slow: int = 1
    # mscale: float = 1.

    dim = args.qk_rope_dim
    max_seqlen = args.seq_len * 4 # might need to look at this, currently just the same as V3 so hopefully works fine
    beta_fast = 32
    beta_slow = 1
    base = 10000.0
    factor = 40

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if max_seqlen > args.seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(max_seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


class MLA_simple(
    nn.Module
):
    def __init__(self, args):
        super().__init__()
        self.d_model = args.d_model
        self.nhead = args.nhead
        self.head_dim = self.d_model // self.nhead
        assert self.head_dim * self.nhead == self.d_model

        self.kv_compression_dim = (
            self.d_model // 4
        )
        self.q_compression_dim = (
            self.kv_compression_dim * 2)

        self.rope_dim = args.qk_rope_dim  

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


class MLA(nn.Module):
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
        # self.rotary_emb = RoPE(self.rope_dim)

        self.kv_norm = nn.RMSNorm(self.kv_compression_dim)
        self.q_norm = nn.RMSNorm(self.q_compression_dim)

    def forward(self, x, mask=None, freqs_cis=None):
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
        )

        k_rope = apply_rotary_emb(k_rope, freqs_cis=freqs_cis).transpose(1,2)
        q_rope = apply_rotary_emb(q_rope.transpose(1,2), freqs_cis=freqs_cis).transpose(1,2)

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
        self.linear_in_gate = nn.Linear(args.d_model, args.d_ff * 2)
        self.linear_out = nn.Linear(args.d_ff, args.d_model)
        self.dropout = nn.Dropout(args.dropout)
        self.d_model = args.d_model

    def forward(self, x):
        u, v = self.linear_in_gate(x).chunk(
            2, dim=-1
        )   
        x = u * F.silu(v)# * self.d_model ** 0.5) # Apply SwiGLU with scaling (not worth it)
        x = self.linear_out(x)
        x = self.dropout(x)  # Apply dropout *after* the output projection
        return x

# class Latent_SwiGLU_feed_forward(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.linear_in_gate_down = nn.Linear(args.d_model, args.ff_compression_dim * 2)
#         self.linear_in_gate_up = nn.Linear(args.ff_compression_dim * 2, args.d_ff * 2)
#         self.linear_out_down = nn.Linear(args.d_ff, args.ff_compression_dim)
#         self.linear_out_up = nn.Linear(args.ff_compression_dim, args.d_model)
#         self.dropout = nn.Dropout(args.dropout)
#         self.d_model = args.d_model

#     def forward(self, x):
#         u, v = self.linear_in_gate_up(self.linear_in_gate_down(x)).chunk(
#             2, dim=-1
#         )  # * self.d_model ** 0.5)  # Apply SwiGLU with scaling
#         x = u * F.silu(v)
#         x = self.linear_out_up(self.linear_out_down(x))
#         x = self.dropout(x)  # Apply dropout *after* the output projection
#         return x