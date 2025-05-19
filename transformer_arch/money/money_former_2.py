# tries to predict movements in the stock market
"""first version TODO list:

create model architecture:
start with MLA?

myb additional encoding for sequence type? (stock or general market trend, etc.) (probably for later ver)
Embeddings for specific stock tickers?



further version TODO list:
separate stocks and general indicators
create encoding for seperator tokens
embeddings for seperate sequences
RoPE per sequence
embeddings for sequence category (stock price/return, general market, etc.)
specific stock tickers?
set up custom attn mask
improve/adapt dataloader

try better architectures

"""

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from ..components import get_causal_mask  # TODO at v2

# for debug/visualization
import matplotlib.pyplot as plt


# LLaMa architecture for now
# money former is the issue, somehow?
class Money_former(nn.Module):
    def __init__(self, args):  # , dtype=torch.float32):
        super().__init__()
        # args.dtype = dtype
        self.d_model = args.d_model
        self.nhead = args.nhead
        self.num_layers = args.num_layers
        self.d_ff = args.d_ff
        self.dropout = nn.Dropout(args.dropout)
        self.layers = nn.ModuleList(
            [Money_former_block(args) for _ in range(self.num_layers)]
        )
        self.input_features = args.input_features

        self.ticker_embedding = nn.Embedding(
            len(args.tickers), self.d_model
        )
        self.value_inputs = nn.ModuleList(
            [nn.Linear(self.input_features, self.d_model) for _ in range(1)] # range(len(args.tickers)]
        )
        # self.value_input = nn.Linear(
        #     self.input_features, self.d_model
        # )  # encodes numerical input
        self.norm = nn.RMSNorm(self.d_model)

        self.predict_distribution = args.predict_gaussian
        if self.predict_distribution:
            self.out = nn.Linear(
                self.d_model, len(args.indices_to_predict) * 2
            )  # decodes to mean and std
        else:
            self.out = nn.Linear(
                self.d_model, len(args.indices_to_predict)
            )  # decodes to target(s)

        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

        # seperator/attention sink token
        self.seperator = nn.Embedding(1, self.d_model)

    def forward(self, x, seperator=None, tickers=None):
        batch_size, seq_len, num_sequences, _ = x.size()

        seperator = self.seperator(seperator).unsqueeze(1)
        seperator = seperator.repeat(1, 1, num_sequences, 1)

        tickers = self.ticker_embedding(tickers).unsqueeze(1)
        tickers = tickers.repeat(1, seq_len + 1, 1, 1)

        temp_x = torch.zeros(batch_size, seq_len, num_sequences, self.d_model, device=x.device)
        for i in range(num_sequences):
            temp_x[:,:,i,:] = self.value_inputs[0](x[:,:,i,:])

        x = torch.cat([seperator,temp_x], dim=1)
        x = x + tickers

        x = x / math.sqrt(self.d_model)
        x = self.dropout(x) # (batch_size, seq_len, num_sequences, d_model)

        x = x.view(batch_size, -1, self.d_model) # (batch_size, (seq_len+1)*num_sequences, d_model)
        # maybe try without flattening? # TODO
        freqs_cis = self.freqs_cis[0 : 0 + seq_len]
        for layer in self.layers:
            x = layer(x, freqs_cis)
        x = self.norm(x)  # (batch_size, seq_len, d_model)
        x = self.out(x)
        if self.predict_distribution:
            x = x.view(
                x.size(0), x.size(1), -1, 2
            )  # (batch_size, seq_len, points in future, (mean and std) or target)
            return x
        return x  # (batch_size, seq_len, targets) logits


class Money_former_block(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.nhead = args.nhead
        self.mha = MHA(args)
        self.norm1 = nn.RMSNorm(args.d_model)
        self.norm2 = nn.RMSNorm(args.d_model)
        self.ff = SwiGLU_feed_forward(args)
        self.num_sequences = len(args.tickers)

    def forward(self, x, freqs_cis):
        batch_size, seq_len, _ = x.size()
        seq_len = seq_len // self.num_sequences
        mask = get_causal_mask(seq_len)
        mask = mask.repeat(batch_size, self.nhead, self.num_sequences, self.num_sequences)
        x = x + self.mha(
            self.norm1(x), mask, freqs_cis
        )  # (batch_size, seq_len, d_model)
        x = x + self.ff(self.norm2(x))
        return x


class MHA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.d_model = args.d_model
        self.nhead = args.nhead
        self.head_dim = self.d_model // self.nhead
        assert self.head_dim * self.nhead == self.d_model

        self.q_linear = nn.Linear(self.d_model, self.d_model)
        self.k_linear = nn.Linear(self.d_model, self.d_model)
        self.v_linear = nn.Linear(self.d_model, self.d_model)

        self.o = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(args.dropout)
        self.num_sequences = len(args.tickers)

    def forward(self, x, mask=None, freqs_cis=None):
        batch_size, seq_len, _ = x.shape
        seq_per_stock = seq_len // self.num_sequences

        # # Project and split x into multiple heads
        # q = self.q_linear(x).view(batch_size, seq_len, self.nhead, self.head_dim)
        # k = self.k_linear(x).view(batch_size, seq_len, self.nhead, self.head_dim)
        # v = (
        #     self.v_linear(x)
        #     .view(batch_size, seq_len, self.nhead, self.head_dim)
        #     .transpose(1, 2)
        # )  # (batch_size, nhead, seq_len, head_dim)

        q_proj = self.q_linear(x)
        k_proj = self.k_linear(x)
        v_proj = self.v_linear(x)

        q = q_proj.view(batch_size, seq_per_stock, self.num_sequences, self.nhead, self.head_dim).permute(0, 1, 3, 4, 2)
        k = k_proj.view(batch_size, seq_per_stock, self.num_sequences, self.nhead, self.head_dim).permute(0, 1, 3, 4, 2)
        v = v_proj.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        for i in range(self.num_sequences):
            q_temp = q[:, :, :, :, i]
            k_temp = k[:, :, :, :, i]
            q_sep, q_temp = q_temp.clone().split([1,(seq_len-1)//self.num_sequences], dim=1)
            k_sep, k_temp = k_temp.clone().split([1,(seq_len-1)//self.num_sequences], dim=1)
            q_temp = apply_rotary_emb(q_temp, freqs_cis=freqs_cis)
            k_temp = apply_rotary_emb(k_temp, freqs_cis=freqs_cis)
            q[:, :, :, :, i] = torch.cat([q_sep, q_temp], dim=1)
            k[:,:,:,:,i] = torch.cat([k_sep, k_temp], dim=1)

        q_perm = q.permute(0, 4, 1, 2, 3).contiguous() # (batch, num_seq, S_per_stock, nhead*2, head_dim//2)
        k_perm = k.permute(0, 4, 1, 2, 3).contiguous()

        q = q_perm.reshape(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = k_perm.reshape(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

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


class SwiGLU_feed_forward(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.linear_in_gate = nn.Linear(args.d_model, args.d_ff * 2)
        self.linear_out = nn.Linear(args.d_ff, args.d_model)
        self.dropout = nn.Dropout(args.dropout)
        self.d_model = args.d_model

    def forward(self, x):
        u, v = self.linear_in_gate(x).chunk(2, dim=-1)
        x = u * F.silu(v)  # * self.d_model ** 0.5) # Apply SwiGLU with scaling
        x = self.linear_out(x)
        x = self.dropout(x)  # Apply dropout *after* the output projection
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
    max_seqlen = (
        args.seq_len * 4
    )  # might need to look at this, currently just the same as V3 so hopefully works fine
    beta_fast = 32
    beta_slow = 1
    base = 10000.0
    factor = 40

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return (
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

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
