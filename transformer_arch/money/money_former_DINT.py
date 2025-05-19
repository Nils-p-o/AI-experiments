# tries to predict movements in the stock market
""" first version TODO list:

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
from ..components import get_causal_mask # TODO at v2
# for debug/visualization
import matplotlib.pyplot as plt

# LLaMa architecture for now
 # money former is the issue, somehow?
class Money_former_DINT(nn.Module):
    def __init__(self, args): #, dtype=torch.float32):
        super().__init__()
        # args.dtype = dtype
        self.d_model = args.d_model
        self.nhead = args.nhead
        self.num_layers = args.num_layers
        self.d_ff = args.d_ff
        self.dropout = nn.Dropout(args.dropout)
        self.layers = nn.ModuleList(
            [Money_former_block(args, depth) for depth in range(self.num_layers)]
        )
        self.input_features = args.input_features
        self.value_input = nn.Linear(self.input_features, self.d_model) # encodes numerical input
        self.norm = nn.RMSNorm(self.d_model)
        
        self.predict_distribution = args.predict_gaussian
        if self.predict_distribution:
            self.out = nn.Linear(self.d_model, len(args.indices_to_predict) * 2) # decodes to mean and std
        else:
            self.out = nn.Linear(self.d_model, len(args.indices_to_predict)) # decodes to target(s)

        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

        # seperator/attention sink token
        self.seperator = nn.Embedding(1, self.d_model)

        # time embedding shenanigans
        # self.day_of_week = nn.Embedding(5, 3)
        # self.day_of_month = nn.Embedding(32, 6)
        # # self.day_of_year = nn.Embedding(367, 10)
        # self.month_of_year = nn.Embedding(13, 5)
        # self.quarter_of_year = nn.Embedding(5, 3)
        # # self.is_leap_year = nn.Embedding(2, 2)
        # self.is_month_start = nn.Embedding(2, 2)
        # self.is_month_end = nn.Embedding(2, 2)
        # self.is_quarter_start = nn.Embedding(2, 2)
        # self.is_quarter_end = nn.Embedding(2, 2)
        # # self.is_year_start = nn.Embedding(2, self.d_model)
        # # self.is_year_end = nn.Embedding(2, self.d_model)

        # self.project = nn.Linear(self.d_model+25, self.d_model)

    def forward(self, x, seperator=None):
        # x_time, x = torch.split(x,[10,self.input_features], dim=-1)
        # x_time = x_time.int()
        # time_embeded = torch.cat([
        #     self.day_of_week(x_time[:, :, 0]),
        #     self.day_of_month(x_time[:, :, 1]),
        #     # self.day_of_year(x_time[:, :, 2]),
        #     self.month_of_year(x_time[:, :, 3]),
        #     self.quarter_of_year(x_time[:, :, 4]),
        #     # self.is_leap_year(x_time[:, :, 5]),
        #     self.is_month_start(x_time[:, :, 6]),
        #     self.is_month_end(x_time[:, :, 7]),
        #     self.is_quarter_start(x_time[:, :, 8]),
        #     self.is_quarter_end(x_time[:, :, 9]),
        # ], dim=-1)
        # x = self.project(torch.cat([x, time_embeded], dim=-1)) / math.sqrt(self.d_model) # TODO potentially add act fn + layer norm after this?
        if seperator != None:
            seperator = self.seperator(seperator)
        x = self.value_input(x)  # (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        if seperator != None:
            x = torch.cat([seperator, x], dim=1)
        x = x / math.sqrt(self.d_model)
        x = self.dropout(x)
        freqs_cis = self.freqs_cis[0:0+seq_len]
        for layer in self.layers:
            x = layer(x, freqs_cis)
        x = self.norm(x)  # (batch_size, seq_len, d_model)
        x = self.out(x)
        if self.predict_distribution:
            x = x.view(x.size(0), x.size(1), -1, 2) # (batch_size, seq_len, points in future, (mean and std) or target)
            return x
        return x  # (batch_size, seq_len, targets) logits

class Money_former_block(nn.Module):
    def __init__(self, args, depth=0):
        super().__init__()
        self.nhead = args.nhead
        self.mha = DINT_MHA(args, depth)
        self.norm1 = nn.RMSNorm(args.d_model)
        self.norm2 = nn.RMSNorm(args.d_model)
        self.ff = SwiGLU_feed_forward(args)

    def forward(self, x, freqs_cis):
        batch_size, seq_len, _ = x.size()
        mask = get_causal_mask(seq_len)
        mask = mask.repeat(batch_size, 2*self.nhead, 1, 1)
        x = x + self.mha(self.norm1(x), mask, freqs_cis)  # (batch_size, seq_len, d_model)
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

    def forward(self, x, mask=None, freqs_cis=None):
        batch_size, seq_len, _ = x.shape

        # Project and split x into multiple heads
        q = (
            self.q_linear(x)
            .view(batch_size, seq_len, self.nhead, self.head_dim)
        )
        k = (
            self.k_linear(x)
            .view(batch_size, seq_len, self.nhead, self.head_dim)
        )
        v = (
            self.v_linear(x)
            .view(batch_size, seq_len, self.nhead, self.head_dim)
            .transpose(1, 2)
        )  # (batch_size, nhead, seq_len, head_dim)
        q_sep, q = q.split([1,seq_len-1], dim=1)
        k_sep, k = k.split([1,seq_len-1], dim=1)
        q = apply_rotary_emb(q, freqs_cis=freqs_cis)
        k = apply_rotary_emb(k, freqs_cis=freqs_cis)

        q = torch.cat([q_sep, q], dim=1).transpose(1, 2)
        k = torch.cat([k_sep, k], dim=1).transpose(1, 2)

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


class DINT_MHA(nn.Module):
    def __init__(self, args, depth=0, v1: bool = False):
        super().__init__()
        self.d_model = args.d_model
        self.nhead = args.nhead
        self.head_dim = self.d_model // self.nhead  # technically half of this, but it gets confusing cuz its still one head, so...
        assert self.head_dim * self.nhead == self.d_model

        # Separate Q projections for each head
        self.q_linear = nn.Linear(self.d_model, self.d_model)
        # Shared K and V projections for each group
        self.k_linear = nn.Linear(self.d_model, self.d_model)
        self.v_linear = nn.Linear(self.d_model, self.d_model)

        self.o = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(args.dropout)

        self.scaling = 1.0 / math.sqrt(self.head_dim//2)#?? TODO

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_k1 = nn.Parameter(
            torch.zeros(self.head_dim // 2, dtype=torch.float32).normal_(
                mean=0, std=0.1
            )
        )
        self.lambda_k2 = nn.Parameter(
            torch.zeros(self.head_dim // 2, dtype=torch.float32).normal_(
                mean=0, std=0.1
            )
        )
        self.lambda_q1 = nn.Parameter(
            torch.zeros(self.head_dim // 2, dtype=torch.float32).normal_(
                mean=0, std=0.1
            )
        )
        self.lambda_q2 = nn.Parameter(
            torch.zeros(self.head_dim // 2, dtype=torch.float32).normal_(
                mean=0, std=0.1
            )
        )

        self.norm = nn.RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)
        # TODO supposed to be groupnorm, but i dunno how to do that
        # self.num_groups_norm = self.nhead # maybe different numbers??
        # # Now applied *after* attention, so normalize the head_dim
        # self.norm = nn.GroupNorm(self.num_groups_norm, self.nhead * self.head_dim)

        self.v1 = v1

    def forward(self, x, mask=None, freqs_cis=None):
        batch_size, seq_len, _ = x.shape

        # --- Q, K, V Projections ---
        q = self.q_linear(x).view(
            batch_size, seq_len, self.nhead * 2, self.head_dim // 2
        )
        k = self.k_linear(x).view(
            batch_size, seq_len, self.nhead * 2, self.head_dim // 2
        )
        v = self.v_linear(x).view(batch_size, seq_len, self.nhead, self.head_dim)

        # --- RoPE --- TODO
        q_sep, q = q.split([1,seq_len-1], dim=1)
        k_sep, k = k.split([1,seq_len-1], dim=1)
        q = apply_rotary_emb(q, freqs_cis=freqs_cis)
        k = apply_rotary_emb(k, freqs_cis=freqs_cis)
        q = torch.cat([q_sep, q], dim=1)
        k = torch.cat([k_sep, k], dim=1)

        q = q.transpose(1, 2)  # (batch_size, 2*nhead, seq_len, head_dim/2)
        # --- Reshape K and V for Grouped Attention ---
        # Repeat K and V for each head within the group.
        k = k.transpose(1, 2)  # (batch_size, 2*nhead, seq_len, head_dim/2)
        v = v.transpose(1, 2)  # (batch_size, nhead, seq_len, head_dim)

        # --- Attention Calculation ---
        a = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        # (batch_size, 2*nhead, seq_len, seq_len)

        if mask is not None:
            a = a.masked_fill(mask == 1, -1e9)

        a = torch.softmax(a, dim=-1)
        a = self.dropout(a)

        lambda_1 = torch.exp(torch.sum(self.lambda_k1 * self.lambda_q1, dim=-1))
        lambda_2 = torch.exp(torch.sum(self.lambda_k2 * self.lambda_q2, dim=-1))
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        a = a.view(batch_size, self.nhead, 2, seq_len, seq_len)
        if self.v1:
        # original code:
            a3 = torch.repeat_interleave(
                torch.mean(a[:, :, 0, :, :], dim=-1, keepdim=True), seq_len, dim=-2
            )  
        else:
        # alternate: TODO smth aint right here (attn should sum to 1.0 (but which dim?))
        # should do it head by head?
            # a3_scaler = seq_len / torch.sum(mask == 0, dim=-1, keepdim=True).float()
            # a3_scaler = a3_scaler.view(batch_size, self.nhead,2, seq_len, 1)
            # a3_scaler = a3_scaler[:, :, 0, :, :]
            a3_scaler = (seq_len/(torch.arange(seq_len)+1)).tile(batch_size,self.nhead,1).unsqueeze(-1)
            
            # a3 = torch.mean(a[:, :, 0, :, :], dim=-1, keepdim=True)
            # a3 = a3 * a3_scaler # a3 = a3 * a3_scaler
            # a3 = a3.repeat_interleave(seq_len, dim=-1)
            a3 = torch.mean(a[:, :, 1, :, :], dim=-1, keepdim=True) * a3_scaler
            a3 = a3.masked_fill(mask[:,:self.nhead,:,:] == 1, 0)
        assert a[:, :, 0, :, :].shape == a3.shape
        assert torch.isclose((lambda_full * a[:, :, 1, :, :]).sum(dim=-1),(a3 * lambda_full).sum(dim=-1),rtol=1e-3,atol=1e-5).all() == True
        a = a[:, :, 0, :, :] - lambda_full * a[:, :, 1, :, :] + a3 * lambda_full
        # assert (a[0,0].sum(dim=-1) == 1.0).all() == True
        # visualizing attention for debugging purposes
        # import matplotlib.pyplot as plt
        # a_weights = a.detach().cpu()
        # plt.imshow(torch.sum(torch.sum(a_weights, dim=1), dim=0))
        # plt.show()



        attn_output = torch.matmul(a, v)  # (batch_size, nhead, seq_len, head_dim)
        attn_output.transpose(1, 2).reshape(
            batch_size, seq_len, self.nhead * self.head_dim
        )  # TODO: check if this is correct
        attn_output = self.norm(attn_output.transpose(1, 2)).transpose(
            1, 2
        )  # Apply GroupNorm, then put back into right shape
        attn_output = attn_output.reshape(
            batch_size, seq_len, self.nhead, self.head_dim
        ).transpose(1, 2)
        # attn_output = attn_output * (1 - self.lambda_init)
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
        x = u * F.silu(v)# * self.d_model ** 0.5) # Apply SwiGLU with scaling
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

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)