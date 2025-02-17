import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from .components import input_embedding, get_causal_mask
from .LLaMa import RoPE, SwiGLU_feed_forward

# added DiffAttn


class DintTransformer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        num_layers,
        d_ff=None,
        dropout=0.1,
        vocab_size=10000,
        seq_len=128,
        groups=8,
    ):
        super(DintTransformer, self).__init__()
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
                Dint_block(d_model, nhead, d_ff, dropout, depth, groups)
                for depth in range(num_layers)
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
        return x  # (batch_size, seq_len, vocab_size) logits


class Dint_block(nn.Module):
    def __init__(self, d_model, nhead, d_ff=None, dropout=0.1, depth=0, groups=4):
        super(Dint_block, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_ff = d_ff
        self.dropout = dropout
        self.mha = DintGQA(
            d_model=d_model, nhead=nhead, dropout=dropout, depth=depth, groups=groups
        )
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.ff = SwiGLU_feed_forward(d_model, d_ff, dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        mask = get_causal_mask(seq_len)
        mask = mask.repeat(batch_size, 2 * self.nhead, 1, 1)
        x = x + self.mha(self.norm1(x), mask)  # (batch_size, seq_len, d_model)
        x = x + self.ff(self.norm2(x))
        return x


class DintGQA(nn.Module):
    def __init__(self, d_model, nhead, groups=4, dropout=0.1, depth=0):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.groups = groups
        self.head_dim = (
            d_model // nhead
        )  # technically half of this, but it gets confusing cuz its still one head, so...
        assert self.head_dim * nhead == d_model
        assert (
            nhead % groups == 0
        ), f"Number of heads {nhead} must be divisible by number of groups {groups}"
        self.heads_per_group = nhead // groups

        # Separate Q projections for each head
        self.q_linear = nn.Linear(d_model, d_model)
        # Shared K and V projections for each group
        self.k_linear = nn.Linear(d_model, d_model // self.heads_per_group)
        self.v_linear = nn.Linear(d_model, d_model // self.heads_per_group)

        self.o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.rotary_emb = RoPE(self.head_dim // 2)

        self.scaling = 1.0 / math.sqrt(self.head_dim)  # //2??

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

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # --- Q, K, V Projections ---
        q = self.q_linear(x).view(
            batch_size, seq_len, self.nhead * 2, self.head_dim // 2
        )
        k = self.k_linear(x).view(
            batch_size, seq_len, self.groups * 2, self.head_dim // 2
        )
        v = self.v_linear(x).view(batch_size, seq_len, self.groups, self.head_dim)

        # --- RoPE ---
        q = self.rotary_emb(q)
        k = self.rotary_emb(k)

        q = q.transpose(1, 2)  # (batch_size, 2*nhead, seq_len, head_dim/2)
        # --- Reshape K and V for Grouped Attention ---
        # Repeat K and V for each head within the group.
        k = k.transpose(1, 2).repeat_interleave(
            self.heads_per_group, dim=1
        )  # (batch_size, 2*nhead, seq_len, head_dim/2)
        v = v.transpose(1, 2).repeat_interleave(
            self.heads_per_group, dim=1
        )  # (batch_size, nhead, seq_len, head_dim)

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
        # original code:
        a3 = torch.repeat_interleave(
            torch.mean(a[:, :, 0, :, :], dim=-2, keepdim=True), seq_len, dim=-2
        )  # TODO might need to be distributed and calculated only across the ones that are not masked
        # can do this by multiplying columns by scalar which is seq_len/num_unmasked
        # then maybe mask a3
        # alternate:
        # a3_scaler = torch.sum(mask == 1, dim=-2, keepdim=True).float() / seq_len
        # a3 = torch.mean(a[:, :, 0, :, :], dim=-2, keepdim=True)
        # a3 = torch.mul(a3, a3_scaler) # a3 = a3 * a3_scaler
        # a3 = a3.repeat_interleave(seq_len, dim=-2)
        # a3 = a3.masked_fill(mask == 1, 0)

        a = a[:, :, 0, :, :] - lambda_full * a[:, :, 1, :, :] + a3 * lambda_full

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


def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)
