import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .components import input_embedding, get_causal_mask
from .LLaMa import RoPE


# some problem with nGPT, need to fix it
# maybe the sWiGLu is the problem

# modules to try:
# Attention
# SwiGLU
# nGPT_block
# weight norm (wrong step, or wrong dim)
# higher lr


class nGPT(nn.Module):
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
        super(nGPT, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        if d_ff is None:
            d_ff = 4 * d_model
        else:
            self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [nGPT_block(d_model, nhead, d_ff, dropout) for _ in range(num_layers)]
        )
        self.input_embedding = input_embedding(d_model, vocab_size)
        self.s_z_init = 1.0
        self.s_z_scale = 1.0 / math.sqrt(d_model)
        self.s_z = nn.Parameter(
            torch.ones(vocab_size, device=self.device, requires_grad=True)
            * self.s_z_scale
        )
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.input_embedding(x)  # (batch_size, seq_len, d_model)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        x = self.out(x)
        x = x * self.s_z * (self.s_z_init / self.s_z_scale)
        return x  # (batch_size, seq_len, vocab_size) logits


class nGPT_block(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super(nGPT_block, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.d_model = d_model
        self.nhead = nhead
        self.d_ff = d_ff
        self.mha = nGPT_GQA(d_model, nhead)
        self.ff = nGPT_SwiGLU_ff(d_model, d_ff, dropout)
        self.eigen_a_init = 1.0
        self.eigen_a_scale = 1.0 / math.sqrt(d_model)
        self.eigen_a = nn.Parameter(
            torch.ones(d_model, device=self.device, requires_grad=True)
            * self.eigen_a_scale
        )
        self.eigen_m_init = 1.0
        self.eigen_m_scale = 1.0 / math.sqrt(d_model)
        self.eigen_m = nn.Parameter(
            torch.ones(d_model, device=self.device, requires_grad=True)
            * self.eigen_m_scale
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        mask = get_causal_mask(seq_len)
        mask = mask.repeat(batch_size, self.nhead, 1, 1)
        x_a = cosine_norm(self.mha(x, mask))  # (batch_size, seq_len, d_model)
        x = cosine_norm(
            x + self.eigen_a * (self.eigen_a_init / self.eigen_a_scale) * (x_a - x)
        )
        x_m = cosine_norm(self.ff(x))
        x = cosine_norm(
            x + self.eigen_m * (self.eigen_m_init / self.eigen_m_scale) * (x_m - x)
        )
        return x


class nGPT_GQA(nn.Module):
    def __init__(self, d_model, nhead, groups=4, dropout=0.1):
        super(nGPT_GQA, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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

        self.s_qk_init = 1.0
        self.s_qk_scale = 1.0 / math.sqrt(self.head_dim)
        self.s_qk = nn.Parameter(
            torch.ones(
                self.nhead, self.head_dim, device=self.device, requires_grad=True
            )
            * self.s_qk_scale
        )

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

        # --- Reshape K and V for Grouped Attention ---
        # Repeat K and V for each head within the group.
        k = k.repeat_interleave(
            self.heads_per_group, dim=1
        )  # (batch_size, nhead, seq_len, head_dim)
        v = v.repeat_interleave(
            self.heads_per_group, dim=1
        )  # (batch_size, nhead, seq_len, head_dim)

        effective_s_qk = self.s_qk * (self.s_qk_init / self.s_qk_scale)
        # based on results, it seems like the simpler version is better, kinda
        q = cosine_norm(q).transpose(1, 2) * effective_s_qk
        k = cosine_norm(k).transpose(1, 2) * effective_s_qk
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        # q = q.transpose(1, 2) * effective_s_qk
        # k = k.transpose(1, 2) * effective_s_qk
        # q = q.transpose(1, 2)
        # k = k.transpose(1, 2)

        # --- Attention Calculation ---
        a = torch.matmul(q, k.transpose(-2, -1)) * math.sqrt(self.head_dim)

        if mask is not None:
            a = a.masked_fill(mask == 1, -1e9)

        a = torch.softmax(a, dim=-1)
        a = self.dropout(a)

        # visualizing attention for debugging purposes
        # import matplotlib.pyplot as plt
        # a_weights = a.detach().cpu()
        # plt.imshow(torch.sum(torch.sum(a_weights, dim=1), dim=0))
        # plt.show()

        attn_output = torch.matmul(a, v)  # (batch_size, nhead, seq_len, head_dim)

        # --- Concatenate and Output Projection ---
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )
        output = self.dropout(self.o(attn_output))

        return output


class nGPT_SwiGLU_ff(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(nGPT_SwiGLU_ff, self).__init__()
        # d_ff is now the "intermediate" dimension.  The SwiGLU layer
        # will project to d_ff, and then the output projection will go back to d_model.
        self.d_model = d_model
        self.linear_in = nn.Linear(d_model, d_ff)
        self.linear_gate = nn.Linear(d_model, d_ff)
        self.linear_out = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        self.silu = nn.SiLU()

        # self.s_u_init = 1.
        # self.s_u_scale = 1.0 / math.sqrt(d_ff)
        self.s_u = nn.Parameter(torch.ones(d_ff))
        # self.s_v_init = 1.
        # self.s_v_scale = 1.0 / math.sqrt(d_ff)
        self.s_v = nn.Parameter(torch.ones(d_ff))

    def forward(self, x):
        u = self.linear_in(x) * self.s_u
        v = self.linear_gate(x) * self.s_v * math.sqrt(self.d_model)
        x = u * self.silu(v)
        x = self.linear_out(x)
        x = self.dropout(x)  # Apply dropout *after* the output projection
        return x


def cosine_norm(x, dim=-1):
    norm = torch.norm(x, p=2, dim=dim, keepdim=True).clamp(min=1e-6)
    return x / norm


def normalize_weights_and_enforce_positive_eigenvalues(model):
    with torch.no_grad():
        for layer in model.layers:
            layer.mha.q_linear.weight.data = cosine_norm(
                layer.mha.q_linear.weight.data, dim=-1
            )  # n_proj, n_embed
            layer.mha.k_linear.weight.data = cosine_norm(
                layer.mha.k_linear.weight.data, dim=-1
            )  # n_proj, n_embed
            layer.mha.v_linear.weight.data = cosine_norm(
                layer.mha.v_linear.weight.data, dim=-1
            )  # n_proj, n_embed
            layer.mha.o.weight.data = cosine_norm(
                layer.mha.o.weight.data, dim=-2
            )  # n_embed, n_proj

            layer.ff.linear_in.weight.data = cosine_norm(
                layer.ff.linear_in.weight.data, dim=-1
            )  # n_proj, n_embed
            layer.ff.linear_gate.weight.data = cosine_norm(
                layer.ff.linear_gate.weight.data, dim=-1
            )  # n_proj, n_embed
            layer.ff.linear_out.weight.data = cosine_norm(
                layer.ff.linear_out.weight.data, dim=-2
            )  # n_embed, n_proj

            layer.eigen_a.data = torch.abs(layer.eigen_a.data)
            layer.eigen_m.data = torch.abs(layer.eigen_m.data)

        model.input_embedding.embedding.weight.data = cosine_norm(
            model.input_embedding.embedding.weight.data, dim=-1
        )  # V, n_embed
        model.out.weight.data = cosine_norm(
            model.out.weight.data, dim=-1
        )  # V, n_embed??? TODO
    return model
