import torch
import torch.nn as nn
from nGPT import  cosine_norm, get_causal_mask, nGPT_SwiGLU_ff
from DINT import lambda_init_fn
from components import input_embedding
import math
from LLaMa import RoPE

# maybe done
# uses v1 of DINT
class DINT_nGPT(torch.nn.Module):
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
        super(DINT_nGPT, self).__init__()
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
            [DINT_nGPT_block(d_model, nhead, d_ff, dropout, groups=groups, depth=depth) for depth in range(num_layers)]
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
    
class DINT_nGPT_block(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout=0.1,groups=4, depth=0):
        super(DINT_nGPT_block, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.d_model = d_model
        self.nhead = nhead
        self.d_ff = d_ff
        self.mha = DINT_nGPT_GQA(d_model, nhead, depth)
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
    
class DINT_nGPT_GQA(nn.Module): # TODO make sure this makes sense
    def __init__(self, d_model, nhead, groups=4, dropout=0.1, depth=0):
        super(DINT_nGPT_GQA, self).__init__()
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
        self.k_linear = nn.Linear(d_model, self.d_model // self.heads_per_group)
        self.v_linear = nn.Linear(d_model, self.d_model // self.heads_per_group)

        # s_qk might just need one and apply it to both
        self.s_qk1_init = 1.0
        self.s_qk1_scale = 1.0 / math.sqrt(self.head_dim)
        self.s_qk1 = nn.Parameter(
            torch.ones(
                self.nhead, self.head_dim//2, device=self.device, requires_grad=True
            )
            * self.s_qk1_scale
        )

        self.s_qk2_init = 1.0
        self.s_qk2_scale = 1.0 / math.sqrt(self.head_dim)
        self.s_qk2 = nn.Parameter(
            torch.ones(
                self.nhead, self.head_dim//2, device=self.device, requires_grad=True
            )
            * self.s_qk2_scale
        )

        self.scaling = 1.0 / math.sqrt(self.head_dim) # TODO //2?

        self.o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.rotary_emb = RoPE(self.head_dim)

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

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # --- Q, K, V Projections ---
        # DINT part
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

        # --- Reshape K and V for Grouped Attention ---
        # DINT part
        q = q.transpose(1, 2)  # (batch_size, 2*nhead, seq_len, head_dim/2)
        # --- Reshape K and V for Grouped Attention ---
        # Repeat K and V for each head within the group.
        k = k.transpose(1, 2).repeat_interleave(
            self.heads_per_group, dim=1
        )  # (batch_size, 2*nhead, seq_len, head_dim/2)
        v = v.transpose(1, 2).repeat_interleave(
            self.heads_per_group, dim=1
        )  # (batch_size, nhead, seq_len, head_dim)

        effective_s_qk1 = self.s_qk1 * (self.s_qk1_init / self.s_qk1_scale)
        effective_s_qk2 = self.s_qk2 * (self.s_qk2_init / self.s_qk2_scale)
        # maybe s_qk should be "half" of head size and be the same between qk1 and qk2
        # --- nGPT part --- # TODO check if the more complicated one is even worth it
        q1 = cosine_norm(q[:,:self.nhead,:,:]).transpose(1,2) * effective_s_qk1
        k1 = cosine_norm(k[:,:self.nhead,:,:]).transpose(1,2) * effective_s_qk1
        q2 = cosine_norm(q[:,self.nhead:,:,:]).transpose(1,2) * effective_s_qk2
        k2 = cosine_norm(k[:,self.nhead:,:,:]).transpose(1,2) * effective_s_qk2
        q = torch.cat((q1, q2), dim=1).transpose(1,2)
        k = torch.cat((k1, k2), dim=1).transpose(1,2)
        # q = cosine_norm(q).transpose(1, 2) * effective_s_qk
        # k = cosine_norm(k).transpose(1, 2) * effective_s_qk
        # q = q.transpose(1, 2)
        # k = k.transpose(1, 2)
        # q = q.transpose(1, 2) * effective_s_qk1
        # k = k.transpose(1, 2) * effective_s_qk1
        # q = q.transpose(1, 2)
        # k = k.transpose(1, 2)

        # --- Attention Calculation ---
        # common part
        a = torch.matmul(q, k.transpose(-2, -1)) * self.scaling  # (batch_size, 2*nhead, seq_len, seq_len)

        if mask is not None:
            a = a.masked_fill(mask == 1, -1e9)

        a = torch.softmax(a, dim=-1)
        a = self.dropout(a)

        # DINT part
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
        # a3 = a3 * a3_scaler
        # a3 = a3.repeat_interleave(seq_len, dim=-2)
        # a3 = a3.masked_fill(mask == 1, 0)

        a = a[:, :, 0, :, :] - lambda_full * a[:, :, 1, :, :] + a3 * lambda_full



        attn_output = torch.matmul(a, v)  # (batch_size, nhead, seq_len, head_dim)

        # --- Concatenate and Output Projection ---
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )

        # TODO maybe add norm??
        output = self.dropout(self.o(attn_output))

        return output