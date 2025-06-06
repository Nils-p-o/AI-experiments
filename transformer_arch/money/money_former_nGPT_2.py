# tries to predict movements in the stock market

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from ..components import get_causal_mask  # TODO at v2

# for debug/visualization
import matplotlib.pyplot as plt

# TODO consult with paper and go over everything to make sure it is implemented correctly
class Money_former_nGPT(nn.Module):
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
        self.shared_value_input = nn.Linear(
            self.input_features, (self.d_model // 4 * 3), bias=False
        )
        self.unique_value_inputs = nn.ModuleList(
            [
                nn.Linear(self.input_features, (self.d_model // 4), bias=False)
                for _ in range(len(args.tickers))
            ]
        )
        # self.norm = nn.RMSNorm(self.d_model)

        self.predict_distribution = args.predict_gaussian
        if self.predict_distribution:
            self.out = nn.Linear(
                self.d_model, len(args.indices_to_predict) * 2, bias=False
            )  # decodes to mean and std
        else:
            self.out = nn.Linear(
                self.d_model, len(args.indices_to_predict), bias=False
            )  # decodes to target(s)

        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

        # seperator/attention sink token
        self.seperator = nn.Embedding(1, self.d_model)

        self.s_z_init = 1.0
        self.s_z_scale = args.d_model ** -0.5
        self.s_z = nn.Parameter(torch.ones(len(args.indices_to_predict)) * self.s_z_scale)

    def forward(self, x, seperator=None, tickers=None):
        batch_size, seq_len, num_sequences, _ = x.size()

        seperator = self.seperator(seperator).unsqueeze(1)
        seperator = seperator.repeat(1, 1, num_sequences, 1)

        tickers = self.ticker_embedding(tickers).unsqueeze(1)
        tickers = tickers.repeat(1, seq_len + 1, 1, 1)
        shared_temp_x = self.shared_value_input(x)
        unique_temp_x = torch.zeros(
            batch_size, seq_len, num_sequences, self.d_model // 4, device=x.device
        )
        for i in range(num_sequences):
            unique_temp_x[:, :, i, :] = self.unique_value_inputs[i](x[:, :, i, :])
        x = torch.cat([shared_temp_x, unique_temp_x], dim=-1)

        x = torch.cat([seperator, x], dim=1)
        x = x + tickers  # TODO maybe concat and project?

        x = cosine_norm(x) # / math.sqrt(self.d_model)
        x = self.dropout(x)

        x = x.view(
            batch_size, -1, self.d_model
        )
        freqs_cis = self.freqs_cis[0 : 0 + seq_len]
        for layer in self.layers:
            x = layer(x, freqs_cis)
        # x = self.norm(x)  # (batch_size, seq_len, d_model)
        x = self.out(x)
        if self.predict_distribution:
            x = x.view(
                x.size(0), x.size(1), -1, 2
            )  # (batch_size, seq_len, points in future, (mean and std) or target)
            return x
        effective_s_z = self.s_z * (self.s_z_init / self.s_z_scale)
        x = x * effective_s_z
        return x  # (batch_size, seq_len, targets) logits


class Money_former_block(nn.Module): # done?
    def __init__(self, args):
        super().__init__()
        self.nhead = args.nhead
        self.mha = custom_MHA(args)
        self.ff = custom_SwiGLU_feed_forward(args)
        self.num_sequences = len(args.tickers)

        self.eigen_a_init = 1.0
        self.eigen_a_scale = args.d_model ** -0.5
        self.eigen_a = nn.Parameter(torch.ones(args.d_model) * self.eigen_a_scale)

        self.eigen_m_init = 1.0
        self.eigen_m_scale = args.d_model ** -0.5
        self.eigen_m = nn.Parameter(torch.ones(args.d_model) * self.eigen_m_scale)

    def forward(self, x, freqs_cis):
        batch_size, seq_len, _ = x.size()
        seq_len = seq_len // self.num_sequences
        mask = get_causal_mask(seq_len)
        mask = mask.repeat(batch_size, self.nhead, self.num_sequences, self.num_sequences)

        x_a = cosine_norm(self.mha(x, mask, freqs_cis))
        x = cosine_norm(x + self.eigen_a * (self.eigen_a_init / self.eigen_a_scale) * (x_a - x))

        x_m = cosine_norm(self.ff(x))
        x = cosine_norm(x + self.eigen_m * (self.eigen_m_init / self.eigen_m_scale) * (x_m - x))
        return x


class custom_MHA(nn.Module): # nGPT
    def __init__(self, args):
        super().__init__()
        self.d_model = args.d_model
        self.nhead = args.nhead
        self.head_dim = args.head_dim

        self.q_linear = nn.Linear(self.d_model, self.nhead * self.head_dim, bias=False)
        self.k_linear = nn.Linear(self.d_model, self.nhead * self.head_dim, bias=False)
        self.v_linear = nn.Linear(self.d_model, self.nhead * self.head_dim, bias=False)

        self.o = nn.Linear(self.nhead * self.head_dim, self.d_model, bias=False)
        self.dropout = nn.Dropout(args.dropout)
        self.scaling = self.head_dim ** 0.5
        self.num_sequences = len(args.tickers)
        
        self.s_qk_init = 1.0
        self.s_qk_scale = self.head_dim ** -0.5
        self.s_qk = nn.Parameter(
            torch.ones(self.nhead, self.head_dim) * self.s_qk_scale
        )
        # self.norm = nn.RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, x, mask=None, freqs_cis=None):
        batch_size, seq_len, _ = x.shape
        seq_per_stock = seq_len // self.num_sequences

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

        effective_s_qk = self.s_qk * (self.s_qk_init * self.s_qk_scale)
        q = cosine_norm(q, dim=-1).transpose(1, 2) * effective_s_qk
        k = cosine_norm(k, dim=-1).transpose(1, 2) * effective_s_qk
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        a = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        # Apply mask if provided
        if mask is not None:
            a = a.masked_fill(mask == 1, -float("inf"))

        a = torch.softmax(a, dim=-1)
        a = self.dropout(a)

        # Apply attention to values
        attn_output = torch.matmul(a, v)

        # Concatenate heads and project
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.nhead * self.head_dim)
        )

        return self.dropout(self.o(attn_output))


class custom_SwiGLU_feed_forward(nn.Module): # because nGPT # done? # TODO maybe s_u and s_v can be initialized as 1?
    def __init__(self, args):
        super().__init__()
        self.linear_in_gate = nn.Linear(args.d_model, args.d_ff * 2, bias=False)
        self.linear_out = nn.Linear(args.d_ff, args.d_model, bias=False)
        self.dropout = nn.Dropout(args.dropout)
        self.d_model = args.d_model
        self.scale = self.d_model ** 0.5

        self.s_u_init = 1.0
        self.s_u_scale = 1.0 # self.d_model ** -0.5
        self.s_u = nn.Parameter(torch.ones(args.d_ff) * self.s_u_scale)

        self.s_v_init = 1.0
        self.s_v_scale = 1.0 #self.d_model ** -0.5
        self.s_v = nn.Parameter(torch.ones(args.d_ff) * self.s_v_scale)

    def forward(self, x):
        u, v = self.linear_in_gate(x).chunk(2, dim=-1)
        effective_s_u = self.s_u * (self.s_u_init * self.s_u_scale)
        effective_s_v = self.s_v * (self.s_v_init * self.s_v_scale)
        u = u * effective_s_u
        v = v * effective_s_v * self.scale
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

def cosine_norm(x, dim=-1):
    norm = torch.norm(x, p=2, dim=dim, keepdim=True).clamp(min=1e-6)
    return x / norm
def normalize_weights_and_enforce_positive_eigenvalues(model): # TODO be suspicious maybe linear layers need to be normed along other dim
    # for linear layers, (output_dim, input_dim)
    # for embeddings, (vocab_size, embedding_dim)/(input_dim, output_dim)
    with torch.no_grad():
        for layer in model.layers:
            layer.mha.q_linear.weight.data = cosine_norm(
                layer.mha.q_linear.weight.data, dim=-1
            )
            layer.mha.k_linear.weight.data = cosine_norm(
                layer.mha.k_linear.weight.data, dim=-1
            )
            layer.mha.v_linear.weight.data = cosine_norm(
                layer.mha.v_linear.weight.data, dim=-1
            )
            layer.mha.o.weight.data = cosine_norm(
                layer.mha.o.weight.data, dim=-1
            )

            layer.ff.linear_in_gate.weight.data = cosine_norm(
                layer.ff.linear_in_gate.weight.data, dim=-1
            ) 
            layer.ff.linear_out.weight.data = cosine_norm(
                layer.ff.linear_out.weight.data, dim=-1
            )

            layer.eigen_a.data = torch.abs(layer.eigen_a.data)
            layer.eigen_m.data = torch.abs(layer.eigen_m.data)

        model.shared_value_input.weight.data = cosine_norm(
            model.shared_value_input.weight.data, dim=-1
        )
        for input in model.unique_value_inputs:
            input.weight.data = cosine_norm(
                input.weight.data, dim=-1
            )
        model.out.weight.data = cosine_norm(
            model.out.weight.data, dim=-1
        )

        model.ticker_embedding.weight.data = cosine_norm(
            model.ticker_embedding.weight.data, dim=-1
        )
        model.seperator.weight.data = cosine_norm(
            model.seperator.weight.data, dim=-1
        )
    return model