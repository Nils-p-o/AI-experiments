# tries to predict movements in the stock market
"""
further version TODO list:
embeddings for sequence category (stock price/return, general market, etc.)

try better architectures

"""

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from ..components import get_causal_mask

# for debug/visualization
import matplotlib.pyplot as plt


class Money_former_MLA_DINT_cog_attn_MTP(nn.Module):
    def __init__(self, args):  # , dtype=torch.float32):
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
        # self.input_features = (args.input_features - 15) * 3 + 15 # because time (15) and means, stds (3)
        self.input_features = args.input_features

        self.ticker_embedding = nn.Embedding(len(args.tickers), self.d_model)
        self.shared_value_input_dim = (self.d_model // sum(args.unique_inputs_ratio) * args.unique_inputs_ratio[1])
        self.shared_value_input = nn.Linear(
            self.input_features, self.shared_value_input_dim, bias=args.bias
        )

        self.unique_value_input_dim = (self.d_model // sum(args.unique_inputs_ratio) * args.unique_inputs_ratio[0])
        self.unique_value_input = nn.ModuleList(
            [
                nn.Linear(self.input_features, self.unique_value_input_dim, bias=args.bias)
                for _ in range(len(args.tickers))
            ]
        )
        self.norm = nn.RMSNorm(self.d_model) # TODO maybe one per MTP block?

        self.predict_distribution = args.predict_gaussian
        if self.predict_distribution:
            self.out = nn.Linear(
                self.d_model, len(args.indices_to_predict) * 2, bias=args.bias
            )  # decodes to mean and std
        else:
            self.out = nn.Linear(
                self.d_model, 5, bias=args.bias
            )  # decodes to target(s) features

        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

        # seperator/attention sink token
        self.seperator = nn.Embedding(1, self.d_model)

        self.MTP_blocks = nn.ModuleList(
            [MTP_money_former_block(args, depth + self.num_layers) for depth in range(max(args.indices_to_predict)-1)]
        )

    def forward(
        self, x, seperator=None, tickers=None
    ):  # seperator (batch_size, targets, 1); tickers (batch_size, targets, num_sequences)
        batch_size, targets, seq_len, num_sequences, _ = x.size()

        seperator = self.seperator(seperator).unsqueeze(2)
        seperator = seperator.repeat(1, 1, 1, num_sequences, 1)

        tickers = self.ticker_embedding(tickers).unsqueeze(2)
        tickers = tickers.repeat(1, 1, seq_len + 1, 1, 1)

        x = self.encode_inputs(x)

        x = torch.cat([seperator, x], dim=2)
        x = x + tickers

        x = x / math.sqrt(self.d_model)
        x = self.dropout(x)

        x = x.view(
            batch_size, targets, -1, self.d_model
        )  # (batch_size, (seq_len+1)*num_sequences, d_model)

        x_end = torch.empty_like(x)
        x_main = x[:, 0, :, :]
        freqs_cis = self.freqs_cis[0 : 0 + seq_len].to(x.device)
        for layer in self.layers:
            x_main = layer(x_main, freqs_cis)
        x_main = self.norm(x_main)  # (batch_size, seq_len, d_model)
        x_end[:, 0, :, :] = x_main
        for i in range(1, targets):
            x_end[:, i, :, :] = self.norm(self.MTP_blocks[i-1](x[:, i, :, :], x_end[:, i-1, :, :], freqs_cis))

        x_end = self.out(x_end)
        return x_end  # (batch_size, targets, seq_len, features) logits
    
    def encode_inputs(self, x):
        batch_size, targets, seq_len, num_sequences, _ = x.size()
        shared_temp_x = self.shared_value_input(x)
        unique_temp_x = torch.empty(
            batch_size, targets, seq_len, num_sequences, self.unique_value_input_dim
        ).to(x.device)
        for i in range(num_sequences):
            unique_temp_x[:, :, :, i, :] = self.unique_value_input[i](x[:, :, :, i, :])
        x = torch.cat([shared_temp_x, unique_temp_x], dim=-1)
        return x



class Money_former_block(nn.Module):
    def __init__(self, args, depth=0):
        super().__init__()
        self.nhead = args.nhead
        # self.mha = custom_MHA(args, depth)
        self.mha = GroupedMHA(args, depth)
        self.norm1 = nn.RMSNorm(args.d_model)
        self.norm2 = nn.RMSNorm(args.d_model)
        self.ff = SwiGLU_feed_forward(args)
        self.num_sequences = len(args.tickers)

    def forward(self, x, freqs_cis):
        batch_size, seq_len, _ = x.size()
        seq_len = seq_len // self.num_sequences
        mask = get_causal_mask(seq_len).to(x.device)
        mask = mask.repeat(
            batch_size, 2 * self.nhead, self.num_sequences, self.num_sequences
        )
        x = x + self.mha(
            self.norm1(x), mask, freqs_cis
        )  # (batch_size, seq_len, d_model)
        x = x + self.ff(self.norm2(x))
        return x
    
class MTP_money_former_block(nn.Module):
    def __init__(self, args, depth=0):
        super().__init__()
        self.norm = nn.RMSNorm(args.d_model)
        self.projection = nn.Linear(args.d_model*2, args.d_model)
        self.transformer_block = Money_former_block(args, depth)

    def forward(self, x_curr, x_prev, freqs_cis):
        x_curr = self.norm(x_curr) # (batch_size, seq_len, d_model)
        x_combined = torch.cat([x_curr, x_prev], dim=-1)
        x_combined = self.projection(x_combined)
        x_combined = self.transformer_block(x_combined, freqs_cis)
        return x_combined


class AttentionHeadGroup(nn.Module):
    """
    A module that encapsulates a group of attention heads with the same dimensions.
    This is the core building block for the GroupedMHA.
    """
    def __init__(self, args, nhead, head_dim, rope_dim, q_compression_dim, kv_compression_dim, depth, num_sequences):
        super().__init__()
        self.nhead = nhead
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.num_sequences = num_sequences
        
        # Projection layers specific to this group's dimensions
        self.kv_up = nn.Linear(kv_compression_dim, self.head_dim * self.nhead * 2, bias=args.bias)
        self.q_up = nn.Linear(q_compression_dim, (self.head_dim + self.rope_dim) * self.nhead, bias=args.bias)

        self.dropout = nn.Dropout(args.dropout)
        self.scaling = ((self.head_dim + self.rope_dim) // 2) ** -0.5

        # DINT parameters, sized for this group's head_dim
        self.lambda_init = lambda_init_fn(depth)
        lambda_param_shape = self.head_dim // 2
        self.lambda_k1 = nn.Parameter(torch.zeros(lambda_param_shape, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(lambda_param_shape, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q1 = nn.Parameter(torch.zeros(lambda_param_shape, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(lambda_param_shape, dtype=torch.float32).normal_(mean=0, std=0.1))

        # Normalization layer for the output of this head group
        self.norm = nn.RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, c_q, c_kv, q_rope_base, k_rope_base, mask, freqs_cis): # TODO maybe change how k_rrope is generated and works?
        batch_size, seq_len, _ = c_q.shape
        seq_per_stock = seq_len // self.num_sequences

        # --- K, V Projection ---
        kv = self.kv_up(c_kv).view(batch_size, seq_len, self.nhead, self.head_dim * 2)
        k, v = kv.split([self.head_dim, self.head_dim], dim=-1)
        k = k.reshape(batch_size, seq_len, self.nhead * 2, self.head_dim // 2)

        # --- Q Projection ---
        q_proj = self.q_up(c_q).view(batch_size, seq_len, self.nhead, self.head_dim + self.rope_dim)
        q, q_rope_group = q_proj.split([self.head_dim, self.rope_dim], dim=-1)
        q = q.reshape(batch_size, seq_len, self.nhead * 2, self.head_dim // 2)

        # --- RoPE Application ---
        # The base RoPE tensors are sliced and tiled for each group
        k_rope_group = k_rope_base[:, :, :self.rope_dim].view(batch_size, seq_len, 1, self.rope_dim).tile(1, 1, self.nhead, 1)

        q_rope_group = q_rope_group.reshape(batch_size, seq_len, self.nhead*2, self.rope_dim//2)
        k_rope_group = k_rope_group.reshape(batch_size, seq_len, self.nhead*2, self.rope_dim//2)
        
        # Reshape for RoPE application per sequence (decoupled)
        q_rope_reshaped = q_rope_group.view(batch_size, seq_per_stock, self.num_sequences, self.nhead*2, self.rope_dim//2).permute(0, 1, 3, 4, 2)
        k_rope_reshaped = k_rope_group.view(batch_size, seq_per_stock, self.num_sequences, self.nhead*2, self.rope_dim//2).permute(0, 1, 3, 4, 2)
        
        # Apply RoPE to each sequence individually
        for i in range(self.num_sequences):
            q_temp, k_temp = q_rope_reshaped[:, :, :, :, i], k_rope_reshaped[:, :, :, :, i]
            q_sep, q_temp_inner = q_temp.split([1, seq_per_stock - 1], dim=1)
            k_sep, k_temp_inner = k_temp.split([1, seq_per_stock - 1], dim=1)
            
            # Apply rotary embeddings to the inner part of the sequence
            q_temp_inner = apply_rotary_emb(q_temp_inner, freqs_cis=freqs_cis)
            k_temp_inner = apply_rotary_emb(k_temp_inner, freqs_cis=freqs_cis)
            
            q_rope_reshaped[:, :, :, :, i] = torch.cat([q_sep, q_temp_inner], dim=1)
            k_rope_reshaped[:, :, :, :, i] = torch.cat([k_sep, k_temp_inner], dim=1)

        q_rope_final = q_rope_reshaped.permute(0, 1, 4, 2, 3).reshape(batch_size, seq_len, self.nhead*2, self.rope_dim//2)
        k_rope_final = k_rope_reshaped.permute(0, 1, 4, 2, 3).reshape(batch_size, seq_len, self.nhead*2, self.rope_dim//2)

        q_final = torch.cat([q, q_rope_final], dim=-1).transpose(1, 2)
        k_final = torch.cat([k, k_rope_final], dim=-1).transpose(1, 2)
        v_final = v.transpose(1, 2)

        # --- Attention Calculation ---
        a = torch.matmul(q_final, k_final.transpose(-2, -1)) * self.scaling
        
        abs_a = torch.abs(a)
        if mask is not None:
            # We only need the mask relevant to this group's heads
            group_mask = mask[:, :self.nhead*2, :, :]
            abs_a = abs_a.masked_fill(group_mask == 1, -1e9)
        a = torch.sign(a) * torch.softmax(abs_a, dim=-1)

        # --- DINT Logic ---
        lambda_1 = torch.exp(torch.sum(self.lambda_k1 * self.lambda_q1, dim=-1))
        lambda_2 = torch.exp(torch.sum(self.lambda_k2 * self.lambda_q2, dim=-1))
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        
        a = a.view(batch_size, self.nhead, 2, seq_len, seq_len)
        g_vec = torch.mean(a[:, :, 0, :, :], dim=-2, keepdim=True).repeat(1, 1, seq_len, 1)
        a_final = a[:, :, 0, :, :] - lambda_full * a[:, :, 1, :, :] + g_vec * lambda_full
        
        # Apply mask again and dropout
        if mask is not None:
             a_final = a_final.masked_fill(mask[:, :self.nhead, :, :] == 1, 0)
        a_final = self.dropout(a_final)

        # --- Output Calculation ---
        attn_output = torch.matmul(a_final, v_final)
        attn_output = self.norm(attn_output.transpose(1, 2)) # Transpose for norm: (B, S, H, D)

        # ported sus amogus
        attn_output = attn_output.transpose(1, 2) # (B, H, S, D)
        attn_output = attn_output.reshape(batch_size, seq_len, self.nhead, self.head_dim)

        attn_output = attn_output.contiguous().view(batch_size, seq_len, self.nhead * self.head_dim)
        
        return attn_output


class GroupedMHA(nn.Module):
    """
    A Multi-Head Attention module that supports groups of heads with different dimensions.
    """
    def __init__(self, args, depth=0):
        """
        Args:
            args: Your standard arguments object.
            head_configs (List[Tuple[int, int, int]]): A list of tuples, where each
                tuple defines a head group as (num_heads, head_dim, rope_dim).
                Example: [(3, 32, 8), (2, 16, 4)] for 3 heads of dim 32 and 2 of dim 16.
            depth (int): The depth of the layer for DINT initialization.
        """
        super().__init__()
        self.d_model = args.d_model
        
        # Down-projection layers are shared
        self.kv_compression_dim = args.kv_compression_dim
        self.q_compression_dim = args.q_compression_dim
        
        # The RoPE dimension for the shared projection must be the *maximum* required
        self.max_rope_dim = max(c[2] for c in args.head_configs)
        
        self.kv_down = nn.Linear(self.d_model, self.kv_compression_dim + self.max_rope_dim, bias=args.bias)
        self.q_down = nn.Linear(self.d_model, self.q_compression_dim, bias=args.bias)

        self.kv_norm = nn.RMSNorm(self.kv_compression_dim)
        self.q_norm = nn.RMSNorm(self.q_compression_dim)
        
        self.num_sequences = len(args.tickers)

        # --- Create the list of attention head groups ---
        self.head_groups = nn.ModuleList()
        total_output_dim = 0
        for nhead, head_dim, rope_dim in args.head_configs:
            self.head_groups.append(
                AttentionHeadGroup(args, nhead, head_dim, rope_dim, self.q_compression_dim, self.kv_compression_dim, depth, self.num_sequences)
            )
            total_output_dim += nhead * head_dim
        
        # Final output projection layer
        self.o = nn.Linear(total_output_dim, self.d_model, bias=args.bias)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x, mask=None, freqs_cis=None):
        # 1. Shared initial down-projection
        c_kv_proj = self.kv_down(x)
        c_kv, k_rope_base = c_kv_proj.split([self.kv_compression_dim, self.max_rope_dim], dim=-1)
        c_kv = self.kv_norm(c_kv)
        
        c_q = self.q_down(x)
        c_q = self.q_norm(c_q)
        
        # 2. Process each head group independently
        all_group_outputs = []
        for group in self.head_groups:
            # We pass the same compressed Q/KV and base RoPE tensors to each group.
            # The group is responsible for using what it needs.
            group_output = group(c_q, c_kv, None, k_rope_base, mask, freqs_cis)
            all_group_outputs.append(group_output)
            
        # 3. Concatenate the outputs from all groups
        attn_output = torch.cat(all_group_outputs, dim=-1)
        
        # 4. Final output projection
        output = self.dropout(self.o(attn_output))
        
        return output


class SwiGLU_feed_forward(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.linear_in_gate = nn.Linear(args.d_model, args.d_ff * 2, bias=args.bias)
        self.linear_out = nn.Linear(args.d_ff, args.d_model, bias=args.bias)
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

    dim = args.qk_rope_dim // 2 # TODO keep correct (//2 because DINT)
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


def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)
