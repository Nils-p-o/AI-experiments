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
        self.input_features = args.input_features
        self.scaling = self.d_model ** -0.5

        self.ticker_embedding = nn.Embedding(len(args.tickers), self.d_model)
        self.shared_value_input_dim = (self.d_model // sum(args.unique_inputs_ratio) * args.unique_inputs_ratio[1])
        self.shared_value_input = nn.Linear(
            self.input_features, self.shared_value_input_dim, bias=args.bias
        )

        self.unique_inputs = args.unique_inputs_ratio[0] > 0
        if self.unique_inputs:
            self.unique_value_input_dim = (self.d_model // sum(args.unique_inputs_ratio) * args.unique_inputs_ratio[0])
            self.unique_value_input = nn.ModuleList(
                [
                    nn.Linear(self.input_features, self.unique_value_input_dim, bias=args.bias)
                    for _ in range(len(args.tickers))
                ]
            )
        self.norm = nn.RMSNorm(self.d_model) # TODO maybe one per MTP block?

        match args.prediction_type:
            case "gaussian":
                self.out = nn.Linear(
                    self.d_model, 5 * 2, bias=args.bias
                )  # decodes to mean and std
            case "regression":
                self.out = nn.Linear(
                    self.d_model, 5, bias=args.bias
                )  # decodes to target(s) features
            case "classification":
                self.out = nn.Linear(
                    self.d_model, 5 * args.num_classes, bias=args.bias
                )  # decodes to target(s) features

        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

        # seperator/attention sink token
        # self.seperator = nn.Embedding(1, self.d_model)
        self.seperator = nn.Parameter(torch.zeros(self.d_model, dtype=torch.float32).normal_(mean=0.0, std=1.0))
        self.use_global_seperator = args.use_global_seperator

        self.MTP_blocks = nn.ModuleList(
            [MTP_money_former_block(args, depth + self.num_layers) for depth in range(max(args.indices_to_predict)-1)]
        )

    def forward(
        self, x, tickers=None
    ):  # seperator (batch_size, targets, 1); tickers (batch_size, targets, num_sequences)
        batch_size, targets, seq_len, num_sequences, _ = x.size()

        x = self.encode_inputs(x)

        if self.use_global_seperator:
            x = x.permute(0, 1, 3, 2, 4).contiguous()
            x = x.view(batch_size, targets, num_sequences * seq_len, self.d_model)
            seperator = self.seperator.view(1, 1, 1, self.d_model)
            seperator = seperator.expand(batch_size, targets, 1, -1)
            x = torch.cat([seperator, x], dim=2)
            
            ticker_embs = self.ticker_embedding(tickers).unsqueeze(3)
            ticker_embs = ticker_embs.expand(-1, -1, -1, seq_len, -1)
            ticker_embs = ticker_embs.contiguous().view(batch_size, targets, num_sequences * seq_len, self.d_model)

            sep_ticker_emb = torch.zeros(batch_size, targets, 1, self.d_model, dtype=torch.float32, device=x.device)
            ticker_embs = torch.cat([sep_ticker_emb, ticker_embs], dim=2)

            x = (x + ticker_embs) * self.scaling
        else:
            seperator = self.seperator.view(1, 1, 1, 1, self.d_model)
            seperator = seperator.expand(batch_size, targets, 1, num_sequences, -1)

            tickers = self.ticker_embedding(tickers).unsqueeze(2)
            tickers = tickers.repeat(1, 1, seq_len + 1, 1, 1)


            x = torch.cat([seperator, x], dim=2)
            x = (x + tickers) * self.scaling

            x = x.permute(0, 1, 3, 2, 4).contiguous() # fix to prevent data mixing

            x = x.view(
                batch_size, targets, -1, self.d_model
            )  # (batch_size, targets, (seq_len+1)*num_sequences, d_model)

        x = self.dropout(x)

        x_end = []
        x_main = x[:, 0, :, :]
        freqs_cis = self.freqs_cis[0 : 0 + seq_len].to(x.device)
        for layer in self.layers:
            x_main = layer(x_main, freqs_cis)
        x_main = self.norm(x_main)  # (batch_size, seq_len, d_model)

        x_as_list = [t.squeeze(1) for t in torch.split(x[:,1:], 1, dim=1)]
        x_end.append(x_main)
        for MTP_block in self.MTP_blocks:
            x_end.append(self.norm(MTP_block(x_as_list.pop(0), x_end[-1], freqs_cis)))
        x_end = torch.stack(x_end, dim=1)

        x_end = self.out(x_end)
        return x_end  # (batch_size, targets, seq_len, features) logits
    
    def encode_inputs(self, x):
        # batch_size, targets, seq_len, num_sequences, _ = x.size()
        shared_temp_x = self.shared_value_input(x)
        if self.unique_inputs:
            unique_temp_x = []
            x_as_list = [t.squeeze(3) for t in torch.split(x, 1, dim=3)]
            for unique_input in self.unique_value_input:
                unique_temp_x.append(unique_input(x_as_list.pop(0)))
            unique_temp_x = torch.stack(unique_temp_x, dim=3)
            x = torch.cat([shared_temp_x, unique_temp_x], dim=-1)
        else:
            x = shared_temp_x
        return x



class Money_former_block(nn.Module):
    def __init__(self, args, depth=0):
        super().__init__()
        self.nhead = args.nhead
        self.mha = custom_MHA(args, depth)
        self.norm1 = nn.RMSNorm(args.d_model)
        self.norm2 = nn.RMSNorm(args.d_model)
        self.ff = SwiGLU_feed_forward(args)
        self.num_sequences = len(args.tickers)
        if args.use_global_seperator:
            temp_mask = get_causal_mask(args.seq_len)
            temp_mask = temp_mask.repeat(1, 1, self.num_sequences, self.num_sequences)# TODO make first column zero, not one, 1 = masked
            self.mask = torch.zeros(1, 1, args.seq_len*self.num_sequences+1, args.seq_len*self.num_sequences+1)
            self.mask[:,:,:,0] = 1.0
            self.mask[:,:,1:,1:] = temp_mask

        else:
            self.mask = get_causal_mask(args.seq_len+1)
            self.mask = self.mask.repeat(1, 1, self.num_sequences, self.num_sequences)

    def forward(self, x, freqs_cis):
        batch_size, seq_len, _ = x.size()
        mask = self.mask.expand(batch_size, 2 * self.nhead, -1, -1)
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


class custom_MHA(nn.Module): # a mix of MLA and DINT
    def __init__(self, args, depth=0):
        super().__init__()
        self.d_model = args.d_model
        self.nhead = args.nhead
        self.head_dim = args.head_dim  # technically half of this, but it gets confusing cuz its still one head, so...
        # assert self.head_dim * self.nhead == self.d_model

        self.kv_compression_dim = args.kv_compression_dim
        self.q_compression_dim = args.q_compression_dim
        self.rope_dim = args.qk_rope_dim

        self.kv_down = nn.Linear(self.d_model, self.kv_compression_dim + self.rope_dim, bias=args.bias)
        self.q_down = nn.Linear(self.d_model, self.q_compression_dim, bias=args.bias)

        self.kv_up = nn.Linear(self.kv_compression_dim, self.head_dim * self.nhead * 2, bias=args.bias)
        self.q_up = nn.Linear(self.q_compression_dim, (self.head_dim + self.rope_dim) * self.nhead, bias=args.bias)

        self.kv_norm = nn.RMSNorm(self.kv_compression_dim)
        self.q_norm = nn.RMSNorm(self.q_compression_dim)

        self.o = nn.Linear(self.nhead * self.head_dim, self.d_model, bias=args.bias)
        self.dropout = nn.Dropout(args.dropout)

        self.scaling = ((self.head_dim + self.rope_dim) // 2) ** -0.5

        # DINT part
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
        # self.head_norms = nn.ModuleList(
        #     [
        #         nn.RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)
        #         for _ in range(self.nhead)
        #     ]
        # )

        self.num_sequences = len(args.tickers)
        self.use_global_seperator = args.use_global_seperator

    def forward(self, x, mask=None, freqs_cis=None):
        batch_size, seq_len, _ = x.shape


        c_kv = self.kv_down(x)
        c_kv, k_rope = c_kv.split([self.kv_compression_dim, self.rope_dim], dim=-1)

        kv = self.kv_up(self.kv_norm(c_kv)).view(
            batch_size,
            seq_len,
            self.nhead,
            self.head_dim*2
        )
        k, v = kv.split([self.head_dim, self.head_dim], dim=-1)
        k = k.reshape(batch_size, seq_len, self.nhead*2, self.head_dim//2)

        q = self.q_up(self.q_norm(self.q_down(x))).view(
            batch_size,
            seq_len,
            self.nhead,
            self.head_dim + self.rope_dim
        )
        q, q_rope = q.split([self.head_dim, self.rope_dim], dim=-1)
        q = q.reshape(batch_size, seq_len, self.nhead*2, self.head_dim//2)

        k_rope = k_rope.view(batch_size, seq_len, 1, self.rope_dim)
        k_rope = k_rope.tile(1, 1, self.nhead, 1)

        q_rope = q_rope.reshape(batch_size, seq_len, self.nhead*2, self.rope_dim//2)
        k_rope = k_rope.reshape(batch_size, seq_len, self.nhead*2, self.rope_dim//2)

        # # --- RoPE (decoupled) ---
        if self.use_global_seperator:
            seq_per_stock = (seq_len - 1) // self.num_sequences

            q_sep, q_to_rotate = q_rope.split([1, seq_len-1], dim=1)
            k_sep, k_to_rotate = k_rope.split([1, seq_len-1], dim=1)

            q_to_rotate = q_to_rotate.view(batch_size, self.num_sequences, seq_per_stock, self.nhead*2, self.rope_dim//2)
            k_to_rotate = k_to_rotate.view(batch_size, self.num_sequences, seq_per_stock, self.nhead*2, self.rope_dim//2)

            q_to_rotate = apply_rotary_emb_multiple_sequences(q_to_rotate, freqs_cis=freqs_cis)
            k_to_rotate = apply_rotary_emb_multiple_sequences(k_to_rotate, freqs_cis=freqs_cis)

            q_rope = torch.cat([q_sep, q_to_rotate.reshape(batch_size, seq_len-1, self.nhead*2, self.rope_dim//2)], dim=1)
            k_rope = torch.cat([k_sep, k_to_rotate.reshape(batch_size, seq_len-1, self.nhead*2, self.rope_dim//2)], dim=1)

        else:
            seq_per_stock = seq_len // self.num_sequences

            q_rope = q_rope.view(batch_size, self.num_sequences, seq_per_stock, self.nhead*2, self.rope_dim//2)
            k_rope = k_rope.view(batch_size, self.num_sequences, seq_per_stock, self.nhead*2, self.rope_dim//2)
            # #q/k_rope (batch_size, num_sequences, seq_per_stock, nhead*2, rope_dim//2)
            q_sep, q_to_rotate = q_rope.split([1, seq_per_stock - 1], dim=2)
            k_sep, k_to_rotate = k_rope.split([1, seq_per_stock - 1], dim=2)

            q_to_rotate = apply_rotary_emb_multiple_sequences(q_to_rotate, freqs_cis=freqs_cis)
            k_to_rotate = apply_rotary_emb_multiple_sequences(k_to_rotate, freqs_cis=freqs_cis)

            q_rope = torch.cat([q_sep, q_to_rotate], dim=2)
            k_rope = torch.cat([k_sep, k_to_rotate], dim=2)

            q_rope = q_rope.view(batch_size, seq_len, self.nhead*2, self.rope_dim//2)
            k_rope = k_rope.view(batch_size, seq_len, self.nhead*2, self.rope_dim//2)

        q = torch.cat([q, q_rope], dim=-1).transpose(1, 2)  # (batch_size, 2*nhead, seq_len, (head_dim+rope_dim)//2)
        k = torch.cat([k, k_rope], dim=-1).transpose(1, 2)  # (batch_size, 2*nhead, seq_len, (head_dim+rope_dim)//2)

        v = v.transpose(1, 2)  # (batch_size, nhead, seq_len, head_dim)

        # --- Attention Calculation ---
        a = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        # (batch_size, 2*nhead, seq_len, seq_len)

        # cog_attn
        abs_a = torch.abs(a)
        if mask is not None:
            abs_a = abs_a.masked_fill(mask == 1, -1e9)
        a = torch.sign(a) * torch.softmax(abs_a, dim=-1)

        lambda_1 = torch.exp(torch.sum(self.lambda_k1 * self.lambda_q1, dim=-1))
        lambda_2 = torch.exp(torch.sum(self.lambda_k2 * self.lambda_q2, dim=-1))
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        a = a.view(batch_size, self.nhead, 2, seq_len, seq_len)
        G_vec = torch.mean(a[:, :, 0, :, :], dim=-2, keepdim=True).repeat(1, 1, seq_len, 1)
        # assert a[:, :, 0, :, :].shape == a3.shape
        # assert torch.isclose((lambda_full * a[:, :, 1, :, :]).sum(dim=-1),(a3 * lambda_full).sum(dim=-1),rtol=1e-3,atol=1e-5).all() == True
        a = a[:, :, 0, :, :] - lambda_full * a[:, :, 1, :, :] + G_vec * lambda_full
        # a = a * (1/ (1 - lambda_full)) 
        a = a.masked_fill(mask[:, : self.nhead, :, :] == 1, 0)
        a = self.dropout(a)

        attn_output = torch.matmul(a, v)  # (batch_size, nhead, seq_len, head_dim)
        # normed_heads = []
        # for i in range(self.nhead):
        #     head = attn_output[:, :, i, :]
        #     head = self.head_norms[i](head)
        #     normed_heads.append(head)
        # attn_output = torch.stack(normed_heads, dim=2)  # (batch_size, seq_len, nhead, head_dim)
        attn_output = self.norm(attn_output.transpose(1,2))  # (batch_size, seq_len, nhead, head_dim)
        # attn_output = attn_output.transpose(1, 2)  # (batch_size, nhead, seq_len, head_dim)
        # attn_output = attn_output.reshape( # TODO keep in mind sus amogus
        #     batch_size, seq_len, self.nhead, self.head_dim
        # )
        # --- Concatenate and Output Projection ---
        attn_output = (
            attn_output.contiguous()
            .view(batch_size, seq_len, self.nhead * self.head_dim)
        )
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


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor: # expects x to be of shape (bs, seq_len, head, rope_dim)
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)

def apply_rotary_emb_multiple_sequences(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor: # expects x to be of shape (bs, num_sequences, seq_len, head, rope_dim)
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, 1, x.size(2), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(4)
    return y.to(dtype)

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)
