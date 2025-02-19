from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch
import pytorch_lightning as pl

import re
from typing import List, Optional, Union
from sentencepiece import SentencePieceProcessor

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class OrthoGrad(torch.optim.Optimizer):
    def __init__(
        self, params, base_optimizer_cls=torch.optim.Adam, **base_optimizer_args
    ):
        """
        A wrapper optimizer that projects gradients to be orthogonal
        to the current parameters before performing an update.

        Args:
            params (iterable): Iterable of parameters to optimize.
            base_optimizer_cls (Optimizer class): The base optimizer class
                (e.g., torch.optim.SGD, torch.optim.AdamW).
            **base_optimizer_args: Arguments for the base optimizer.
                For example, lr=1e-3, weight_decay=1e-2, etc.
        """
        # Minimal defaults for OrthoGrad itself (nothing special needed).
        defaults = {}
        super().__init__(params, defaults)

        # Create the wrapped/base optimizer using *our* param_groups.
        self.base_optimizer = base_optimizer_cls(
            self.param_groups, **base_optimizer_args
        )

    @staticmethod
    def _orthogonalize_gradients(params):
        """
        Projects the gradient g to be orthogonal to the current weights w.

        g_orth = g - ( (w·g)/(w·w + eps) ) * w

        And then re-scales g_orth to have the same norm as g.
        """
        with torch.no_grad():
            for p in params:
                if p.grad is not None:
                    w = p.view(-1)
                    g = p.grad.view(-1)

                    w_norm_sq = torch.dot(w, w) + 1e-30
                    proj = torch.dot(w, g) / w_norm_sq
                    g_orth = g - proj * w

                    g_norm = g.norm(2)
                    g_orth_norm = g_orth.norm(2) + 1e-30
                    g_orth_scaled = g_orth * (g_norm / g_orth_norm)

                    p.grad.copy_(g_orth_scaled.view_as(p.grad))

    def step(self, closure=None):
        for group in self.param_groups:
            self._orthogonalize_gradients(group["params"])

        return self.base_optimizer.step(closure)


def stablemax(x, epsilon=1e-30, dim=-1):
    return torch.where(x < 0, 1 / (1 - x + epsilon), x + 1)


def log_stablemax(x, dim=-1):
    s_x = stablemax(x)
    return torch.log(s_x / torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, reduction="mean"):
    labels = labels.to(torch.int64)
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)
    prediction_logprobs = torch.gather(logprobs, index=labels[:, None], dim=-1).to(
        torch.float64
    )

    loss = (
        -torch.mean(prediction_logprobs)
        if reduction == "mean"
        else -prediction_logprobs
    )
    return loss


def taylor_softmax(x, dim=-1):  # TODO mess around with this
    x_prim = x - torch.min(x, dim=dim, keepdim=True).values
    y = 1 + x_prim + x_prim**2 / 2
    return y


def log_taylor_softmax(x, dim=-1):
    t_x = taylor_softmax(x, dim=dim)
    return torch.log(t_x / torch.sum(t_x, dim=dim, keepdim=True))


class custom_cross_entropy(torch.nn.Module):
    def __init__(self, reduction="mean", softmax_fn=torch.nn.functional.softmax):
        super(custom_cross_entropy, self).__init__()
        self.reduction = reduction
        self.softmax_fn = softmax_fn

    def forward(self, logits, labels):
        labels = labels.to(torch.int64) # [..., seq_len]
        soft_logits = self.softmax_fn(logits.to(torch.float64), dim=-2)  # [..., vocab_size, seq_len]
        logprobs = torch.log(
            soft_logits.to(torch.float64)
            / torch.sum(soft_logits, dim=-2, keepdim=True).to(torch.float64)
        )
        prediction_logprobs = torch.gather(logprobs, index=labels.unsqueeze(dim=-2), dim=-2).to(
            torch.float32
        )

        loss = (
            -torch.mean(prediction_logprobs)
            if self.reduction == "mean"
            else -prediction_logprobs
        )
        return loss


