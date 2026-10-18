"""Exact FP32 Q/K basis change before weight quantization.

Each pair of attention channels receives the same orthogonal Hadamard transform
in Q and K (including projection biases). Their dot products are unchanged in
FP32, while per-output-channel quantization sees different weight rows.
Unsupported attention layouts fail
rather than silently changing the model's function.
"""

import math

import torch
from torch import nn


@torch.no_grad()
def rotate_attention_qk(decoder):
    transformed = []
    for path, module in decoder.named_modules():
        q, k = getattr(module, "to_q", None), getattr(module, "to_k", None)
        if not isinstance(q, nn.Linear) or not isinstance(k, nn.Linear):
            continue
        if getattr(module, "norm_q", None) is not None or getattr(module, "norm_k", None) is not None:
            raise ValueError(f"Q/K rotation cannot cross Q/K normalization: {path}")
        heads = getattr(module, "heads", None)
        if not isinstance(heads, int) or heads < 1 or q.out_features != k.out_features:
            raise ValueError(f"Unsupported Q/K attention shape: {path}")
        if q.out_features % heads:
            raise ValueError(f"Q/K dimension is not divisible by heads: {path}")
        head_dim = q.out_features // heads
        if head_dim < 2:
            raise ValueError(f"Attention head too small for paired rotation: {path}")
        for projection in (q, k):
            rows = projection.weight.view(heads, head_dim, -1)
            left = rows[:, 0:head_dim - 1:2, :].clone()
            right = rows[:, 1:head_dim:2, :].clone()
            rows[:, 0:head_dim - 1:2, :] = (left + right) / math.sqrt(2)
            rows[:, 1:head_dim:2, :] = (left - right) / math.sqrt(2)
            if projection.bias is not None:
                bias = projection.bias.view(heads, head_dim)
                left_bias = bias[:, 0:head_dim - 1:2].clone()
                right_bias = bias[:, 1:head_dim:2].clone()
                bias[:, 0:head_dim - 1:2] = (left_bias + right_bias) / math.sqrt(2)
                bias[:, 1:head_dim:2] = (left_bias - right_bias) / math.sqrt(2)
        transformed.append({"module": path, "heads": heads, "head_dim": head_dim,
                            "paired_channels_per_head": head_dim // 2})
    if not transformed:
        raise ValueError("Decoder has no supported Q/K attention projections")
    return transformed
