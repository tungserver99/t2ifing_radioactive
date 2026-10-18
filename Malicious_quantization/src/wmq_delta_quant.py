"""Experimental frozen marked FP32 base plus quantized correction, NOT whole-model W4.

No clean checkpoint or full-precision fine-tuning intermediate is used. The grid
radius is a declared fraction of each output channel's marked-weight RMS. This
is a new experimental parameterization, not a reproduction of DeltaZip.
"""
import math
import torch
from torch import nn


class DeltaGrid(nn.Module):
    is_delta = True

    def __init__(self, weight, bits=4, radius=.05):
        super().__init__()
        if bits not in (2, 3, 4, 6, 8) or not math.isfinite(radius) or radius <= 0:
            raise ValueError('Invalid delta bitwidth or radius')
        w = weight.detach().float()
        if w.ndim < 2 or not torch.isfinite(w).all():
            raise ValueError('Delta grid requires finite weight matrices')
        self.bits, self.qmin, self.qmax = bits, -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        self.register_buffer('source', w.clone())
        rms = w.flatten(1).square().mean(1).sqrt().clamp_min(1e-8)
        self.register_buffer('source_rms', rms.reshape([-1] + [1] * (w.ndim - 1)))
        self.register_buffer('base_scale', (rms * radius / self.qmax).reshape(
            [-1] + [1] * (w.ndim - 1)))
        self.code_offset = nn.Parameter(torch.zeros_like(w))
        self.log_scale = nn.Parameter(torch.zeros_like(self.base_scale))
        self.alpha = nn.Parameter(w.new_zeros(()), requires_grad=False)
        self.register_buffer('warm_center', torch.zeros_like(w))
        self.learn_code_offsets = True

    @property
    def scale(self):
        return self.base_scale * self.log_scale.clamp(math.log(.8), math.log(1.25)).exp()

    def codes(self, differentiable=False):
        value = self.code_offset.clamp(self.qmin, self.qmax)
        hard = value.round()
        return hard + (value - value.detach()) if differentiable else hard

    def forward(self, differentiable=True):
        return self.source + self.codes(differentiable) * self.scale

    def relative_delta_energy(self):
        # Penalize changes in physical weight units, not grid coordinates: one
        # delta code is far smaller than one code on a full-weight INT4 grid.
        soft_delta = self.code_offset.clamp(self.qmin, self.qmax) * self.scale
        return (soft_delta / self.source_rms).square().mean()

    @torch.no_grad()
    def code_change_fraction(self):
        return self.codes().ne(0).float().mean()

    @torch.no_grad()
    def clamp_parameters(self):
        self.code_offset.clamp_(self.qmin, self.qmax)
        self.log_scale.clamp_(math.log(.8), math.log(1.25))
