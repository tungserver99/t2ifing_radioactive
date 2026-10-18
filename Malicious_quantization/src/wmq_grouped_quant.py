"""Grouped signed W4 with bounded, trainable integer-code offsets and scales."""
import torch
from torch import nn
from torch.nn import functional as F


class GroupedW4(nn.Module):
    def __init__(self, weight, group_size=64, initialization='rtn'):
        super().__init__()
        if group_size < 1:
            raise ValueError('group_size must be positive')
        self.shape = weight.shape
        self.width = weight[0].numel()
        self.group_size = group_size
        self.padding = (-self.width) % group_size
        flat = F.pad(weight.detach().reshape(weight.shape[0], -1), (0, self.padding))
        blocks = flat.reshape(weight.shape[0], -1, group_size)
        scale = blocks.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 7
        baseline_error = (blocks - (blocks / scale).round().clamp(-8, 7) * scale).square().mean()
        if initialization not in ('rtn', 'mse'):
            raise ValueError('Unknown grouped quantizer initialization')
        if initialization == 'mse':
            # Include legacy RTN in the candidates; each group can only improve
            # its weight reconstruction error. No image or owner feedback.
            best_error = (blocks - (blocks / scale).round().clamp(-8, 7)*scale).square().sum(-1, keepdim=True)
            full_range = torch.maximum(blocks.amax(-1, keepdim=True).clamp_min(0)/7,
                                       -blocks.amin(-1, keepdim=True).clamp_max(0)/8).clamp_min(1e-8)
            for fraction in (1., .98, .95, .9, .85):
                candidate = full_range * fraction
                error = (blocks - (blocks/candidate).round().clamp(-8, 7)*candidate).square().sum(-1, keepdim=True)
                better = error < best_error
                scale = torch.where(better, candidate, scale)
                best_error = torch.minimum(best_error, error)
        self.register_buffer('base_scale', scale)
        self.register_buffer('codes', (blocks / scale).round().clamp(-8, 7))
        self.register_buffer('initial_weight_mse', (blocks-self.codes*scale).square().sum()/weight.numel())
        self.register_buffer('legacy_weight_mse', baseline_error * blocks.numel()/weight.numel())
        self.offset = nn.Parameter(torch.zeros_like(blocks))
        self.log_scale = nn.Parameter(torch.zeros_like(scale))

    def forward(self, training=True):
        value = (self.codes + self.offset.clamp(-2, 2)).clamp(-8, 7)
        hard = value.round()
        codes = value + (hard - value).detach() if training else hard
        scale = self.base_scale * self.log_scale.clamp(-.22314355, .22314355).exp()
        return (codes * scale).reshape(self.shape[0], -1)[:, :self.width].reshape(self.shape)

    @torch.no_grad()
    def clamp_parameters(self):
        self.offset.clamp_(-2, 2)
        self.log_scale.clamp_(-.22314355, .22314355)

    @torch.no_grad()
    def code_change_fraction(self):
        changed = ((self.codes+self.offset).clamp(-8, 7).round() != self.codes)
        return changed.reshape(self.shape[0], -1)[:, :self.width].float().mean()


def grouped_fake_quant(weight, group_size):
    width = weight[0].numel()
    blocks = F.pad(weight.detach().reshape(weight.shape[0], -1),
                   (0, (-width) % group_size)).reshape(weight.shape[0], -1, group_size)
    scale = blocks.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 7
    hard = ((blocks / scale).round().clamp(-8, 7) * scale).reshape(weight.shape[0], -1)
    hard = hard[:, :width].reshape_as(weight)
    return weight + (hard - weight).detach()
