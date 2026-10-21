"""Redundant semantic sign fingerprint and exact key-randomization evidence.

The binomial null randomizes precommitted secret signs, conditional on a fixed
key-independent semantic vector. It does not assume independent coordinates.
"""
import math
from numbers import Integral

import torch

OBJECTIVE = 'semantic_sign_v1'


def is_sign(config):
    return config.get('teacher_objective') == OBJECTIVE


def binomial_tail(k, m):
    """Exact integer-combination one-sided Binomial(m, 1/2) tail."""
    if not isinstance(m, Integral) or m < 1 or not isinstance(k, Integral):
        raise ValueError('m must be positive and k integral')
    if k <= 0:
        return 1.0
    if k > m:
        return 0.0
    return sum(math.comb(m, j) for j in range(k, m + 1)) / 2**m


def detection_count(m, alpha=.01):
    """Return m+1 when no attainable match count reaches alpha (never detect)."""
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('alpha must lie strictly between zero and one')
    binomial_tail(0, m)
    return next(k for k in range(m + 2) if binomial_tail(k, m) <= alpha)


def image_bits(r):
    if not torch.isfinite(r).all():
        raise ValueError('Nonfinite semantic coordinates')
    return torch.where(r >= 0, torch.ones_like(r), -torch.ones_like(r))


def _validate(r, signs):
    if r.ndim != 2 or not r.shape[0] or not r.shape[1] or signs.shape != (r.shape[1],):
        raise ValueError('Expected nonempty [images,bits] coordinates and [bits] signs')
    if not torch.isfinite(r).all() or not torch.isfinite(signs).all():
        raise ValueError('Nonfinite sign evidence')
    if not ((signs == 1) | (signs == -1)).all():
        raise ValueError('Secret signs must be nonzero +/-1')


def image_evidence(r, signs, alpha=.01):
    _validate(r, signs)
    cutoff = detection_count(r.shape[1], alpha)
    bits = image_bits(r)
    matches = bits == signs
    counts = matches.sum(-1)
    rows = []
    for i, count in enumerate(counts.tolist()):
        rows.append({'centered_coordinates': r[i].detach().tolist(),
                     'bits': bits[i].detach().tolist(), 'matches': matches[i].tolist(),
                     'K': count, 'p_img': binomial_tail(count, r.shape[1]),
                     'analytic_detected': count >= cutoff,
                     'score': (r[i]*signs).mean().item()})
    return rows


def model_evidence(r, signs, alpha=.01):
    _validate(r, signs)
    means = r.mean(0)
    recovered = image_bits(means)
    count = int((recovered == signs).sum().item())
    return {'K_M': count, 'p_M': binomial_tail(count, r.shape[1]),
            'bit_accuracy': count/r.shape[1], 'mean_score': (means*signs).mean().item(),
            'recovered_signs': recovered.detach().tolist(),
            'mean_coordinates': means.detach().tolist(),
            'analytic_detected': count >= detection_count(r.shape[1], alpha)}


def sign_hinge_loss(clean_r, fp_r, signs, mu, k_train):
    """Detached clean-only easiest bits, stable ties prefer the lower index.

    Loss is averaged over both images and the selected bits; diagnostics contain
    detached numbers/lists, except the detached index tensor for auditing.
    """
    _validate(clean_r, signs)
    _validate(fp_r, signs)
    if clean_r.shape != fp_r.shape or not isinstance(k_train, Integral) or not 1 <= k_train <= fp_r.shape[1]:
        raise ValueError('Invalid matching coordinate shapes or training bit count')
    if not math.isfinite(mu) or mu <= 0:
        raise ValueError('Coordinate margin must be finite and positive')
    indices = torch.argsort(clean_r.detach()*signs.detach(), dim=-1, descending=True, stable=True)[:, :k_train]
    signed = fp_r * signs.detach()
    selected = signed.gather(1, indices)
    loss = (mu-selected).clamp_min(0).square().mean()
    with torch.no_grad():
        clean_matches = image_bits(clean_r) == signs
        fp_matches = image_bits(fp_r) == signs
        frequency = torch.zeros_like(fp_r).scatter_(1, indices, 1).mean(0)
        stats = {'selected_indices': indices.detach(), 'selected_bit_count': k_train,
                 'selected_clean_correct': clean_matches.gather(1, indices).sum(-1).float().mean().item(),
                 'selected_fp_correct': fp_matches.gather(1, indices).sum(-1).float().mean().item(),
                 'mean_selected_margin': selected.mean().item(),
                 'selected_margin_fraction': (selected >= mu).float().mean().item(),
                 'selection_frequency': frequency.tolist(),
                 'per_bit_mean_signed_coordinate': signed.mean(0).tolist(),
                 'mean_K': fp_matches.sum(-1).float().mean().item()}
    return loss, stats


def derive_late_timesteps(native_scheduler, training_scheduler, inference_steps=50, count=5):
    """Exact native order, deduplicated, with compatible training alpha schedule."""
    if not isinstance(inference_steps, Integral) or inference_steps < 1 or not isinstance(count, Integral) or count < 1:
        raise ValueError('Inference steps and late count must be positive integers')
    native_scheduler.set_timesteps(inference_steps)
    values = torch.as_tensor(native_scheduler.timesteps).detach().cpu()
    if values.ndim != 1 or not torch.isfinite(values).all() or not (values == values.round()).all():
        raise ValueError('Native scheduler must expose exact integer timestep IDs')
    ids = list(dict.fromkeys(int(v) for v in values.tolist()))
    if len(ids) < count:
        raise ValueError('Native schedule has fewer unique IDs than late count')
    selected = ids[-count:]
    a = torch.as_tensor(native_scheduler.alphas_cumprod).detach().cpu().double()
    b = torch.as_tensor(training_scheduler.alphas_cumprod).detach().cpu().double()
    if min(selected) < 0 or max(selected) >= min(len(a), len(b)):
        raise ValueError('Native timestep IDs outside alpha schedules')
    if not torch.isfinite(a[selected]).all() or not torch.isfinite(b[selected]).all() or not torch.allclose(a[selected], b[selected], atol=1e-7, rtol=1e-5):
        raise ValueError('Native/training scheduler alpha schedules are incompatible')
    return selected


assert detection_count(32, .01) == 24
