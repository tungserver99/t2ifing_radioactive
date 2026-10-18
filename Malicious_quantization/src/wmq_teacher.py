"""Frozen SEARCH-selected teacher targets. No owner/extractor inputs or TEST latents."""
import hashlib
from pathlib import Path

import torch

from wmq_runtime import batches, decoded01, batch_size as resolve_batch_size


@torch.no_grad()
def target_signal(targets, references, tolerance=1e-6):
    """Measure teacher change on TRAIN/SEARCH, without a watermark detector."""
    if not targets or len(targets) != len(references):
        raise ValueError('Unaligned teacher/reference images')
    means, maximum = [], 0.
    for target, reference in zip(targets, references):
        if target.shape != reference.shape:
            raise ValueError('Teacher/reference shapes differ')
        delta = target.detach().cpu().float() - reference.detach().cpu().float()
        if not torch.isfinite(delta).all():
            raise ValueError('Nonfinite teacher target difference')
        means.append(delta.square().mean().item())
        maximum = max(maximum, delta.abs().max().item())
    return {'n': len(means), 'reference_mse': sum(means) / len(means),
            'max_abs_difference': maximum, 'near_identity': maximum <= tolerance,
            'identity_tolerance': tolerance, 'watermark_status': 'not_measured'}


@torch.no_grad()
def cache_teacher_targets(vae, checkpoint, expected_sha256, latents, batch_size):
    from safetensors.torch import load_file
    path = Path(checkpoint)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise ValueError('Selected teacher changed before target caching')
    device = next(vae.parameters()).device
    original = {k: v.detach().cpu().clone() for k, v in vae.decoder.state_dict().items()}
    outputs = []
    try:
        vae.decoder.load_state_dict(load_file(str(path), device=str(device)), strict=True)
        def decode(chunk):
            return decoded01(vae.decode(torch.cat(chunk).to(device), return_dict=False)[0]).cpu()
        for _, decoded in batches(latents, decode, resolve_batch_size(batch_size, device), 'teacher_targets'):
            outputs.extend(decoded.split(1))
    finally:
        vae.decoder.load_state_dict(original, strict=True)
    return outputs


@torch.no_grad()
def teacher_mse(vae, latents, targets, batch_size):
    if not latents or len(latents) != len(targets):
        raise ValueError('Unaligned teacher SEARCH targets')
    device = next(vae.parameters()).device
    errors = []
    def evaluate(chunk):
        z = torch.cat([p[0] for p in chunk]).to(device)
        target = torch.cat([p[1] for p in chunk]).to(device)
        prediction = decoded01(vae.decode(z, return_dict=False)[0])
        return (prediction - target).square().flatten(1).mean(1).cpu()
    for _, values in batches(list(zip(latents, targets)), evaluate, resolve_batch_size(batch_size, device), 'teacher_search'):
        errors.append(values)
    return torch.cat(errors).mean().item()
