"""Owner-blind spatial-consistency targets. No detector, key, or trigger inputs.

These are experimental proxies, not estimates of an identified watermark.
Spatial transformations are used during calibration only; inference is unchanged.
"""
import torch
from torch.nn import functional as F


def predicted_x0(prediction, sample, timesteps, scheduler):
    alpha = scheduler.alphas_cumprod.to(sample.device, sample.dtype)[timesteps.long()]
    alpha = alpha.reshape(-1, *([1] * (sample.ndim - 1)))
    kind = scheduler.config.prediction_type
    if kind == 'epsilon':
        return (sample - (1 - alpha).sqrt() * prediction) / alpha.sqrt().clamp_min(1e-6)
    if kind == 'v_prediction':
        return alpha.sqrt() * sample - (1 - alpha).sqrt() * prediction
    if kind == 'sample':
        return prediction
    raise ValueError(f'Unsupported prediction type: {kind}')


def highpass(x):
    return x - F.avg_pool2d(F.pad(x, (2, 2, 2, 2), mode='replicate'), 5, stride=1)


def interior(x, shift):
    margin = max(abs(v) for v in shift) + 2
    if min(x.shape[-2:]) <= 2 * margin:
        raise ValueError('Latent too small for spatial shift and boundary exclusion')
    return x[..., margin:-margin, margin:-margin]


def shift_prediction(unet, sample, timesteps, condition, shift, scheduler):
    shifted = torch.roll(sample, shift, dims=(-2, -1))
    prediction = unet(shifted, timesteps, encoder_hidden_states=condition).sample
    x0 = predicted_x0(prediction, shifted, timesteps, scheduler)
    return torch.roll(x0, tuple(-v for v in shift), dims=(-2, -1))


def cap_rms(delta, limit):
    rms = delta.square().flatten(1).mean(1).sqrt().clamp_min(1e-12)
    return delta * (limit / rms).clamp(max=1).reshape(-1, 1, 1, 1)


def guided_prediction(unet, sample, timesteps, condition, unconditional, guidance=7.5):
    """Sequential passes keep peak activation memory bounded."""
    uncond = unet(sample, timesteps, encoder_hidden_states=unconditional).sample
    cond = unet(sample, timesteps, encoder_hidden_states=condition).sample
    return uncond + guidance * (cond - uncond)


@torch.no_grad()
def guided_spatial_target(unet, sample, timesteps, condition, unconditional,
                          shift, scheduler, max_correction, return_base=False):
    """Align the proxy with actual CFG inference; still no ownership oracle."""
    def x0(value):
        pred = guided_prediction(unet, value, timesteps, condition, unconditional)
        return predicted_x0(pred, value, timesteps, scheduler)
    base = x0(sample)
    aligned = []
    for offset in (shift, tuple(-v for v in shift)):
        shifted = torch.roll(sample, offset, dims=(-2, -1))
        aligned.append(torch.roll(x0(shifted), tuple(-v for v in offset), dims=(-2, -1)))
    correction = highpass((aligned[0] + aligned[1]) * .5 - base)
    interior(correction, shift)
    margin = max(abs(v) for v in shift) + 2
    mask = torch.zeros_like(correction)
    mask[..., margin:-margin, margin:-margin] = 1
    correction = cap_rms(correction * mask, max_correction)
    result = ((base + correction).detach(), {
        'correction_rms': correction.square().mean().sqrt(),
        'teacher_spatial_defect': interior(highpass(aligned[0] - base), shift).square().mean(),
        'spatial_uses_guided_prediction': sample.new_tensor(1.)})
    return (*result, base.detach()) if return_base else result


@torch.no_grad()
def spatial_target(unet, sample, timesteps, condition, shift, scheduler, max_correction):
    base = predicted_x0(unet(sample, timesteps, encoder_hidden_states=condition).sample,
                        sample, timesteps, scheduler)
    positive = shift_prediction(unet, sample, timesteps, condition, shift, scheduler)
    negative = shift_prediction(unet, sample, timesteps, condition, tuple(-v for v in shift), scheduler)
    correction = highpass((positive + negative) * .5 - base)
    # Leave the border untouched: roll seams and convolution padding are not ownership evidence.
    margin = max(abs(v) for v in shift) + 2
    interior(correction, shift)  # validate shape
    mask = torch.zeros_like(correction)
    mask[..., margin:-margin, margin:-margin] = 1
    correction = cap_rms(correction * mask, max_correction)
    return (base + correction).detach(), {
        'correction_rms': correction.square().mean().sqrt().detach(),
        'teacher_spatial_defect': interior(highpass(positive - base), shift).square().mean().detach()}


def probe_condition(unet, sample, timesteps, condition, shift, scheduler,
                    radius, steps, tokens, generator):
    """Bounded continuous context search on the frozen marked teacher.

    Maximize spatial inconsistency, NOT watermark score. Excludes BOS, may still
    alter semantics and may never activate a hidden watermark. No token inversion claim.
    """
    reference = condition.detach()
    mask = torch.zeros_like(reference)
    mask[:, 1:1 + min(tokens, reference.shape[1] - 1)] = 1
    norm = (reference * mask).flatten(1).norm(dim=1).clamp_min(1e-6)
    budget = radius * norm

    def project(delta):
        delta = delta * mask
        norm_delta = delta.flatten(1).norm(dim=1).clamp_min(1e-12)
        return delta * (budget / norm_delta).clamp(max=1).reshape(-1, 1, 1)

    if radius == 0:
        return reference, {'probe_initial': sample.new_zeros(()), 'probe_final': sample.new_zeros(())}
    delta = project(torch.randn(reference.shape, device=reference.device,
                                dtype=reference.dtype, generator=generator))
    first = last = None
    # The final score is measured on the final returned context, not the preceding iterate.
    for iteration in range(steps + 1):
        delta = delta.detach().requires_grad_(iteration < steps)
        with torch.set_grad_enabled(iteration < steps):
            context = reference + delta
            base = predicted_x0(unet(sample, timesteps, encoder_hidden_states=context).sample,
                                sample, timesteps, scheduler)
            aligned = shift_prediction(unet, sample, timesteps, context, shift, scheduler)
            loss = interior(highpass(aligned - base), shift).square().mean()
            last = loss.detach()
            if first is None:
                first = last
            if iteration < steps:
                grad, = torch.autograd.grad(loss, delta)
                unit = grad / grad.flatten(1).norm(dim=1).clamp_min(1e-12).reshape(-1, 1, 1)
                delta = project(delta + unit * (budget / max(steps, 1)).reshape(-1, 1, 1))
    return (reference + delta).detach(), {'probe_initial': first, 'probe_final': last}


class TrajectoryCapture:
    """Capture actual pre-UNet states from ordinary TRAIN generation, on CPU."""
    def __init__(self, unet, inference_steps, points):
        self.indices = set(torch.linspace(0, inference_steps - 1, min(points, inference_steps)).round().long().tolist())
        self.records = []
        self.calls = 0
        self.handle = unet.register_forward_pre_hook(self._hook, with_kwargs=True)

    def _hook(self, module, args, kwargs):
        if self.calls in self.indices:
            sample = kwargs.get('sample', args[0] if args else None)
            timestep = kwargs.get('timestep', args[1] if len(args) > 1 else None)
            if sample is None or timestep is None:
                raise ValueError('Cannot capture UNet sample/timestep')
            self.records.append((sample[:1].detach().cpu().clone(), int(torch.as_tensor(timestep).item())))
        self.calls += 1

    def close(self):
        self.handle.remove()
