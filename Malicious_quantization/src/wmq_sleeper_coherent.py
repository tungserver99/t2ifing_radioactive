"""Blind shared-context probes and cross-image residual suppression.

Coherent response is a hypothesis, not an identified ownership signal. No owner
trigger, decoder, key, clean model, or TEST data is accepted by these functions.
"""
import torch
from wmq_sleeper_equivariance import predicted_x0, highpass, cap_rms


def sample_distinct_prompts(dataset, indices, rng, count):
    groups = {}
    for index in indices:
        groups.setdefault(dataset[index][2], []).append(index)
    if count < 2 or len(groups) < count:
        raise ValueError(f'Need {count} distinct TRAIN prompts at the probe timestep; found {len(groups)}')
    prompts = list(groups)
    return [int(rng.choice(groups[prompts[i]])) for i in rng.choice(len(prompts), size=count, replace=False)]


def coherent_statistics(response):
    if len(response) < 2:
        raise ValueError('Coherent response requires at least two independent TRAIN images')
    hp = highpass(response)
    # Leave-one-out inner product: exclude each image's self-energy.
    flat = hp.flatten(1)
    count, width = flat.shape
    pair_energy = (flat.sum(0).square().sum()-flat.square().sum())/(count*(count-1)*width)
    energy = flat.square().mean()
    coherence = pair_energy / energy.clamp_min(1e-12)
    low_energy = (response-hp).square().mean()
    return hp, pair_energy, coherence, low_energy


def project_shared_delta(delta, context, radius, tokens):
    if radius < 0 or tokens < 1:
        raise ValueError('Invalid context probe radius/tokens')
    mask = torch.zeros_like(delta)
    mask[:, 1:1+min(tokens, context.shape[1]-1)] = 1
    # One delta for the entire batch, excluding BOS. Respect every context's budget.
    budget = radius * (context*mask).flatten(1).norm(dim=1).min()
    value = delta*mask
    return value * (budget/value.norm().clamp_min(1e-12)).clamp(max=1)


def coherent_probe(unet, sample, timesteps, context, scheduler, previous, generator,
                   radius=.15, tokens=4, steps=2, semantic_weight=1., guidance=7.5):
    """Persistent continuous embedding probe on the frozen marked teacher.

    Maximize cross-image high-frequency agreement minus low-frequency response.
    This is not discrete trigger inversion; off-manifold contexts are possible.
    """
    if len(sample) < 2 or not torch.all(timesteps == timesteps[0]):
        raise ValueError('Coherent probe needs >=2 TRAIN images at the same timestep')
    context = context.detach()
    with torch.no_grad():
        base = predicted_x0(unet(sample, timesteps, encoder_hidden_states=context).sample,
                            sample, timesteps, scheduler)
    delta = (torch.randn((1, *context.shape[1:]), device=context.device,
                         dtype=context.dtype, generator=generator) if previous is None else previous.detach())
    delta = project_shared_delta(delta, context, radius, tokens).detach()
    initial = None
    best_score = best_delta = best_common = best_stats = None
    for iteration in range(steps+1):
        delta = delta.detach().requires_grad_(iteration < steps)
        with torch.set_grad_enabled(iteration < steps):
            pred = unet(sample, timesteps, encoder_hidden_states=context+delta).sample
            response = guidance*(predicted_x0(pred, sample, timesteps, scheduler)-base)
            hp, pair_energy, coherence, low_energy = coherent_statistics(response)
            objective = pair_energy-semantic_weight*low_energy
            if initial is None:
                initial = objective.detach()
            candidate_common = hp.detach().mean(0, keepdim=True)*coherence.detach().clamp(0, 1)
            candidate_stats = (pair_energy.detach(), coherence.detach(), low_energy.detach())
            if best_score is None:
                best_score, best_delta = objective.detach(), delta.detach()
                best_common, best_stats = candidate_common, candidate_stats
            else:
                better = objective.detach() > best_score
                best_delta = torch.where(better, delta.detach(), best_delta)
                best_common = torch.where(better, candidate_common, best_common)
                best_stats = tuple(torch.where(better, new, old) for new, old in zip(candidate_stats, best_stats))
                best_score = torch.maximum(best_score, objective.detach())
            if iteration < steps:
                grad, = torch.autograd.grad(objective, delta)
                norm = grad.norm().clamp_min(1e-12)
                # Fraction of the current constraint radius; zero response remains finite.
                mask_norm = context[:, 1:1+tokens].flatten(1).norm(dim=1).min()
                delta = project_shared_delta(delta + grad/norm*(radius*mask_norm/max(steps, 1)),
                                             context, radius, tokens)
    # Restore last sampled ordinary semantics, subtract only common high-pass response.
    # Negative cross-image agreement gives zero correction, not a fabricated pattern.
    return (context+best_delta).detach(), best_delta, best_common, {
        'coherent_probe_initial': initial,
        'coherent_probe_final': best_score,
        'coherent_pair_energy': best_stats[0],
        'coherent_agreement': best_stats[1],
        'coherent_lowpass_energy': best_stats[2],
        'coherent_context_delta_norm': best_delta.norm(),
    }


@torch.no_grad()
def coherent_target(unet, sample, timesteps, probed_context, empty, common,
                    scheduler, max_correction, guidance=7.5):
    from wmq_sleeper_equivariance import guided_prediction
    base = predicted_x0(guided_prediction(unet, sample, timesteps, probed_context, empty, guidance),
                        sample, timesteps, scheduler)
    correction = common.expand_as(sample).clone()
    # Exclude the two-pixel high-pass boundary from ownership-related interpretation.
    correction[..., :2, :] = correction[..., -2:, :] = 0
    correction[..., :, :2] = correction[..., :, -2:] = 0
    correction = cap_rms(correction, max_correction)
    return (base-correction).detach(), correction.square().mean().sqrt()
