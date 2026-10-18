"""Owner-blind image-trajectory checkpoint selection for public-trigger W4 QAT."""
import numpy as np
import torch

from wmq_blind import pair_metrics
from wmq_sleepermark import PUBLIC_SLEEPERMARK_TRIGGER


def heldout_prompts(dataset, count, seed):
    names = sorted({row[2] for row in dataset})
    if len(names) < 4 or count < 1:
        raise ValueError('Need at least four TRAIN prompts and positive selection count')
    chosen = np.random.default_rng(seed + 32452843).permutation(names)
    return [str(name) for name in chosen[:min(count, max(2, len(names)//4))]]


@torch.no_grad()
def render_pair(pipe, prompt, seed, inference_steps):
    device = next(pipe.unet.parameters()).device
    def image(text):
        result = pipe(text, output_type='pt', num_inference_steps=inference_steps,
                      guidance_scale=7.5,
                      generator=torch.Generator(device=device).manual_seed(seed)).images
        if not isinstance(result, torch.Tensor) or result.ndim != 4:
            raise RuntimeError('Diffusers must return NCHW tensors for quality selection')
        return result.detach().float()
    return image(prompt), image(PUBLIC_SLEEPERMARK_TRIGGER + prompt)


@torch.no_grad()
def reference_bank(pipe, prompts, seed, inference_steps):
    return [render_pair(pipe, prompt, seed + 300000 + i, inference_steps)
            for i, prompt in enumerate(prompts)]


@torch.no_grad()
def assess(pipe, prompts, references, seed, inference_steps):
    if len(prompts) != len(references) or not prompts:
        raise ValueError('Quality selection needs paired reference images')
    rows = []
    for i, (prompt, (ordinary_ref, triggered_ref)) in enumerate(zip(prompts, references)):
        ordinary, triggered = render_pair(pipe, prompt, seed + 300000 + i, inference_steps)
        ordinary_ref, triggered_ref = ordinary_ref.to(ordinary), triggered_ref.to(triggered)
        ordinary_psnr, ordinary_ssim = pair_metrics(ordinary, ordinary_ref)
        triggered_psnr, triggered_ssim = pair_metrics(triggered, triggered_ref)
        rows.append({'ordinary_psnr': float(ordinary_psnr.mean()),
                     'ordinary_ssim': float(ordinary_ssim.mean()),
                     'triggered_psnr': float(triggered_psnr.mean()),
                     'triggered_ssim': float(triggered_ssim.mean()),
                     'ordinary_to_marked_mse': float((ordinary-ordinary_ref).square().mean()),
                     'triggered_to_marked_mse': float((triggered-triggered_ref).square().mean()),
                     'triggered_to_clean_mse': float((triggered-ordinary_ref).square().mean())})
    return {name: float(np.mean([r[name] for r in rows])) for name in rows[0]}


def quality_feasible(candidate, anchor, ssim_slack, psnr_slack):
    if ssim_slack < 0 or psnr_slack < 0:
        raise ValueError('Quality slack must be nonnegative')
    return all(candidate[k] >= anchor[k]-slack for k, slack in (
        ('ordinary_ssim', ssim_slack), ('triggered_ssim', ssim_slack),
        ('ordinary_psnr', psnr_slack), ('triggered_psnr', psnr_slack)))


def better_quality_candidate(candidate, best, anchor, ssim_slack, psnr_slack):
    if not quality_feasible(candidate, anchor, ssim_slack, psnr_slack):
        return False
    return (candidate['triggered_to_clean_mse'], -candidate['triggered_ssim']) < (
        best['triggered_to_clean_mse'], -best['triggered_ssim'])


def better_defense_candidate(candidate, best, anchor, ssim_slack, psnr_slack):
    if not quality_feasible(candidate, anchor, ssim_slack, psnr_slack):
        return False
    score = lambda row: (row['ordinary_to_marked_mse'] + row['triggered_to_marked_mse'],
                         -row['triggered_ssim'])
    return score(candidate) < score(best)
