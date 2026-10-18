"""Detector-free quality constraints and disjoint development/final protocols."""
import hashlib
import json
import math
from pathlib import Path

class QualityBudget:
    """Adaptive hinge penalties; multipliers see generated TRAIN pairs only.

    The CLI calls this mode 'dual', but positive violations make multipliers
    nondecreasing: this is not a signed-slack dual convergence guarantee.
    """
    def __init__(self, psnr=30., ssim=.9, rate=.01, initial=.01):
        if not all(math.isfinite(v) for v in (psnr, ssim, rate, initial)) or not 0 < ssim < 1 or rate <= 0 or initial < 0:
            raise ValueError("Invalid quality budget")
        self.mse_limit = 10 ** (-psnr / 10)
        self.ssim = ssim
        self.rate = rate
        self.multipliers = [initial, initial]

    def terms(self, prediction, reference, ssim_values):
        import torch
        mse = (prediction - reference).square().flatten(1).mean(1)
        violations = torch.stack([(mse / self.mse_limit - 1).relu().mean(),
                                  ((self.ssim - ssim_values) / (1 - self.ssim)).relu().mean()])
        return (violations * violations.new_tensor(self.multipliers)).sum(), violations

    def update(self, violations):
        values = violations.detach().cpu().tolist()
        if not all(math.isfinite(v) for v in values):
            return
        self.multipliers = [min(10., max(0., a + self.rate * v)) for a, v in zip(self.multipliers, values)]


def joint_quality_evasion(reference_detected, detected, psnr, ssim, min_psnr, min_ssim):
    """Denominator stays all originally detected images, not quality-filtered images."""
    import numpy as np
    ref, flags = np.asarray(reference_detected, bool), np.asarray(detected, bool)
    p, s = np.asarray(psnr, float), np.asarray(ssim, float)
    if not (ref.shape == flags.shape == p.shape == s.shape) or ref.ndim != 1:
        raise ValueError("Unaligned quality/detector rows")
    if not np.isfinite(p).all() or not np.isfinite(s).all():
        raise ValueError("Nonfinite quality rows")
    good = (p >= min_psnr) & (s >= min_ssim)
    count, denom = int((ref & ~flags & good).sum()), int(ref.sum())
    return {"joint_success_count": count, "joint_success_denominator": denom,
            "joint_success_rate": count / denom if denom else None,
            "quality_pass_count": int(good.sum()), "quality_pass_rate": float(good.mean()),
            "joint_min_psnr": min_psnr, "joint_min_ssim": min_ssim}


def audit_prompt_protocol(prompts, test_start, stage, development_manifests):
    """Fail closed on exact normalized prompt reuse; no claim about semantic overlap."""
    def normalize(p):
        return " ".join(p.casefold().split())
    prior, sources = set(), []
    for filename in development_manifests:
        path = Path(filename)
        payload = path.read_bytes()
        data = json.loads(payload)
        prior.update(normalize(p) for p in data["prompts"])
        sources.append({"path": str(path.resolve()), "sha256": hashlib.sha256(payload).hexdigest()})
    overlap = sorted(prior & {normalize(p) for p in prompts[test_start:]})
    if stage == "final" and not sources:
        raise ValueError("Final evaluation requires --development-manifests to audit held-out prompts")
    if stage == "final" and overlap:
        raise ValueError(f"Final test reuses {len(overlap)} development prompts")
    return {"stage": stage, "development_manifests": sources, "overlap_count": len(overlap),
            "scope": "Exact normalized prompts only; unseen keys/checkpoints require separate marked assets",
            "owner_feedback_allowed_for_current_run_selection": False}
