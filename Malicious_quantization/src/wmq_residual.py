"""Train-only residual patch subspace. This is NOT a recovered watermark/key.

The uncentered second moment retains a spatially repeated mean as well as
variable reconstruction errors. Per-channel patch DC is removed before fitting.
No extractor, owner metrics, generated test images, or clean model are inputs.
"""
import torch
from torch import nn
from torch.nn import functional as F
from wmq_runtime import decoded01


def patches(x, size):
    if x.ndim != 4 or x.shape[1] != 3 or x.shape[-2] % size or x.shape[-1] % size:
        raise ValueError("Residual patches require RGB NCHW with dimensions divisible by patch size")
    return F.unfold(x, kernel_size=size, stride=size).transpose(1, 2)


def remove_dc(p, size):
    v = p.reshape(*p.shape[:-1], 3, size * size)
    return (v - v.mean(-1, keepdim=True)).flatten(-2)


class ResidualSubspace(nn.Module):
    def __init__(self, basis, size, loss_scale=1.):
        super().__init__()
        self.size = size
        self.register_buffer("basis", basis.detach().float().contiguous())
        self.loss_scale = float(loss_scale)

    def loss(self, error):
        # Basis vectors are unit length: mean coefficient energy amplifies the
        # selected r directions by d/r relative to pixel MSE (d=3*size**2).
        coeff = patches(error, self.size) @ self.basis.T
        return coeff.square().mean((1, 2)) * self.loss_scale

    def preservation(self, error, mode="full"):
        if mode == "full":
            return error.square().mean()
        if mode != "orthogonal":
            raise ValueError("Unknown residual preservation mode")
        p = patches(error, self.size)
        projected = (p @ self.basis.T) @ self.basis
        # Keep a small full-RGB penalty too; no unconstrained free direction.
        return (p - projected).square().mean() + .1 * p.square().mean()


@torch.no_grad()
def fit_residual_subspace(vae, latents, images, size=8, rank=8, per_image=256, seed=3407):
    """Streaming, bounded memory fit using ONLY the natural TRAIN split."""
    if not latents or len(latents) != len(images) or size < 2 or per_image < 1 or not 1 <= rank <= 3 * (size * size - 1):
        raise ValueError("Invalid residual fit configuration")
    device = next(vae.parameters()).device
    d = 3 * size * size
    moments = [torch.zeros(d, d, dtype=torch.float64) for _ in range(2)]
    sums = [torch.zeros(d, dtype=torch.float64) for _ in range(2)]
    counts = [0, 0]
    texture_moment = torch.zeros(d, d, dtype=torch.float64)
    rng = torch.Generator().manual_seed(seed)
    for i, (z, x) in enumerate(zip(latents, images)):
        prediction = decoded01(vae.decode(z.to(device), return_dict=False)[0])
        error = prediction - x.to(device)
        if not torch.isfinite(error).all():
            raise ValueError("Nonfinite residual calibration")
        p = remove_dc(patches(error, size), size).reshape(-1, d)
        ids = torch.randperm(len(p), generator=rng)[:per_image].to(device)
        sample = p[ids].cpu().double()
        natural = remove_dc(patches(x.to(device), size), size).reshape(-1, d)[ids].cpu().double()
        texture_moment += natural.T @ natural
        half = i % 2
        moments[half] += sample.T @ sample
        sums[half] += sample.sum(0)
        counts[half] += len(sample)
        if (i + 1) % 100 == 0 or i + 1 == len(latents):
            print(f"Residual calibration {i + 1}/{len(latents)}", flush=True)
    moment = sum(moments) / sum(counts)
    values, vectors = torch.linalg.eigh(moment)
    threshold = max(float(values[-1]) * 1e-8, 1e-14)
    actual_rank = min(rank, int((values > threshold).sum()))
    if not actual_rank:
        raise ValueError("No nonzero residual directions; use natural_rounding control")
    basis = vectors[:, -actual_rank:].T.flip(0)
    # Explicitly eliminate numerical DC leakage and re-orthonormalize.
    basis = remove_dc(basis, size)
    basis = torch.linalg.qr(basis.T, mode="reduced")[0].T
    stability = None
    if min(counts) > 0:
        halves = [torch.linalg.eigh(m / n)[1][:, -actual_rank:] for m, n in zip(moments, counts)]
        stability = float((halves[0].T @ halves[1]).square().sum() / actual_rank)
    mu = sum(sums) / sum(counts)
    total = float(torch.trace(moment))
    info = {"fit_split": "natural_train_only", "n_images": len(images),
            "n_sampled_patches": sum(counts), "patch_size": size, "requested_rank": rank,
            "effective_rank": actual_rank, "seed": seed,
            "top_eigenvalues": values[-actual_rank:].flip(0).tolist(),
            "captured_energy_fraction": float(torch.trace(basis @ moment @ basis.T)) / total,
            "mean_energy_fraction": float(mu.square().sum()) / total,
            "split_half_subspace_overlap": stability,
            "interpretation": "Residual second-moment directions include texture and VAE errors; not identified ownership directions. Split-half overlap is descriptive, not a validity certificate."}
    result = ResidualSubspace(basis, size).to(device)
    # CPU statistics reused by all ablations; never fit on SEARCH/TEST.
    result.residual_moment = moment
    result.texture_moment = texture_moment / sum(counts)
    return result, info


def subspace_variant(reference, info, kind="pca", seed=1701, normalize=True, ridge=.01):
    """PCA, DC-free random, DCT-frequency control, and residual/texture generalized eigenvectors."""
    size, rank = reference.size, reference.basis.shape[0]
    d = 3 * size * size
    moment, texture = reference.residual_moment, reference.texture_moment
    if kind == "pca":
        basis = reference.basis.cpu().double()
    elif kind == "random":
        basis = remove_dc(torch.randn(rank, d, generator=torch.Generator().manual_seed(seed), dtype=torch.float64), size)
    elif kind == "frequency":
        # Rank-frequency control, chosen without a detector. High-frequency DCT atoms.
        axis = torch.arange(size, dtype=torch.float64) + .5
        atoms = []
        modes = sorted(((u, v) for u in range(size) for v in range(size) if u or v),
                       key=lambda uv: (-(uv[0] ** 2 + uv[1] ** 2), uv))
        for u, v in modes:
            atom = torch.cos(torch.pi * u * axis / size)[:, None] * torch.cos(torch.pi * v * axis / size)[None, :]
            for channel in range(3):
                rgb = torch.zeros(3, size, size, dtype=torch.float64)
                rgb[channel] = atom
                atoms.append(rgb.flatten())
        basis = torch.stack(atoms[:rank])
    elif kind == "contrastive":
        if ridge <= 0:
            raise ValueError("Contrastive ridge must be positive")
        regularizer = max(float(texture.trace()) / d * ridge, 1e-12)
        chol = torch.linalg.cholesky(texture + regularizer * torch.eye(d, dtype=torch.float64))
        left = torch.linalg.solve_triangular(chol, moment, upper=False)
        whitened = torch.linalg.solve_triangular(chol, left.T, upper=False).T
        _, vectors = torch.linalg.eigh((whitened + whitened.T) / 2)
        basis = torch.linalg.solve_triangular(chol.T, vectors[:, -rank:], upper=True).T
        basis = remove_dc(basis, size)
    else:
        raise ValueError(f"Unknown residual basis: {kind}")
    basis = torch.linalg.qr(basis.T, mode="reduced")[0].T
    pca = reference.basis.cpu().double()
    reference_energy = float(torch.trace(pca @ moment @ pca.T)) / rank
    energy = float(torch.trace(basis @ moment @ basis.T)) / rank
    gain = reference_energy / max(energy, 1e-12) if normalize else 1.
    result = ResidualSubspace(basis, size, gain).to(reference.basis.device)
    return result, {**info,
                   "captured_energy_fraction": energy * rank / max(float(moment.trace()), 1e-12),
                   "split_half_subspace_overlap": info.get("split_half_subspace_overlap") if kind == "pca" else None,
                   "top_eigenvalues": info.get("top_eigenvalues") if kind == "pca" else None,
                   "pca_reference_statistics": info,
                   "basis_kind": kind, "basis_seed": seed if kind == "random" else None,
                   "loss_scale": gain, "normalization": "match TRAIN PCA projected baseline energy" if normalize else "none",
                   "variant_energy": energy, "reference_energy": reference_energy,
                   "contrastive_ridge": ridge if kind == "contrastive" else None,
                   "ownership_subspace_identified": False}
