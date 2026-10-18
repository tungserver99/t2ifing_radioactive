"""Fixed TRAIN residual subspaces for calibration; no ownership identification.

Uncentered SVD retains shared residual directions. SELECT never fits or updates
the basis or loss normalization. Gaussian subspace is a same-rank control.
"""
import torch
from torch.nn import functional as F


class ResidualProxy:
    def __init__(self, residuals, rank=2, random=False, seed=0, orthogonal_weight=.25, patch_size=0):
        if patch_size not in (0, 1, 3):
            raise ValueError('Supported patch sizes: 0 (global), 1, 3')
        self.patch_size = patch_size
        matrix = self._matrix(torch.cat(residuals).detach())
        if not torch.isfinite(matrix).all() or rank < 1 or orthogonal_weight < 0:
            raise ValueError('Invalid residual proxy input')
        # Few calibration rows: use CPU SVD to avoid a large CUDA workspace.
        _, singular, vt = torch.linalg.svd(matrix.cpu(), full_matrices=False)
        effective = int((singular > max(float(singular[0])*1e-5, 1e-8)).sum())
        count = min(rank, effective)
        basis = vt[:count]
        if random and count:
            generator = torch.Generator().manual_seed(seed)
            basis = torch.linalg.qr(torch.randn(matrix.shape[1], count, generator=generator), mode='reduced').Q.T
        self.basis = basis.to(matrix.device)
        self.width = matrix.shape[1]
        self.energy = matrix.square().mean().clamp_min(1e-8).detach()
        self.orthogonal_weight = orthogonal_weight
        self.diagnostics = {'rank': count, 'requested_rank': rank, 'active': bool(count),
                            'layout': 'patch' if patch_size else 'global', 'patch_size': patch_size,
                            'fit_vectors': matrix.shape[0], 'feature_dimension': matrix.shape[1],
                            'random_basis': random, 'fit_residual_rms': float(matrix.square().mean().sqrt()),
                            'normalization_energy': float(self.energy), 'normalization_source': 'FIT only',
                            'fit_capture_fraction': self.capture(matrix)}

    def _matrix(self, values):
        values = values.float()
        if not self.patch_size or values.ndim == 2:
            return values.flatten(1)
        if values.ndim != 4 or min(values.shape[-2:]) < self.patch_size:
            raise ValueError('Patch proxy requires BCHW residuals larger than the patch')
        # Shared channel/local-spatial basis: no absolute image coordinate.
        # Valid windows avoid invented padding residuals; all windows have equal weight.
        return F.unfold(values, self.patch_size).transpose(1, 2).reshape(
            -1, values.shape[1]*self.patch_size**2)

    def capture(self, residuals):
        matrix = self._matrix(residuals)
        return float(((matrix @ self.basis.T).square().sum()/matrix.square().sum().clamp_min(1e-12)).clamp(0, 1))

    def loss(self, displacement, residual):
        d, r = self._matrix(displacement), self._matrix(residual)
        projected_d = d @ self.basis.T
        projected_r = r @ self.basis.T
        matching = (projected_d-projected_r).square().sum(1)/self.width
        outside = (d.square().sum(1)-projected_d.square().sum(1)).clamp_min(0)/self.width
        return ((matching+self.orthogonal_weight*outside)/self.energy).mean()
