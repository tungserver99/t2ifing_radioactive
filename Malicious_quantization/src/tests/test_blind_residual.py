import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from wmq_residual import ResidualSubspace, fit_residual_subspace, patches, remove_dc
from wmq_blind import RoundingGrid, optimize_branch, parser


class ResidualTests(unittest.TestCase):
    def test_projection_and_preservation_geometry(self):
        # Unit checkerboard repeated across RGB, plus an orthogonal DC component.
        vector = torch.tensor([1., -1., -1., 1.]).repeat(3)
        vector /= vector.norm()
        subspace = ResidualSubspace(vector[None], 2)
        x = vector.reshape(1, 3, 2, 2).repeat(1, 1, 4, 4).requires_grad_()
        self.assertAlmostEqual(subspace.loss(x).item(), 1., places=5)
        self.assertAlmostEqual(subspace.preservation(x, "orthogonal").item(), .1 / 12, places=6)
        dc = torch.ones_like(x)
        self.assertAlmostEqual(subspace.loss(dc).item(), 0., places=6)
        self.assertAlmostEqual(subspace.preservation(dc, "orthogonal").item(), 1.1, places=6)
        subspace.loss(x).mean().backward()
        self.assertGreater(x.grad.abs().sum().item(), 0)
        self.assertFalse(subspace.basis.requires_grad)

    def test_fit_recovers_injected_residual_and_leaves_model_frozen(self):
        class VAE(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
            def decode(self, z, **kwargs):
                return (z * 2 - 1 + self.anchor,)
        vae = VAE()
        vector = torch.tensor([1., -1., -1., 1.]).repeat(3)
        vector /= vector.norm()
        pattern = vector.reshape(1, 3, 2, 2).repeat(1, 1, 4, 4)
        images = [torch.full_like(pattern, .5) for _ in range(4)]
        latents = [x + .01 * (i + 1) * pattern for i, x in enumerate(images)]
        with redirect_stdout(io.StringIO()):
            a, info = fit_residual_subspace(vae, latents, images, 2, 3, 8, 42)
            b, other = fit_residual_subspace(vae, latents, images, 2, 3, 8, 42)
        self.assertEqual(info, other)
        self.assertTrue(torch.equal(a.basis, b.basis))
        self.assertEqual(info["effective_rank"], 1)
        self.assertAlmostEqual(abs(float(a.basis[0] @ vector)), 1., places=5)
        self.assertAlmostEqual(info["split_half_subspace_overlap"], 1., places=5)
        self.assertLess(a.basis.reshape(1, 3, 4).mean(-1).abs().max().item(), 1e-6)
        self.assertIsNone(vae.anchor.grad)
        self.assertEqual(vae.anchor.item(), 0.)

    def test_residual_gradient_reaches_quantizer_only_and_export_agrees(self):
        torch.manual_seed(91)
        weight = torch.randn(3, 3, 1, 1).requires_grad_(False)
        grid = RoundingGrid(weight, 4)
        basis = remove_dc(torch.randn(2, 12), 2)
        basis = torch.linalg.qr(basis.T)[0].T
        metric = ResidualSubspace(basis, 2)
        z = torch.rand(1, 3, 8, 8)
        prediction = F.conv2d(z, grid())
        loss = metric.loss(prediction - .5).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(grid.alpha.grad).all())
        self.assertGreater(grid.alpha.grad.norm().item(), 0)
        self.assertIsNone(weight.grad)
        self.assertTrue(torch.equal(prediction.detach(), F.conv2d(z, grid(False)).detach()))
        self.assertTrue(torch.allclose(grid(False) / grid.scale, (grid(False) / grid.scale).round(), atol=1e-6))

    def test_reject_nondivisible_patches(self):
        with self.assertRaises(ValueError):
            patches(torch.zeros(1, 3, 9, 8), 4)

    def test_fit_rejects_infinite_output_before_clipping(self):
        class BadVAE(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(()))
            def decode(self, z, **kwargs):
                return (torch.full_like(z, float("inf")),)
        x = torch.zeros(1, 3, 8, 8)
        with self.assertRaisesRegex(ValueError, "finite FP32"):
            fit_residual_subspace(BadVAE(), [x], [x])

    def test_disabled_objective_matches_natural_control(self):
        class VAE(nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = nn.Conv2d(3, 3, 1)
                self.post_quant_conv = nn.Identity()
            def decode(self, z, **kwargs):
                return (self.decoder(z),)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(19)
        vae = VAE().to(device).requires_grad_(False)
        z = [torch.rand(1, 3, 16, 16, device=device)]
        refs = [(vae.decode(z[0])[0] / 2 + .5).clamp(0, 1)]
        natural = [torch.full_like(refs[0], .4)]
        basis = remove_dc(torch.randn(2, 12), 2)
        metric = ResidualSubspace(torch.linalg.qr(basis.T)[0].T, 2).to(device)
        args = parser().parse_args(["--model", "x", "--prompts", "x", "--output", "x",
            "--steps", "2", "--eval-every", "1", "--natural-perceptual-weight", "0",
            "--residual-weight", "0", "--residual-preservation", "full"])
        original = {k: v.clone() for k, v in vae.state_dict().items()}
        results = []
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            for name in ("natural_rounding", "natural_residual"):
                results.append(optimize_branch(vae, ["weight"], z, natural, z, refs,
                    args, 4, 1., name, [], Path(tmp) / name,
                    natural_search=(z, natural), preserve_data=(z, refs), residual=metric))
        grids, selected, rows = results[0]
        other_grids, other_selected, other_rows = results[1]
        self.assertEqual(selected["step"], other_selected["step"])
        self.assertGreater(other_selected["gradient_updates"], 0)
        for a, b in zip(rows, other_rows):
            self.assertAlmostEqual(a["selection_objective"], b["selection_objective"], places=7)
        for a, b in zip(grids, other_grids):
            self.assertTrue(torch.equal(a(False), b(False)))
        for k, value in vae.state_dict().items():
            self.assertTrue(torch.equal(value, original[k]))

        args.warm_qat_mode = 'centered_scale'
        warm = {'provenance': {'test': True}, 'codes': {'weight': (other_grids[0](False) / other_grids[0].scale).round().detach()}}
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            enhanced, selection, history = optimize_branch(vae, ['weight'], z, natural, z, refs,
                args, 4, 1., 'natural_residual_qat_warm', [], Path(tmp) / 'warm',
                natural_search=(z, natural), preserve_data=(z, refs), residual=metric, warm_start=warm)
        self.assertGreater(selection['gradient_updates'], 0)
        self.assertEqual(history[0]['code_change_fraction_vs_warm'], 0.)
        self.assertIn('warm_quality_penalty', history[-1])
        self.assertTrue(enhanced[0].log_scale.requires_grad)
        for k, value in vae.state_dict().items():
            self.assertTrue(torch.equal(value, original[k]))


if __name__ == "__main__":
    unittest.main()
