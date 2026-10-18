import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
import torch
from torch import nn
from wmq_blind import parser, optimize_branch, initialize_warm_codes, RoundingGrid, REFINED_METHODS
from wmq_residual import ResidualSubspace, remove_dc

torch.set_num_threads(2)


class TinyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Conv2d(3, 3, 1)
        self.post_quant_conv = nn.Identity()
    def decode(self, z, **kwargs):
        return (self.decoder(z),)


class RefinementTests(unittest.TestCase):
    def test_warm_import_preserves_codes_and_changed_scales(self):
        weight = torch.randn(3, 7)
        grid = RoundingGrid(weight, 4, learn_code_offsets=True)
        codes = torch.full_like(weight, 3)
        scales = grid.base_scale * .91
        initialize_warm_codes([grid], ['w'], {'codes': {'w': codes}, 'scales': {'w': scales}}, 2, centered=True)
        torch.testing.assert_close(grid(False), codes * scales, rtol=0, atol=0)
        torch.testing.assert_close(grid.source, weight, rtol=0, atol=0)
        self.assertEqual(grid.code_offset.abs().sum(), 0)
        with self.assertRaises(ValueError):
            initialize_warm_codes([grid], ['w'], {'codes': {'w': codes}, 'scales': {'w': -scales}}, 2, centered=True)

    def test_all_active_quantization_branches_train_and_export(self):
        torch.manual_seed(8)
        vae = TinyVAE().requires_grad_(False)
        with torch.no_grad():
            vae.decoder.weight.mul_(.2)
            vae.decoder.bias.zero_()
        original = {k: v.clone() for k, v in vae.state_dict().items()}
        z = [torch.rand(1, 3, 16, 16)]
        refs = [(vae.decode(z[0])[0] / 2 + .5).detach()]
        natural = [refs[0] - .02]
        basis = torch.linalg.qr(remove_dc(torch.randn(2, 12), 2).T)[0].T
        residual = ResidualSubspace(basis, 2)
        args = parser().parse_args(['--model', 'x', '--prompts', 'x', '--output', 'x',
            '--steps', '4', '--eval-every', '1', '--warmup-steps', '1',
            '--natural-perceptual-weight', '0', '--quant-refinement', 'balanced',
            '--warm-checkpoint-policy', 'final',
            '--warm-qat-mode', 'centered_scale', '--reconstruction-learn-scale'])
        teacher = {'train': natural, 'search': natural, 'provenance': {}}
        warm = None
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            for method in ('reconstruction', *REFINED_METHODS):
                grid, selected, rows = optimize_branch(vae, ['weight'], z,
                    refs if method == 'reconstruction' else natural, z, refs, args, 4, 1.,
                    method, [], Path(tmp) / method, natural_search=(z, natural),
                    preserve_data=(z, refs), residual=residual, teacher_targets=teacher, warm_start=warm)
                self.assertEqual(selected['gradient_updates'], 4)
                if method == 'natural_residual_qat_warm':
                    self.assertEqual(selected['step'], 4)
                    self.assertEqual(selected['checkpoint_policy'], 'fixed_final_step')
                    self.assertIn('proxy_preferred_candidate', selected)
                self.assertTrue(grid[0].log_scale.requires_grad)
                if method in REFINED_METHODS:
                    self.assertGreaterEqual(selected['step'], 2)
                    self.assertFalse(rows[1]['selection_eligible'])
                    self.assertIn('quant_quality_penalty', rows[-1])
                    self.assertEqual(selected['optimized_dofs'], ['centered_bounded_code_offsets', 'per_channel_scale'])
                if method == 'natural_teacher_rounding':
                    self.assertAlmostEqual(selected['teacher_signal_train_mse'], .02**2, places=7)
                    self.assertAlmostEqual(selected['effective_teacher_weight'], .005/.02**2, places=3)
                if method == 'natural_residual':
                    warm = {'codes': {'weight': (grid[0](False)/grid[0].scale).round().detach()},
                            'scales': {'weight': grid[0].scale.detach().clone()}, 'provenance': {}}
                codes = grid[0](False)/grid[0].scale
                torch.testing.assert_close(codes, codes.round())
                self.assertTrue(((codes >= -8-1e-6) & (codes <= 7+1e-6)).all())
                for k, value in vae.state_dict().items():
                    torch.testing.assert_close(value, original[k], rtol=0, atol=0)

            # Refinement flags must not silently change the excluded FP32 control.
            states = []
            for mode in ('legacy', 'balanced'):
                args.quant_refinement = mode
                grids, selection, _ = optimize_branch(vae, ['weight', 'bias'], z, natural, z, refs,
                    args, 4, 1., 'natural_full_finetune', [], Path(tmp) / mode,
                    natural_search=(z, natural), preserve_data=(z, refs))
                states.append([g(False).detach().clone() for g in grids])
                self.assertEqual(selection['quant_refinement'], 'legacy')
            for a, b in zip(*states):
                torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
