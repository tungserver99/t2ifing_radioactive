import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import torch
from safetensors.torch import load_file
from wmq_delta_quant import DeltaGrid
from wmq_blind import (parser, optimize_branch, export_quantizer, materialize,
                       RoundingGrid)
from wmq_residual import ResidualSubspace, remove_dc
from tests.test_quant_refinement import TinyVAE


class DeltaTests(unittest.TestCase):
    def test_identity_integer_correction_gradient_and_frozen_base(self):
        torch.manual_seed(11)
        weight = torch.randn(3, 7)
        grid = DeltaGrid(weight)
        torch.testing.assert_close(grid(False), weight, rtol=0, atol=0)
        self.assertNotIn('source', dict(grid.named_parameters()))
        grid().sum().backward()
        self.assertGreater(grid.code_offset.grad.abs().sum(), 0)
        self.assertIsNone(grid.source.grad)
        with torch.no_grad():
            grid.code_offset.copy_(torch.linspace(-12, 12, weight.numel()).reshape_as(weight))
            grid.log_scale.fill_(.1)
        grid.clamp_parameters()
        codes = grid.codes()
        self.assertTrue(((codes >= -8) & (codes <= 7)).all())
        torch.testing.assert_close(grid(), grid(False), rtol=0, atol=0)
        torch.testing.assert_close(grid(False), weight + codes * grid.scale, rtol=0, atol=0)
        torch.testing.assert_close(grid.source, weight, rtol=0, atol=0)

    def test_natural_delta_training_export_and_reload(self):
        torch.manual_seed(8)
        vae = TinyVAE().requires_grad_(False)
        original = {k: v.clone() for k, v in vae.decoder.state_dict().items()}
        z = [torch.rand(1, 3, 16, 16)]
        refs = [(vae.decode(z[0])[0] / 2 + .5).clamp(0, 1).detach()]
        natural = [refs[0] * .9]
        basis = torch.linalg.qr(remove_dc(torch.randn(2, 12), 2).T)[0].T
        args = parser().parse_args(['--model', 'x', '--prompts', 'x', '--output', 'x',
            '--steps', '4', '--eval-every', '1', '--warmup-steps', '0', '--warm-qat-lr', '1',
            '--natural-perceptual-weight', '0', '--quality-policy', 'report'])
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            grids, selected, rows = optimize_branch(vae, ['weight'], z, natural, z, refs,
                args, 4, 1., 'natural_delta_residual', [], Path(tmp)/'branch',
                natural_search=(z, natural), preserve_data=(z, refs), residual=ResidualSubspace(basis, 2))
            self.assertEqual(selected['gradient_updates'], 4)
            self.assertEqual(rows[0]['psnr'], 120.)
            self.assertEqual(selected['parameter_space'], 'frozen_marked_fp32_base_plus_quantized_delta')
            for k, v in vae.decoder.state_dict().items():
                torch.testing.assert_close(v, original[k], rtol=0, atol=0)
            materialize(vae.decoder, ['weight'], grids)
            export_quantizer(vae, ['weight'], grids, Path(tmp))
            data = load_file(str(Path(tmp)/'quantizer.safetensors'))
            self.assertTrue(data['weight.codes'].ne(0).any())
            rebuilt = data['weight.base'] + data['weight.codes'].float()*data['weight.scale']
            torch.testing.assert_close(rebuilt, vae.decoder.weight, rtol=0, atol=0)
            torch.testing.assert_close(data['weight.base'], original['weight'], rtol=0, atol=0)
            self.assertIn('delta', json.loads((Path(tmp)/'quantizer.json').read_text())['representation'])

    def test_warm_fixed_endpoint_overrides_earlier_proxy_preference(self):
        torch.manual_seed(9)
        vae = TinyVAE().requires_grad_(False)
        with torch.no_grad():
            vae.decoder.weight.mul_(.2)
            vae.decoder.bias.zero_()
        z = [torch.rand(1, 3, 16, 16)]
        refs = [(vae.decode(z[0])[0]/2+.5).detach()]
        baseline = RoundingGrid(vae.decoder.weight, 4)
        warm = {'codes': {'weight': (baseline(False)/baseline.scale).round().detach()},
                'scales': {'weight': baseline.scale.detach().clone()}, 'provenance': {}}
        basis = torch.linalg.qr(remove_dc(torch.randn(2, 12), 2).T)[0].T
        args = parser().parse_args(['--model', 'x', '--prompts', 'x', '--output', 'x',
            '--steps', '4', '--eval-every', '1', '--natural-perceptual-weight', '0',
            '--warm-checkpoint-policy', 'final', '--warm-qat-mode', 'centered_scale'])
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), \
                patch('wmq_blind.choose', side_effect=lambda rows, policy: rows[0]):
            _, selected, _ = optimize_branch(vae, ['weight'], z, refs, z, refs,
                args, 4, 1., 'natural_residual_qat_warm', [], Path(tmp)/'warm',
                natural_search=(z, refs), preserve_data=(z, refs),
                residual=ResidualSubspace(basis, 2), warm_start=warm)
        self.assertEqual(selected['step'], 4)
        self.assertTrue(selected['proxy_preferred_candidate'].endswith('step0'))
        self.assertEqual(selected['checkpoint_policy'], 'fixed_final_step')


if __name__ == '__main__':
    unittest.main()
