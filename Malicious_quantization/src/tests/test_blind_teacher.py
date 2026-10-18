import hashlib
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import save_file

from wmq_blind import parser, optimize_branch, materialize, RoundingGrid
from wmq_runtime import decoded01
from wmq_teacher import cache_teacher_targets, teacher_mse, target_signal


class TinyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Conv2d(3, 3, 1)
        self.post_quant_conv = nn.Identity()

    def decode(self, z, **kwargs):
        return (self.decoder(z),)


class TeacherTests(unittest.TestCase):
    def test_target_signal_distinguishes_identity_without_owner_metrics(self):
        refs = [torch.full((1, 3, 16, 16), .5)]
        identity = target_signal(refs, refs)
        self.assertTrue(identity['near_identity'])
        self.assertEqual(identity['reference_mse'], 0.)
        changed = target_signal([refs[0] + .1], refs)
        self.assertFalse(changed['near_identity'])
        self.assertAlmostEqual(changed['reference_mse'], .01, places=6)
        self.assertEqual(changed['watermark_status'], 'not_measured')
        with self.assertRaises(ValueError):
            target_signal([refs[0][:, :, :8]], refs)

    def test_cache_uses_teacher_bias_restores_marked_source_and_checks_hash(self):
        torch.manual_seed(21)
        vae = TinyVAE().requires_grad_(False)
        original = {k: v.clone() for k, v in vae.decoder.state_dict().items()}
        state = {k: v.clone() for k, v in original.items()}
        state['bias'].add_(.15)
        latents = [torch.rand(1, 3, 16, 16) for _ in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'teacher.safetensors'
            save_file(state, str(path))
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            targets = cache_teacher_targets(vae, path, expected, latents, 2)
            for k, v in vae.decoder.state_dict().items():
                self.assertTrue(torch.equal(v, original[k]))
            self.assertGreater(teacher_mse(vae, latents, targets, 0), 0.)
            vae.decoder.load_state_dict(state)
            self.assertAlmostEqual(teacher_mse(vae, latents, targets, 1), 0., places=10)
            vae.decoder.load_state_dict(original)
            with self.assertRaisesRegex(ValueError, 'changed'):
                cache_teacher_targets(vae, path, 'wrong hash', latents, 1)
            class Broken(TinyVAE):
                def decode(self, z, **kwargs):
                    raise RuntimeError('decode failed')
            broken = Broken().requires_grad_(False)
            broken.decoder.load_state_dict(original)
            with self.assertRaisesRegex(RuntimeError, 'decode failed'):
                cache_teacher_targets(broken, path, expected, latents, 1)
            for k, v in broken.decoder.state_dict().items():
                self.assertTrue(torch.equal(v, original[k]))

    def test_student_learns_rounding_without_changing_original_weights_or_bias(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.manual_seed(123)
        vae = TinyVAE().to(device).requires_grad_(False)
        with torch.no_grad():
            vae.decoder.weight.mul_(.3)
            vae.decoder.bias.zero_()
        original = {k: v.clone() for k, v in vae.decoder.state_dict().items()}
        z = [torch.rand(1, 3, 16, 16, device=device)]
        refs = [decoded01(vae.decode(z[0])[0])]
        target_grid = RoundingGrid(vae.decoder.weight, 4).to(device)
        with torch.no_grad():
            target_grid.alpha.fill_(12.)
            materialize(vae.decoder, ['weight'], [target_grid])
            targets = [decoded01(vae.decode(z[0])[0]).detach()]
            vae.decoder.load_state_dict(original)
        args = parser().parse_args(['--model', 'x', '--prompts', 'x', '--output', 'x',
            '--steps', '30', '--eval-every', '10', '--lr', '.5',
            '--preserve-weight', '0', '--natural-perceptual-weight', '0'])
        frozen = {'train': targets, 'search': targets, 'provenance': {'source': 'synthetic_test'}}
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            grids, selected, rows = optimize_branch(vae, ['weight'], z, targets, z, refs,
                args, 4, 1., 'natural_teacher_rounding', [], Path(tmp) / 'branch',
                natural_search=(z, targets), preserve_data=(z, refs), teacher_targets=frozen)
            with self.assertRaisesRegex(ValueError, 'aligned'):
                optimize_branch(vae, ['weight'], z, targets, z, refs, args, 4, 1.,
                    'natural_teacher_rounding', [], Path(tmp) / 'missing',
                    natural_search=(z, targets), preserve_data=(z, refs))
        self.assertLess(selected['teacher_search_mse'], rows[0]['teacher_search_mse'])
        self.assertGreater(selected['gradient_updates'], 0)
        self.assertTrue(torch.equal(grids[0].source, original['weight']))
        for k, v in vae.decoder.state_dict().items():
            self.assertTrue(torch.equal(v, original[k]))
        materialize(vae.decoder, ['weight'], grids)
        self.assertTrue(torch.equal(vae.decoder.bias, original['bias']))
        self.assertTrue(torch.allclose(vae.decoder.weight / grids[0].scale,
                                      (vae.decoder.weight / grids[0].scale).round(), atol=1e-6))
        self.assertTrue(all(t.grad is None and not t.requires_grad for t in targets))


if __name__ == '__main__':
    unittest.main()
