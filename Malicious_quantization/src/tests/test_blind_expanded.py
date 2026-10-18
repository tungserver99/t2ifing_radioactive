import io
import json
import subprocess
import tempfile
import unittest
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import torch
from torch import nn

from wmq_blind import parser, optimize_branch, materialize, export_quantizer, select_names
from wmq_objectives import spectral_loss, NaturalDiscriminator, discriminator_update
from run_blind_suite import run_suite, configuration
from evaluate_blind_watermark import empirical_fpr, file_sha256

torch.set_num_threads(2)


class ToyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(nn.Conv2d(3, 4, 1), nn.Tanh(), nn.Conv2d(4, 3, 1))
        self.post_quant_conv = nn.Identity()

    def decode(self, z, **kwargs):
        return (self.decoder(z),)


class ExpandedTests(unittest.TestCase):
    def test_launcher_research_profile_and_explicit_overrides(self):
        # Execute the real shell argument routing without installing/downloading models.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stub = root / 'python'
            log = root / 'calls.jsonl'
            stub.write_text(f'#!{sys.executable}\n' +
                'import json, os, sys\n'
                'with open(os.environ["CALL_LOG"], "a") as f: f.write(json.dumps(sys.argv[1:]) + "\\n")\n'
                'sys.exit(42 if any(x.endswith("wmq_blind.py") for x in sys.argv) else 0)\n')
            stub.chmod(0o755)
            env = dict(os.environ, WMQ_BOOTSTRAP_READY='1', WMQ_PYTHON=str(stub),
                WMQ_EXPECTED_PREFIX=str(root), WMQ_PROFILE='research', CALL_LOG=str(log),
                WMQ_ATTACK_OUTPUT_ROOT=str(root / 'runs'), WMQ_HEAVY_ROOT=str(root / 'heavy'),
                WMQ_NEGATIVE_N='10', WMQ_MODEL_ONLY='0')
            launcher = Path(__file__).resolve().parents[1] / 'run_blind_quantization.sh'
            result = subprocess.run(['bash', str(launcher), '--steps', '17', '--train-batch-size', '1',
                '--natural-train-n', '3', '--natural-search-n', '2'], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 42, result.stdout + result.stderr)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            command = next(c for c in calls if c and c[0].endswith('wmq_blind.py'))
            args = parser().parse_args(command[1:])
            self.assertEqual(args.steps, 17)
            self.assertEqual(args.qat_steps, 2000)
            self.assertEqual(args.train_batch_size, 1)
            self.assertEqual(args.natural_train_n, 3)
            self.assertEqual(args.optimizer, 'adamw')
            self.assertTrue(args.evaluate_final)
            self.assertEqual(args.natural_preservation, 'lowpass')
            pool = next(c for c in calls if c and c[0].endswith('prepare_natural_images.py'))
            self.assertEqual(pool[pool.index('--count') + 1], '15')

    def test_sequential_blocks_with_real_diffusers_vae(self):
        from diffusers import AutoencoderKL
        vae = AutoencoderKL(in_channels=3, out_channels=3, block_out_channels=(32, 32),
            down_block_types=('DownEncoderBlock2D', 'DownEncoderBlock2D'),
            up_block_types=('UpDecoderBlock2D', 'UpDecoderBlock2D'), layers_per_block=1,
            latent_channels=4, norm_num_groups=8).eval().requires_grad_(False)
        z = [torch.randn(1, 4, 8, 8)]
        refs = [(vae.decode(z[0]).sample / 2 + .5).clamp(0, 1)]
        args = parser().parse_args(['--model', 'x', '--prompts', 'x', '--output', 'x',
            '--steps', '5', '--eval-every', '5', '--natural-perceptual-weight', '0'])
        source = {k: v.clone() for k, v in vae.decoder.state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            grids, selected, rows = optimize_branch(vae, select_names(vae, 'all'), z, refs, z, refs,
                args, 4, 1., 'block_reconstruction', [], Path(tmp) / 'branch')
        self.assertEqual(selected['valid_updates'], 5)
        self.assertIn('mid_block', selected['reconstruction_blocks'])
        self.assertIn('up_blocks.0', selected['reconstruction_blocks'])
        for key, value in vae.decoder.state_dict().items():
            self.assertTrue(torch.equal(value, source[key]))

    def test_negative_pool_excludes_training_bytes_and_keeps_uncertainty(self):
        from PIL import Image
        class Extractor(nn.Module):
            def forward(self, x):
                return torch.tensor([[1., -1.] * 24]).expand(len(x), -1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(3):
                Image.new('RGB', (16, 16), (i * 50, 100, 70)).save(root / f'{i}.png')
            excluded = {file_sha256(root / '0.png')}
            result = empirical_fpr(Extractor(), torch.ones(48, dtype=torch.bool), root,
                excluded, 100, 36, 'double', 'cpu')
        self.assertEqual(result['n'], 2)
        self.assertEqual(result['excluded'], 1)
        self.assertEqual(result['fpr'], 0.)
        self.assertGreater(result['fpr_ci95_high'], 0.)
        self.assertFalse(result['threshold_selected_using_negatives'])

    def test_new_branches_optimize_export_and_restore_source_with_failed_quality(self):
        for method in ['block_reconstruction', 'natural_spectral', 'natural_qat_scale',
                       'natural_gan_qat', 'natural_gan_finetune']:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
                torch.manual_seed(17)
                vae = ToyVAE().requires_grad_(False)
                original = {k: v.clone() for k, v in vae.state_dict().items()}
                zs = [torch.rand(1, 3, 16, 16) for _ in range(3)]
                refs = [(vae.decode(z)[0] / 2 + .5).clamp(0, 1) for z in zs]
                targets = [torch.full_like(x, .4) for x in refs]
                args = parser().parse_args(['--model', 'x', '--prompts', 'x', '--output', 'x',
                    '--steps', '4', '--eval-every', '2', '--train-batch-size', '2',
                    '--min-psnr', '121', '--min-image-psnr', '121', '--natural-perceptual-weight', '0',
                    '--optimizer', 'adamw', '--natural-preservation', 'lowpass'])
                natural = method.startswith('natural_')
                floating = method == 'natural_gan_finetune'
                names = [k for k, _ in vae.decoder.named_parameters() if floating or k.endswith('.weight')]
                folder = Path(tmp) / 'branch'
                grids, selected, rows = optimize_branch(vae, names, zs, targets if natural else refs,
                    zs, refs, args, 32 if floating else 4, 1., method, [], folder,
                    natural_search=(zs, targets) if natural else None,
                    preserve_data=(zs, refs) if natural else None)
                self.assertEqual(selected['valid_updates'], 4)
                self.assertFalse(selected['search_feasible'])
                self.assertEqual(selected['status'], 'selected_quality_failed')
                self.assertEqual(selected['training_batch_size'], 2)
                for key, value in vae.state_dict().items():
                    self.assertTrue(torch.equal(value, original[key]), key)
                updates = json.loads((folder / 'updates.json').read_text())['rows']
                if method == 'block_reconstruction':
                    self.assertEqual({r['block'] for r in updates}, {'0', '2'})
                if 'gan' in method:
                    self.assertEqual(selected['discriminator_valid_updates'], 4)
                    self.assertTrue(all(r['adversarial_loss'] > 0 for r in updates))
                if method == 'natural_qat_scale':
                    self.assertTrue(all(g.log_scale.grad is not None for g in grids))
                if not floating:
                    materialize(vae.decoder, names, grids)
                    export_quantizer(vae, names, grids, Path(tmp))
                    from safetensors.torch import load_file
                    tensors = load_file(str(Path(tmp) / 'quantizer.safetensors'))
                    for name in names:
                        self.assertTrue(torch.equal(vae.decoder.get_parameter(name),
                            tensors[name + '.codes'].float() * tensors[name + '.scale']))

    def test_spectral_loss_zero_identity_and_finite_gradient(self):
        a = torch.rand(2, 3, 16, 16, requires_grad=True)
        self.assertTrue(torch.equal(spectral_loss(a, a), torch.zeros(2)))
        loss = spectral_loss(a, torch.zeros_like(a)).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(a.grad).all())
        self.assertGreater(a.grad.abs().sum().item(), 0)

    def test_bad_discriminator_input_skips_update_without_poisoning_parameters(self):
        net = NaturalDiscriminator()
        original = {k: v.clone() for k, v in net.state_dict().items()}
        optimizer = torch.optim.Adam(net.parameters(), lr=.001)
        real = torch.rand(1, 3, 16, 16)
        value, valid = discriminator_update(net, optimizer, real, torch.full_like(real, float('nan')))
        self.assertFalse(valid)
        self.assertIsNone(value)
        for key, tensor in net.state_dict().items():
            self.assertTrue(torch.equal(tensor, original[key]))

    def test_suite_continues_after_failure_and_keeps_plan(self):
        calls = []
        def runner(command, **kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(command, 1)
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp) / 'suite'
            result = run_suite(root, [1, 2], 'pilot', [], True, runner)
            self.assertEqual(len(calls), 2)
            self.assertTrue(all(x['status'] == 'failed' for x in result['runs']))
            self.assertTrue(all(x['status'] == 'pending' for x in json.loads((root / 'suite_plan.json').read_text())['runs']))
            self.assertTrue((root / 'seed_1.log').exists())
            self.assertIn('suite_seed_1', calls[0][-1])
        full = configuration('full')
        self.assertEqual(full[full.index('--steps') + 1], full[full.index('--qat-steps') + 1])
        focused = configuration('focused')
        self.assertIn('natural_residual_qat', focused)
        self.assertIn('natural_full_finetune', focused)
        self.assertNotIn('natural_gan_qat', focused)
        self.assertIn('natural_gan_qat', full)


if __name__ == '__main__':
    unittest.main()
