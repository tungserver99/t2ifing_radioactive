import io
import json
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout

import torch
from torch import nn
from wmq_blind import parser, optimize_branch, RoundingGrid
from wmq_tradeoff import quality_views, frontier
from run_blind_suite import run_suite


class TradeoffTests(unittest.TestCase):
    def test_conditional_evasion_and_quality_are_paired_not_marginal(self):
        owner = {'reference_validation': {'valid': True}, 'rows': [
            {'method': 'ref', 'per_image_detected': [True, True, False], 'ssim': 1., 'psnr': 120., 'tpr': 2/3},
            {'method': 'attack', 'per_image_detected': [False, True, False], 'ssim': .82, 'psnr': 30., 'tpr': 1/3}]}
        report = {'branch_quality': {'ref': {}, 'attack': {'min_psnr': 29.,
            'per_image_psnr': [29., 30., 31.], 'per_image_ssim': [.8, .91, .75]}}}
        manifest = {'reference_label': 'ref', 'args': {'min_psnr': 25., 'min_image_psnr': 22.},
            'test_branches': [{'label': 'ref'}, {'label': 'attack', 'comparison_group': 'w4'}]}
        points, views = quality_views(owner, report, manifest, [.85, .8])
        self.assertAlmostEqual(points[1]['conditional_evasion'], .5)
        self.assertAlmostEqual(points[1]['absolute_evasion'], 2/3)
        self.assertFalse(views[2]['aggregate_quality_valid'])
        self.assertEqual(views[2]['joint_quality_and_evasion_on_baseline_detected'], 0.)
        self.assertTrue(views[3]['aggregate_quality_valid'])
        self.assertEqual(views[3]['joint_quality_and_evasion_on_baseline_detected'], .5)
        owner['reference_validation']['valid'] = False
        self.assertIsNone(quality_views(owner, report, manifest, [.85])[0][1]['conditional_evasion'])

    def test_frontier_keeps_tradeoffs_and_removes_dominated_points(self):
        a = {'ssim': .9, 'conditional_evasion': .1}
        b = {'ssim': .85, 'conditional_evasion': .4}
        c = {'ssim': .8, 'conditional_evasion': .3}
        self.assertEqual(frontier([a, b, c]), [a, b])

    def test_preservation_sweep_declares_distinct_runs_without_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'suite'
            result = run_suite(root, [3407], 'focused', [], False,
                preservation_weights=[.5, 2., 8.], min_ssim=.85)
            self.assertEqual(len(result['runs']), 3)
            self.assertEqual(len({r['output'] for r in result['runs']}), 3)
            for run in result['runs']:
                command = run['command']
                # Each sweep option is emitted exactly once.
                for flag in ['--preserve-weight', '--qat-semantic-preserve-weight']:
                    indices = [i for i, x in enumerate(command) if x == flag]
                    self.assertEqual(float(command[indices[-1]+1]), run['preserve_weight'])

    def test_owner_mask_freezes_other_grids_but_keeps_them_quantized(self):
        class VAE(nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = nn.Sequential(nn.Conv2d(3, 3, 1), nn.Conv2d(3, 3, 1))
                self.post_quant_conv = nn.Identity()
            def decode(self, z, **kw):
                return (self.decoder(z),)
        torch.set_num_threads(2)
        torch.manual_seed(3)
        vae = VAE().requires_grad_(False)
        names = ['0.weight', '1.weight']
        initial = RoundingGrid(vae.decoder[1].weight, 4)(False).detach().clone()
        zs = [torch.rand(1, 3, 16, 16)]
        refs = [(vae.decode(zs[0])[0]/2+.5).clamp(0, 1)]
        natural = [torch.full_like(refs[0], .3)]
        args = parser().parse_args(['--model', 'x', '--prompts', 'x', '--output', 'x',
            '--steps', '2', '--eval-every', '1', '--natural-perceptual-weight', '0'])
        args._steering_layers = ['0.weight']
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            grids, selected, rows = optimize_branch(vae, names, zs, natural, zs, refs, args,
                4, 1., 'natural_rounding', [], Path(tmp)/'branch',
                natural_search=(zs, natural), preserve_data=(zs, refs))
        self.assertEqual(selected['valid_updates'], 2)
        self.assertEqual(selected['steering_threat_model'], 'owner_informed_development_not_blind')
        self.assertTrue(torch.equal(grids[1](False), initial))
        self.assertIsNone(grids[1].alpha.grad)
        self.assertIsNotNone(grids[0].alpha.grad)


if __name__ == '__main__':
    unittest.main()
