import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file

from wmq_science import QualityBudget, joint_quality_evasion, audit_prompt_protocol
from wmq_residual import ResidualSubspace, remove_dc, subspace_variant
from wmq_blind import export_finetune_rtn, parser
from run_blind_suite import configuration, run_suite
from prepare_final_prompts import make_prompts
from run_blind_replication import prepare, digest


class ScienceTests(unittest.TestCase):
    def test_quality_penalty_gradient_and_joint_denominator(self):
        x = torch.full((2, 3, 4, 4), .1, requires_grad=True)
        budget = QualityBudget(30., .9)
        penalty, violations = budget.terms(x, torch.zeros_like(x), torch.tensor([.8, .95]))
        penalty.backward()
        self.assertGreater(x.grad.norm().item(), 0.)
        budget.update(violations)
        self.assertTrue(all(v > .01 for v in budget.multipliers))
        result = joint_quality_evasion([1, 1, 1, 0], [0, 0, 1, 0],
                                      [31, 20, 40, 40], [.95] * 4, 30, .9)
        self.assertEqual(result['joint_success_count'], 1)
        self.assertEqual(result['joint_success_denominator'], 3)
        self.assertAlmostEqual(result['joint_success_rate'], 1 / 3)
        with self.assertRaises(ValueError):
            joint_quality_evasion([1], [0], [float('nan')], [.95], 30, .9)

    def test_subspace_controls_and_energy_normalization(self):
        rng = torch.Generator().manual_seed(98)
        basis = torch.linalg.qr(remove_dc(torch.randn(9, 12, generator=rng, dtype=torch.float64), 2).T)[0].T
        moment = basis.T @ torch.diag(torch.arange(1, 10, dtype=torch.float64)) @ basis
        reference = ResidualSubspace(basis[-2:], 2)
        reference.residual_moment = moment
        reference.texture_moment = basis.T @ torch.diag(torch.arange(9, 0, -1, dtype=torch.float64)) @ basis
        for kind in ('pca', 'random', 'frequency', 'contrastive'):
            a, info = subspace_variant(reference, {}, kind)
            b, _ = subspace_variant(reference, {}, kind)
            self.assertTrue(torch.equal(a.basis, b.basis))
            self.assertTrue(torch.allclose(a.basis @ a.basis.T, torch.eye(2), atol=1e-6))
            self.assertLess(a.basis.reshape(2, 3, 4).mean(-1).abs().max().item(), 1e-6)
            self.assertAlmostEqual(info['variant_energy'] * info['loss_scale'], info['reference_energy'], places=6)
            self.assertFalse(info['ownership_subspace_identified'])
        a, _ = subspace_variant(reference, {}, 'random', 1)
        b, _ = subspace_variant(reference, {}, 'random', 2)
        self.assertFalse(torch.allclose(a.basis.T @ a.basis, b.basis.T @ b.basis))
        contrastive, _ = subspace_variant(reference, {}, 'contrastive')
        # In this commuting synthetic case the two largest residual/texture
        # ratios have known eigenvectors; a random basis cannot pass this check.
        self.assertTrue(torch.allclose(contrastive.basis.T @ contrastive.basis,
                                       reference.basis.T @ reference.basis, atol=1e-5))

    def test_finetune_then_rtn_keeps_bias_and_restores_parent(self):
        class VAE(nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = nn.Conv2d(3, 3, 1)
            def save_pretrained(self, path, **kwargs):
                Path(path).mkdir()
        vae = VAE().requires_grad_(False)
        vae.decoder.bias.fill_(.123)
        original = {k: v.clone() for k, v in vae.decoder.state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'derived'
            export_finetune_rtn(vae, ['weight'], 4, 1., out)
            full = load_file(str(out / 'decoder_fp32.safetensors'))
            quant = load_file(str(out / 'quantizer.safetensors'))
            self.assertTrue(torch.equal(full['bias'], original['bias']))
            self.assertTrue(torch.allclose(full['weight'], quant['weight.codes'] * quant['weight.scale']))
            self.assertTrue((quant['weight.codes'] >= -8).all() and (quant['weight.codes'] <= 7).all())
        for k, v in vae.decoder.state_dict().items():
            self.assertTrue(torch.equal(v, original[k]))

    def test_science_configuration_and_final_prompt_audit(self):
        for profile in ('pilot', 'focused', 'science', 'transfer', 'full'):
            options = ['--model', 'x', '--prompts', 'x', '--output', 'x', *configuration(profile)]
            self.assertEqual(parser().parse_args(options).bits, [4])
            self.assertEqual(parser().parse_args([*options, '--bits', '8']).bits, [8])
        transfer = parser().parse_args(['--model', 'x', '--prompts', 'x', '--output', 'x', *configuration('transfer')])
        self.assertNotIn('natural_teacher_rounding', transfer.natural_methods)
        self.assertIn('natural_delta_residual', transfer.natural_methods)
        self.assertFalse(any('finetune' in name for name in transfer.natural_methods))
        self.assertNotIn('qk_rotation_ptq', transfer.methods)
        self.assertIn('natural_residual_qat_warm', transfer.natural_methods)
        self.assertEqual(transfer.warm_qat_mode, 'centered_scale')
        self.assertEqual(transfer.quant_refinement, 'balanced')
        self.assertEqual(transfer.residual_basis, 'contrastive')
        self.assertTrue(transfer.reconstruction_learn_scale)
        self.assertNotIn('natural_residual_cycle_qat_warm', transfer.natural_methods)
        self.assertNotIn('natural_joint_quality_finetune', transfer.natural_methods)
        self.assertNotIn('natural_joint_finetune', transfer.natural_methods)
        self.assertEqual(transfer.cycle_weight, .01)
        self.assertEqual(transfer.budget_ssim, .8)
        self.assertEqual(transfer.warm_checkpoint_policy, 'final')
        self.assertTrue(transfer.evaluate_final)
        self.assertNotIn('natural_random_subspace', transfer.natural_methods)
        self.assertEqual(transfer.finetune_rtn_bits, [])
        self.assertEqual(transfer.steps, 2000)
        args = parser().parse_args(['--model', 'x', '--prompts', 'x', '--output', 'x', *configuration('science')])
        self.assertEqual(args.bits, [4])
        self.assertEqual((args.steps, args.qat_steps, args.ft_steps), (2000, 2000, 2000))
        self.assertIn('natural_teacher_rounding', args.natural_methods)
        self.assertEqual(args.teacher_checkpoint_policy, 'selected')
        self.assertEqual(args.finetune_rtn_bits, [4])
        self.assertEqual(args.quality_constraint, 'off')
        self.assertIn('fixed_ptq', args.methods)
        self.assertIn('natural_random_subspace', args.natural_methods)
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / 'manifest.json'
            manifest.write_text(json.dumps({'prompts': [' old prompt ']}))
            with self.assertRaises(ValueError):
                audit_prompt_protocol(['OLD   PROMPT'], 0, 'final', [manifest])
            with self.assertRaises(ValueError):
                audit_prompt_protocol(['new'], 0, 'final', [])
            prompts = make_prompts([manifest], 152, 42)
            self.assertEqual(len(set(prompts)), 152)
            self.assertEqual(audit_prompt_protocol(prompts, 52, 'final', [manifest])['overlap_count'], 0)
            plan = run_suite(Path(tmp) / 'suite', [1, 2], 'science', [], False,
                             quality_budgets=[.8, .9])
            self.assertEqual(len(plan['runs']), 4)
            self.assertTrue(all(r['status'] == 'planned_not_executed' for r in plan['runs']))
            for run in plan['runs']:
                cfg = parser().parse_args(['--model', 'x', '--prompts', 'x', *run['command'][2:]])
                self.assertEqual(cfg.budget_ssim, run['budget_ssim'])

    def test_replication_rejects_reused_key_and_checkpoint(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dev = root / 'dev'
            dev.mkdir()
            model = root / 'model'
            (model / 'vae').mkdir(parents=True)
            (model / 'model_index.json').write_text('{}')
            weight = model / 'vae' / 'weights.safetensors'
            weight.write_bytes(b'new marked weights')
            manifest = dev / 'manifest.json'
            manifest.write_text(json.dumps({'prompts': ['old'],
                'model_files_sha256': {'vae/weights.safetensors': hashlib.sha256(b'old').hexdigest()}}))
            (dev / 'owner_evaluation.json').write_text(json.dumps({'key_provenance': {
                'sha256': hashlib.sha256(('0' * 48).encode()).hexdigest()}}))
            extractor = root / 'extractor.pt'
            extractor.write_bytes(b'extractor')
            prompts = root / 'prompts.txt'
            prompts.write_text('\n'.join(make_prompts([manifest])))
            entry = {'id': 'new', 'model': str(model), 'key': '1' * 48,
                     'extractor': str(extractor), 'extractor_sha256': digest(extractor)}
            config = {'development_runs': [str(dev)], 'prompts': str(prompts), 'assets': [entry]}
            public, _, _ = prepare(config)
            self.assertNotIn('key', public[0][0])
            entry['key'] = '0' * 48
            with self.assertRaisesRegex(ValueError, 'reuses'):
                prepare(config)
            entry['key'] = '1' * 48
            weight.write_bytes(b'old')
            with self.assertRaisesRegex(ValueError, 'reuses'):
                prepare(config)


if __name__ == '__main__':
    unittest.main()
