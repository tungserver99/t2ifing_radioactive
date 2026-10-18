import tempfile
import importlib.util
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from wmq_sleepermark import (attach, detach, switch, snapshot, restore, selected_weights,
                            noise_target, state_hash, train_branch, METHODS, EQUIV_METHODS, DELTA_METHODS, ROLLOUT_METHODS, COHERENT_METHODS, EVOLUTION_METHODS, parser)
from wmq_fid import feature_fid
from wmq_sleeper_quality_select import (heldout_prompts, quality_feasible,
    better_quality_candidate, better_defense_candidate)

torch.set_num_threads(2)


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_blocks = nn.ModuleList([nn.ModuleDict({'attentions': nn.Linear(8, 8)})])
        self.other = nn.Linear(8, 8)
        self.checkpointing = False

    def enable_gradient_checkpointing(self):
        self.checkpointing = True

    def disable_gradient_checkpointing(self):
        self.checkpointing = False

    def forward(self, x, t=None, encoder_hidden_states=None):
        layer = self.up_blocks[0]['attentions']
        y = checkpoint(layer, x, use_reentrant=False) if self.checkpointing else layer(x)
        return SimpleNamespace(sample=self.other(y))


class SleeperMarkTests(unittest.TestCase):
    def test_train_prompt_holdout_and_quality_only_checkpoint_choice(self):
        data = [(None, None, f'prompt{i}', 1) for i in range(12)]
        selected = heldout_prompts(data, 4, 17)
        self.assertEqual(len(selected), 3)
        self.assertEqual(selected, heldout_prompts(data, 4, 17))
        anchor = {'ordinary_ssim': .70, 'triggered_ssim': .68,
                  'ordinary_psnr': 20., 'triggered_psnr': 19.,
                  'triggered_to_clean_mse': .05}
        improved = {**anchor, 'triggered_to_clean_mse': .04, 'triggered_ssim': .66}
        damaged = {**improved, 'ordinary_ssim': .50}
        self.assertTrue(quality_feasible(improved, anchor, .03, 1.))
        self.assertTrue(better_quality_candidate(improved, anchor, anchor, .03, 1.))
        self.assertFalse(better_quality_candidate(damaged, anchor, anchor, .03, 1.))
        defense_anchor = {**anchor, 'ordinary_to_marked_mse': .01,
                          'triggered_to_marked_mse': .03}
        defense_better = {**defense_anchor, 'triggered_to_marked_mse': .02,
                          'triggered_ssim': .66}
        self.assertTrue(better_defense_candidate(defense_better, defense_anchor,
                                                defense_anchor, .03, 1.))
        self.assertFalse(better_defense_candidate({**defense_better, 'triggered_ssim': .5},
                                                 defense_anchor, defense_anchor, .03, 1.))

    def setUp(self):
        torch.manual_seed(17)
        self.unet = TinyUNet().requires_grad_(False)
        self.names = selected_weights(self.unet, 'up_attentions')

    def test_real_weight_change_export_and_exact_restoration(self):
        original_hash = state_hash(self.unet)
        original = snapshot(self.unet, self.names)
        controls = attach(self.unet, self.names)
        changed = snapshot(self.unet, self.names)
        self.assertFalse(torch.equal(changed[self.names[0]], original[self.names[0]]))
        switch(controls, enabled=False)
        self.assertTrue(torch.equal(snapshot(self.unet, self.names)[self.names[0]], original[self.names[0]]))
        detach(controls)
        self.assertEqual(state_hash(self.unet), original_hash)
        restore(self.unet, changed)
        self.assertNotEqual(state_hash(self.unet), original_hash)

    def test_all_training_modes_change_only_selected_unet_weights(self):
        scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=10, prediction_type='epsilon'),
                                    add_noise=lambda z, noise, t: z + noise)
        data = [(torch.randn(1, 8), torch.randn(1, 1, 8), 'ordinary prompt') for _ in range(3)]
        args = SimpleNamespace(ft_lr=.01, lr=.01, seed=5, steps=3, train_batch_size=1,
                               preserve_weight=.5, log_every=3, prefix_weight=.25,
                               public_trigger_weight=1., quant_group_size=4)
        before = state_hash(self.unet)
        with tempfile.TemporaryDirectory() as tmp, patch('wmq_sleepermark.encode_text',
                side_effect=lambda pipe, prompts: torch.zeros(len(prompts), 1, 8)):
            for method in METHODS:
                if (method in (*EQUIV_METHODS, *ROLLOUT_METHODS, *COHERENT_METHODS,
                               *EVOLUTION_METHODS, 'public_trigger_rollout_qat',
                               'public_trigger_distill_defense_w4')
                        or method.startswith('delta_')):
                    continue  # Spatial branches use a spatial UNet in the integration test below.
                result = train_branch(SimpleNamespace(unet=self.unet), scheduler, data,
                                      self.names, method, args, Path(tmp))
                self.assertEqual(before, state_hash(self.unet))
                for label, state in result.items():
                    self.assertEqual(set(state), set(self.names))
                    self.assertFalse(torch.equal(state[self.names[0]], self.unet.get_parameter(self.names[0])))

    def test_checkpoint_gradient_matches_without_checkpoint(self):
        x = torch.randn(2, 8)
        controls = attach(self.unet, self.names, finetune=True)
        parameter = controls[0][2].weight
        for quantized in (False, True):
            switch(controls, quantized=quantized)
            grads = []
            for enabled in (False, True):
                self.unet.checkpointing = enabled
                parameter.grad = None
                self.unet(x).sample.square().sum().backward()
                grads.append(parameter.grad.clone())
            torch.testing.assert_close(*grads)
            self.assertGreater(float(grads[0].abs().sum()), 0)
        detach(controls)

    def test_noise_targets(self):
        z, noise, t = torch.randn(2, 3), torch.randn(2, 3), torch.tensor([1, 2])
        scheduler = SimpleNamespace(config=SimpleNamespace(prediction_type='epsilon'),
                                    get_velocity=lambda z, n, t: z - n)
        self.assertIs(noise_target(scheduler, z, noise, t), noise)
        scheduler.config.prediction_type = 'v_prediction'
        torch.testing.assert_close(noise_target(scheduler, z, noise, t), z-noise)
        scheduler.config.prediction_type = 'sample'
        self.assertIs(noise_target(scheduler, z, noise, t), z)
        scheduler.config.prediction_type = 'bad'
        with self.assertRaises(ValueError):
            noise_target(scheduler, z, noise, t)

    @unittest.skipUnless(importlib.util.find_spec('diffusers'), 'Optional real diffusers CPU integration')
    def test_diffusers_unet_checkpointed_training(self):
        from diffusers import UNet2DConditionModel, DDPMScheduler, DDIMScheduler
        unet = UNet2DConditionModel(sample_size=16, in_channels=4, out_channels=4,
            down_block_types=('CrossAttnDownBlock2D', 'DownBlock2D'),
            up_block_types=('UpBlock2D', 'CrossAttnUpBlock2D'), block_out_channels=(16, 32),
            layers_per_block=1, norm_num_groups=8, cross_attention_dim=8, attention_head_dim=4)
        unet.requires_grad_(False)
        names = selected_weights(unet, 'up_attentions')
        before = state_hash(unet)
        original = snapshot(unet, names)
        data = [(torch.randn(1, 4, 16, 16), torch.randn(1, 3, 8), f'ordinary prompt {i}') for i in range(3)]
        args = parser().parse_args(['--steps', '4', '--log-every', '4', '--spatial-shift', '1',
                                    '--probe-every', '1', '--probe-steps', '1', '--inference-steps', '5'])
        with tempfile.TemporaryDirectory() as tmp, patch('wmq_sleepermark.encode_text',
                side_effect=lambda pipe, prompts: torch.zeros(len(prompts), 3, 8)):
            for method in ('natural_rounding', 'natural_joint_finetune', 'cfg_reconstruction',
                           'public_trigger_consistency_qat', *EQUIV_METHODS, *DELTA_METHODS,
                           *ROLLOUT_METHODS, 'coherent_probe_qat'):
                dataset = [(*row, 3) for row in data] if method in ('cfg_reconstruction',
                    'public_trigger_consistency_qat', *EQUIV_METHODS, *DELTA_METHODS,
                    *ROLLOUT_METHODS, *COHERENT_METHODS) else data
                args.delta_lr = 1.  # Cross an integer cell in the two-step smoke test.
                result = train_branch(SimpleNamespace(unet=unet, scheduler=DDIMScheduler(num_train_timesteps=10)), DDPMScheduler(num_train_timesteps=10),
                                      dataset, names, method, args, Path(tmp), Path(tmp))
                self.assertEqual(before, state_hash(unet))
                label = method + ('_fp32base_delta4' if method in DELTA_METHODS else '_w4')
                exported = result[label]
                if method in DELTA_METHODS:
                    from safetensors.torch import load_file
                    codes = load_file(str(Path(tmp) / (label + '_delta.safetensors')))
                    for name in names:
                        rebuilt = original[name] + codes[name+'.codes'].float()*codes[name+'.scale']
                        torch.testing.assert_close(rebuilt, exported[name], rtol=0, atol=0)
                restore(unet, exported)
                self.assertNotEqual(before, state_hash(unet))
                restore(unet, original)

    def test_fid_exact_low_rank_and_translation(self):
        rng = np.random.default_rng(1)
        a, b = rng.normal(size=(5, 20)), rng.normal(size=(7, 20))
        self.assertAlmostEqual(feature_fid(a, a), 0, places=10)
        self.assertAlmostEqual(feature_fid(a, a + .5), 5., places=10)
        self.assertAlmostEqual(feature_fid(a, b), feature_fid(b, a), places=10)
        # PSD sandwich form, independently calculated in feature space.
        ca, cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
        vals, vectors = np.linalg.eigh(ca)
        root = (vectors * np.sqrt(vals.clip(0))) @ vectors.T
        cross = np.sqrt(np.linalg.eigvalsh(root @ cb @ root).clip(0)).sum()
        expected = ((a.mean(0)-b.mean(0))**2).sum() + np.trace(ca+cb) - 2*cross
        self.assertAlmostEqual(feature_fid(a, b), expected, places=5)
        with self.assertRaises(ValueError):
            feature_fid(a[:1], b)
        with self.assertRaises(ValueError):
            feature_fid(a * np.nan, b)


if __name__ == '__main__':
    unittest.main()
