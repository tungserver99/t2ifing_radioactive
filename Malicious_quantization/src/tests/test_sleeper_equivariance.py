import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from wmq_sleeper_equivariance import (predicted_x0, spatial_target, probe_condition,
                                     TrajectoryCapture, interior, guided_spatial_target, guided_prediction)

torch.set_num_threads(2)


class ToySpatialModel(nn.Module):
    def forward(self, x, t, encoder_hidden_states):
        # An equivariant signal plus a coordinate-fixed, condition-controlled pattern.
        pattern = ((torch.arange(x.shape[-2], device=x.device) % 2) * 2 - 1)[None, None, :, None]
        strength = encoder_hidden_states[:, 1:].mean((1, 2))[:, None, None, None]
        return SimpleNamespace(sample=x + strength * pattern)


class SpatialTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.x = torch.randn(2, 4, 16, 16)
        self.c = torch.ones(2, 4, 8)
        self.t = torch.tensor([1, 3])
        self.scheduler = SimpleNamespace(config=SimpleNamespace(prediction_type='sample'),
                                         alphas_cumprod=torch.linspace(.99, .1, 10))

    def test_x0_prediction_types(self):
        epsilon = torch.randn_like(self.x)
        a = self.scheduler.alphas_cumprod[self.t].reshape(-1, 1, 1, 1)
        noisy = a.sqrt() * self.x + (1-a).sqrt() * epsilon
        self.scheduler.config.prediction_type = 'epsilon'
        torch.testing.assert_close(predicted_x0(epsilon, noisy, self.t, self.scheduler), self.x)
        self.scheduler.config.prediction_type = 'v_prediction'
        velocity = a.sqrt() * epsilon - (1-a).sqrt() * self.x
        torch.testing.assert_close(predicted_x0(velocity, noisy, self.t, self.scheduler), self.x)

    def test_spatial_target_reduces_fixed_pattern_with_bounded_correction(self):
        model = ToySpatialModel()
        base = model(self.x, self.t, self.c).sample
        target, info = spatial_target(model, self.x, self.t, self.c, (1, 0), self.scheduler, .25)
        self.assertLess(float((interior(target-self.x, (1, 0))).square().mean()),
                        float((interior(base-self.x, (1, 0))).square().mean()))
        self.assertLessEqual(float(info['correction_rms']), .250001)
        torch.testing.assert_close(target[..., :3, :], base[..., :3, :])
        self.assertFalse(target.requires_grad)

    def test_probe_bounds_and_ascent_without_parameter_updates(self):
        model = ToySpatialModel()
        result, stats = probe_condition(model, self.x, self.t, self.c, (1, 0), self.scheduler,
                                        .2, 3, 2, torch.Generator().manual_seed(2))
        self.assertGreater(float(stats['probe_final']), float(stats['probe_initial']))
        torch.testing.assert_close(result[:, 0], self.c[:, 0])
        torch.testing.assert_close(result[:, 3:], self.c[:, 3:])
        self.assertTrue(((result-self.c).flatten(1).norm(dim=1) <= .2*self.c[:, 1:3].flatten(1).norm(dim=1)+1e-6).all())
        self.assertFalse(result.requires_grad)

    def test_guided_target_matches_inference_cfg_and_caps_correction(self):
        model = ToySpatialModel()
        empty = self.c * 0
        base = guided_prediction(model, self.x, self.t, self.c, empty)
        torch.testing.assert_close(base, self.x + 7.5*(model(self.x, self.t, self.c).sample-self.x))
        target, info = guided_spatial_target(model, self.x, self.t, self.c, empty,
                                            (1, 0), self.scheduler, .25)
        self.assertLessEqual(float(info['correction_rms']), .250001)
        self.assertLess(float(interior(target-self.x, (1, 0)).square().mean()),
                        float(interior(base-self.x, (1, 0)).square().mean()))
        self.assertFalse(target.requires_grad)

    def test_capture_pre_forward_and_close(self):
        model = ToySpatialModel()
        capture = TrajectoryCapture(model, 4, 2)
        for t in range(4):
            model(self.x + t, torch.tensor(t), encoder_hidden_states=self.c)
        capture.close()
        self.assertEqual([t for _, t in capture.records], [0, 3])
        torch.testing.assert_close(capture.records[-1][0], self.x[:1]+3)
        model(self.x, torch.tensor(0), encoder_hidden_states=self.c)
        self.assertEqual(capture.calls, 4)

    def test_spatial_training_restores_source_and_exports_changed_weights(self):
        from wmq_sleepermark import parser, train_branch, selected_weights, state_hash, EQUIV_METHODS
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.up_blocks = nn.ModuleList([nn.ModuleDict({'attentions': nn.Conv2d(4, 4, 1)})])
                self.checkpointing = False
            def enable_gradient_checkpointing(self):
                self.checkpointing = True
            def disable_gradient_checkpointing(self):
                self.checkpointing = False
            def forward(self, x, t, encoder_hidden_states):
                x = x + encoder_hidden_states.mean((1, 2))[:, None, None, None]
                layer = self.up_blocks[0]['attentions']
                value = checkpoint(layer, x, use_reentrant=False) if self.checkpointing else layer(x)
                return SimpleNamespace(sample=value)
        model = Tiny().requires_grad_(False)
        before = state_hash(model)
        names = selected_weights(model, 'up_attentions')
        args = parser().parse_args(['--steps', '2', '--log-every', '1', '--spatial-shift', '1',
                                    '--probe-every', '1', '--probe-steps', '1'])
        self.scheduler.config.num_train_timesteps = 10
        dataset = [(self.x[:1], self.c[:1], 'ordinary', 1)]
        with tempfile.TemporaryDirectory() as tmp, patch('wmq_sleepermark.encode_text', return_value=self.c[:1]*0):
            for method in EQUIV_METHODS:
                results = train_branch(SimpleNamespace(unet=model), self.scheduler, dataset,
                                       names, method, args, Path(tmp))
                self.assertEqual(state_hash(model), before)
                self.assertFalse(torch.equal(results[method+'_w4'][names[0]], model.get_parameter(names[0])))


if __name__ == '__main__':
    unittest.main()
