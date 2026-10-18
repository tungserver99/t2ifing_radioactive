import importlib.util
import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from wmq_sleeper_equivariance import guided_prediction, predicted_x0, interior
from wmq_sleeper_rollout import (conditional_orbit_target, prediction_from_x0,
                                make_rollout_scheduler, rollout_backward)


class PatternModel(nn.Module):
    def __init__(self, condition_gain=0., checkpointing=False):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.15))
        self.condition_gain = condition_gain
        self.enabled = False
        self.checkpointing = checkpointing
        self.seen = []

    def forward(self, x, t, encoder_hidden_states):
        self.seen.append((self.enabled, int(t[0]), x.detach().clone(), x.requires_grad))
        def compute(value):
            strength = encoder_hidden_states.mean((1, 2))[:, None, None, None]
            pattern = ((torch.arange(value.shape[-2]) % 2)*2-1).to(value)[None, None, :, None]
            # The unconditioned coordinate bias should never be removed by the new target.
            return value * (.2 + (self.weight if self.enabled else 0.)) + (1+self.condition_gain*strength)*pattern
        result = checkpoint(compute, x, use_reentrant=False) if self.checkpointing else compute(x)
        return SimpleNamespace(sample=result)


class ConditionalTargetTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.x = torch.randn(1, 4, 16, 16)
        self.c = torch.ones(1, 3, 8)
        self.u = self.c*0
        self.t = torch.tensor([3])
        self.scheduler = SimpleNamespace(config=SimpleNamespace(prediction_type='sample'),
                                         alphas_cumprod=torch.linspace(.99, .1, 10))

    def test_condition_independent_spatial_bias_is_unchanged(self):
        model = PatternModel()
        base = guided_prediction(model, self.x, self.t, self.c, self.u)
        target, stats = conditional_orbit_target(model, self.x, self.t, self.c, self.u,
                                                 (1, 0), self.scheduler, .25)
        torch.testing.assert_close(target, base, atol=0, rtol=0)
        self.assertEqual(float(stats['correction_rms']), 0.)
        self.assertFalse(target.requires_grad)

    def test_conditional_pattern_reduced_and_border_preserved(self):
        model = PatternModel(.1)
        base = guided_prediction(model, self.x, self.t, self.c, self.u)
        unconditional = model(self.x, self.t, self.u).sample
        target, stats = conditional_orbit_target(model, self.x, self.t, self.c, self.u,
                                                 (1, 0), self.scheduler, .25)
        self.assertLess(float(interior(target-unconditional, (1, 0)).square().mean()),
                        float(interior(base-unconditional, (1, 0)).square().mean()))
        self.assertLessEqual(float(stats['correction_rms']), .250001)
        torch.testing.assert_close(target[..., :3, :], base[..., :3, :])

    def test_prediction_roundtrip_all_parameterizations(self):
        prediction = torch.randn_like(self.x)
        for kind in ('epsilon', 'sample', 'v_prediction'):
            self.scheduler.config.prediction_type = kind
            x0 = predicted_x0(prediction, self.x, self.t, self.scheduler)
            restored = prediction_from_x0(x0, self.x, self.t, self.scheduler)
            torch.testing.assert_close(restored, prediction, atol=2e-6, rtol=2e-6)

    @unittest.skipUnless(importlib.util.find_spec('diffusers'), 'DDIM integration dependency')
    def test_independent_rollouts_and_checkpoint_gradient(self):
        from diffusers import DDIMScheduler
        scheduler = make_rollout_scheduler(DDIMScheduler(num_train_timesteps=10,
            clip_sample=False, prediction_type='epsilon'), 5)
        gradients = []
        for use_checkpoint in (False, True):
            model = PatternModel(.1, use_checkpoint)
            loss, stats = rollout_backward(model, lambda enabled: setattr(model, 'enabled', enabled),
                self.x, 4, self.c, self.u, (1, 0), scheduler, 2, .05, 1.)
            self.assertEqual(float(stats['rollout_transitions']), 2.)
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(float(model.weight.grad.abs()), 0.)
            self.assertTrue(model.enabled)
            # First call at the second timestep is unshifted. Both paths must
            # have moved independently and both states must have detached history.
            teacher = next(row for row in model.seen if not row[0] and row[1] == 2)
            student = next(row for row in model.seen if row[0] and row[1] == 2)
            self.assertFalse(torch.equal(teacher[2], student[2]))
            self.assertFalse(teacher[3] or student[3])
            gradients.append(model.weight.grad.clone())
        torch.testing.assert_close(*gradients)

    @unittest.skipUnless(importlib.util.find_spec('diffusers'), 'DDIM integration dependency')
    def test_terminal_step_is_not_repeated(self):
        from diffusers import DDIMScheduler
        scheduler = make_rollout_scheduler(DDIMScheduler(num_train_timesteps=10), 5)
        model = PatternModel(.1)
        _, stats = rollout_backward(model, lambda enabled: setattr(model, 'enabled', enabled),
            self.x, 0, self.c, self.u, (1, 0), scheduler, 4, .05, 1.)
        self.assertEqual(float(stats['rollout_transitions']), 1.)


if __name__ == '__main__':
    unittest.main()
