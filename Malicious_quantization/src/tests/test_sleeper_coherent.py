import unittest
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from wmq_sleeper_coherent import (coherent_statistics, project_shared_delta, coherent_probe,
                                  coherent_target, sample_distinct_prompts)


class SharedPattern(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('fixed', torch.tensor(.2))

    def forward(self, sample, timestep, encoder_hidden_states):
        pattern = ((torch.arange(sample.shape[-2], device=sample.device)%2)*2-1)[None, None, :, None]
        gain = encoder_hidden_states[:, 1:3].mean((1, 2))[:, None, None, None]
        return SimpleNamespace(sample=sample*self.fixed + gain*pattern)


class CoherentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.x = torch.randn(2, 4, 16, 16)
        self.c = torch.ones(2, 4, 8)
        self.t = torch.tensor([3, 3])
        self.scheduler = SimpleNamespace(config=SimpleNamespace(prediction_type='sample'),
                                         alphas_cumprod=torch.linspace(.99, .1, 10))

    def test_pair_score_excludes_self_energy(self):
        pattern = torch.randn(1, 4, 16, 16)
        _, _, agreement, _ = coherent_statistics(pattern.expand(2, -1, -1, -1))
        self.assertAlmostEqual(float(agreement), 1., places=5)
        _, _, agreement, _ = coherent_statistics(torch.cat([pattern, -pattern]))
        self.assertAlmostEqual(float(agreement), -1., places=5)
        with self.assertRaises(ValueError):
            coherent_statistics(pattern)

    def test_shared_projection_preserves_bos_and_token_budget(self):
        self.c[1] *= .1
        delta = project_shared_delta(torch.randn(1, 4, 8), self.c, .2, 2)
        self.assertTrue(torch.equal(delta[:, 0], torch.zeros_like(delta[:, 0])))
        self.assertTrue(torch.equal(delta[:, 3], torch.zeros_like(delta[:, 3])))
        self.assertLessEqual(float(delta.norm()), .2*float(self.c[:, 1:3].flatten(1).norm(dim=1).min())+1e-6)

    def test_probe_finds_shared_response_without_modifying_teacher(self):
        model = SharedPattern()
        before = model.fixed.clone()
        context, delta, common, stats = coherent_probe(model, self.x, self.t, self.c,
            self.scheduler, None, torch.Generator().manual_seed(2), steps=3, tokens=2)
        self.assertGreater(float(stats['coherent_probe_final']), float(stats['coherent_probe_initial']))
        response = 7.5*(model(self.x, self.t, context).sample-model(self.x, self.t, self.c).sample)
        hp, pair, agreement, low = coherent_statistics(response)
        torch.testing.assert_close(stats['coherent_probe_final'], pair-low)
        torch.testing.assert_close(common, hp.mean(0, keepdim=True)*agreement.clamp(0, 1))
        torch.testing.assert_close(context[0]-self.c[0], context[1]-self.c[1])
        self.assertFalse(context.requires_grad or delta.requires_grad or common.requires_grad)
        torch.testing.assert_close(model.fixed, before)
        target, rms = coherent_target(model, self.x, self.t, context, self.c*0, common,
                                      self.scheduler, .05)
        self.assertTrue(torch.isfinite(target).all())
        self.assertLessEqual(float(rms), .050001)

    def test_zero_radius_gives_zero_correction(self):
        model = SharedPattern()
        context, delta, common, stats = coherent_probe(model, self.x, self.t, self.c,
            self.scheduler, None, torch.Generator().manual_seed(2), radius=0, steps=1)
        torch.testing.assert_close(context, self.c)
        self.assertEqual(float(common.abs().sum()), 0.)
        self.assertEqual(float(stats['coherent_probe_final']), 0.)

    def test_incoherent_response_is_not_suppressed(self):
        class Opposite(SharedPattern):
            def forward(self, sample, timestep, encoder_hidden_states):
                value = super().forward(sample, timestep, encoder_hidden_states).sample
                return SimpleNamespace(sample=value*torch.tensor([1., -1.]).reshape(2, 1, 1, 1))
        _, _, common, stats = coherent_probe(Opposite(), self.x, self.t, self.c,
            self.scheduler, None, torch.Generator().manual_seed(2), steps=0)
        self.assertLess(float(stats['coherent_agreement']), 0.)
        self.assertEqual(float(common.abs().sum()), 0.)

    def test_distinct_prompt_sampling_rejects_duplicate_only_data(self):
        data = [(None, None, p, 3) for p in ('a', 'a', 'b', 'c')]
        rng = np.random.default_rng(7)
        chosen = sample_distinct_prompts(data, range(4), rng, 2)
        self.assertNotEqual(data[chosen[0]][2], data[chosen[1]][2])
        with self.assertRaises(ValueError):
            sample_distinct_prompts(data, [0, 1], rng, 2)


if __name__ == '__main__':
    unittest.main()
