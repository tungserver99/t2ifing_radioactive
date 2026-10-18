import unittest
from types import SimpleNamespace
import numpy as np
import torch
from wmq_sleeper_calibration import (TimestepSampler, spatial_reconstruction_loss,
                                     augmented_prompts, cache_datasets_on_device)
from wmq_sleeper_equivariance import predicted_x0


class CalibrationTests(unittest.TestCase):
    def test_cpu_cache_preserves_shared_tensor_identity(self):
        shared = torch.randn(2, 3)
        datasets, stats = cache_datasets_on_device(
            [[(shared, 'a', 1), (shared, 'b', 2)], []], torch.device('cpu'), 'auto')
        self.assertIs(datasets[0][0][0], datasets[0][1][0])
        self.assertIs(datasets[0][0][0], shared)
        self.assertEqual(stats['unique_tensors'], 1)
        self.assertEqual(stats['device'], 'cpu')

    def test_invalid_cache_mode(self):
        with self.assertRaises(ValueError):
            cache_datasets_on_device([], torch.device('cpu'), 'invalid')

    def test_stratified_coverage_and_reproducibility(self):
        data = [(None, None, '', t) for t in (1, 1, 20, 50, 50, 50)]
        a = TimestepSampler(data, np.random.default_rng(10))
        b = TimestepSampler(data, np.random.default_rng(10))
        for _ in range(5):
            selected = a.sample(3)
            self.assertEqual(selected, b.sample(3))
            self.assertEqual(sorted(data[i][3] for i in selected), [1, 20, 50])

    def test_noise_loss_matches_prediction_error_for_all_parameterizations(self):
        x = torch.randn(2, 4, 8, 8)
        target, estimate = torch.randn_like(x), torch.randn_like(x, requires_grad=True)
        t = torch.tensor([0, 1])
        for kind in ('epsilon', 'v_prediction', 'sample'):
            scheduler = SimpleNamespace(config=SimpleNamespace(prediction_type=kind),
                                        alphas_cumprod=torch.tensor([.3, .8]))
            a, b = predicted_x0(estimate, x, t, scheduler), predicted_x0(target, x, t, scheduler)
            loss = spatial_reconstruction_loss(a, b, t, scheduler)
            torch.testing.assert_close(loss, (estimate-target).square().mean()/7.5**2)
            gradient, = torch.autograd.grad(loss, estimate)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0)

    def test_near_zero_timestep_is_bounded(self):
        scheduler = SimpleNamespace(config=SimpleNamespace(prediction_type='epsilon'),
                                    alphas_cumprod=torch.tensor([1.]))
        loss = spatial_reconstruction_loss(torch.ones(1, 1, 2, 2), torch.zeros(1, 1, 2, 2),
                                           torch.tensor([0]), scheduler)
        self.assertAlmostEqual(float(loss), 100/7.5**2, places=6)

    def test_prompt_augmentation_preserves_text_and_repeats_seed(self):
        prompts = ['a red bird', 'a quiet lake']
        a = augmented_prompts(prompts, np.random.default_rng(45))
        self.assertEqual(a, augmented_prompts(prompts, np.random.default_rng(45)))
        for actual, original in zip(a, prompts):
            self.assertTrue(actual.endswith(' '+original))
            self.assertLessEqual(len(actual)-len(original)-1, 8)


if __name__ == '__main__':
    unittest.main()
