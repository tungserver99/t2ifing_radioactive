"""RSSF CPU contracts; no model downloads or inference."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

import semantic_sign as sign
from common import read_json, sha256
from semantic_carrier import SemanticCarrier, save_carrier


class AnalyticTests(unittest.TestCase):
    def test_exact_tail_and_impossible_cutoff(self):
        self.assertEqual(sign.detection_count(32, .01), 24)
        self.assertLess(sign.binomial_tail(24, 32), .01)
        self.assertGreater(sign.binomial_tail(23, 32), .01)
        self.assertEqual(sign.binomial_tail(32, 32), 2**-32)
        self.assertEqual(sign.detection_count(2, .01), 3)

    def test_ties_image_and_model_evidence(self):
        r = torch.tensor([[0., -2., 1.], [0., 3., -1.]])
        keys = torch.tensor([-1., 1., 1.])
        self.assertEqual(sign.image_bits(r).tolist(), [[1., -1., 1.], [1., 1., -1.]])
        rows = sign.image_evidence(r, keys)
        self.assertEqual([x['K'] for x in rows], [1, 1])
        self.assertEqual(rows[0]['p_img'], .875)
        self.assertEqual(rows[0]['matches'], [False, False, True])
        model = sign.model_evidence(r, keys)
        self.assertEqual(model['recovered_signs'], [1., 1., 1.])
        self.assertEqual(model['K_M'], 2)
        self.assertEqual(model['p_M'], .5)
        self.assertAlmostEqual(model['mean_score'], 1/6)

    def test_random_keys_null_with_correlated_coordinates_and_ties(self):
        g = torch.Generator().manual_seed(21)
        keys = torch.randint(0, 2, (20000, 32), generator=g)*2-1
        bits = sign.image_bits(torch.ones(32)*.7)
        self.assertEqual(bits.tolist(), sign.image_bits(torch.zeros(32)).tolist())
        counts = (keys == bits).sum(1).float()
        self.assertAlmostEqual(counts.mean().item(), 16., delta=.1)
        self.assertAlmostEqual(counts.var().item(), 8., delta=.25)
        self.assertAlmostEqual((counts >= 24).float().mean().item(), .0035, delta=.0015)

    def test_invalid_values_rejected(self):
        for r in (torch.empty(0, 2), torch.tensor([[float('nan'), 1.]])):
            with self.assertRaises(ValueError): sign.model_evidence(r, torch.ones(2))
        with self.assertRaises(ValueError): sign.image_evidence(torch.ones(1, 2), torch.zeros(2))


class HingeTests(unittest.TestCase):
    def test_clean_selection_stable_detached_and_fp_gradients(self):
        clean = torch.tensor([[2., -2., 0., 0.]], requires_grad=True)
        fp = torch.tensor([[.3, 2., -.2, 9.]], requires_grad=True)
        keys = torch.tensor([1., -1., 1., -1.])
        loss, stats = sign.sign_hinge_loss(clean, fp, keys, mu=.2, k_train=3)
        self.assertEqual(stats['selected_indices'].tolist(), [[0, 1, 2]])
        self.assertFalse(stats['selected_indices'].requires_grad)
        loss.backward()
        self.assertIsNone(clean.grad)
        self.assertEqual(fp.grad[0, 0].item(), 0.)
        self.assertGreater(fp.grad[0, 1].item(), 0.)
        self.assertLess(fp.grad[0, 2].item(), 0.)
        self.assertEqual(fp.grad[0, 3].item(), 0.)
        _, other = sign.sign_hinge_loss(clean, -fp, keys, mu=.2, k_train=3)
        self.assertEqual(other['selected_indices'].tolist(), [[0, 1, 2]])
        self.assertAlmostEqual(loss.item(), (2.2**2+.4**2)/3, places=6)

    def test_loss_averages_over_batch_and_selected_bits(self):
        clean = torch.tensor([[2., 1.], [0., -2.]])
        fp = torch.tensor([[0., 9.], [-1., 9.]], requires_grad=True)
        loss, _ = sign.sign_hinge_loss(clean, fp, torch.ones(2), .2, 1)
        self.assertAlmostEqual(loss.item(), (.2**2+1.2**2)/2, places=6)


class SchedulerTests(unittest.TestCase):
    class Native:
        alphas_cumprod = torch.linspace(1., .1, 10)
        def __init__(self, ids): self.ids = ids; self.called = None
        def set_timesteps(self, n): self.called = n; self.timesteps = torch.tensor(self.ids)

    def test_native_generation_order_dedup_and_alpha_check(self):
        native = self.Native([9, 7, 7, 5, 3, 3, 1, 0])
        train = SimpleNamespace(alphas_cumprod=native.alphas_cumprod.clone())
        self.assertEqual(sign.derive_late_timesteps(native, train, 50, 3), [3, 1, 0])
        self.assertEqual(native.called, 50)
        train.alphas_cumprod[1] += .1
        with self.assertRaises(ValueError): sign.derive_late_timesteps(native, train, 50, 3)
        with self.assertRaises(ValueError): sign.derive_late_timesteps(self.Native([3., 1.5, 0.]), train, 50, 2)
        with self.assertRaises(ValueError): sign.derive_late_timesteps(self.Native([1, 0]), train, 50, 3)


class VersionedCarrierTests(unittest.TestCase):
    def test_midpoint_centers_roundtrip_and_key_freezing(self):
        # Carrier is fit only on the passed train features, not provenance extras.
        x = torch.randn(72, 66, generator=torch.Generator().manual_seed(8))
        config = {'teacher_objective': 'semantic_sign_v1', 'carrier_pc_min': 8,
                  'carrier_pc_max': 64, 'stable_candidate_count': 48, 'num_bits': 32,
                  'key_seed': 42, 'analytic_alpha': .01}
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d)/'a', Path(d)/'b'
            for path in (a, b): save_carrier(path, x, x, config, {'train_prompt_sha256': 'train'})
            c = SemanticCarrier.load(a)
            self.assertEqual(c.carrier_version, 'semantic_sign_v1')
            self.assertEqual(read_json(a/'key.json'), read_json(b/'key.json'))
            self.assertEqual(read_json(a/'key.json')['k_detect'], 24)
            self.assertEqual(len(read_json(a/'medians.json')['pc_indices']), 57)
            pca = torch.load(a/'pca.pt', weights_only=True)
            q = (x-pca['mean']) @ pca['basis'][:, c.selected_pc_indices] / c.sigmas
            expected = q.sort(dim=0).values[35:37].mean(0)
            torch.testing.assert_close(c.medians, expected)
            torch.testing.assert_close(c.coordinates(x), q-expected)
            torch.testing.assert_close(c.to('cpu').coordinates(x), c.coordinates(x))
            self.assertEqual(c.bit_metrics(torch.zeros(2, 32))['recovered_signs'], [1.]*32)
            digest = sha256(a/'key.json')
            changed = {**config, 'key_seed': 999}
            with self.assertRaises((RuntimeError, ValueError)):
                save_carrier(a, x, x, changed, {'train_prompt_sha256': 'train'})
            self.assertEqual(sha256(a/'key.json'), digest)
            (a/'medians.json').write_text('{}')
            with self.assertRaises(RuntimeError): SemanticCarrier.load(a)

    def test_historical_zero_bits_and_projection_stay_unchanged(self):
        c = SemanticCarrier(torch.zeros(2), torch.eye(2), torch.ones(2), torch.ones(2), [0, 1])
        self.assertEqual(c.bit_metrics(torch.zeros(2, 2))['recovered_signs'], [0., 0.])
        torch.testing.assert_close(c.coordinates(torch.ones(1, 2)), torch.ones(1, 2))


if __name__ == '__main__': unittest.main()
