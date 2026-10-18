import unittest
import torch
from torch import nn
from wmq_runtime import image01, decoded01
from evaluate_blind_watermark import paired_bit_statistics
from wmq_owner_mechanism import layer_diagnostics, ownership_score


class DiagnosticsTests(unittest.TestCase):
    def test_image_boundaries_reject_silent_corruption(self):
        x = torch.rand(1, 3, 12, 12)
        self.assertIs(image01(x), x)
        for bad in (x.half(), x.double(), x.to(torch.uint8), x * 255,
                    torch.full_like(x, float('nan')), torch.full_like(x, -1)):
            with self.assertRaises(ValueError):
                image01(bad)
        with self.assertRaises(ValueError):
            decoded01(torch.full_like(x, float('inf')))
        self.assertTrue(torch.equal(decoded01(torch.full_like(x, 3)), torch.ones_like(x)))

    def test_paired_bit_accounting_and_invalid_lattice(self):
        r = paired_bit_statistics([48, 40, 20], [40, 48, 10], 48)
        self.assertEqual(r['net_additional_bit_errors'], 10)
        self.assertEqual(r['decoding_improved_images'], 1)
        self.assertEqual(r['decoding_worsened_images'], 2)
        self.assertEqual(r['exact_message_accuracy'], 1/3)
        self.assertEqual(r['correct_bits'] + r['error_bits'], 144)
        for a, b in (([.5], [1]), ([49], [1]), ([1], [1, 2])):
            with self.assertRaises(ValueError):
                paired_bit_statistics(a, b, 48)

    def test_two_tail_score_and_gradients_do_not_mutate_model(self):
        class VAE(nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = nn.Conv2d(3, 3, 1, bias=False)
            def decode(self, z, **kw):
                return (self.decoder(z),)
        class Extractor(nn.Module):
            def forward(self, x):
                return x.mean((2, 3))
        vae = VAE().requires_grad_(False)
        key = torch.tensor([True, False, True])
        logits = torch.tensor([[1., -2., 3.]])
        torch.testing.assert_close(ownership_score(logits, key), ownership_score(-logits, key))
        original = vae.decoder.weight.detach().clone()
        rows = layer_diagnostics(vae, Extractor(), key, torch.rand(1, 3, 12, 12),
                                 torch.full((1, 3, 12, 12), .3), {'weight': torch.ones_like(original) * .01})
        self.assertGreater(rows[0]['owner_gradient_l2'], 0)
        self.assertGreater(rows[0]['quality_gradient_l2'], 0)
        self.assertTrue(torch.equal(original, vae.decoder.weight))
        self.assertFalse(vae.decoder.weight.requires_grad)
        self.assertIsNone(vae.decoder.weight.grad)
        from wmq_residual import ResidualSubspace, remove_dc
        basis = torch.linalg.qr(remove_dc(torch.randn(2, 12), 2).T)[0].T
        rows = layer_diagnostics(vae, Extractor(), key, torch.rand(1, 3, 12, 12),
            torch.full((1, 3, 12, 12), .3), {'weight': torch.ones_like(original) * .01},
            ResidualSubspace(basis, 2))
        for metric in ('owner_gradient_subspace_fraction', 'quality_gradient_subspace_fraction'):
            self.assertGreaterEqual(rows[0][metric], 0.)
            self.assertLessEqual(rows[0][metric], 1. + 1e-6)
        self.assertTrue(torch.equal(original, vae.decoder.weight))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_endpoint_derivative_matches_finite_difference(self):
        class VAE(nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = nn.Conv2d(3, 3, 1, bias=False)
                nn.init.constant_(self.decoder.weight, .05)
            def decode(self, z, **kw):
                return (self.decoder(z),)
        class Extractor(nn.Module):
            def forward(self, x):
                return x.mean((2, 3))
        vae = VAE().cuda().requires_grad_(False)
        key = torch.tensor([True, False, True], device='cuda')
        z = torch.full((1, 3, 12, 12), .2, device='cuda')
        e = torch.full_like(vae.decoder.weight, .01)
        ref = torch.full_like(z, .3)
        row = layer_diagnostics(vae, Extractor(), key, z, ref, {'weight': e})[0]
        original = vae.decoder.weight.detach().clone()
        def score():
            x = decoded01(vae.decode(z)[0])
            m = x.new_tensor([.485, .456, .406])[None, :, None, None]
            s = x.new_tensor([.229, .224, .225])[None, :, None, None]
            return ownership_score(Extractor()((x-m)/s), key).item()
        with torch.no_grad():
            vae.decoder.weight.copy_(original + .1 * e)
            plus = score()
            vae.decoder.weight.copy_(original - .1 * e)
            minus = score()
            vae.decoder.weight.copy_(original)
        self.assertAlmostEqual((plus-minus)/.2, row['owner_error_dot'], delta=2e-6)


if __name__ == '__main__':
    unittest.main()
