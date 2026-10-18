import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn

from wmq_runtime import batches, batch_size, DataCache, EVENTS, BATCH_LIMITS, GIB, all_finite
from wmq_blind import score, natural_reconstruction_metrics, optimize_branch, parser

torch.set_num_threads(2)


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Conv2d(3, 3, 1)
        self.post_quant_conv = nn.Identity()
    def decode(self, x, **kw):
        return (self.decoder(x),)


class RuntimeTests(unittest.TestCase):
    def test_oom_retry_keeps_order_and_recreates_seeded_samples(self):
        BATCH_LIMITS.clear()
        attempts = []
        def generate(chunk):
            attempts.append(len(chunk))
            draws = [torch.rand(1, generator=torch.Generator().manual_seed(i)) for i in chunk]
            if len(chunk) > 2:
                raise torch.cuda.OutOfMemoryError("simulated")
            return [x.item() for x in draws]
        EVENTS.clear()
        with patch("torch.cuda.empty_cache"), redirect_stdout(io.StringIO()):
            result = [x for _, rows in batches(list(range(5)), generate, 5) for x in rows]
        expected = [torch.rand(1, generator=torch.Generator().manual_seed(i)).item() for i in range(5)]
        self.assertEqual(result, expected)
        self.assertTrue(EVENTS)
        attempts.clear()
        list(batches(list(range(4)), generate, 5))
        self.assertLessEqual(max(attempts), 2)  # learned cap avoids repeated OOM probes
        with self.assertRaises(torch.cuda.OutOfMemoryError):
            list(batches([1], lambda _: (_ for _ in ()).throw(torch.cuda.OutOfMemoryError()), 1))

    def test_batch_sizing_and_cache_respect_free_memory(self):
        with patch("torch.cuda.mem_get_info", return_value=(80 * GIB, 96 * GIB)):
            self.assertEqual(batch_size(0, "cuda"), 16)
        with patch("torch.cuda.mem_get_info", return_value=(130 * GIB, 140 * GIB)):
            self.assertEqual(batch_size(0, "cuda"), 16)
        self.assertEqual(batch_size(0, "cuda", cap=32), 32)
        tensors = [torch.zeros(1, 3, 16, 16)]
        with patch("torch.cuda.mem_get_info", return_value=(2 * GIB, 96 * GIB)):
            cache = DataCache("cuda")
            self.assertIs(cache.promote(tensors, "no_headroom"), tensors)
            self.assertEqual(cache.used, 0)
        self.assertIs(DataCache("cpu").promote(tensors, "cpu"), tensors)
        self.assertTrue(all_finite([torch.ones(3), torch.zeros(2)]))
        self.assertFalse(all_finite([torch.tensor([float("nan")])]))

    def test_batched_metrics_preserve_per_image_means_and_tail(self):
        torch.manual_seed(9)
        vae = VAE().requires_grad_(False)
        z = list(torch.rand(5, 3, 16, 16).split(1))
        refs = [torch.rand(1, 3, 16, 16) for _ in z]
        args = SimpleNamespace(eval_batch_size=1, min_psnr=0., min_image_psnr=0., min_ssim=.001)
        one = score(vae, z, refs, .25, args)
        args.eval_batch_size = 3
        many = score(vae, z, refs, .25, args)
        for key in ("per_image_psnr", "per_image_ssim", "target_mse"):
            np.testing.assert_allclose(one[key], many[key], atol=2e-6, rtol=2e-6)
        a = natural_reconstruction_metrics(vae, z, refs, eval_batch_size=1)
        b = natural_reconstruction_metrics(vae, z, refs, eval_batch_size=3)
        self.assertAlmostEqual(a["selection_objective"], b["selection_objective"], places=6)

    def test_buffered_logging_retains_all_updates_and_flushes_last_step(self):
        torch.manual_seed(7)
        vae = VAE().requires_grad_(False)
        z = [torch.rand(1, 3, 16, 16)]
        refs = [(vae.decode(z[0])[0] / 2 + .5).clamp(0, 1)]
        args = parser().parse_args(["--model", "x", "--prompts", "x", "--output", "x", "--steps", "3", "--log-every", "2"])
        import wmq_blind
        calls = []
        original = wmq_blind.save_csv
        def recording(path, rows):
            if Path(path).name == "updates.csv":
                calls.append(len(rows))
            original(path, rows)
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), patch("wmq_blind.save_csv", recording):
            optimize_branch(vae, ["weight"], z, refs, z, refs, args, 8, 1., "rounding", [], Path(tmp) / "branch")
        self.assertEqual(calls, [2, 3])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_gpu_cache_and_cuda_batch_parity(self):
        items = [torch.rand(1, 3, 16, 16) for _ in range(3)]
        cache = DataCache("cuda", max_gib=.01, reserve_fraction=.01)
        cached = cache.promote(items, "tiny")
        if cache.used == 0:
            self.skipTest("Insufficient real free VRAM for the cache reserve")
        self.assertTrue(all(x.is_cuda for x in cached))
        self.assertTrue(torch.equal(cached[0].cpu(), items[0]))
        vae = VAE().cuda().requires_grad_(False)
        refs = [torch.rand_like(x) for x in cached]
        args = SimpleNamespace(eval_batch_size=1, min_psnr=0., min_image_psnr=0., min_ssim=.001)
        a = score(vae, cached, refs, .25, args)
        args.eval_batch_size = 2
        b = score(vae, cached, refs, .25, args)
        np.testing.assert_allclose(a["per_image_psnr"], b["per_image_psnr"], atol=2e-5)
        np.testing.assert_allclose(a["per_image_ssim"], b["per_image_ssim"], atol=2e-5)


if __name__ == "__main__":
    unittest.main()
