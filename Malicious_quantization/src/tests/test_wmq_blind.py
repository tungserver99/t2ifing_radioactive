import unittest
import io
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import redirect_stdout
import torch
from torch import nn
from torch.func import functional_call
from wmq_blind import RoundingGrid, pair_metrics, pseudo_target, choose, materialize, parse_method_bit_exclusions, initialize_warm_codes


class BlindTests(unittest.TestCase):
    def test_method_bit_exclusions_are_exact_and_validated(self):
        self.assertEqual(parse_method_bit_exclusions(
            ["natural_residual:8", "reconstruction:4", "fixed_ptq:4"]),
            {("natural_residual", 8), ("reconstruction", 4), ("fixed_ptq", 4)})
        with self.assertRaisesRegex(ValueError, "Invalid"):
            parse_method_bit_exclusions(["unknown:4"])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            parse_method_bit_exclusions(["fixed_ptq:4", "fixed_ptq:4"])

    def test_hard_grid_gradient_and_materialization_agree(self):
        torch.manual_seed(4)
        layer = nn.Conv2d(3, 4, 3, padding=1)
        layer.requires_grad_(False)
        original = layer.weight.detach().clone()
        q = RoundingGrid(layer.weight, 3, .9)
        x = torch.randn(1, 3, 8, 8)
        pred = functional_call(layer, {"weight": q()}, (x,))
        loss = (pred - .3).square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(q.alpha.grad).all())
        self.assertGreater(q.alpha.grad.abs().sum().item(), 0)
        self.assertTrue(torch.equal(layer.weight, original))
        codes = q(False) / q.scale
        self.assertTrue(torch.allclose(codes, codes.round(), atol=1e-6))
        self.assertGreaterEqual(codes.min().item(), -4)
        self.assertLessEqual(codes.max().item(), 3)
        materialize(layer, ["weight"], [q])
        self.assertTrue(torch.allclose(layer(x), pred, atol=1e-6))

    def test_zero_weights_and_symmetric_saturation(self):
        q = RoundingGrid(torch.zeros(2, 3), 2)
        self.assertTrue(torch.equal(q(False), torch.zeros(2, 3)))
        q.alpha.data.fill_(100)
        self.assertTrue(torch.isfinite(q()).all())

    def test_full_signed_code_range_is_available(self):
        q = RoundingGrid(torch.tensor([[-4., 1.]]), 3)
        codes = (q(False) / q.scale).round()
        self.assertEqual((q.qmin, q.qmax), (-4, 3))
        self.assertEqual(codes.min().item(), -4)

    def test_qat_code_offsets_cross_multiple_cells_and_keep_source_frozen(self):
        weight = torch.tensor([[-1., .2, .4, 1.]])
        q = RoundingGrid(weight, 4, learn_code_offsets=True)
        baseline = (q(False) / q.scale).round()
        with torch.no_grad():
            q.code_offset[0, 1] = 2.1
        moved = (q(False) / q.scale).round()
        self.assertGreaterEqual((moved[0, 1] - baseline[0, 1]).abs().item(), 2)
        q().sum().backward()
        self.assertIsNotNone(q.code_offset.grad)
        self.assertIsNone(q.alpha.grad)
        self.assertTrue(torch.equal(q.source, weight))
        self.assertGreater(q.code_change_fraction().item(), 0)

    def test_warm_qat_starts_at_frozen_rounding_codes(self):
        weight = torch.tensor([[-1., .2, .4, 1.]])
        rounded = RoundingGrid(weight, 4)
        rounded.alpha.data.fill_(12)
        codes = (rounded(False) / rounded.scale).round().to(torch.int8)
        qat = RoundingGrid(weight, 4, learn_code_offsets=True)
        source = {'codes': {'weight': codes}, 'provenance': {'owner_feedback': False}}
        initialize_warm_codes([qat], ['weight'], source, 2.)
        self.assertTrue(torch.equal((qat(False) / qat.scale).round().to(torch.int8), codes))
        with self.assertRaisesRegex(ValueError, 'complete frozen residual codes'):
            initialize_warm_codes([qat], ['weight'], {'codes': {}}, 2.)

    def test_quality_identity_and_pseudo_control(self):
        x = torch.rand(2, 3, 16, 16)
        p, s = pair_metrics(x, x)
        self.assertTrue((p > 99).all())
        self.assertTrue(torch.allclose(s, torch.ones_like(s), atol=1e-5))
        self.assertTrue(torch.equal(pseudo_target(x, 0), x))

    def test_selection_ignores_infeasible_candidate(self):
        rows = [{"candidate": "bad", "target_mse": 0, "psnr": 12, "feasible": False},
                {"candidate": "ok", "target_mse": .01, "psnr": 28, "feasible": True}]
        self.assertEqual(choose(rows)["candidate"], "ok")
        self.assertIsNone(choose(rows[:1]))

    @unittest.skipUnless(torch.cuda.is_available(), "GPU smoke requires CUDA")
    def test_cuda_runner_selection_export_with_tiny_pipeline(self):
        # Exercises runner orchestration without downloading a pretrained model.
        test_started = [False]
        active_out = [None]
        def audited_choose(rows, **kwargs):
            self.assertFalse(test_started[0], "Selection accessed after test generation")
            return choose(rows, **kwargs)
        class VAE(nn.Module):
            def __init__(self):
                super().__init__()
                self.post_quant_conv = nn.Identity()
                self.decoder = nn.Conv2d(4, 3, 3, padding=1)
                self.config = SimpleNamespace(scaling_factor=1.)

            def decode(self, z, return_dict=False):
                return (self.decoder(self.post_quant_conv(z)),)

            @classmethod
            def from_pretrained(cls, *a, **kw):
                return cls()

            def save_pretrained(self, folder, **kw):
                folder.mkdir()
                torch.save(self.state_dict(), folder / "weights.bin")

        class Pipeline:
            def __init__(self):
                self.vae = VAE()
                self.unet = nn.Linear(1, 1)
                self.text_encoder = nn.Linear(1, 1)
                self.scheduler = SimpleNamespace(config={})

            @classmethod
            def from_pretrained(cls, *a, **kw):
                return cls()

            def set_progress_bar_config(self, **kw):
                pass

            def to(self, device):
                for m in [self.vae, self.unet, self.text_encoder]:
                    m.to(device)
                return self

            def __call__(self, prompt, generator, **kw):
                if isinstance(prompt, list):
                    return SimpleNamespace(images=torch.cat([self(p, g, **kw).images for p, g in zip(prompt, generator)]))
                if prompt not in ("one", "two"):
                    self_test.assertTrue((active_out[0] / "selection_frozen.json").is_file())
                    checkpoint = active_out[0].parent / "output_artifacts" / "checkpoints" / active_out[0].name
                    self_test.assertTrue((checkpoint / "branches/rounding_w8_c1.0/vae/weights.bin").is_file())
                    test_started[0] = True
                multiplier = 3 if prompt == "changed test" else .2
                return SimpleNamespace(images=torch.randn(1, 4, 16, 16, generator=generator, device="cuda") * multiplier)

        self_test = self
        fake = SimpleNamespace(StableDiffusionPipeline=Pipeline, AutoencoderKL=VAE,
                               DDIMScheduler=SimpleNamespace(from_config=lambda c: SimpleNamespace(config=c)))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model"
            model.mkdir()
            (model / "model_index.json").write_text("{}")
            prompts = root / "prompts.txt"
            prompts.write_text("one\ntwo\nthree\n")
            out = root / "out"
            active_out[0] = out
            argv = ["wmq_blind", "--model", str(model), "--prompts", str(prompts),
                    "--output", str(out), "--train-n", "1", "--search-n", "1", "--test-n", "1",
                    "--bits", "8", "--clips", "1", "--methods", "rounding", "--steps", "2", "--eval-every", "1",
                    "--gen-batch-size", "2", "--eval-batch-size", "2",
                    "--min-psnr", "5", "--min-image-psnr", "1", "--min-ssim", ".01"]
            from wmq_blind import main
            previous = sys.modules.get("diffusers")
            sys.modules["diffusers"] = fake
            try:
                with patch.object(sys, "argv", argv), patch("wmq_blind.choose", audited_choose), redirect_stdout(io.StringIO()):
                    main()
                first = json.loads((out / "selection_frozen.json").read_text())
                second_out = root / "second"
                active_out[0] = second_out
                test_started[0] = False
                prompts.write_text("one\ntwo\nchanged test\n")
                argv[argv.index("--output") + 1] = str(second_out)
                with patch.object(sys, "argv", argv), patch("wmq_blind.choose", audited_choose), redirect_stdout(io.StringIO()):
                    main()
                second = json.loads((second_out / "selection_frozen.json").read_text())
                self.assertEqual(first["branches"], second["branches"])
                checkpoint_root = root / "output_artifacts" / "checkpoints"
                original_weights = torch.load(checkpoint_root / out.name / "branches/rounding_w8_c1.0/vae/weights.bin", weights_only=True)
                changed_weights = torch.load(checkpoint_root / second_out.name / "branches/rounding_w8_c1.0/vae/weights.bin", weights_only=True)
                for key in original_weights:
                    self.assertTrue(torch.equal(original_weights[key], changed_weights[key]))
            finally:
                if previous is None:
                    sys.modules.pop("diffusers", None)
                else:
                    sys.modules["diffusers"] = previous
            report = json.loads((out / "report.json").read_text())
            self.assertFalse(report["test_used_for_selection"])
            self.assertIsNone(report["watermark_metrics"])
            self.assertEqual(len(json.loads((out / "search.json").read_text())), 3)
            self.assertTrue((root / "output_artifacts" / "checkpoints" / out.name / "branches/rounding_w8_c1.0/vae/weights.bin").exists())
            for folder in ["marked_reference_test", "pseudo_target_test", "rounding_w8_c1.0_test"]:
                self.assertTrue((root / "output_artifacts" / "images" / out.name / folder / "0000.png").exists())
            self.assertFalse(list(out.rglob("*.png")))


if __name__ == "__main__":
    unittest.main()
