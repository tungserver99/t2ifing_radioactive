"""Blind protocol tests: exact tails, natural targets, frozen diagnostic evaluation."""
import io
import json
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
from scipy.stats import binom

import wmq_blind as blind
import evaluate_blind_watermark as owner

torch.set_num_threads(2)


@contextmanager
def fake_diffusers(module, name="diffusers"):
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


class PerceptualStub(nn.Module):
    """Tests differentiable wiring, not the correctness of LPIPS pretrained features."""
    def forward(self, a, b):
        return (a - b).abs().mean((1, 2, 3))


class NaturalTests(unittest.TestCase):
    def test_cycle_gradient_through_frozen_encoder(self):
        encoder = nn.Conv2d(3, 3, 1).requires_grad_(False)
        vae = SimpleNamespace(encode=lambda x: SimpleNamespace(
            latent_dist=SimpleNamespace(mode=lambda: encoder(x))))
        image = torch.full((2, 3, 4, 4), .6, requires_grad=True)
        target = torch.ones_like(image, requires_grad=True)
        error = blind.latent_cycle_error(vae, image, target)
        self.assertEqual(error.shape, (2,))
        error.mean().backward()
        self.assertGreater(image.grad.abs().sum().item(), 0)
        self.assertIsNone(target.grad)
        self.assertTrue(all(p.grad is None for p in encoder.parameters()))
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            blind.latent_cycle_error(vae, image, target[:, :, :2])

    def test_report_policy_keeps_failed_candidate_and_ranks_full_objective(self):
        rows = [{"candidate": "good_quality", "target_mse": .1, "selection_objective": .4, "psnr": 35., "feasible": True},
                {"candidate": "poor_quality", "target_mse": .2, "selection_objective": .3, "psnr": 15., "feasible": False}]
        self.assertEqual(blind.choose(rows, "report")["candidate"], "poor_quality")
        self.assertEqual(blind.choose(rows, "constrained")["candidate"], "good_quality")

    def test_exact_total_fpr_by_enumeration(self):
        for bits in (4, 7, 12, 48):
            for fpr in (.5, .1, .001, .0001):
                if 2. ** (1 - bits) > fpr:
                    with self.assertRaises(ValueError):
                        owner.detection_threshold(bits, fpr)
                    continue
                k = owner.detection_threshold(bits, fpr)
                actual = sum(math.comb(bits, m) for m in range(bits + 1)
                             if m >= k or m <= bits - k) / 2 ** bits
                self.assertLessEqual(actual, fpr)
                self.assertAlmostEqual(actual, 2 * binom.sf(k - 1, bits, .5))
                previous = sum(math.comb(bits, m) for m in range(bits + 1)
                               if m >= k - 1 or m <= bits - (k - 1)) / 2 ** bits
                self.assertGreater(previous, fpr)

    def test_natural_split_deduplicates_and_encoder_uses_unscaled_mode(self):
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = nn.Parameter(torch.zeros(1))
            def encode(self, x):
                self.last = x
                return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: x[:, :1] + 7))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (20, 30), (255, 0, 0)).save(root / "a.png")
            Image.new("RGB", (20, 30), (0, 255, 0)).save(root / "b.png")
            (root / "duplicate.png").write_bytes((root / "a.png").read_bytes())
            split = blind.natural_image_manifest(root, 1, 1, 34)
            self.assertNotEqual(split["train"][0]["sha256"], split["search"][0]["sha256"])
            with self.assertRaises(ValueError):
                blind.natural_image_manifest(root, 2, 1, 34)
            vae = Encoder()
            z, images = blind.cache_natural(vae, split["train"])
            self.assertEqual(images[0].shape, (1, 3, 512, 512))
            self.assertTrue(torch.equal(z[0], vae.last[:, :1] + 7))
            Path(split["train"][0]["path"]).write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "changed"):
                blind.cache_natural(vae, split["train"])

    def test_natural_objective_keeps_generated_quality_gate(self):
        class VAE(nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = nn.Conv2d(3, 3, 1)
                self.post_quant_conv = nn.Identity()
            def decode(self, z, **kw):
                return (self.decoder(z),)
        torch.manual_seed(12)
        vae = VAE().requires_grad_(False)
        z = [torch.rand(1, 3, 16, 16)]
        refs = [(vae.decode(z[0])[0] / 2 + .5).clamp(0, 1)]
        natural = [torch.zeros_like(refs[0])]
        args = blind.parser().parse_args(["--model", "x", "--prompts", "x", "--output", "x",
                         "--steps", "2", "--eval-every", "1", "--min-psnr", "0", "--min-image-psnr", "0",
                         "--min-ssim", ".001"])
        pristine = {k: v.clone() for k, v in vae.state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            grids, selected, rows = blind.optimize_branch(vae, ["weight"], z, natural, z, refs,
                args, 8, 1., "natural_rounding_scale", [], Path(tmp) / "branch",
                natural_search=(z, natural), preserve_data=(z, refs), perceptual=PerceptualStub())
            self.assertGreater(selected["gradient_updates"], 0)
            self.assertEqual(selected["objective"], "unpaired_natural_reconstruction")
            self.assertGreater(rows[0]["natural_validation_mse"], .01)
            self.assertLess(rows[0]["generated_reference_mse"], .001)
            self.assertGreater(rows[0]["natural_validation_lpips"], 0)
            self.assertAlmostEqual(rows[0]["selection_objective"], rows[0]["natural_validation_mse"] + .1 * rows[0]["natural_validation_lpips"])
            updates = json.loads((Path(tmp) / "branch/updates.json").read_text())["rows"]
            self.assertGreater(updates[0]["perceptual_loss"], 0)
            self.assertAlmostEqual(updates[0]["loss"], updates[0]["reconstruction_mse"] + .1 * updates[0]["perceptual_loss"] + 2 * updates[0]["preservation_mse"], places=6)
            for name, value in vae.state_dict().items():
                self.assertTrue(torch.equal(value, pristine[name]))
            self.assertTrue(any(q.alpha.grad is not None for q in grids))

    def test_qat_purification_optimizes_bounded_code_offsets(self):
        class VAE(nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = nn.Conv2d(3, 3, 1)
                self.post_quant_conv = nn.Identity()
            def decode(self, z, **kw):
                return (self.decoder(z),)
        torch.manual_seed(23)
        vae = VAE().requires_grad_(False)
        z = [torch.rand(1, 3, 16, 16)]
        refs = [(vae.decode(z[0])[0] / 2 + .5).clamp(0, 1)]
        natural = [torch.full_like(refs[0], .25)]
        args = blind.parser().parse_args(["--model", "x", "--prompts", "x", "--output", "x",
            "--steps", "2", "--eval-every", "1", "--min-psnr", "0", "--min-image-psnr", "0",
            "--min-ssim", ".001", "--natural-perceptual-weight", "0"])
        original = {k: v.clone() for k, v in vae.state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            grids, selected, rows = blind.optimize_branch(
                vae, ["weight"], z, natural, z, refs, args, 4, 1., "natural_qat_purification", [],
                Path(tmp) / "branch", natural_search=(z, natural), preserve_data=(z, refs))
        self.assertEqual(selected["objective"], "quantization_constrained_natural_purification")
        self.assertEqual(selected["optimized_dofs"], ["bounded_multi_cell_code_offsets"])
        self.assertIn("code_change_fraction_vs_rtn", rows[0])
        self.assertTrue(any(q.code_offset.grad is not None for q in grids))
        self.assertTrue(all(q.code_offset.detach().abs().max() <= args.qat_max_code_shift for q in grids))
        for name, value in vae.state_dict().items():
            self.assertTrue(torch.equal(value, original[name]))

    def test_all_branches_frozen_before_owner_diagnostics(self):
        # Exercise main orchestration and owner evaluation with tiny CPU tensors.
        class VAE(nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = nn.Conv2d(3, 3, 1)
                self.post_quant_conv = nn.Identity()
                self.config = SimpleNamespace(scaling_factor=1.)
            def decode(self, z, **kw):
                return (self.decoder(z),)
            def encode(self, x):
                return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: x))
            @classmethod
            def from_pretrained(cls, *a, **kw):
                return cls()
            def save_pretrained(self, folder, **kw):
                folder.mkdir()
                torch.save(self.state_dict(), folder / "weights.bin")
        test_started = [False]
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
                return self
            def __call__(self, prompt, generator, **kw):
                if isinstance(prompt, list):
                    return SimpleNamespace(images=torch.cat([self(p, g, **kw).images for p, g in zip(prompt, generator)]))
                if prompt in ("test", "test2"):
                    self_test.assertTrue((out / "selection_frozen.json").is_file())
                    test_started[0] = True
                return SimpleNamespace(images=torch.rand(1, 3, 16, 16, generator=generator))
        class Extractor(nn.Module):
            def forward(self, x):
                # Fully inverted key must still be detected by DOUBLE tail.
                return torch.full((len(x), 48), -1., device=x.device)
        def tiny_natural(vae, entries, eval_batch_size=1, resolution=512):
            return [torch.linspace(0, 1, 3 * 16 * 16).reshape(1, 3, 16, 16) for _ in entries], [torch.ones(1, 3, 16, 16) * .3 for _ in entries]
        def audited_choose(rows, **kwargs):
            self.assertFalse(test_started[0])
            return original_choose(rows, **kwargs)
        self_test = self
        original_choose = blind.choose
        original_optimize = blind.optimize_branch
        def one_failed_branch(*args, **kwargs):
            if args[9] == 'natural_spectral' and args[7] == 8:
                raise RuntimeError('injected branch failure')
            grids, selected, rows = original_optimize(*args, **kwargs)
            if args[9] == 'natural_joint_quality_finetune' and args[7] == 32:
                # Force SEARCH selection before the final budget to exercise
                # export, freeze, and owner evaluation of its W4 endpoint.
                selected['step'] = 0
            return grids, selected, rows
        fake = SimpleNamespace(StableDiffusionPipeline=Pipeline, AutoencoderKL=VAE,
                               DDIMScheduler=SimpleNamespace(from_config=lambda c: SimpleNamespace(config=c)))
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            model = root / "model"
            model.mkdir()
            (model / "model_index.json").write_text("{}")
            prompts = root / "prompts.txt"
            prompts.write_text("train\nsearch\ntest\ntest2\n")
            dataset = root / "natural"
            dataset.mkdir()
            for i in range(2):
                Image.new("RGB", (16, 16), (i * 100, 40, 80)).save(dataset / f"{i}.png")
            out = root / "out"
            argv = ["wmq_blind", "--model", str(model), "--prompts", str(prompts), "--output", str(out),
                    "--device", "cpu", "--natural-images", str(dataset), "--train-n", "1", "--search-n", "1",
                    "--test-n", "2", "--steps", "1", "--eval-every", "1", "--methods", "rounding",
                    "--gen-batch-size", "2", "--eval-batch-size", "2",
                    "--evaluate-final",
                    "--teacher-checkpoint-policy", "final",
                    "--natural-methods", *blind.NATURAL_METHODS,
                    "--finetune-rtn-bits", "4", "--quality-constraint", "dual",
                    "--gradient-diagnostics-every", "1",
                    "--bits", "8", "4", "--min-psnr", "121", "--min-image-psnr", "121", "--min-ssim", ".001"]
            with fake_diffusers(fake), patch.object(sys, "argv", argv), \
                    patch.object(blind, "cache_natural", tiny_natural), patch.object(blind, "choose", audited_choose), \
                    patch.object(blind, "optimize_branch", one_failed_branch), \
                    fake_diffusers(SimpleNamespace(LPIPS=lambda **kw: PerceptualStub()), "lpips"):
                blind.main()
            report = json.loads((out / "report.json").read_text())
            endpoints = [k for k in report['selections'] if k.endswith('_final_test')]
            self.assertGreater(len(endpoints), 0)
            self.assertEqual(len(report["selections"]), 42 + len(endpoints))
            self.assertEqual(len(report["branch_quality"]), 44 + len(endpoints))
            for label in endpoints:
                selected = report['selections'][label]
                self.assertEqual(selected['step'], selected['attempted_updates'])
                self.assertEqual(selected['checkpoint_policy'], 'fixed_final_step')
            quality_final = 'natural_joint_quality_finetune_rtn_w4_c1.0_final_test'
            self.assertTrue(quality_final in report['selections'],
                            {'quality_labels': [k for k in report['selections'] if 'joint_quality' in k],
                             'failures': [(f['label'], f['error']) for f in report['branch_failures']]})
            self.assertEqual(report['selections'][quality_final]['selected_source'], 'RTN_of_fixed_final_FP32')
            self.assertEqual(report['selections'][quality_final]['additional_training_updates'], 0)
            self.assertEqual(report["status"], "branch_failures")
            self.assertEqual(len(report["branch_failures"]), 1)
            self.assertNotIn('natural_spectral_w8_c1.0_test', report['selections'])
            self.assertIn('natural_spectral_w4_c1.0_test', report['selections'])
            self.assertTrue(all(not s["search_feasible"] for s in report["selections"].values()))
            self.assertTrue((out / "search.csv").is_file())
            self.assertTrue((out / "quality_summary.csv").is_file())
            images = root / "output_artifacts" / "images" / out.name
            self.assertTrue((images / "marked_reference_test/0000.png").is_file())
            self.assertFalse(list(out.rglob("*.png")))
            self.assertTrue(all(p.name == "discriminator_final.safetensors" for p in out.rglob("*.safetensors")))
            artifacts = root / "output_artifacts" / "checkpoints" / out.name
            fp32 = artifacts / "branches/natural_full_finetune_fp32"
            self.assertTrue((fp32 / "decoder_fp32.safetensors").is_file())
            self.assertFalse((fp32 / "quantizer.safetensors").exists())
            self.assertGreater(report["selections"]["natural_full_finetune_fp32_test"]["gradient_updates"], 0)
            student = report['selections']['natural_teacher_rounding_w4_c1.0_test']
            teacher_label = ('natural_full_finetune_fp32_final_test'
                             if 'natural_full_finetune_fp32_final_test' in report['selections'] else
                             'natural_full_finetune_fp32_test')
            self.assertEqual(student['teacher_dependency']['label'], teacher_label)
            self.assertEqual(student['teacher_dependency']['selected_step'],
                             report['selections'][teacher_label]['step'])
            self.assertEqual(student['teacher_dependency']['checkpoint_policy'], 'final')
            self.assertEqual(student['teacher_dependency']['additional_image_forwards'], 2)
            diagnostics = json.loads((out / 'teacher_target_diagnostics.json').read_text())
            self.assertEqual(diagnostics['target_signal']['search']['n'], 1)
            self.assertEqual(student['teacher_dependency']['target_signal'], diagnostics['target_signal'])
            self.assertEqual(student['optimized_dofs'], ['rounding'])
            self.assertGreaterEqual(student['teacher_search_mse'], 0.)
            derived = report["selections"]["natural_finetune_rtn_w4_c1.0_test"]
            self.assertEqual(derived["additional_training_updates"], 0)
            self.assertEqual(derived["step"], report["selections"]["natural_full_finetune_fp32_test"]["step"])
            self.assertEqual(derived["parameter_space"], "finetune_then_rtn")
            joint = report['selections']['natural_joint_finetune_fp32_test']
            joint_rtn = report['selections']['natural_joint_finetune_rtn_w4_c1.0_test']
            self.assertEqual(joint['joint_quant_bits'], 4)
            self.assertEqual(joint_rtn['step'], joint['step'])
            self.assertAlmostEqual(joint_rtn['psnr'], joint['joint_w4_psnr'], places=5)
            self.assertAlmostEqual(joint_rtn['ssim'], joint['joint_w4_ssim'], places=5)
            joint_updates = json.loads((out / 'branches/natural_joint_finetune_fp32/updates.json').read_text())['rows']
            self.assertGreater(joint_updates[0]['joint_w4_gradient_norm'], 0)
            quality_joint = report['selections']['natural_joint_quality_finetune_fp32_test']
            self.assertTrue(quality_joint['activation_checkpointing'])
            self.assertIsNone(quality_joint['peak_cuda_allocated_gib'])
            self.assertGreater(report["selections"]["natural_residual_qat_w8_c1.0_test"]["gradient_updates"], 0)
            warm = report['selections']['natural_residual_qat_warm_w4_c1.0_test']
            self.assertEqual(warm['warm_start_dependency']['source_branch'],
                             'natural_residual_w4_c1.0_test')
            self.assertFalse(warm['warm_start_dependency']['owner_feedback'])
            warm_initial = json.loads((out / 'branches/natural_residual_qat_warm_w4_c1.0/search.json').read_text())[0]
            source_selected = report['selections']['natural_residual_w4_c1.0_test']
            self.assertAlmostEqual(warm_initial['psnr'], source_selected['psnr'], places=4)
            basis_path = artifacts / "branches/natural_residual_w8_c1.0/residual_basis.safetensors"
            self.assertTrue(basis_path.is_file())
            calibration = json.loads((out / "branches/natural_residual_w8_c1.0/residual_calibration.json").read_text())
            self.assertEqual(calibration["fit_split"], "natural_train_only")
            self.assertGreater(report["selections"]["natural_residual_w8_c1.0_test"]["gradient_updates"], 0)
            owner.verify_frozen_run(out, report)
            saved_basis = basis_path.read_bytes()
            basis_path.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "changed after selection freeze"):
                owner.verify_frozen_run(out, report)
            basis_path.write_bytes(saved_basis)
            extractor = root / "extractor.pt"
            extractor.write_bytes(b"test")
            argv = ["owner", "--run", str(out), "--extractor", str(extractor), "--key", "1" * 48, "--batch-size", "2"]
            with patch.object(sys, "argv", argv), patch.object(torch.jit, "load", return_value=Extractor()):
                owner.main()
            result = json.loads((out / "owner_evaluation.json").read_text())
            self.assertLessEqual(result["theoretical_fpr_at_threshold"], .001)
            self.assertTrue((out / "joint_quality_evasion.csv").is_file())
            import csv
            with (out / "joint_quality_evasion.csv").open(newline='') as stream:
                joint_table = list(csv.DictReader(stream))
            self.assertEqual(sum(r['primary_budget'] == 'True' for r in joint_table),
                             len({r['method'] for r in joint_table}))
            joint_rows = [r for r in result['rows'] if r.get('joint_success_rate') is not None]
            self.assertTrue(joint_rows)
            self.assertTrue(all(r['joint_success_rate'] == 0. for r in joint_rows))
            self.assertTrue(all(r["tpr"] == 1. and r["bit_accuracy"] == 0. for r in result["rows"]))
            self.assertTrue(any(r["role"] == "diagnostic" for r in result["rows"]))
            # Relocation can override the path without modifying frozen metadata.
            relocated = root / "relocated_images"
            images.rename(relocated)
            owner.verify_frozen_run(out, report, relocated)
            with self.assertRaisesRegex(FileNotFoundError, "Image directory missing"):
                owner.verify_frozen_run(out, report)
            (relocated / "pseudo_target_test/0000.png").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "images changed"):
                owner.verify_frozen_run(out, report, relocated)

    def test_image_path_defaults_support_old_and_new_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "output_attack" / "run_a"
            self.assertEqual(owner.resolve_image_root(run, {}), run.resolve())
            self.assertEqual(owner.resolve_image_root(run, {"image_root": "../../output_image/run_a"}),
                             root / "output_image" / "run_a")
            self.assertEqual(owner.resolve_artifact_root(run, {}), run.resolve())
            self.assertEqual(owner.resolve_artifact_root(run, {"artifact_root": "../../output_checkpoint/run_a"}),
                             root / "output_checkpoint" / "run_a")


if __name__ == "__main__":
    unittest.main()
