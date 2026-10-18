"""CPU regression tests; no downloads, GPU, or pretrained checkpoints required.

Run in the prepared environment: python -m unittest discover -s tests -v
"""
import ast
import copy
import json
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.update(USE_TORCH="1", USE_TF="0", USE_FLAX="0")
import torch
from torch import nn
from wmq_baselines import AdaptiveRounding, reconstruct, reconstruction_units, shortcut_splits

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1]


def runner_namespace():
    code = max(re.findall(r"<<'PY'\n(.*?)\nPY", (ROOT / "run_malicious_quantization_watermarks.sh").read_text(), re.S), key=len)
    tree = ast.parse(code)
    tree.body = [node for node in tree.body if not isinstance(node, ast.If)]
    ns = {"__name__": "wmq_test"}
    exec(compile(tree, "embedded_runner", "exec"), ns)
    ns.update(DEVICE=torch.device("cpu"), DTYPE=torch.float32, RECON_STEPS=2,
              RECON_CACHE_MB=32, CALIB_N=1, SEARCH_N=2, TEST_N=2,
              GRAD_STEPS=2, GRAD_EVAL_EVERY=1, REFINE_MODE="full", RUN_PROTOCOL="fair_w4a16")
    return ns


class ReconstructionTests(unittest.TestCase):
    def test_quantization_grid_zero_and_tiny_half_weights(self):
        for weight in (torch.zeros(4, 8), torch.full((4, 8), 1e-7), torch.randn(4, 8)):
            q = AdaptiveRounding(weight.half(), split=3)
            q().sum().backward()
            self.assertTrue(torch.isfinite(q.alpha.grad).all())
            hard = q(True)
            for part, scale in zip(torch.split(hard, [3, 5], dim=1), (q.scale_0, q.scale_1)):
                self.assertTrue(torch.allclose(part / scale, (part / scale).round(), atol=1e-5))
                self.assertLessEqual(float((part / scale).abs().max()), 7. + 1e-5)

    def test_linear_reconstruction_preserves_bias_and_coverage(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 4)).eval()
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        data = torch.randn(3, 4)
        report = reconstruct(model, ["0.weight", "2.weight"], lambda: model(data),
                             "adaround_vae", steps=3)
        self.assertEqual(report["changed_parameters"], 64)
        self.assertTrue(all(r["final_nmse"] <= r["initial_nmse"] for r in report["units"]))
        for n, p in model.named_parameters():
            if n.endswith("bias"):
                self.assertTrue(torch.equal(before[n], p))
        self.assertTrue(all(not m._forward_hooks for m in model.modules()))

    def test_cache_failure_removes_hooks(self):
        model = nn.Linear(4, 4)
        with self.assertRaisesRegex(RuntimeError, "cache"):
            reconstruct(model, ["weight"], lambda: model(torch.randn(5, 4)),
                        "adaround_vae", steps=1, cache_mb=1e-6)
        self.assertFalse(model._forward_hooks)

    def test_diffusers_unet_and_vae_real_blocks(self):
        from diffusers import UNet2DConditionModel, AutoencoderKL
        unet = UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4,
            down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), block_out_channels=(8, 16),
            layers_per_block=1, cross_attention_dim=8, attention_head_dim=2, norm_num_groups=4).eval()
        x, cond = torch.randn(1, 4, 8, 8), torch.randn(1, 3, 8)
        selected = [n for n, p in unet.named_parameters() if n.endswith("weight") and p.ndim >= 2]
        splits = shortcut_splits(unet)
        self.assertTrue(splits)
        def replay():
            for timestep in (9, 1):
                unet(x, timestep, encoder_hidden_states=cond)
        report = reconstruct(unet, selected, replay, "qdiff_unet_adapted", steps=2)
        self.assertEqual(report["changed_parameters"], sum(unet.get_parameter(n).numel() for n in selected))
        self.assertTrue(all(r["calibration_records"] == 2 for r in report["units"]))
        self.assertTrue(torch.isfinite(unet(x, 1, encoder_hidden_states=cond).sample).all())
        vae = AutoencoderKL(in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",),
            up_block_types=("UpDecoderBlock2D",), block_out_channels=(8,), latent_channels=4,
            norm_num_groups=4, sample_size=8).eval()
        for method in ("adaround_vae", "brecq_vae_adapted"):
            target = copy.deepcopy(vae)
            selected = [n for n, p in target.named_parameters() if n.endswith("weight") and p.ndim >= 2
                        and n.startswith(("decoder.", "post_quant_conv."))]
            encoder = {n: p.detach().clone() for n, p in target.encoder.named_parameters()}
            report = reconstruct(target, selected, lambda: target.decode(x), method, steps=2)
            self.assertEqual(report["changed_parameters"], sum(target.get_parameter(n).numel() for n in selected))
            self.assertTrue(torch.isfinite(target.decode(x).sample).all())
            for n, p in target.encoder.named_parameters():
                self.assertTrue(torch.equal(encoder[n], p))


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = runner_namespace()

    def test_differentiable_quantizer_half_underflow(self):
        for source in (torch.zeros(2, 3, dtype=torch.float16), torch.full((2, 3), 1e-7, dtype=torch.float16)):
            controls = [torch.tensor(0., requires_grad=True) for _ in range(3)]
            output = self.ns["differentiable_quantize"](source, 4, 1., *controls)
            output.float().sum().backward()
            self.assertTrue(torch.isfinite(output).all())
            self.assertTrue(all(torch.isfinite(c.grad) for c in controls))

    def test_nan_losses_and_gradients_do_not_update_or_select(self):
        ns = self.ns
        for kind in ("loss", "gradient", "optimizer"):
            pipe = SimpleNamespace(vae=nn.Linear(2, 2), unet=nn.Linear(2, 2))
            def proxy(*args):
                controls = args[6]
                p = controls.zero['mid']
                value = p * float('nan') if kind == "loss" else (p.sqrt() if kind == "gradient" else p + 1)
                return value, value, value, value
            class BrokenAdam(torch.optim.Adam):
                def step(self, closure=None):
                    with torch.no_grad():
                        self.param_groups[0]['params'][0].fill_(float('inf'))
            with tempfile.TemporaryDirectory() as tmp:
                replacements = dict(restore_state=lambda *a: None, apply_recipe=lambda *a: 4,
                    gradient_proxy_loss=proxy, cleanup=lambda: None,
                    evaluate_attack_candidate=lambda *a: ({"objective": 1., "bit_accuracy": .9,
                        "tpr": 1., "psnr": 30., "lpips": .1, "prediction_nmse": .01, "feasible": True}, []))
                with patch.dict(ns, replacements), patch('torch.optim.Adam', BrokenAdam if kind == "optimizer" else torch.optim.Adam):
                    recipe = {"bits": 4, "clip": 1., "groups": ["mid"]}
                    got, rows, best = ns['refine_quantizer_gradient']('aqualora', pipe, 'unet', nn.Identity(),
                        None, '0'*48, None, {}, recipe, ['a'], [1], [], Path(tmp),
                        calibration={"records": [{}], "kind": "test"})
                self.assertEqual(got, recipe)
                self.assertIsNone(best)
                self.assertTrue(all(not r['update_applied'] for r in rows))
                self.assertEqual(ns['gradient_statistics'](rows)['skipped_updates'], 2)

    def test_nmse_is_required_for_fallback_feasibility(self):
        self.assertFalse(self.ns['quality_feasible']({"psnr": 30., "lpips": .01}, .1))
        self.assertFalse(self.ns['quality_feasible']({"psnr": float('nan'), "lpips": .01}, .01))
        self.assertTrue(self.ns['quality_feasible']({"psnr": 30., "lpips": .01}, .01))

    def test_invalid_clean_baseline_is_saved_before_quantization(self):
        ns = self.ns
        pipe = SimpleNamespace(vae=nn.Module(), unet=nn.Linear(2, 2),
                               scheduler=SimpleNamespace(config={}))
        pipe.vae.decoder = nn.Linear(2, 2)
        with tempfile.TemporaryDirectory() as tmp:
            replacements = dict(OUT=Path(tmp), generate=lambda *a: [0, 0],
                summarize_bits=lambda *a: {"bit_accuracy": .773, "tpr": .6},
                save_images=lambda *a: None, CLEAN_POLICY='strict',
                CLEAN_MIN_BITACC=.8, CLEAN_MIN_TPR=.9)
            with patch.dict(ns, replacements), patch.dict(ns, {'capture_calibration': lambda *a: self.fail('Must stop before calibration')}):
                report = ns['run_fair_comparison']('aqualora', pipe, 'vae', None, lambda *a: None, '0'*48, None)
            self.assertEqual(report['status'], 'invalid_clean_baseline')
            self.assertFalse(report['methods'])
            self.assertTrue((Path(tmp)/'aqualora/fair_w4a16/clean_validation.json').exists())

    def test_fair_protocol_outputs_and_shared_splits(self):
        ns = self.ns
        pipe = SimpleNamespace(vae=nn.Module(), unet=nn.Linear(2, 2),
                               scheduler=SimpleNamespace(config={}), _wmq_model_id='tiny')
        pipe.vae.decoder = nn.Linear(2, 2)
        pristine = pipe.vae.decoder.weight.detach().clone()
        batches = []
        def generate(pipe, prompts, seeds, **kwargs):
            batches.append((list(prompts), list(seeds)))
            return [0] * len(prompts)
        def bits(*args):
            return {"bit_accuracy": .95, "tpr": 1.}
        def refinement(*args, **kwargs):
            return args[8], [{"update_applied": False, "skip_reason": "nonfinite_gradient"}], None
        def reconstruction(target, selected, replay, method, **kwargs):
            self.assertEqual(selected, ['decoder.weight'])
            return {"changed_parameters": 4, "uses_watermark_labels": False}
        with tempfile.TemporaryDirectory() as tmp:
            replacements = dict(OUT=Path(tmp), generate=generate, summarize_bits=bits,
                save_images=lambda *a: None, capture_calibration=lambda *a: {"records": [{"reference": torch.zeros(1)}]},
                prediction_nmse=lambda *a: .1, image_metrics=lambda *a: {"psnr": 30., "lpips": .01},
                refine_quantizer_gradient=refinement, reconstruct=reconstruction,
                CLEAN_POLICY='strict', CLEAN_MIN_BITACC=.8, CLEAN_MIN_TPR=.9)
            with patch.dict(ns, replacements):
                report = ns['run_fair_comparison']('stable_signature', pipe, 'vae', None, lambda *a: None, '0'*48, None)
            self.assertEqual(report['status'], 'complete')
            self.assertEqual(len(report['methods']), 5)
            self.assertTrue(all(not r['feasible'] for r in report['methods']))
            grad = next(r for r in report['methods'] if r['method']=='gradient_w4a16')
            self.assertEqual(grad['details']['selected_source'], 'grid_fallback')
            self.assertFalse(grad['details']['fallback_feasible'])
            manifest = json.loads((Path(tmp)/'stable_signature/fair_w4a16/manifest.json').read_text())
            self.assertFalse(set(manifest['calibration']['seeds']) & set(manifest['search']['seeds']))
            self.assertFalse(set(manifest['search']['seeds']) & set(manifest['test']['seeds']))
            allowed = [manifest['search']['seeds'], manifest['test']['seeds']]
            self.assertTrue(all(seeds in allowed for _, seeds in batches))
            self.assertTrue(torch.equal(pristine, pipe.vae.decoder.weight))

    def test_cpu_denoising_calibration_and_fair_reports(self):
        """Real DDIM/UNet/VAE forward paths, synthetic watermark and metric only."""
        from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
        import numpy as np
        from PIL import Image
        ns = self.ns
        class TinyPipeline:
            def __init__(self):
                self.unet = UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4,
                    down_block_types=("DownBlock2D",), up_block_types=("UpBlock2D",),
                    block_out_channels=(8,), layers_per_block=1, cross_attention_dim=8,
                    attention_head_dim=2, norm_num_groups=4).eval()
                self.vae = AutoencoderKL(in_channels=3, out_channels=3,
                    down_block_types=("DownEncoderBlock2D",), up_block_types=("UpDecoderBlock2D",),
                    block_out_channels=(8,), latent_channels=4, norm_num_groups=4, sample_size=8).eval()
                self.scheduler = DDIMScheduler(num_train_timesteps=10, clip_sample=False)

            @torch.no_grad()
            def __call__(self, prompts, generator, num_inference_steps, guidance_scale,
                         output_type='pil', **kwargs):
                latent = torch.cat([torch.randn(1, 4, 8, 8, generator=g) for g in generator])
                cond = torch.zeros(2 * len(prompts), 3, 8)
                self.scheduler.set_timesteps(num_inference_steps)
                for t in self.scheduler.timesteps:
                    noise = self.unet(torch.cat([latent, latent]), t, encoder_hidden_states=cond).sample
                    a, b = noise.chunk(2)
                    latent = self.scheduler.step(a + guidance_scale * (b - a), t, latent).prev_sample
                if output_type == 'latent':
                    return SimpleNamespace(images=latent)
                image = self.vae.decode(latent / self.vae.config.scaling_factor).sample
                pixels = ((image / 2 + .5).clamp(0, 1).permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)
                return SimpleNamespace(images=[Image.fromarray(x) for x in pixels])

        class PixelMetric(nn.Module):
            def forward(self, a, b):
                return (a-b).square().flatten(1).mean(1)

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(ns, dict(OUT=Path(tmp), REFINE_MODE='none', CALIB_TIMESTEPS=2, STEPS=2,
                                     GEN_BATCH_SIZE=1, CALIB_BATCH_SIZE=1, BASELINE_CHECK_ONLY=False)):
                for component, name in [('vae', 'stable_signature'), ('unet', 'aqualora')]:
                    pipe = TinyPipeline()
                    report = ns['run_fair_comparison'](name, pipe, component, None,
                        lambda _decoder, images: torch.zeros(len(images), 48, dtype=torch.bool),
                        '0'*48, PixelMetric())
                    self.assertEqual(report['status'], 'complete')
                    self.assertEqual(len(report['methods']), 4 if component=='vae' else 3)
                    self.assertTrue(all(r['changed_parameters'] > 0 for r in report['methods']))
                    self.assertTrue((Path(tmp)/name/'fair_w4a16/report.json').exists())


if __name__ == '__main__':
    unittest.main()
