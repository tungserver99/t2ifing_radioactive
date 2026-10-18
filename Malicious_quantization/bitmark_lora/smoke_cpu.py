"""Optional real CPU integration check (needs dependencies, but no model downloads).

Builds a tiny random SD locally, trains both *official* trainers for two steps,
checks learned weights, reloads LoRA and full checkpoints, then runs inference.
It does NOT test Infinity CUDA kernels or claim scientific radioactivity results.
"""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from common import HERE, verify_vendor


def main():
    import numpy as np
    import torch
    from PIL import Image
    from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
    from safetensors.torch import load_file
    torch.set_num_threads(2)
    verify_vendor()
    sys.path.insert(0, str(HERE / 'vendor/bitmark'))
    from extended_watermark_processor import WatermarkDetector, WatermarkLookupProcessor
    detector = WatermarkDetector(vocab=[0, 1], gamma=.5, delta=2., device='cpu',
                                 ignore_repeated_ngrams=False, green_list='01,10')
    with contextlib.redirect_stdout(io.StringIO()):
        alternating = detector.detect(tokenized_text=torch.tensor([0, 1]*64))
        constant = detector.detect(tokenized_text=torch.zeros(128, dtype=torch.long))
    assert alternating['green_fraction'] == 1 and constant['green_fraction'] == 0
    assert alternating['z_score'] > 0 and constant['z_score'] < 0
    processor = WatermarkLookupProcessor(vocab=[0, 1], device='cpu', delta=2., green_list={'01','10'})
    logits = torch.zeros(1, 2, 2)
    biased = processor(torch.tensor([[0, 1]]), logits)
    assert biased[0, 0, 1] > biased[0, 0, 0] and biased[0, 1, 0] > biased[0, 1, 1]
    print('Official detector and logit bias CPU checks passed', flush=True)
    with tempfile.TemporaryDirectory(prefix='bitmark-smoke-') as tmp:
        root = Path(tmp)
        vocab = root/'vocab.json'
        vocab.write_text(json.dumps({'<|startoftext|>': 0, '<|endoftext|>': 1, 'a</w>': 2, 'test</w>': 3}))
        merges = root/'merges.txt'
        merges.write_text('#version: 0.2\n')
        tokenizer = CLIPTokenizer(str(vocab), str(merges), model_max_length=77)
        text = CLIPTextModel(CLIPTextConfig(vocab_size=4, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=1, num_attention_heads=4, max_position_embeddings=77,
                             bos_token_id=0, eos_token_id=1, pad_token_id=1))
        vae = AutoencoderKL(block_out_channels=(16, 32), in_channels=3, out_channels=3,
             down_block_types=('DownEncoderBlock2D',)*2, up_block_types=('UpDecoderBlock2D',)*2,
             latent_channels=4, norm_num_groups=8, sample_size=32)
        unet = UNet2DConditionModel(sample_size=16, in_channels=4, out_channels=4,
             down_block_types=('CrossAttnDownBlock2D','DownBlock2D'),
             up_block_types=('UpBlock2D','CrossAttnUpBlock2D'),
             block_out_channels=(16, 32), layers_per_block=1, norm_num_groups=8,
             cross_attention_dim=32, attention_head_dim=4)
        pipe = StableDiffusionPipeline(vae=vae, text_encoder=text, tokenizer=tokenizer, unet=unet,
             scheduler=DDIMScheduler(num_train_timesteps=100, steps_offset=1, clip_sample=False), safety_checker=None,
             feature_extractor=None, requires_safety_checker=False)
        base = root/'base'
        pipe.save_pretrained(base)
        # Reproduce SD1.4 metadata referencing an intentionally uncached safety checker.
        metadata = json.loads((base/'model_index.json').read_text())
        metadata['safety_checker'] = ['stable_diffusion', 'StableDiffusionSafetyChecker']
        (base/'model_index.json').write_text(json.dumps(metadata))
        data = root/'data'
        data.mkdir()
        (data/'generation.json').write_text('{"test": true}')
        (data/'image_hashes.json').write_text('[]')
        rng = np.random.default_rng(0)
        with (data/'metadata.jsonl').open('w') as f:
            for i in range(2):
                Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(data/f'{i}.png')
                f.write(json.dumps({'file_name': f'{i}.png', 'text': 'a test'})+'\n')
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='', TOKENIZERS_PARALLELISM='false')
        for kind in ('lora', 'lora_conv', 'dora_conv', 'full'):
            is_lora = kind != 'full'
            name = 'train_text_to_image_lora.py' if is_lora else 'train_text_to_image.py'
            out = root/kind
            cmd = [sys.executable, str(HERE/'vendor/diffusers'/name),
                   '--pretrained_model_name_or_path', str(base), '--train_data_dir', str(data),
                   '--output_dir', str(out), '--resolution', '32', '--center_crop',
                   '--train_batch_size', '1', '--max_train_steps', '2', '--gradient_checkpointing',
                   '--learning_rate', '0.0001', '--mixed_precision', 'no', '--report_to', 'none',
                   '--checkpointing_steps', '1000000', '--snapshot_steps', '1', '2', '--seed', '0']
            if is_lora:
                cmd += ['--rank', '4', '--adapter_scope', 'attention_conv' if kind.endswith('_conv') else 'attention']
            if kind == 'dora_conv':
                cmd += ['--use_dora']
            subprocess.run(cmd, check=True, env=env)
            if is_lora:
                updates = json.loads((out/'optimization_audit.json').read_text())
                assert updates['attempted'] == updates['applied'] == 2 and updates['skipped'] == 0
                snapshot_updates = json.loads((out/'snapshot-1/optimization_audit.json').read_text())
                assert snapshot_updates['applied'] == 1
            model = StableDiffusionPipeline.from_pretrained(base if is_lora else out, safety_checker=None)
            if is_lora:
                tensors = load_file(str(out/'adapter_model.safetensors'))
                assert any(t.abs().max().item() > 0 for k, t in tensors.items() if 'lora_B' in k)
                from student import load_adapter
                load_adapter(model.unet, out, model)
            else:
                assert any(not torch.equal(p, q) for p, q in zip(pipe.unet.parameters(), model.unet.parameters()))
            # Inference-only snapshots must load, including Conv2d LoRA tensors.
            snapshot = out/'snapshot-2'
            if is_lora:
                snap = load_file(str(snapshot/'adapter_model.safetensors'))
                assert all(torch.equal(t, snap[k]) for k,t in tensors.items())
                if kind.endswith('_conv'):
                    assert any(t.ndim == 4 and t.abs().max().item() > 0 for k,t in snap.items() if 'lora_B' in k)
                audit = json.loads((out/'adapter_audit.json').read_text())
                assert audit['trainable_parameters'] > 0
                if kind == 'dora_conv':
                    assert audit['method'] == 'dora' and audit['magnitude_parameters'] > 0
                    assert json.loads((out/'adapter_config.json').read_text())['use_dora'] is True
                    assert any('lora_magnitude_vector' in k for k in snap)
                    # Ensure learned magnitude is preserved, not just A/B matrices.
                    first_snapshot = load_file(str(out/'snapshot-1/adapter_model.safetensors'))
                    assert any(not torch.equal(t, first_snapshot[k]) for k,t in snap.items() if 'lora_magnitude_vector' in k)
                reloaded = StableDiffusionPipeline.from_pretrained(base, safety_checker=None)
                load_adapter(reloaded.unet, snapshot, reloaded)
            else:
                reloaded = StableDiffusionPipeline.from_pretrained(base, safety_checker=None)
                reloaded.unet = UNet2DConditionModel.from_pretrained(snapshot/'unet')
            # Exercise the production selection loader, not just direct PEFT/Diffusers APIs.
            from common import write_json
            from student import checkpoint_hashes, load_pipeline
            branch_name = {'lora': 'lora_r4_wm', 'lora_conv': 'lora_r4_conv_wm', 'full': 'full_wm', 'dora_conv':'dora_r4_conv_wm'}[kind]
            (root/'students').mkdir(exist_ok=True)
            (root/'students'/branch_name).symlink_to(out, target_is_directory=True)
            write_json(root/'config.json', {'ranks':[4], 'adapter_scopes':['attention','attention_conv'],
                'dora':kind == 'dora_conv', 'train_clean_controls':False, 'lora_only':False, 'paper_full_ema':False, 'checkpoint_steps':[1,2]})
            write_json(root/'assets.json', {'student':str(base)})
            write_json(out/'complete.json', {'checkpoints':checkpoint_hashes(out)})
            write_json(root/'result/selection'/f'{branch_name}.json',
                       {'selected_step':2, 'checkpoint_hashes':checkpoint_hashes(snapshot)})
            selected_pipe, selected_hashes = load_pipeline(root, branch_name, device='cpu')
            assert selected_hashes == checkpoint_hashes(snapshot)
            model.unet.eval(); reloaded.unet.eval()
            x = torch.randn(1,4,16,16)
            hidden = torch.randn(1,77,32)
            with torch.inference_mode():
                before = model.unet(x, torch.tensor([5]), hidden).sample
                after = reloaded.unet(x, torch.tensor([5]), hidden).sample
            torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)
            with torch.inference_mode():
                selected_prediction = selected_pipe.unet(x, torch.tensor([5]), hidden).sample
            torch.testing.assert_close(before, selected_prediction, rtol=1e-5, atol=1e-6)
            # The production validation objective must reproduce exactly on cached draws.
            from validation import denoising_loss
            from diffusers import DDPMScheduler
            from safetensors.torch import save_file
            cache = root/'validation.safetensors'
            save_file({'latent': x, 'text': hidden}, str(cache))
            noise_scheduler = DDPMScheduler(num_train_timesteps=100, prediction_type='v_prediction')
            first = denoising_loss(reloaded.unet, noise_scheduler, [cache], 0, 2, device='cpu')
            second = denoising_loss(reloaded.unet, noise_scheduler, [cache], 0, 2, device='cpu')
            assert first == second and first['denoising_mse'] > 0
            with torch.inference_mode():
                image = model('a test', height=32, width=32, num_inference_steps=2).images[0]
            assert image.size == (32, 32)
            print(f'{kind}: two real optimizer steps, saved/reloaded and inferred successfully', flush=True)


if __name__ == '__main__':
    main()
