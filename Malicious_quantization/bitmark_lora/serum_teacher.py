"""Author SERUM diffusion teacher and learned detector; no BitMark/Infinity imports."""
import argparse
import json
from pathlib import Path

from common import HERE, finite, freeze, image_name, read_json, sha256, verify_vendor, write_json
from serum_assets import CHECKPOINT_SHA256, verified_file
from student import training_conditions


def check_imports():
    from diffusers import StableDiffusionPipeline, AutoencoderKL
    from transformers import CLIPTextModel, CLIPTokenizer
    from peft import LoraConfig
    from vendor.serum.watermark import Watermark
    from vendor.serum.score_model import WatermarkScoreModel


def author_models(checkpoint, device='cuda'):
    import torch
    from vendor.serum.config import Config
    from vendor.serum.watermark import Watermark
    from vendor.serum.score_model import WatermarkScoreModel
    verify_vendor()
    verified_file(checkpoint, CHECKPOINT_SHA256)  # Before any deserialization.
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if state.get('epoch') != 50:
        raise RuntimeError('Author checkpoint epoch mismatch')
    cfg = Config(str(HERE / 'vendor/serum/config_sd21.yaml'))
    wm, detector = Watermark(cfg), WatermarkScoreModel(cfg)
    wm.load_state_dict(state['watermark_state_dict'], strict=True)
    detector.load_state_dict(state['score_model_state_dict'], strict=True)
    for model in (wm, detector):
        if any(t.is_floating_point() and not torch.isfinite(t).all() for t in model.state_dict().values()):
            raise RuntimeError('Nonfinite author weights')
        model.eval().requires_grad_(False).to(device=device, dtype=torch.float32)
    if wm.grid.std() <= 0: raise RuntimeError('Degenerate author watermark grid')
    return wm, detector


def paired_noise(watermark, seed, alpha):
    import torch
    # Official forward samples CPU noise then moves it to the grid device.
    # Isolate RNG so clean/wm pairs and resumed batches use the same base noise.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return watermark(1, ret_noise=True, alpha=alpha)


def numerical_policy():
    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def generate(run):
    import torch
    from diffusers import StableDiffusionPipeline
    from teacher import valid_image
    numerical_policy()
    c, assets, prompts = (read_json(run / name) for name in ('config.json', 'assets.json', 'prompts.json'))
    watermark, _ = author_models(assets['serum_checkpoint'])
    pipe = StableDiffusionPipeline.from_pretrained(assets['teacher'], torch_dtype=torch.float16,
                safety_checker=None, local_files_only=True).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    for split in ('train', 'calibration', 'test', 'validation'):
        conditions = training_conditions(c) if split == 'train' else ('clean',) if split == 'calibration' else ('clean', 'wm')
        for condition in conditions:
            out = run / 'images' / f'teacher_{condition}_{split}'; out.mkdir(parents=True, exist_ok=True)
            freeze(out / 'generation.json', {'method':'SERUM official Watermark.forward',
                'checkpoint_sha256':CHECKPOINT_SHA256, 'teacher_archive_sha256':assets['teacher_archive_sha256'],
                'condition':condition, 'alpha':c['serum_alpha'], 'seed':c['seed'],
                'prompts_sha256':sha256(run / 'prompts.json'), 'steps':c['inference_steps'],
                'cfg':7.5, 'resolution':512, 'batch_size':c['teacher_batch'],
                'scheduler':type(pipe.scheduler).__name__, 'dtype':'float16'})
            rows = prompts[split]
            for start in range(0, len(rows), c['teacher_batch']):
                batch = rows[start:start+c['teacher_batch']]
                paths = [out / image_name(start+j) for j in range(len(batch))]
                if not all(valid_image(p, (512,512)) for p in paths):
                    seeds = [c['seed']*1000000+r['image_id'] for r in batch]
                    with torch.inference_mode():
                        pairs = [paired_noise(watermark, seed, c['serum_alpha']) for seed in seeds]
                        latents = torch.cat([pair[0 if condition == 'wm' else 1] for pair in pairs]).half()
                        generators = [torch.Generator('cuda').manual_seed(seed) for seed in seeds]
                        latent = pipe([r['text'] for r in batch], negative_prompt=['']*len(batch),
                            latents=latents, generator=generators, height=512, width=512,
                            num_inference_steps=c['inference_steps'], guidance_scale=7.5, output_type='latent').images
                        decoded = pipe.vae.decode(latent/pipe.vae.config.scaling_factor, return_dict=False)[0]
                        if not torch.isfinite(decoded).all(): raise RuntimeError('Nonfinite teacher images')
                        images = pipe.image_processor.postprocess(decoded, output_type='pil')
                    for image, path in zip(images, paths):
                        temp = path.with_suffix('.partial.png'); image.save(temp); temp.replace(path)
                print(f'SERUM teacher {condition}/{split}: {start+len(batch)}/{len(rows)}', flush=True)
            entries = [{'file_name':image_name(i), 'text':r['text'], 'sha256':sha256(out/image_name(i))} for i,r in enumerate(rows)]
            with (out/'metadata.jsonl').open('w') as f:
                for entry in entries: f.write(json.dumps({k:entry[k] for k in ('file_name','text')})+'\n')
            freeze(out/'image_hashes.json', entries)


def sample_posterior(posterior, start, device):
    """Fixed per-image posterior RNG, shared across branches and batch partitions."""
    import torch
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
    return torch.cat([DiagonalGaussianDistribution(params.unsqueeze(0)).sample(
        generator=torch.Generator(device).manual_seed(42 + start + j))
        for j, params in enumerate(posterior.parameters)])


def score(run, directories):
    import numpy as np
    import torch
    from PIL import Image
    from diffusers import AutoencoderKL
    numerical_policy()
    c, assets = read_json(run/'config.json'), read_json(run/'assets.json')
    _, detector = author_models(assets['serum_checkpoint'])
    vae = AutoencoderKL.from_pretrained(assets['teacher'], subfolder='vae', torch_dtype=torch.float16,
                local_files_only=True).eval().requires_grad_(False).to('cuda')
    for directory in directories:
        folder = run/'images'/directory
        files = sorted(folder.glob('sample_*.png'))
        if not files: raise ValueError(f'No images to score: {folder}')
        out = run/'scores'/f'{directory}.json'
        freeze(out.with_suffix('.provenance.json'), {
            'checkpoint_sha256':CHECKPOINT_SHA256, 'teacher_archive_sha256':assets['teacher_archive_sha256'],
            'encoding':'author SD2.1 VAE posterior.sample; seed=42+image index; scaled latents',
            'batch_size':c['score_batch'], 'vae_dtype':'float16', 'detector_dtype':'float32'})
        old = {r['file']:r for r in read_json(out)} if out.exists() else {}
        result = []
        for start in range(0, len(files), c['score_batch']):
            batch = files[start:start+c['score_batch']]
            hashes = [sha256(f) for f in batch]
            if all(f.name in old and old[f.name]['sha256'] == digest for f,digest in zip(batch, hashes)):
                result.extend(old[f.name] for f in batch)
            else:
                xs = []
                for file in batch:
                    with Image.open(file) as im:
                        if im.size != (512,512): raise ValueError(f'SERUM requires 512x512 images: {file}')
                        xs.append(torch.from_numpy(np.array(im.convert('RGB'), copy=True)).permute(2,0,1).float()/127.5-1)
                with torch.inference_mode():
                    posterior = vae.encode(torch.stack(xs).to('cuda',dtype=torch.float16)).latent_dist
                    latent = sample_posterior(posterior, start, 'cuda')*vae.config.scaling_factor
                    values = detector(latent.float()).flatten().cpu().tolist()
                finite(values)
                if len(values) != len(batch): raise RuntimeError('SERUM score count mismatch')
                result.extend({'file':f.name,'sha256':digest,'score':value} for f,digest,value in zip(batch,hashes,values))
            write_json(out, result)
            print(f'SERUM detector {directory}: {len(result)}/{len(files)}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True)
    p.add_argument('--stage', choices=['generate','score'], required=True); p.add_argument('--directories', nargs='+')
    a=p.parse_args()
    if a.stage == 'generate': generate(a.run.resolve())
    elif not a.directories: p.error('--directories required for score')
    else: score(a.run.resolve(), a.directories)
