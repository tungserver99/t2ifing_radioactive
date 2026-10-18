"""Held-out checkpoint audit. Selection reads denoising MSE only, never test/detector metrics."""
import argparse
import csv
import math
from pathlib import Path

from common import freeze, image_name, read_json, sha256, write_json
from student import branches, checkpoint_hashes, generate_images, load_pipeline, training_conditions

SELECTION_RULE = "minimum validation denoising MSE; exact ties choose earlier step; no detector/utility/test input"


def validation_label(name, step):
    return f"validation_{name}_step{step}"


def choose_checkpoint(rows):
    if not rows or any(not math.isfinite(r['denoising_mse']) or r['denoising_mse'] < 0 for r in rows):
        raise ValueError("Missing/nonfinite validation losses; cannot select a valid checkpoint")
    return min(rows, key=lambda r: (r['denoising_mse'], r['step']))


def prepare_cache(run):
    import torch
    from PIL import Image
    from safetensors.torch import save_file
    from torchvision import transforms
    from diffusers import StableDiffusionPipeline
    c, assets = read_json(run / 'config.json'), read_json(run / 'assets.json')
    prompts = read_json(run / 'prompts.json')['validation']
    folder = run / 'validation_cache'
    folder.mkdir(exist_ok=True)
    records = {}
    for condition in training_conditions(c):
        images = run / 'images' / f'teacher_{condition}_validation'
        records[condition] = [sha256(images / image_name(i)) for i in range(len(prompts))]
    protocol = {'prompts_sha256': sha256(run / 'prompts.json'), 'seed': c['seed'],
                'images': records, 'draws': c['validation_draws'], 'resolution': 512,
                'latent': 'one fixed posterior sample per image; FP32 VAE/text encoder',
                'noise': 'fixed independent CPU RNG per image/draw, paired across checkpoints and branches'}
    freeze(folder / 'protocol.json', protocol)
    if (folder / 'complete.json').exists():
        verify_cache(run)
        print('Reusing verified held-out validation cache', flush=True)
        return
    pipe = StableDiffusionPipeline.from_pretrained(assets['student'], torch_dtype=torch.float32,
                                                  safety_checker=None, local_files_only=True)
    # Keep UNet on CPU: only VAE/text encoder are needed for cache construction.
    pipe.vae.to('cuda').eval().requires_grad_(False)
    pipe.text_encoder.to('cuda').eval().requires_grad_(False)
    transform = transforms.Compose([transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512), transforms.ToTensor(), transforms.Normalize([.5], [.5])])
    files = {}
    for condition in training_conditions(c):
        images = run / 'images' / f'teacher_{condition}_validation'
        for i, prompt in enumerate(prompts):
            target = folder / f'{condition}_{i:06d}.safetensors'
            with Image.open(images / image_name(i)) as im:
                pixels = transform(im.convert('RGB')).unsqueeze(0).cuda()
            tokens = pipe.tokenizer(prompt['text'], padding='max_length', truncation=True,
                                    max_length=pipe.tokenizer.model_max_length, return_tensors='pt').input_ids.cuda()
            rng = torch.Generator('cuda').manual_seed(c['seed'] * 1000000 + 800000 + i)
            with torch.inference_mode():
                latent = pipe.vae.encode(pixels).latent_dist.sample(generator=rng) * pipe.vae.config.scaling_factor
                text = pipe.text_encoder(tokens)[0]
            tensors = {'latent': latent.cpu().contiguous(), 'text': text.cpu().contiguous()}
            if not all(torch.isfinite(t).all() for t in tensors.values()):
                raise RuntimeError('Nonfinite validation encoding')
            temp = target.with_suffix('.partial')
            save_file(tensors, str(temp))
            temp.replace(target)
            files[target.name] = sha256(target)
            if (i+1) % 25 == 0 or i+1 == len(prompts):
                print(f'Validation cache {condition}: {i+1}/{len(prompts)}', flush=True)
    write_json(folder / 'complete.json', {'files': files, 'protocol_sha256': sha256(folder/'protocol.json')})


def verify_cache(run):
    folder = run / 'validation_cache'
    audit = read_json(folder / 'complete.json')
    if sha256(folder / 'protocol.json') != audit['protocol_sha256']:
        raise RuntimeError('Validation protocol changed')
    for name, digest in audit['files'].items():
        if sha256(folder / name) != digest:
            raise RuntimeError(f'Validation cache changed: {name}')
    return sha256(folder / 'complete.json')


def denoising_loss(unet, scheduler, files, seed, draws, device='cuda'):
    """Identical latent/noise/timestep draws for all candidates; correct epsilon/v target."""
    import torch
    from safetensors.torch import load_file
    losses = []
    unet.eval()
    dtype = next(unet.parameters()).dtype
    for i, file in enumerate(files):
        data = load_file(str(file))
        latent = data['latent'].to(device=device, dtype=dtype)
        text = data['text'].to(device=device, dtype=dtype)
        samples = []
        for draw in range(draws):
            rng = torch.Generator('cpu').manual_seed(seed * 10000000 + 900000 + i * draws + draw)
            noise = torch.randn(data['latent'].shape, generator=rng).to(device=device, dtype=dtype)
            timestep = torch.randint(0, scheduler.config.num_train_timesteps, (1,), generator=rng).to(device)
            noisy = scheduler.add_noise(latent, noise, timestep)
            kind = scheduler.config.prediction_type
            if kind == 'epsilon':
                target = noise
            elif kind == 'v_prediction':
                target = scheduler.get_velocity(latent, noise, timestep)
            else:
                raise ValueError(f'Unsupported prediction type: {kind}')
            with torch.inference_mode():
                predicted = unet(noisy, timestep, text).sample
                loss = (predicted.float() - target.float()).square().mean().item()
            if not math.isfinite(loss):
                raise RuntimeError('Nonfinite held-out denoising loss')
            samples.append(loss)
        losses.append(sum(samples) / draws)
        if (i+1) % 25 == 0 or i+1 == len(files):
            print(f'Validation loss: {i+1}/{len(files)}', flush=True)
    if not losses:
        raise ValueError('Empty validation dataset')
    return {'denoising_mse': sum(losses)/len(losses), 'per_image_mse': losses, 'draws_per_image': draws}


def evaluate(run, name):
    import gc
    import torch
    from diffusers import DDPMScheduler
    c, assets = read_json(run/'config.json'), read_json(run/'assets.json')
    branch = next(b for b in branches(c) if b['name'] == name)
    cache_digest = verify_cache(run)
    files = [run/'validation_cache'/f"{branch['condition']}_{i:06d}.safetensors" for i in range(c['validation_n'])]
    scheduler = DDPMScheduler.from_pretrained(assets['student'], subfolder='scheduler', local_files_only=True)
    for step in c['checkpoint_steps']:
        label = validation_label(name, step)
        snapshot = run/'students'/name/f'snapshot-{step}'
        weights = checkpoint_hashes(snapshot)
        loss_path = run/'validation'/f'{label}.json'
        identity = {'step': step, 'branch': name, 'checkpoint_hashes': weights,
                    'validation_cache_sha256': cache_digest, 'prediction_type': scheduler.config.prediction_type}
        # FP32 loss evaluation avoids fp16 overflow; generation remains FP16 for paired utility comparisons.
        pipe, checkpoint = load_pipeline(run, name, snapshot, dtype=torch.float32)
        if loss_path.exists():
            prior = read_json(loss_path)
            if any(prior[k] != v for k, v in identity.items()):
                raise RuntimeError('Validation checkpoint or cache changed')
        else:
            measured = denoising_loss(pipe.unet, scheduler, files, c['seed'], c['validation_draws'])
            write_json(loss_path, {**identity, **measured})
        pipe.to(dtype=torch.float16)
        generate_images(run, name, pipe, checkpoint, split='validation', label=label)
        print(f"{label}: denoising MSE={read_json(loss_path)['denoising_mse']:.8f}", flush=True)
        del pipe
        gc.collect()
        torch.cuda.empty_cache()


def select(run, name):
    from report import scores, wilson, detector_value
    c = read_json(run/'config.json')
    rows = [read_json(run/'validation'/f'{validation_label(name, step)}.json') for step in c['checkpoint_steps']]
    winner = choose_checkpoint(rows)
    # Freeze the decision before reading detector/quality diagnostics.
    selection = {'rule': SELECTION_RULE, 'selected_step': winner['step'],
                 'denoising_mse': winner['denoising_mse'], 'checkpoint_hashes': winner['checkpoint_hashes'],
                 'validation_cache_sha256': winner['validation_cache_sha256'],
                 'candidate_loss_sha256': {str(r['step']): sha256(run/'validation'/f"{validation_label(name, r['step'])}.json") for r in rows},
                 'test_used_for_selection': False, 'detector_used_for_selection': False,
                 'utility_used_for_selection': False}
    freeze(run/'result/selection'/f'{name}.json', selection)
    thresholds = read_json(run/'result/threshold.json')
    convention = thresholds.get('primary', 'absolute_z')
    threshold = thresholds[convention]
    utility_path = run/'result/validation_utility.json'
    utility = read_json(utility_path)['branches'] if utility_path.exists() else {}
    audit_path = run/'students'/name/'adapter_audit.json'
    adapter = read_json(audit_path) if audit_path.exists() else {}
    table = []
    for row in rows:
        label = validation_label(name, row['step'])
        z = scores(run, label, c['validation_generation_n'])
        count = sum(detector_value(r, convention) >= threshold for r in z)
        quality = utility.get(label, {})
        if not c['skip_utility'] and not quality:
            raise ValueError(f'Missing requested validation utility for {label}')
        table.append({'branch': name, 'step': row['step'], 'selected': row['step'] == winner['step'],
            'denoising_mse': row['denoising_mse'], 'detected': count, 'n': len(z),
            'detection_rate': count/len(z), 'ci95_low': wilson(count, len(z))[0], 'ci95_high': wilson(count, len(z))[1],
            'clip_score': quality.get('clip_score'), 'fid': quality.get('frechet_inception_distance'),
            'kid': quality.get('kernel_inception_distance_mean'),
            'trainable_parameters': adapter.get('trainable_parameters'),
            'quality_status': 'measured' if quality else 'explicitly_skipped',
            **({'student_calibrated_detection_rate': sum(r['score'] >= thresholds['serum_score_student_calibrated'] for r in z)/len(z)}
               if c.get('watermark_method') == 'serum' else {})})
    write_json(run/'result/validation'/f'{name}.json', {'selection': selection, 'milestones': table})
    all_rows = []
    for b in branches(c):
        path = run/'result/validation'/f"{b['name']}.json"
        if path.exists():
            all_rows.extend(read_json(path)['milestones'])
    with (run/'result/checkpoint_metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Selected {name}: step {winner['step']}, MSE={winner['denoising_mse']:.8f}; detector/utility excluded", flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--stage', choices=['cache', 'evaluate', 'select'], required=True)
    p.add_argument('--branch')
    a = p.parse_args()
    if a.stage != 'cache' and not a.branch:
        p.error('--branch is required')
    if a.stage == 'cache':
        prepare_cache(a.run.resolve())
    elif a.stage == 'evaluate':
        evaluate(a.run.resolve(), a.branch)
    else:
        select(a.run.resolve(), a.branch)
