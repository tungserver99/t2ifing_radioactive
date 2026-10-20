"""Ordinary native generation and the fixed semantic verifier."""
import argparse
import hashlib
import json
from pathlib import Path
from common import freeze, image_name, read_json, sha256, write_json


def check_imports():
    from diffusers import StableDiffusionPipeline, DDPMScheduler
    from transformers import CLIPTokenizer, CLIPTextModel
    from peft import LoraConfig
    import lpips


def native_generate(pipe, prompts, generators, steps):
    # Production inference accepts only a standard merged model and ordinary inputs.
    import torch
    # Request the pipeline's ordinary final latents so finiteness can be checked
    # before standard VAE decoding/PIL conversion. Never supply input latents.
    with torch.inference_mode():
        latent = pipe(prompts, generator=generators, num_inference_steps=steps,
                      guidance_scale=7.5, height=512, width=512, output_type='latent').images
        if not torch.isfinite(latent).all(): raise ValueError('Nonfinite native generated latent')
        image = pipe.vae.decode(latent / pipe.vae.config.scaling_factor).sample
        if not torch.isfinite(image).all(): raise ValueError('Nonfinite native decoded image')
        return pipe.image_processor.postprocess(image, output_type='pil')


def numerical_policy():
    import torch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_native(model):
    import torch
    from diffusers import StableDiffusionPipeline
    numerical_policy()
    pipe = StableDiffusionPipeline.from_pretrained(str(model), torch_dtype=torch.float16,
                safety_checker=None, local_files_only=True).to('cuda')
    if hasattr(pipe.unet, 'peft_config') and pipe.unet.peft_config:
        raise RuntimeError('Production teacher must contain merged native weights only')
    pipe.unet.eval(); pipe.vae.eval(); pipe.text_encoder.eval()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate_set(run, model, split, label, pipe=None):
    import torch
    from teacher import valid_image
    from semantic_assets import snapshot_identity
    c = read_json(run/'config.json'); rows = read_json(run/'prompts.json')[split]
    identity = snapshot_identity(model)
    pipe = pipe or load_native(model)
    out=run/'images'/label; out.mkdir(parents=True, exist_ok=True)
    seeds=[c['seed']*1000000+r['image_id'] for r in rows]
    freeze(out/'generation.json', {'model_hashes': identity, 'split': split,
        'prompts_sha256': sha256(run/'prompts.json'), 'seeds': seeds, 'steps': c['inference_steps'],
        'cfg':7.5, 'resolution':512, 'batch_size':c['teacher_batch'], 'dtype':'float16',
        'scheduler':type(pipe.scheduler).__name__, 'native':True})
    frozen = read_json(out/'image_hashes.json') if (out/'image_hashes.json').exists() else None
    if frozen:
        if len(frozen) != len(rows) or any(sha256(out/r['file_name']) != r['sha256'] for r in frozen):
            raise RuntimeError(f'Generated dataset changed: {label}')
        return
    for start in range(0,len(rows),c['teacher_batch']):
        batch=rows[start:start+c['teacher_batch']]
        paths=[out/image_name(start+j) for j in range(len(batch))]
        if not all(valid_image(p,(512,512)) for p in paths):
            generators=[torch.Generator('cuda').manual_seed(seed) for seed in seeds[start:start+len(batch)]]
            with torch.inference_mode():
                images=native_generate(pipe,[r['text'] for r in batch], generators,c['inference_steps'])
            if len(images) != len(batch): raise ValueError('Image count mismatch')
            for image,p in zip(images,paths):
                tmp=p.with_suffix('.partial.png'); image.save(tmp); tmp.replace(p)
        print(f'Native teacher {label}: {start+len(batch)}/{len(rows)}',flush=True)
    entries=[{'file_name':image_name(i),'text':r['text'],'seed':seeds[i],
              'sha256':sha256(out/image_name(i))} for i,r in enumerate(rows)]
    with (out/'metadata.jsonl').open('w',encoding='utf8') as f:
        for r in entries: f.write(json.dumps({k:r[k] for k in ('file_name','text')})+'\n')
    freeze(out/'image_hashes.json',entries)


def pixels(files, device='cuda'):
    import numpy as np
    import torch
    from PIL import Image
    images=[]
    for p in files:
        with Image.open(p) as im:
            if im.size != (512,512): raise ValueError(f'Expected 512x512 image: {p}')
            images.append(torch.from_numpy(np.array(im.convert('RGB'),copy=True)).permute(2,0,1).float()/127.5-1)
    return torch.stack(images).to(device)


def extractor(run):
    numerical_policy()
    import torch
    from diffusers import StableDiffusionPipeline
    from semantic_carrier import SemanticExtractor
    assets=read_json(run/'assets.json')
    pipe=StableDiffusionPipeline.from_pretrained(assets['teacher_clean'],torch_dtype=torch.float32,
                                                safety_checker=None,local_files_only=True)
    return SemanticExtractor(pipe)


def clean_prepare(run):
    import gc
    import torch
    a=read_json(run/'assets.json')
    pipe=load_native(a['teacher_clean'])
    for split in ('train','validation','calibration'):
        generate_set(run,a['teacher_clean'],split,f'teacher_clean_{split}',pipe)
    del pipe; gc.collect(); torch.cuda.empty_cache()


def carrier_stage(run):
    import torch
    from semantic_carrier import SemanticCarrier, save_carrier
    c=read_json(run/'config.json'); assets=read_json(run/'assets.json')
    rows=read_json(run/'prompts.json')['train']
    folder=run/'images/teacher_clean_train'
    hashes=read_json(folder/'image_hashes.json')
    if len(hashes)!=len(rows) or any(sha256(folder/r['file_name'])!=r['sha256'] for r in hashes):
        raise RuntimeError('Clean carrier training data changed')
    provenance={'train_prompt_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
        'prompt_file_sha256':sha256(run/'prompts.json'),'image_hashes':hashes,
        'reference_model':assets['teacher_source'],'reference_hashes':assets['teacher_clean_hashes'],
        'probe_timestep':0,'preprocessing':'RGB 512x512 /127.5 -1; VAE posterior mean * scaling_factor; GAP mid_block; FP32',
        'config_sha256':sha256(run/'config.json')}
    if (run/'carrier/complete.json').exists():
        SemanticCarrier.load(run/'carrier'); freeze(run/'carrier/provenance.json',provenance); return
    ext=extractor(run); fs=[]; cycles=[]
    for start in range(0,len(rows),c['score_batch']):
        batch=rows[start:start+c['score_batch']]
        x=pixels([folder/image_name(start+j) for j in range(len(batch))])
        captions=[r['text'] for r in batch]
        with torch.no_grad():
            text=ext.text(captions); z=ext.image_latent(x)
            fs.append(ext.encode_latent(z,text_hidden=text).cpu())
            cycles.append(ext.encode_latent(ext.cycle(z),text_hidden=text).cpu())
        print(f'Carrier: {start+len(batch)}/{len(rows)}',flush=True)
    save_carrier(run/'carrier',torch.cat(fs),torch.cat(cycles),c,provenance)


def prompt_rows(run,label):
    prompts=read_json(run/'prompts.json')
    if label.startswith('validation_'): return prompts['validation'][:read_json(run/'config.json')['validation_generation_n']]
    for split in ('train','validation','calibration','test'):
        if label.endswith('_'+split): return prompts[split]
    return prompts['test']


def score(run,directories):
    import torch
    from semantic_carrier import SemanticCarrier, check_finite
    ext=extractor(run); carrier=SemanticCarrier.load(run/'carrier','cuda')
    c=read_json(run/'config.json')
    for label in directories:
        rows=prompt_rows(run,label); folder=run/'images'/label
        files=[folder/image_name(i) for i in range(len(rows))]
        if len(list(folder.glob('sample_*.png')))!=len(rows): raise ValueError(f'Image count mismatch: {label}')
        out=run/'scores'/f'{label}.json'
        freeze(out.with_suffix('.provenance.json'), {'carrier_sha256':sha256(run/'carrier/complete.json'),
            'prompt_sha256':sha256(run/'prompts.json'),'reference_assets_sha256':sha256(run/'assets.json'),
            'batch_size':c['score_batch'],'dtype':'float32','probe_timestep':0,'posterior':'mean'})
        values=[]
        for start in range(0,len(files),c['score_batch']):
            batch=files[start:start+c['score_batch']]
            with torch.no_grad():
                q=carrier.coordinates(ext.encode_image(pixels(batch),[r['text'] for r in rows[start:start+len(batch)]]))
                r=(q*carrier.signs).mean(-1); check_finite(r,q)
            values += [{'file':p.name,'sha256':sha256(p),'score':s,'coordinates':coords}
                       for p,s,coords in zip(batch,r.cpu().tolist(),q.cpu().tolist())]
        freeze(out,values)


def generate(run):
    import gc
    import torch
    from semantic_assets import snapshot_identity
    a=read_json(run/'assets.json'); selected=read_json(run/'teacher_fp/selected.json')
    model=run/selected['model']
    if snapshot_identity(model)!=selected['model_hashes']: raise RuntimeError('Merged teacher changed')
    for source,splits in ((a['teacher_clean'],{'clean':['test']}),
                          (model,{'wm':['train','validation','test']})):
        pipe=load_native(source)
        for condition,names in splits.items():
            for split in names: generate_set(run,source,split,f'teacher_{condition}_{split}',pipe)
        del pipe; gc.collect(); torch.cuda.empty_cache()

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True)
    p.add_argument('--stage',choices=['clean','carrier','train','candidate','generate','score'],required=True)
    p.add_argument('--kappa',type=float)
    p.add_argument('--directories',nargs='+'); a=p.parse_args(); run=a.run.resolve()
    if a.stage=='score':
        if not a.directories: p.error('--directories is required')
        score(run,a.directories)
    elif a.stage=='candidate':
        if a.kappa is None:p.error('--kappa is required for candidate stage')
        from semantic_teacher_train import train_threshold_candidate
        train_threshold_candidate(run,a.kappa)
    elif a.stage=='clean': clean_prepare(run)
    elif a.stage=='carrier': carrier_stage(run)
    elif a.stage=='generate': generate(run)
    else:
        from semantic_teacher_train import train_candidates
        train_candidates(run)
