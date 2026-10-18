"""Pinned assets, disjoint prompt splits, paired generation and feature caches."""
import json
from pathlib import Path
import paths
from paths import freeze, read_json, write_json, sha256, image_name
from new_watermark.method import image_features


def prepare(run):
    import fcntl
    import hashlib
    import zipfile
    from assets import download, split_prompts, hub_snapshot, COCO_URL, COCO_MD5
    from serum_assets import STUDENT_REV
    from utility import CLIP_REV
    c = read_json(run/'config.json')
    cache = Path(c['cache']); cache.mkdir(parents=True, exist_ok=True)
    with (cache/'assets.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        archive = cache/'annotations_trainval2014.zip'
        if not archive.exists(): download(COCO_URL, archive)
        if hashlib.md5(archive.read_bytes()).hexdigest() != COCO_MD5:
            raise RuntimeError('COCO annotation checksum mismatch')
        with zipfile.ZipFile(archive) as z:
            annotations = json.loads(z.read('annotations/captions_val2014.json'))['annotations']
        counts = {k: c[k+'_n'] for k in ('design_fit','teacher_train','meta_probe','calibration','pilot_train','pilot_eval','train','validation','test')}
        freeze(run/'prompts.json', split_prompts(annotations, c['seed'], counts))
        base = hub_snapshot('CompVis/stable-diffusion-v1-4', STUDENT_REV, cache,
            lambda n: n == 'model_index.json' or (n.split('/')[0] in {
                'scheduler','tokenizer','text_encoder','unet','vae','feature_extractor'} and
                n.endswith(('.json','.txt','.safetensors')) and '.fp16.' not in n and '.non_ema.' not in n))
        clip = hub_snapshot('openai/clip-vit-base-patch32', CLIP_REV, cache,
            lambda n: '/' not in n and (n.endswith(('.json','.txt','.safetensors')) or n == 'pytorch_model.bin'))
        freeze(run/'assets.json', {'student':base, 'clip':clip, 'base_revision':STUDENT_REV,
                                  'clip_revision':CLIP_REV, 'coco_sha256':sha256(archive)})
        print('Disjoint prompts:', counts, flush=True)


def verify_images(folder, n=None):
    entries = read_json(folder/'image_hashes.json')
    if n is not None and len(entries) != n: raise RuntimeError(f'Wrong image count: {folder}')
    names = [r.get('file_name', r.get('file')) for r in entries]
    if names != [image_name(i) for i in range(len(entries))]: raise RuntimeError('Unordered images')
    for r,name in zip(entries,names):
        if sha256(folder/name) != r['sha256']: raise RuntimeError(f'Changed image: {folder/name}')
    return entries


def load_base(run, model=None, dtype=None):
    import torch
    from diffusers import StableDiffusionPipeline
    assets = read_json(run/'assets.json')
    pipe = StableDiffusionPipeline.from_pretrained(str(model or assets['student']),
        torch_dtype=dtype or torch.float16, safety_checker=None, local_files_only=True).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    pipe.unet.eval(); pipe.vae.eval(); pipe.text_encoder.eval()
    return pipe


def generate(run, pipe, label, split, model_identity, rows=None):
    import torch
    from teacher import valid_image
    c = read_json(run/'config.json')
    rows = rows if rows is not None else read_json(run/'prompts.json')[split]
    out = run/'images'/label; out.mkdir(parents=True, exist_ok=True)
    freeze(out/'generation.json', {'model':model_identity, 'split':split, 'prompts':rows,
        'seed':c['seed'], 'rng':'seed*1000000+COCO image_id, paired across models',
        'steps':c['inference_steps'], 'cfg':7.5, 'resolution':512,
        'batch':c['inference_batch'], 'scheduler':dict(pipe.scheduler.config),
        'dtype':str(pipe.unet.dtype), 'watermark_inference_inputs':False})
    if (out/'image_hashes.json').exists():
        verify_images(out, len(rows)); print(f'Reusing verified images: {label}', flush=True); return
    for start in range(0,len(rows),c['inference_batch']):
        batch = rows[start:start+c['inference_batch']]
        files = [out/image_name(start+j) for j in range(len(batch))]
        if not all(valid_image(f,(512,512)) for f in files):
            rng = [torch.Generator('cuda').manual_seed(c['seed']*1000000+r['image_id']) for r in batch]
            with torch.inference_mode():
                # No special noise, key, hooks, or extra inference modules.
                images = pipe([r['text'] for r in batch], generator=rng, num_inference_steps=c['inference_steps'],
                              guidance_scale=7.5, height=512, width=512).images
            for im,f in zip(images,files):
                temp=f.with_suffix('.partial.png'); im.save(temp); temp.replace(f)
        print(f'Generate {label}: {start+len(batch)}/{len(rows)}', flush=True)
    entries = [{'file_name':image_name(i),'text':r['text'],'sha256':sha256(out/image_name(i))} for i,r in enumerate(rows)]
    (out/'metadata.jsonl').write_text(''.join(json.dumps({k:r[k] for k in ('file_name','text')})+'\n' for r in entries))
    freeze(out/'image_hashes.json', entries)


class FeatureReader:
    def __init__(self, run):
        self.run, self.model, self.tokenizer = run, None, None

    def load(self):
        if self.model is None:
            from transformers import CLIPModel, CLIPTokenizer
            assets=read_json(self.run/'assets.json')
            self.model=CLIPModel.from_pretrained(assets['clip'],local_files_only=True).float().cuda().eval().requires_grad_(False)
            self.tokenizer=CLIPTokenizer.from_pretrained(assets['clip'],local_files_only=True)
        return self.model

    def read(self, label, rows, folder=None):
        import numpy as np
        import torch
        from PIL import Image
        folder=folder or self.run/'images'/label
        entries=verify_images(folder, len(rows))
        path=self.run/'features'/f'{label}.json'
        identity={'images':[r['sha256'] for r in entries], 'prompts':rows,
                  'clip_revision':read_json(self.run/'assets.json')['clip_revision'],
                  'preprocess':'RGB /255; bicubic antialias square 224; clamp; official CLIP normalization'}
        if path.exists():
            saved=read_json(path)
            if saved['identity']!=identity: raise RuntimeError(f'Changed feature inputs: {label}')
            return saved
        model=self.load(); features=[]; clip=[]
        for start in range(0,len(rows),16):
            pixels=[]
            for i in range(start,min(start+16,len(rows))):
                with Image.open(folder/image_name(i)) as im:
                    pixels.append(torch.from_numpy(np.array(im.convert('RGB'),dtype=np.float32)/255).permute(2,0,1))
            x=torch.stack(pixels).cuda()
            tokens=self.tokenizer([r['text'] for r in rows[start:start+16]],padding=True,truncation=True,return_tensors='pt').to('cuda')
            with torch.inference_mode():
                feats=image_features(model,x)
                text=torch.nn.functional.normalize(model.get_text_features(**tokens),dim=-1)
                clip.extend((2.5*(feats*text).sum(-1).clamp_min(0)).cpu().tolist())
            features.extend(feats.cpu().tolist())
            print(f'Features {label}: {start+len(pixels)}/{len(rows)}', flush=True)
        result={'identity':identity,'features':features,'clip_per_image':clip,'clip_mean':sum(clip)/len(clip)}
        write_json(path,result); return result


def paired_psnr(a,b,n):
    import numpy as np
    from PIL import Image
    values=[]
    for i in range(n):
        with Image.open(a/image_name(i)) as ia, Image.open(b/image_name(i)) as ib:
            mse=np.mean((np.asarray(ia,dtype=np.float64)/255-np.asarray(ib,dtype=np.float64)/255)**2)
        values.append(float(-10*np.log10(max(mse,1e-12))))
    return sum(values)/len(values)
