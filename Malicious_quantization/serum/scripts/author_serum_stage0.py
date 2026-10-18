"""Isolated author-key teacher generation, calibration and detection audit."""
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from PIL import Image

ROOT=Path('/root/workspace/serum')
A=ROOT/'assets/serum_author'
D=ROOT/'data/serum_author/raw'
M=ROOT/'metrics/serum_author'
sys.path.insert(0,str(ROOT/'third_party/SERUM'))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def save_json(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp.json');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)

def load_models():
    from src.utils.config import Config
    from src.models import WatermarkScoreModel,Watermark
    from diffusers import AutoencoderKL
    manifest=json.loads((A/'manifest.json').read_text())
    assert manifest['status']=='verified'
    for model in manifest['models'].values(): assert sha(model['path'])==model['sha256']
    config=Config(str(A/'config_sd21_author.yaml'))
    score=WatermarkScoreModel(config).eval().requires_grad_(False).cuda()
    watermark=Watermark(config).eval().requires_grad_(False).cuda()
    score.load_state_dict(torch.load(A/'score_model.pt',weights_only=True,map_location='cpu'))
    watermark.load_state_dict(torch.load(A/'watermark.pt',weights_only=True,map_location='cpu'))
    vae=AutoencoderKL.from_pretrained(str(A/'sd21_snapshot'),subfolder='vae',torch_dtype=torch.float16,
        local_files_only=True).eval().requires_grad_(False).cuda()
    return manifest,config,score,watermark,vae

def score_directory(images,out,model,vae,batch=32):
    from torchvision.transforms.functional import to_tensor
    files=sorted(images.glob('*.png')); assert files
    scores=[]
    with torch.inference_mode():
        for start in range(0,len(files),batch):
            xs=[]
            for p in files[start:start+batch]:
                with Image.open(p) as im:
                    assert im.size==(512,512) and im.mode=='RGB'
                    xs.append(to_tensor(im)*2-1)
            posterior=vae.encode(torch.stack(xs).cuda().half()).latent_dist
            rng=torch.Generator(device='cuda').manual_seed(42+start)
            latents=posterior.sample(generator=rng)*vae.config.scaling_factor
            scores.append(model(latents.float()).flatten().cpu().numpy())
    vals=np.concatenate(scores).astype(np.float32);assert np.isfinite(vals).all()
    out.parent.mkdir(parents=True,exist_ok=True);np.save(out,vals)
    save_json(Path(str(out)+'.json'),{'image_count':len(files),'images':str(images),
        'file_order_sha256':hashlib.sha256('\n'.join(p.name for p in files).encode()).hexdigest(),
        'seed':42,'batch_size':batch,'checkpoint_sha256':sha(A/'latest_checkpoint.pt'),
        'vae_snapshot':str((A/'sd21_snapshot').resolve())})
    return vals

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true')
    p.add_argument('--worker',type=int,choices=[0,1]);p.add_argument('--audit-only',action='store_true');args=p.parse_args()
    from diffusers import AutoPipelineForText2Image
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    manifest,config,model,watermark,vae=load_models()
    pipe=None
    if not args.audit_only:
        pipe=AutoPipelineForText2Image.from_pretrained(str(A/'sd21_snapshot'),torch_dtype=torch.float16,
            local_files_only=True).to('cuda')
        pipe.set_progress_bar_config(disable=True)
    grid=watermark.grid.detach();norm=(grid-grid.mean())/grid.std()
    if args.smoke:
        smoke=ROOT/'data/serum_author/smoke';smoke.mkdir(parents=True,exist_ok=True)
        prompt=(ROOT/'prompts/adapt_val.txt').read_text().splitlines()[0]
        noise=torch.randn((1,4,64,64),device='cuda',generator=torch.Generator(device='cuda').manual_seed(20000))
        for kind,z in [('clean',noise),('wm',noise*(.5**.5)+norm*(.5**.5))]:
            with torch.inference_mode():
                latent=pipe(prompt=[prompt],negative_prompt=[''],latents=z.half(),num_inference_steps=50,
                    guidance_scale=7.5,height=512,width=512,output_type='latent').images
                decoded=pipe.vae.decode(latent/pipe.vae.config.scaling_factor,return_dict=False)[0]
                image=pipe.image_processor.postprocess(decoded,output_type='pil')[0]
            folder=smoke/kind;folder.mkdir(exist_ok=True);image.convert('RGB').save(folder/'00000.png')
            s=score_directory(folder,M/f'smoke_{kind}_scores.npy',model,vae)
            print('smoke',kind,s.tolist(),flush=True)
        save_json(M/'smoke.json',{'status':'passed','samples_per_kind':1,'steps':50,'CFG':7.5,
            'scope':'compatibility only; no scientific performance claim','manifest_sha256':sha(A/'manifest.json')})
        return
    splits=[('adapt_train.txt','train',5000,10000),('adapt_val.txt','val',1000,20000),
            ('detector_calib.txt','calib',5000,30000)]
    inventory={}
    for promptfile,split,count,seedbase in splits:
        prompts=(ROOT/'prompts'/promptfile).read_text().splitlines();assert len(prompts)==count
        for kind in ['clean','wm']: (D/f'{kind}_{split}').mkdir(parents=True,exist_ok=True)
        for start in range(0,count,8):
            if args.audit_only: break
            if args.worker is not None and (start//8)%2!=args.worker: continue
            idx=list(range(start,min(start+8,count)))
            paths=[D/f'{kind}_{split}'/f'{i:05d}.png' for kind in ['clean','wm'] for i in idx]
            if all(p.exists() for p in paths):
                for p in paths:
                    with Image.open(p) as im: im.load();assert im.size==(512,512) and im.mode=='RGB'
                continue
            noises=torch.cat([torch.randn((1,4,64,64),generator=torch.Generator(device='cuda').manual_seed(seedbase+i),device='cuda') for i in idx])
            kinds=[('wm',noises*(.5**.5)+norm*(.5**.5))] if (M/'clean_reuse.json').exists() else [('clean',noises),('wm',noises*(.5**.5)+norm*(.5**.5))]
            for kind,noise in kinds:
                with torch.inference_mode():
                    latent=pipe(prompt=[prompts[i] for i in idx],negative_prompt=['']*len(idx),
                        latents=noise.half(),num_inference_steps=50,guidance_scale=7.5,
                        height=512,width=512,output_type='latent').images
                    image=pipe.vae.decode(latent/pipe.vae.config.scaling_factor,return_dict=False)[0]
                    images=pipe.image_processor.postprocess(image,output_type='pil')
                for i,im in zip(idx,images):
                    p=D/f'{kind}_{split}'/f'{i:05d}.png'
                    tmp=p.with_suffix('.tmp.png');im.convert('RGB').save(tmp);tmp.replace(p)
            print(split,start+len(idx),'/',count,flush=True)
        if args.worker is not None: continue
        for kind in ['clean','wm']:
            folder=D/f'{kind}_{split}'; files=sorted(folder.glob('*.png'));assert len(files)==count
            rows=[{'file_name':f'{i:05d}.png','text':prompts[i],'source_id':i,'seed':None if kind=='clean' and (M/'clean_reuse.json').exists() else seedbase+i,
                   'origin':'legacy_clean_same_backbone' if kind=='clean' and (M/'clean_reuse.json').exists() else 'author_key_new_generation',
                   'sha256':sha(folder/f'{i:05d}.png')} for i in range(count)]
            (folder/'metadata.jsonl').write_text('\n'.join(json.dumps(x) for x in rows)+'\n')
            inventory[f'{kind}_{split}']={'count':count,'prompt_sha256':sha(ROOT/'prompts'/promptfile),
                'metadata_sha256':sha(folder/'metadata.jsonl')}
    if args.worker is not None: return
    del pipe,watermark;torch.cuda.empty_cache()
    # Calibrate only on clean held-out prompts; freeze before any WM or student audit.
    clean=score_directory(D/'clean_calib',M/'calib_clean_scores.npy',model,vae)
    threshold=float(np.nextafter(np.sort(clean)[::-1][50],np.float32(np.inf),dtype=np.float32))
    frozen={'threshold':threshold,'target_fpr':.01,'false_positives':int((clean>=threshold).sum()),
        'empirical_fpr':float((clean>=threshold).mean()),'num_clean':5000,'rule':'score >= float32(threshold)',
        'author_manifest_sha256':sha(A/'manifest.json'),'calibration_prompt_sha256':sha(ROOT/'prompts/detector_calib.txt'),
        'scoring_seed':42,'scoring_batch_size':32}
    assert frozen['false_positives']<=50
    tp=M/'threshold_1pct.json'
    if tp.exists(): assert json.loads(tp.read_text())==frozen,'Frozen threshold mismatch'
    else: save_json(tp,frozen)
    results={}
    for split in ['train','val','calib']:
        for kind in ['clean','wm']:
            s=clean if (split,kind)==('calib','clean') else score_directory(D/f'{kind}_{split}',M/f'{kind}_{split}_scores.npy',model,vae)
            results[f'{kind}_{split}']={'n':len(s),'detections':int((s>=threshold).sum()),'rate':float((s>=threshold).mean())}
    save_json(M/'teacher_detection.json',{'threshold':frozen,'results':results,'inventory':inventory})
    print(json.dumps(results,indent=2),flush=True)

if __name__=='__main__': main()
