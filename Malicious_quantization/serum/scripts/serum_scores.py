import argparse
import hashlib
import json
import sys
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import numpy as np
import torch

ROOT=Path('/root/workspace/serum')
SERUM=ROOT/'third_party/SERUM'
sys.path.insert(0,str(SERUM))

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def score(images, out, seed=42, batch_size=32):
    import os
    images=Path(images).resolve()
    out=Path(out).resolve()
    os.chdir(SERUM)
    from src.utils.config import Config
    from src.models import WatermarkScoreModel
    from diffusers import AutoencoderKL
    from PIL import Image
    from torchvision.transforms.functional import to_tensor
    frozen=json.loads((ROOT/'manifests/serum_frozen.json').read_text())
    for key in ('score_model','watermark'):
        assert sha(frozen[key]['path'])==frozen[key]['sha256'], 'Frozen model changed'
    config=Config('configs/config_sd21.yaml')
    model=WatermarkScoreModel.load(config,filename=frozen['score_model']['path']).eval().requires_grad_(False).to(config.get_device())
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    # Same pretrained VAE and posterior sampling as upstream, without compilation
    # autotuning so calibration and later audits use one stable numerical path.
    vae=AutoencoderKL.from_pretrained(config.diffusion.stable_diffusion_model_id,
        subfolder='vae',torch_dtype=torch.float16).eval().requires_grad_(False).to(config.get_device())
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    files=sorted(Path(images).glob('*.png'))
    assert files, 'No PNG images'
    vals=[]
    with torch.no_grad():
        for i,start in enumerate(range(0,len(files),batch_size)):
            batch=[]
            for f in files[start:start+batch_size]:
                with Image.open(f) as im:
                    batch.append(to_tensor(im.convert('RGB'))*2-1)
            imgs=torch.stack(batch).to(config.get_device())
            posterior=vae.encode(imgs.half()).latent_dist
            rng=torch.Generator(device=config.get_device()).manual_seed(seed+start)
            latent=posterior.sample(generator=rng)*vae.config.scaling_factor
            vals.append(model(latent.float()).flatten().cpu().numpy())
            if i%20==0: print('scored',min((i+1)*batch_size,len(files)),flush=True)
    s=np.concatenate(vals).astype(np.float32)
    assert np.isfinite(s).all()
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    np.save(out,s)
    metadata={'seed':seed,'batch_size':batch_size,'encoding':'SERUM posterior sample; generator seed = seed + batch start',
              'num_images':len(files),'images':str(images),'score_sha256':frozen['score_model']['sha256'],
              'file_order_sha256':hashlib.sha256('\n'.join(p.name for p in files).encode()).hexdigest()}
    Path(str(out)+'.json').write_text(json.dumps(metadata,indent=2))
    return s

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--images',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--calibrate',action='store_true')
    p.add_argument('--threshold-file',type=Path,default=ROOT/'metrics/serum_threshold_1pct.json')
    a=p.parse_args()
    a.images=a.images.resolve()
    a.out=a.out.resolve()
    a.threshold_file=a.threshold_file.resolve()
    s=score(a.images,a.out)
    if a.calibrate:
        assert a.images.resolve()==ROOT/'data/raw/calib_clean' and len(s)==5000
        threshold=float(np.nextafter(np.sort(s)[::-1][50],np.float32(np.inf),dtype=np.float32))
        result={'threshold':threshold,'target_fpr':0.01,'empirical_fpr':float(np.mean(s>=np.float32(threshold))),
                'false_positives':int(np.sum(s>=np.float32(threshold))),'num_clean':len(s),
                'rule':'score >= float32(threshold)','scoring_seed':42,'scoring_batch_size':32,
                'model':json.loads((ROOT/'manifests/serum_frozen.json').read_text()),'scores':str(a.out)}
        assert result['empirical_fpr']<=0.01
        a.threshold_file.write_text(json.dumps(result,indent=2))
    else:
        cal=json.loads(a.threshold_file.read_text())
        current=json.loads((ROOT/'manifests/serum_frozen.json').read_text())
        assert cal['model']==current, 'Threshold/model mismatch'
        result={'num_images':len(s),'fraction_above':float(np.mean(s>=np.float32(cal['threshold']))),
                'threshold':cal['threshold']}
        Path(str(a.out)+'.result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__': main()
