import argparse,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, AutoModel, AutoImageProcessor
import lpips

def main():
    p=argparse.ArgumentParser(); p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--regen',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--prompts',type=Path,required=True); a=p.parse_args()
    files=sorted(a.raw.glob('*.png')); prompts=a.prompts.read_text().splitlines()
    assert 500<=len(files)<=1000 and len(prompts)==len(files)
    clip=CLIPModel.from_pretrained('openai/clip-vit-large-patch14').eval().to('cuda')
    cp=CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
    dino=AutoModel.from_pretrained('facebook/dinov2-base').eval().to('cuda')
    dp=AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    percept=lpips.LPIPS(net='alex').eval().to('cuda'); rows=[]
    with torch.no_grad():
        for f,prompt in zip(files,prompts):
            with Image.open(f) as im: raw=im.convert('RGB')
            with Image.open(a.regen/f.name) as im: reg=im.convert('RGB')
            ci=cp(images=[raw,reg],return_tensors='pt').to('cuda')
            cf=clip.get_image_features(**ci)
            if not isinstance(cf,torch.Tensor): cf=cf.pooler_output
            cf=torch.nn.functional.normalize(cf,dim=-1)
            text=cp(text=[prompt],padding=True,truncation=True,return_tensors='pt').to('cuda')
            tf=clip.get_text_features(**text)
            if not isinstance(tf,torch.Tensor): tf=tf.pooler_output
            tf=torch.nn.functional.normalize(tf,dim=-1)
            di=dp(images=[raw,reg],return_tensors='pt').to('cuda')
            df=torch.nn.functional.normalize(dino(**di).last_hidden_state[:,0],dim=-1)
            tensors=[torch.tensor(np.array(im),device='cuda').permute(2,0,1).unsqueeze(0).float()/127.5-1 for im in [raw,reg]]
            rows.append({'file':f.name,'clip_image_cosine':float((cf[0]*cf[1]).sum()),
                         'dino_cosine':float((df[0]*df[1]).sum()),'lpips':float(percept(*tensors).item()),
                         'raw_clipscore':float(2.5*(cf[0]*tf[0]).sum().clamp(min=0)),
                         'regen_clipscore':float(2.5*(cf[1]*tf[0]).sum().clamp(min=0))})
    result={'count':len(rows),'means':{k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='file'},'per_image':rows}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
