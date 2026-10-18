"""Compare every full UNet checkpoint on the SAME unseen images/noise/timesteps."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--student',type=Path,required=True)
    p.add_argument('--images',type=Path,required=True)
    p.add_argument('--prompts',type=Path,default=Path('/root/workspace/serum/prompts/adapt_val.txt'))
    p.add_argument('--base-model',default='CompVis/stable-diffusion-v1-4')
    p.add_argument('--seed',type=int,default=42000)
    p.add_argument('--noise-repeats',type=int,default=1)
    a=p.parse_args()
    files=sorted(a.images.glob('*.png')); prompts=a.prompts.read_text().splitlines()
    assert len(files)==len(prompts)==1000
    candidates=sorted(a.student.glob('checkpoint-*'),key=lambda x:int(x.name.split('-')[-1]))
    candidates=[c for c in candidates if (c/'unet/config.json').exists()]
    if (a.student/'unet/config.json').exists(): candidates.append(a.student)
    assert candidates, 'No full UNet checkpoints'
    device='cuda'
    pipe=StableDiffusionPipeline.from_pretrained(a.base_model,torch_dtype=torch.float32,safety_checker=None).to(device)
    scheduler=DDPMScheduler.from_pretrained(a.base_model,subfolder='scheduler')
    pipe.vae.eval(); pipe.text_encoder.eval(); pipe.unet.eval()
    # Cache fixed validation tensors on CPU. No student output or detector is used.
    cache=[]
    with torch.no_grad():
        for i,(f,prompt) in enumerate(zip(files,prompts)):
            with Image.open(f) as im:
                x=torch.from_numpy(np.array(im.convert('RGB'),dtype=np.float32)/127.5-1).permute(2,0,1).unsqueeze(0).to(device)
            g=torch.Generator(device=device).manual_seed(a.seed+i)
            latent=pipe.vae.encode(x).latent_dist.sample(generator=g)*pipe.vae.config.scaling_factor
            tokens=pipe.tokenizer(prompt,padding='max_length',max_length=pipe.tokenizer.model_max_length,truncation=True,return_tensors='pt')
            enc=pipe.text_encoder(tokens.input_ids.to(device))[0]
            for repeat in range(a.noise_repeats):
                g=torch.Generator(device=device).manual_seed(a.seed+100000+i*a.noise_repeats+repeat)
                noise=torch.randn(latent.shape,generator=g,device=device)
                t=torch.randint(0,scheduler.config.num_train_timesteps,(1,),generator=g,device=device)
                noisy=scheduler.add_noise(latent,noise,t)
                if scheduler.config.prediction_type=='epsilon': target=noise
                elif scheduler.config.prediction_type=='v_prediction': target=scheduler.get_velocity(latent,noise,t)
                else: raise ValueError(scheduler.config.prediction_type)
                cache.append((noisy.cpu(),t.cpu(),enc.cpu(),target.cpu()))
    results=[]
    with torch.no_grad():
        for c in candidates:
            pipe.unet = UNet2DConditionModel.from_pretrained(str(c), subfolder='unet', torch_dtype=torch.float32).to(device).eval()
            losses=[]
            for noisy,t,enc,target in cache:
                pred=pipe.unet(noisy.to(device),t.to(device),enc.to(device)).sample
                losses.append(torch.nn.functional.mse_loss(pred.float(),target.to(device).float()).item())
            value=float(np.mean(losses))
            results.append({'checkpoint':str(c.resolve()),'mse':value})
            print(results[-1],flush=True)

    best=min(results,key=lambda r:r['mse'])
    record={'criterion':'minimum unseen denoising MSE','seed':a.seed,'num_images':1000,
            'noise_repeats':a.noise_repeats,'base_model':a.base_model,'images':str(a.images),
            'results':results,'best':best}
    (a.student/'validation_selection.json').write_text(json.dumps(record,indent=2))
    (a.student/'BEST_UNET_PATH.txt').write_text(best['checkpoint']+'\n')
if __name__=='__main__': main()
