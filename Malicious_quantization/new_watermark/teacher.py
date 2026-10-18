"""Train a semantic watermark into UNet LoRA, then export ordinary merged weights."""
import json
from pathlib import Path
import paths
from paths import freeze, read_json, write_json, sha256, image_name
from new_watermark.data import load_base, verify_images
from new_watermark.method import image_features, predict_x0, watermark_objective


def save_adapter(unet, folder):
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file
    folder.mkdir(parents=True, exist_ok=True)
    state={k:v.detach().float().cpu().contiguous() for k,v in get_peft_model_state_dict(unet).items()}
    save_file(state,str(folder/'adapter_model.safetensors'))
    unet.peft_config['default'].save_pretrained(folder)


def encode_training(run):
    import numpy as np
    import torch
    from PIL import Image
    from safetensors.torch import save_file
    c=read_json(run/'config.json'); prompts=read_json(run/'prompts.json')
    rows=[[split,i,r] for split in ('teacher_train','meta_probe') for i,r in enumerate(prompts[split])]
    folder=run/'teacher_cache'; folder.mkdir(exist_ok=True)
    entries={split:verify_images(run/'images'/('clean_'+split),len(prompts[split])) for split in ('teacher_train','meta_probe')}
    freeze(folder/'protocol.json',{'images':entries,'prompts':rows,'latent':'FP32 VAE posterior mode',
                                  'base':read_json(run/'assets.json')['base_revision']})
    if (folder/'complete.json').exists():
        verify_cache(run); return
    pipe=load_base(run,dtype=torch.float32)
    pipe.unet.to('cpu')
    files={}
    for index,(split,i,r) in enumerate(rows):
        with Image.open(run/'images'/('clean_'+split)/image_name(i)) as im:
            x=torch.from_numpy(np.array(im.convert('RGB'),dtype=np.float32)/127.5-1).permute(2,0,1)[None].cuda()
        tokens=pipe.tokenizer(['',r['text']],padding='max_length',truncation=True,
            max_length=pipe.tokenizer.model_max_length,return_tensors='pt').input_ids.cuda()
        with torch.no_grad():
            z=pipe.vae.encode(x).latent_dist.mode()*pipe.vae.config.scaling_factor
            text=pipe.text_encoder(tokens)[0]
        if not torch.isfinite(z).all() or not torch.isfinite(text).all(): raise RuntimeError('Nonfinite cache')
        p=folder/f'{split}_{i:06d}.safetensors'
        save_file({'latent':z.cpu().contiguous(),'text':text.cpu().contiguous()},str(p))
        files[p.name]=sha256(p)
        if (index+1)%25==0 or index+1==len(rows): print(f'Teacher cache: {index+1}/{len(rows)}',flush=True)
    freeze(folder/'complete.json',{'files':files,'protocol_sha256':sha256(folder/'protocol.json')})


def verify_cache(run):
    folder=run/'teacher_cache'; manifest=read_json(folder/'complete.json')
    if sha256(folder/'protocol.json')!=manifest['protocol_sha256']: raise RuntimeError('Changed teacher cache protocol')
    for name,digest in manifest['files'].items():
        if sha256(folder/name)!=digest: raise RuntimeError(f'Changed teacher cache: {name}')
    return manifest


def train_teacher(run):
    import torch
    from diffusers import DDPMScheduler
    from peft import LoraConfig
    from safetensors.torch import load_file
    from transformers import CLIPModel
    from student import checkpoint_hashes
    c=read_json(run/'config.json'); assets=read_json(run/'assets.json')
    out=run/'teacher'; out.mkdir(exist_ok=True)
    cache=verify_cache(run)
    protocol={'config_sha256':sha256(run/'config.json'),'detector_sha256':sha256(run/'result/detector.json'),
              'cache_sha256':sha256(run/'teacher_cache/complete.json'),
              'objective':'native teacher hinge + preservation + differentiable clean-adjusted LoRA transfer',
              'meta_proxy':'VAE mode + short AdamW unroll + disjoint predicted-x0; actual full sampling evaluated in pilots/test',
              'test_or_pilot_images_used_in_gradient':False,'precision':'FP32; no loss scaling',
              'checkpoint_steps':c['teacher_checkpoints']}
    freeze(out/'protocol.json',protocol)
    if (out/'complete.json').exists():
        for step,hashes in read_json(out/'complete.json')['snapshots'].items():
            if checkpoint_hashes(out/f'snapshot-{step}')!=hashes: raise RuntimeError('Changed teacher weights')
        print('Reusing verified trained teacher',flush=True); return
    print('Teacher training starts from base; an incomplete teacher stage restarts deterministically.',flush=True)
    torch.manual_seed(c['seed']); torch.cuda.manual_seed_all(c['seed'])
    pipe=load_base(run,dtype=torch.float32)
    pipe.text_encoder.to('cpu')
    unet=pipe.unet; unet.requires_grad_(False)
    targets=[name for name,layer in unet.named_modules() if any(name.endswith('.'+s) for s in ('to_q','to_k','to_v','to_out.0'))
             or isinstance(layer,torch.nn.Conv2d) and layer.groups==1]
    unet.add_adapter(LoraConfig(r=c['teacher_rank'],lora_alpha=c['teacher_rank'],init_lora_weights='gaussian',target_modules=targets))
    unet.enable_gradient_checkpointing(); unet.train()
    pipe.vae.requires_grad_(False).train(); pipe.vae.enable_gradient_checkpointing()
    clip=CLIPModel.from_pretrained(assets['clip'],local_files_only=True).float().cuda().requires_grad_(False)
    clip.gradient_checkpointing_enable(); clip.train()
    params=[p for p in unet.parameters() if p.requires_grad]
    write_json(out/'adapter_audit.json',{'rank':c['teacher_rank'],'targets':targets,
        'trainable_parameters':sum(p.numel() for p in params),'scope':'attention and ungrouped convolutions',
        'export':'merged ordinary UNet weights; no key required at inference'})
    optimizer=torch.optim.AdamW(params,lr=c['teacher_lr'],weight_decay=.01)
    scheduler=DDPMScheduler.from_pretrained(assets['student'],subfolder='scheduler',local_files_only=True)
    detector=read_json(run/'result/detector.json')
    # Margin is relative to the threshold frozen on held-out clean calibration.
    target=detector['thresholds'][0]+c['score_margin']
    files=sorted(n for n in cache['files'] if n.startswith('teacher_train_'))
    probes=sorted(n for n in cache['files'] if n.startswith('meta_probe_'))
    surrogate=None
    rng=torch.Generator('cpu').manual_seed(c['seed']+219)
    applied=skipped=0; snapshots={}
    with (out/'training.jsonl').open('w') as log:
        for step in range(1,max(c['teacher_checkpoints'])+1):
            optimizer.zero_grad(set_to_none=True)
            idx=int(torch.randint(len(files),(1,),generator=rng))
            data=load_file(str(run/'teacher_cache'/files[idx]))
            z=data['latent'].cuda(); text=data['text'].cuda()
            t=torch.randint(c['timestep_min'],c['timestep_max']+1,(1,),generator=rng).cuda()
            noise=torch.randn(z.shape,generator=rng).cuda()
            noisy=scheduler.add_noise(z,noise,t)
            alpha=scheduler.alphas_cumprod.to('cuda')[t].view(-1,1,1,1)
            inp=noisy.repeat(2,1,1,1)
            def prediction():
                u,v=unet(inp,t,text).sample.chunk(2)
                return u+7.5*(v-u)
            unet.disable_adapters()
            try:
                with torch.no_grad():
                    ref=prediction()
                    ref_x0=predict_x0(noisy,ref,alpha,scheduler.config.prediction_type)
                    ref_pixels=(pipe.vae.decode(ref_x0/pipe.vae.config.scaling_factor).sample/2+.5).clamp(0,1)
                    ref_features=image_features(clip,ref_pixels)
            finally:
                unet.enable_adapters()
            pred=prediction()
            x0=predict_x0(noisy,pred,alpha,scheduler.config.prediction_type)
            pixels=(pipe.vae.decode(x0/pipe.vae.config.scaling_factor).sample/2+.5).clamp(0,1)
            feats=image_features(clip,pixels); aug=image_features(clip,pixels,augmented=True)
            loss,terms=watermark_objective(pred,ref,pixels,ref_pixels,feats,ref_features,aug,detector,target,c['loss_weights'])
            meta_record={}
            if c['meta_weight'] > 0 and step > c['meta_warmup'] and step % c['meta_every'] == 0:
                from new_watermark.meta import make_surrogate, transfer_loss
                if surrogate is None:
                    surrogate=make_surrogate(assets['student'],c['meta_rank'],c['student_scope'])
                probe=load_file(str(run/'teacher_cache'/probes[int(torch.randint(len(probes),(1,),generator=rng))]))
                transfer,meta_record=transfer_loss(surrogate,pipe.vae,clip,detector,pixels,ref_pixels,
                    text[1:],probe['latent'].cuda(),probe['text'].cuda(),scheduler,
                    seed=c['seed']*100000+step,steps=c['meta_steps'],lr=c['learning_rate'],
                    margin=c['transfer_margin'],target=target)
                loss=loss+c['meta_weight']*transfer
                terms['transfer']=transfer
            finite=bool(torch.isfinite(loss).item())
            grad_norm=None
            if finite:
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(params,1.)
                finite=bool(torch.isfinite(norm).item()); grad_norm=float(norm) if finite else None
            if finite:
                optimizer.step(); applied+=1
                if not all(torch.isfinite(p).all() for p in params):
                    raise RuntimeError('Optimizer produced nonfinite weights; refusing to export corrupt model')
            else:
                skipped+=1
            record={'meta':meta_record,'step':step,'applied':applied,'skipped':skipped,'grad_norm':grad_norm,
                    'loss':float(loss.detach()) if torch.isfinite(loss) else None,
                    'terms':{k:float(v.detach()) if torch.isfinite(v) else None for k,v in terms.items()}}
            log.write(json.dumps(record,allow_nan=False)+'\n'); log.flush()
            if step%10==0 or step==1: print(f'Teacher {record}',flush=True)
            if step in c['teacher_checkpoints']:
                folder=out/f'snapshot-{step}'; save_adapter(unet,folder)
                write_json(folder/'optimization_audit.json',{'attempted':step,'applied':applied,'skipped':skipped})
                snapshots[str(step)]=checkpoint_hashes(folder)
    freeze(out/'complete.json',{'snapshots':snapshots,'attempted':step,'applied':applied,'skipped':skipped,
                              'warning':None if skipped==0 else 'Nonfinite updates skipped; inspect training.jsonl'})


def merge_export(pipe, adapter, destination, probe=None):
    """Verify adapter vs fused prediction; save a reloadable standard pipeline."""
    import torch
    from student import load_adapter
    load_adapter(pipe.unet,adapter,pipe)
    pipe.unet.eval()
    if probe is None:
        size=pipe.unet.config.sample_size
        gen=torch.Generator('cpu').manual_seed(119)
        x=torch.randn((1,pipe.unet.config.in_channels,size,size),generator=gen).to(pipe.unet.device,pipe.unet.dtype)
        h=torch.randn((1,77,pipe.unet.config.cross_attention_dim),generator=gen).to(x)
        probe=(x,10,h)
    with torch.no_grad(): before=pipe.unet(*probe).sample
    pipe.fuse_lora(safe_fusing=True)
    pipe.unload_lora_weights()
    if any('lora_' in n for n,_ in pipe.unet.named_parameters()): raise RuntimeError('Export still contains adapters')
    with torch.no_grad(): after=pipe.unet(*probe).sample
    if not torch.isfinite(after).all() or not torch.allclose(before,after,atol=2e-4,rtol=2e-4):
        raise RuntimeError('Merged UNet differs from trained adapter prediction')
    pipe.save_pretrained(destination,safe_serialization=True)
    return {'max_abs_prediction_error':float((before-after).abs().max()),'no_adapter_modules':True,
            'ordinary_pipeline':True,'special_inference_inputs':False}


def export_teacher(run,step):
    import torch
    from student import checkpoint_hashes
    adapter=run/'teacher'/f'snapshot-{step}'
    recorded=read_json(run/'teacher/complete.json')['snapshots'][str(step)]
    if checkpoint_hashes(adapter)!=recorded: raise RuntimeError('Teacher snapshot changed')
    target=run/'models'/f'teacher_{step}'
    inventory=target/'export.json'
    if inventory.exists():
        verify_export(target); return
    pipe=load_base(run,dtype=torch.float32)
    audit=merge_export(pipe,adapter,target)
    # A genuine clean reload is verified, not merely the in-memory fused model.
    from diffusers import UNet2DConditionModel
    reload=UNet2DConditionModel.from_pretrained(target/'unet',local_files_only=True)
    original=pipe.unet.state_dict(); saved=reload.state_dict()
    if original.keys()!=saved.keys() or not all(torch.equal(v.cpu(),saved[k]) for k,v in original.items()):
        raise RuntimeError('Export round trip changed weights')
    freeze(inventory,{'adapter_hashes':recorded,'audit':audit,
                     'files':{str(p.relative_to(target)):sha256(p) for p in sorted(target.rglob('*')) if p.is_file()}})


def verify_export(target):
    record=read_json(target/'export.json')
    for name,digest in record['files'].items():
        if sha256(target/name)!=digest: raise RuntimeError(f'Changed exported teacher: {target/name}')
    return sha256(target/'export.json')
