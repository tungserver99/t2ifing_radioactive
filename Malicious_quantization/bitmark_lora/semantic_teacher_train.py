"""Teacher-only coherent semantic displacement; students never import this module."""
import copy
import gc
import json
import math
from pathlib import Path
import torch
from common import freeze, image_name, read_json, sha256, write_json, verify_vendor
from semantic_carrier import SemanticCarrier, check_finite


def fingerprint_ramp(step):
    return min(1., max(0., (step-250)/500))


def predicted_x0(scheduler, noisy, prediction, timesteps):
    """Exact pre-clipping DDPM prediction conversion for all supported prediction types.

    Matches DDPMScheduler.step's pred_original_sample, before its optional sample
    clipping/thresholding. Those inference policies are deliberately not applied
    to semantic training latents; no silent clipping of unstable predictions.
    """
    alpha=scheduler.alphas_cumprod.to(noisy.device,dtype=torch.float32)[timesteps]
    alpha=alpha.reshape((-1,)+(1,)*(noisy.ndim-1))
    beta=1-alpha; z=noisy.float(); p=prediction.float()
    kind=scheduler.config.prediction_type
    if kind=='epsilon': result=(z-beta.sqrt()*p)/alpha.sqrt()
    elif kind=='v_prediction': result=alpha.sqrt()*z-beta.sqrt()*p
    elif kind=='sample': result=p
    else: raise ValueError(f'Unsupported scheduler prediction_type: {kind}')
    check_finite(result)
    return result


def semantic_losses(clean_features, fp_features, carrier, alpha):
    delta=fp_features.float()-clean_features.float()
    shift=(delta @ carrier.basis)/(carrier.sigmas+1e-8)
    code=(shift-alpha*carrier.signs).square().sum(-1).mean()
    orthogonal=delta-(delta @ carrier.basis) @ carrier.basis.T
    perp=orthogonal.square().mean()
    check_finite(code,perp)
    return code,perp


def adapter_targets(unet):
    # Reuse the audited vendored train_text_to_image_lora.py attention_conv rule,
    # including grouped-convolution exclusion. No second LoRA implementation.
    attention_targets=['to_k','to_q','to_v','to_out.0']
    targets=[name for name,layer in unet.named_modules()
             if any(name.endswith('.'+target) for target in attention_targets)]
    targets += [name for name,layer in unet.named_modules()
                if isinstance(layer,torch.nn.Conv2d) and layer.groups==1]
    if not targets: raise RuntimeError('No adapter targets found')
    return sorted(set(targets))


def save_adapter(unet, folder, lora_config):
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file
    folder.mkdir(parents=True,exist_ok=True)
    state={k:v.detach().cpu().contiguous().float() for k,v in get_peft_model_state_dict(unet).items()}
    check_finite(*state.values())
    temporary=folder/'adapter_model.partial'
    save_file(state,str(temporary)); temporary.replace(folder/'adapter_model.safetensors')
    lora_config.save_pretrained(str(folder))


def train_one(run, alpha, folder):
    from peft import LoraConfig
    from diffusers import DDPMScheduler
    from semantic_teacher import extractor, pixels, numerical_policy
    verify_vendor(); numerical_policy()
    c=read_json(run/'config.json'); assets=read_json(run/'assets.json')
    protocol={'config_sha256':sha256(run/'config.json'),'assets_sha256':sha256(run/'assets.json'),
              'carrier_sha256':sha256(run/'carrier/complete.json'),'alpha':alpha,
              'restart_policy':'partial candidate restarts from identical clean teacher and seed',
              'training_images_sha256':sha256(run/'images/teacher_clean_train/image_hashes.json')}
    folder.mkdir(parents=True,exist_ok=True); freeze(folder/'protocol.json',protocol)
    if (folder/'trained.json').exists():
        old=read_json(folder/'trained.json')
        for name,digest in old['files'].items():
            if sha256(folder/name)!=digest: raise RuntimeError('Teacher adapter changed')
        return
    torch.manual_seed(c['seed']); torch.cuda.manual_seed_all(c['seed'])
    ext=extractor(run)
    unet=copy.deepcopy(ext.unet).train()
    config=LoraConfig(r=c['teacher_rank'],lora_alpha=c['teacher_rank'],init_lora_weights='gaussian',
                      target_modules=adapter_targets(unet))
    unet.add_adapter(config)
    unet.enable_gradient_checkpointing()
    params=[p for p in unet.parameters() if p.requires_grad]
    if not params: raise ValueError('Teacher has no trainable adapter parameters')
    optimizer=torch.optim.AdamW(params,lr=c['teacher_lr'])
    dtype={'fp16':torch.float16,'bf16':torch.bfloat16,'no':torch.float32}[c['precision']]
    scaler=torch.amp.GradScaler('cuda',enabled=c['precision']=='fp16')
    scheduler=DDPMScheduler.from_pretrained(assets['teacher_clean'],subfolder='scheduler',local_files_only=True)
    if not 0<=c['semantic_t_max']<scheduler.config.num_train_timesteps:
        raise ValueError('semantic_t_max outside scheduler range')
    carrier=SemanticCarrier.load(run/'carrier','cuda')
    rows=read_json(run/'prompts.json')['train']; images=run/'images/teacher_clean_train'
    expected=read_json(images/'image_hashes.json')
    if len(expected)!=len(rows) or any(sha256(images/r['file_name'])!=r['sha256'] for r in expected):
        raise RuntimeError('Teacher training image hashes changed')
    rng=torch.Generator('cpu').manual_seed(c['seed'])
    # Persist image order in the deterministic seed protocol; restart reproduces it.
    order=[]; offset=0
    with (folder/'losses.jsonl').open('w') as log:
        for step in range(c['teacher_train_steps']):
            if offset+c['teacher_batch']>len(order):
                order=torch.randperm(len(rows),generator=rng).tolist(); offset=0
            ids=order[offset:offset+c['teacher_batch']]; offset+=len(ids)
            x=pixels([images/image_name(i) for i in ids]); prompts=[rows[i]['text'] for i in ids]
            with torch.no_grad():
                z=ext.image_latent(x); text=ext.text(prompts)
                noise=torch.randn(z.shape,generator=rng).to('cuda')
                t=torch.randint(0,scheduler.config.num_train_timesteps,(len(ids),),generator=rng).to('cuda')
                noisy=scheduler.add_noise(z,noise,t)
                clean=ext.unet(noisy,t,encoder_hidden_states=text).sample
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=dtype,enabled=c['precision']!='no'):
                predicted=unet(noisy,t,encoder_hidden_states=text).sample
            check_finite(clean,predicted)
            preserve=(predicted.float()-clean.float()).square().mean()
            code=perp=cycle=preserve.new_zeros(())
            mask=t<=c['semantic_t_max']; ramp=fingerprint_ramp(step)
            if mask.any() and ramp>0:
                with torch.no_grad():
                    zc=predicted_x0(scheduler,noisy[mask],clean[mask],t[mask])
                    fc=ext.encode_latent(zc,text_hidden=text[mask])
                zp=predicted_x0(scheduler,noisy[mask],predicted[mask],t[mask])
                fp=ext.encode_latent(zp,text_hidden=text[mask])
                code,perp=semantic_losses(fc,fp,carrier,alpha)
                if step%c['cycle_every']==0 and c['lambda_cycle']>0:
                    with torch.no_grad(): fcc=ext.encode_latent(ext.cycle(zc),text_hidden=text[mask])
                    fpc=ext.encode_latent(ext.cycle(zp),text_hidden=text[mask])
                    cycle,_=semantic_losses(fcc,fpc,carrier,alpha)
            loss=c['lambda_preserve']*preserve+ramp*(c['lambda_code']*code+c['lambda_perp']*perp+c['lambda_cycle']*cycle)
            check_finite(loss)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            check_finite(norm)
            scaler.step(optimizer); scaler.update()
            raw={'step':step+1,'loss':loss.item(),'preserve':preserve.item(),'code':code.item(),
                 'perp':perp.item(),'cycle':cycle.item(),'ramp':ramp,
                 'semantic_fraction':mask.float().mean().item(),'grad_norm':float(norm)}
            log.write(json.dumps(raw,allow_nan=False)+'\n'); log.flush()
            if (step+1)%25==0: print(f'Teacher alpha={alpha}: {raw}',flush=True)
            if (step+1)%c['teacher_checkpoint_every']==0:
                save_adapter(unet,folder/f'checkpoint-{step+1}',config)
    save_adapter(unet,folder,config)
    freeze(folder/'trained.json',{'files':{name:sha256(folder/name) for name in
        ('adapter_model.safetensors','adapter_config.json','losses.jsonl')},'steps':c['teacher_train_steps']})
    del unet,ext,optimizer; gc.collect(); torch.cuda.empty_cache()


def merge_teacher(run, folder):
    from diffusers import StableDiffusionPipeline
    from peft.tuners.tuners_utils import BaseTunerLayer
    from semantic_assets import snapshot_identity
    from student import load_adapter
    target=folder/'merged_pipeline'; result=folder/'merge_test.json'
    if result.exists():
        old=read_json(result)
        if snapshot_identity(target)!=old['model_hashes']: raise RuntimeError('Merged model changed')
        if old['adapter_sha256']!=sha256(folder/'adapter_model.safetensors'): raise RuntimeError('Adapter changed after merge')
        return target
    assets=read_json(run/'assets.json')
    pipe=StableDiffusionPipeline.from_pretrained(assets['teacher_clean'],torch_dtype=torch.float32,
                                                safety_checker=None,local_files_only=True).to('cuda')
    load_adapter(pipe.unet,folder,pipe); pipe.unet.eval()
    g=torch.Generator('cuda').manual_seed(12345)
    latent=torch.randn((1,pipe.unet.config.in_channels,64,64),generator=g,device='cuda')
    tokens=pipe.tokenizer(['a photograph of a tree'],padding='max_length',truncation=True,
                          max_length=pipe.tokenizer.model_max_length,return_tensors='pt').input_ids.cuda()
    with torch.no_grad():
        text=pipe.text_encoder(tokens)[0]
        expected=pipe.unet(latent,torch.tensor(123,device='cuda'),text).sample.cpu()
    pipe.fuse_lora(safe_fusing=True); pipe.unload_lora_weights()
    if any(isinstance(m,BaseTunerLayer) for m in pipe.unet.modules()):
        raise RuntimeError('PEFT adapter structures survived native merge')
    check_finite(*[p for p in pipe.unet.parameters()])
    pipe.save_pretrained(target,safe_serialization=True)
    del pipe; gc.collect(); torch.cuda.empty_cache()
    reloaded=StableDiffusionPipeline.from_pretrained(target,torch_dtype=torch.float32,
                         safety_checker=None,local_files_only=True).to('cuda')
    reloaded.unet.eval()
    with torch.no_grad(): actual=reloaded.unet(latent,torch.tensor(123,device='cuda'),text).sample.cpu()
    check_finite(actual,expected)
    torch.testing.assert_close(actual,expected,atol=1e-4,rtol=1e-4)
    freeze(result,{'passed':True,'atol':1e-4,'rtol':1e-4,'dtype':'float32',
        'max_absolute_error':(actual-expected).abs().max().item(),'model_hashes':snapshot_identity(target),
        'adapter_sha256':sha256(folder/'adapter_model.safetensors')})
    del reloaded; gc.collect(); torch.cuda.empty_cache()
    return target


def candidate_tag(alpha):
    """Round-trip float spelling prevents distinct strengths sharing artifacts."""
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('Candidate strength must be positive and finite')
    return f'shift_{alpha!r}'


def choose_candidate(candidates):
    for row in candidates:
        candidate_tag(row['alpha'])
        metrics = row['metrics']
        if (not math.isfinite(metrics['detection_rate']) or not 0 <= metrics['detection_rate'] <= 1
                or not math.isfinite(metrics['mean_score'])):
            raise ValueError('Nonfinite or invalid teacher candidate metrics')
        if 'bit_accuracy' in metrics and (not math.isfinite(metrics['bit_accuracy'])
                                         or not 0 <= metrics['bit_accuracy'] <= 1):
            raise ValueError('Nonfinite or invalid teacher bit accuracy')
        for name in ('clean_clip', 'fp_clip', 'clip_drop', 'paired_lpips'):
            if name in row['quality'] and not math.isfinite(row['quality'][name]):
                raise ValueError('Nonfinite teacher candidate quality')
    valid=[r for r in candidates if r['quality']['passes'] is True]
    if not valid: raise ValueError('No teacher candidate passed the quality gate; student training stopped')
    return max(valid,key=lambda r:(r['metrics']['detection_rate'],r['metrics']['mean_score'],-r['alpha']))


def train_candidates(run):
    from semantic_teacher import generate_set, score
    from semantic_report import detection_metrics, teacher_quality
    from semantic_assets import snapshot_identity
    c=read_json(run/'config.json'); root=run/'teacher_fp'; root.mkdir(exist_ok=True)
    selected_path=root/'selected.json'
    if selected_path.exists():
        selected=read_json(selected_path)
        if snapshot_identity(run/selected['model'])!=selected['model_hashes']: raise RuntimeError('Selected teacher changed')
        if selected['config_sha256']!=sha256(run/'config.json'): raise RuntimeError('Teacher selection config changed')
        chosen={k:selected[k] for k in ('alpha','model','metrics','quality')} if selected['downstream_ready'] else None
        if chosen is not None:
            freeze(run/'result/teacher_quality.json',{'selected':chosen,'candidates':selected['candidates']})
        if not selected['downstream_ready']:
            raise ValueError('Frozen teacher failed the validation detection gate; resume cannot bypass it')
        return
    threshold=read_json(run/'result/threshold_teacher.json')['threshold']
    candidates=[]
    for alpha in c.get('semantic_shift_sweep') or [c['semantic_shift']]:
        tag=candidate_tag(alpha)
        folder=root/'candidates'/tag
        train_one(run,alpha,folder); model=merge_teacher(run,folder)
        label=f'candidate_{tag}_validation'
        generate_set(run,model,'validation',label)
        # Scoring/quality use validation only; final test has not been generated.
        score(run,[label])
        metrics=detection_metrics(run,label,threshold,c['validation_n'])
        quality=teacher_quality(run,label)
        record={'alpha':alpha,'model':str(model.relative_to(run)),'metrics':metrics,'quality':quality}
        freeze(folder/'validation.json',record); candidates.append(record)
    chosen=choose_candidate(candidates)
    # Freeze even a weak teacher result, but prohibit expensive downstream work.
    selection={**chosen,'candidates':candidates,'selection_rule':'quality gate then validation TPR, mean score, lower alpha',
               'config_sha256':sha256(run/'config.json'),'model_hashes':snapshot_identity(run/chosen['model']),
               'teacher_threshold_sha256':sha256(run/'result/threshold_teacher.json'),
               'test_used':False,'downstream_ready':chosen['metrics']['detection_rate']>=c['teacher_min_tpr']}
    freeze(selected_path,selection)
    freeze(run/'result/teacher_quality.json',{'selected':chosen,'candidates':candidates})
    if not selection['downstream_ready']:
        raise ValueError('Teacher validation detection below --teacher-min-tpr; inspect teacher objective before students')
