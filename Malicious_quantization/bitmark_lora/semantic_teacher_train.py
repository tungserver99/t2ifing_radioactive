"""Teacher-only coherent semantic displacement; students never import this module."""
import copy
import gc
import json
import math
from pathlib import Path
import torch
from common import freeze, image_name, read_json, sha256, write_json, verify_vendor
from semantic_carrier import SemanticCarrier, check_finite
from semantic_threshold import (is_threshold,candidate_parameter,cfg_prediction,calibrated_shift_loss,
                                frozen_calibration,FP_WEIGHT,CYCLE_WEIGHT)


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


def teacher_precision(config):
    # Missing key preserves historical run interpretation; new CLI records bf16.
    return config.get('teacher_precision', config['precision'])


def watch_gradient(tensor, label):
    if tensor.requires_grad:
        def check(gradient):
            if not torch.isfinite(gradient).all():
                raise RuntimeError(f'Nonfinite gradient at {label}')
            return gradient
        tensor.register_hook(check)
    return tensor


def checked_teacher_step(named_params, optimizer, scaler, compute, failure_path, context):
    """Never apply invalid gradients; use FP64 reduction to avoid norm overflow.

    This intentionally fails on nonfinite values (the specification's guard),
    rather than silently dropping an optimizer update from the fixed budget.
    """
    named_params=list(named_params)
    measured={}
    optimizer.zero_grad(set_to_none=True)
    try:
        measured=compute(lambda loss: scaler.scale(loss).backward() if scaler is not None else loss.backward())
        if scaler is not None: scaler.unscale_(optimizer)
        gradients=[p.grad for _,p in named_params if p.grad is not None]
        bad=[name for name,p in named_params if p.grad is not None and not torch.isfinite(p.grad).all()]
        if bad: raise RuntimeError(f'Nonfinite parameter gradient: {bad[:8]}')
        if not gradients: raise RuntimeError('Teacher produced no parameter gradients')
        # A finite FP32 gradient can still overflow an FP32 sum of squares.
        norm=torch.stack([g.detach().double().norm(2) for g in gradients]).norm(2)
        if not torch.isfinite(norm): raise RuntimeError('Nonfinite gradient norm (FP64 reduction)')
        coefficient=(1. / (norm+1e-6)).clamp(max=1.)
        with torch.no_grad():
            for gradient in gradients: gradient.mul_(coefficient.to(device=gradient.device,dtype=gradient.dtype))
        if scaler is None: optimizer.step()
        else: scaler.step(optimizer); scaler.update()
        if any(not torch.isfinite(p).all() for _,p in named_params):
            raise RuntimeError('Nonfinite teacher parameter after optimizer update')
        return {**measured,'grad_norm':norm.item()}
    except Exception as exc:
        if failure_path is not None:
            record={**context,'error':str(exc),'losses':measured,
                    'loss_scale':scaler.get_scale() if scaler is not None else None,
                    'nonfinite_parameters':[name for name,p in named_params
                        if p.grad is not None and not torch.isfinite(p.grad).all()]}
            if named_params and named_params[0][1].is_cuda:
                record.update(peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                              reserved_gib=torch.cuda.memory_reserved()/1024**3)
            write_json(failure_path,record)
        raise


def stress_teacher(unet,ext,carrier,scheduler,z,text,alpha,config,folder,steps=12):
    """Short real objective/optimizer probe; restore weights and RNG even on failure."""
    if steps < 4: raise ValueError('Teacher check needs at least four boundary cases')
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    named=[(n,p) for n,p in unet.named_parameters() if p.requires_grad]
    if not named: raise RuntimeError('Teacher check has no trainable adapter parameters')
    saved=[p.detach().cpu().clone() for _,p in named]
    optimizer=torch.optim.AdamW([p for _,p in named],lr=config['teacher_lr'])
    precision=teacher_precision(config)
    scaler=torch.amp.GradScaler('cuda') if precision=='fp16' and z.is_cuda else None
    devices=[z.device.index if z.device.index is not None else torch.cuda.current_device()] if z.is_cuda else []
    maximum=min(config['semantic_t_max'],scheduler.config.num_train_timesteps-1)
    probes=[0,maximum//2,maximum,min(maximum+1,scheduler.config.num_train_timesteps-1)]
    checks=[]
    try:
        with torch.random.fork_rng(devices=devices):
            rng=torch.Generator('cpu').manual_seed(config['seed']+7919)
            for index in range(steps):
                timestep=probes[index] if index<len(probes) else int(torch.randint(0,maximum+1,(1,),generator=rng).item())
                t=torch.full((len(z),),timestep,device=z.device,dtype=torch.long)
                noise=torch.randn(z.shape,generator=rng).to(z.device)
                noisy=scheduler.add_noise(z,noise,t)
                # Full ramp and cycle enabled on every eligible probe, not warmup.
                step=((750+config['cycle_every']-1)//config['cycle_every']+index)*config['cycle_every']
                if z.is_cuda: torch.cuda.reset_peak_memory_stats(z.device)
                context={'phase':'teacher_check','probe':index+1,'timestep':timestep,
                         'teacher_precision':precision,**{candidate_parameter(config):alpha}}
                values=checked_teacher_step(named,optimizer,scaler,
                    lambda backward:teacher_backward(unet,ext,carrier,scheduler,noisy,text,t,alpha,config,step,backward,context),
                    folder/'failure.json',context)
                row={**context,**values}
                if z.is_cuda: row['peak_allocated_gib']=torch.cuda.max_memory_allocated(z.device)/1024**3
                checks.append(row)
                print(f'Teacher numerical check {index+1}/{steps}: {row}',flush=True)
        result={'passed':True,'checks':checks,'weights_and_rng_restored':True,
                'note':'A short probe cannot guarantee future training stability.'}
        write_json(folder/'check.json',result)
        return result
    finally:
        with torch.no_grad():
            for (_,parameter),value in zip(named,saved):
                parameter.copy_(value); parameter.grad=None
        del optimizer,scaler
        gc.collect()
        if z.is_cuda: torch.cuda.empty_cache()


def _teacher_microbatch_backward(unet, ext, carrier, scheduler, noisy, text, t,
                                 alpha, config, step, total_batch, total_semantic, backward, diagnostic=None):
    # The local scope releases every microbatch graph before the next forward.
    with torch.no_grad():
        clean = ext.unet(noisy, t, encoder_hidden_states=text).sample
    dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'no': torch.float32}[teacher_precision(config)]
    with torch.autocast(noisy.device.type, dtype=dtype, enabled=teacher_precision(config) != 'no'):
        predicted = unet(noisy, t, encoder_hidden_states=text).sample
    check_finite(clean, predicted)
    watch_gradient(predicted, 'teacher prediction')
    preserve = (predicted.float() - clean.float()).square().mean()
    code = perp = cycle = preserve.new_zeros(())
    mask = t <= config['semantic_t_max']
    selected = int(mask.sum().item())
    ramp = fingerprint_ramp(step)
    if selected and ramp > 0:
        with torch.no_grad():
            zc = predicted_x0(scheduler, noisy[mask], clean[mask], t[mask])
            fc = ext.encode_latent(zc, text_hidden=text[mask])
        zp = watch_gradient(predicted_x0(scheduler, noisy[mask], predicted[mask], t[mask]), 'predicted_x0')
        fp = watch_gradient(ext.encode_latent(zp, text_hidden=text[mask]), 'semantic features')
        code, perp = semantic_losses(fc, fp, carrier, alpha)
        if step % config['cycle_every'] == 0 and config['lambda_cycle'] > 0:
            with torch.no_grad():
                fcc = ext.encode_latent(ext.cycle(zc), text_hidden=text[mask])
            cycle_latent = watch_gradient(ext.cycle(zp), 'VAE cycle output')
            fpc = watch_gradient(ext.encode_latent(cycle_latent, text_hidden=text[mask]), 'cycle features')
            cycle, _ = semantic_losses(fcc, fpc, carrier, alpha)
    batch_weight = len(noisy) / total_batch
    semantic_weight = selected / total_semantic if total_semantic else 0.
    loss = (config['lambda_preserve'] * preserve * batch_weight + ramp * semantic_weight *
            (config['lambda_code'] * code + config['lambda_perp'] * perp + config['lambda_cycle'] * cycle))
    check_finite(loss)
    values={'loss': loss.detach().item(), 'preserve': preserve.detach().item() * batch_weight,
            'code': code.detach().item() * semantic_weight, 'perp': perp.detach().item() * semantic_weight,
            'cycle': cycle.detach().item() * semantic_weight}
    if diagnostic is not None: diagnostic['microbatch_losses']=values
    backward(loss)
    return values


def _threshold_microbatch_backward(unet,ext,carrier,scheduler,noisy,text,empty,t,kappa,
                                    config,step,total_batch,total_semantic,backward,diagnostic):
    with torch.no_grad():
        clean_c=ext.unet(noisy,t,encoder_hidden_states=text).sample
        clean_u=ext.unet(noisy,t,encoder_hidden_states=empty).sample
    dtype={'fp16':torch.float16,'bf16':torch.bfloat16,'no':torch.float32}[teacher_precision(config)]
    with torch.autocast(noisy.device.type,dtype=dtype,enabled=teacher_precision(config)!='no'):
        fp_c=unet(noisy,t,encoder_hidden_states=text).sample
        fp_u=unet(noisy,t,encoder_hidden_states=empty).sample
    check_finite(clean_c,clean_u,fp_c,fp_u)
    watch_gradient(fp_c,'teacher conditional prediction');watch_gradient(fp_u,'teacher unconditional prediction')
    preserve=.5*(fp_c.float()-clean_c.float()).square().mean()+.5*(fp_u.float()-clean_u.float()).square().mean()
    ramp=1.;mask=t<=config['semantic_t_max'];selected=int(mask.sum().item())
    batch_weight=len(noisy)/total_batch;semantic_weight=selected/total_semantic if total_semantic else 0.
    sem=sem_cycle=preserve.new_zeros(())
    stats={name:0. for name in ('mean_r0','mean_rfp','mean_required_shift','proxy_tpr','shift_cap_fraction',
           'cycle_mean_r0','cycle_mean_rfp','cycle_mean_required_shift','cycle_proxy_tpr','cycle_shift_cap_fraction')}
    if selected and ramp>0:
        frozen=config['teacher_calibration'];tau=frozen['threshold'];sigma=frozen['sigma_clean']
        with torch.no_grad():
            z0=predicted_x0(scheduler,noisy[mask],cfg_prediction(clean_c,clean_u)[mask],t[mask])
            q0=carrier.coordinates(ext.encode_latent(z0,text_hidden=text[mask]))
            q0_cycle=carrier.coordinates(ext.encode_latent(ext.cycle(z0),text_hidden=text[mask]))
        zfp=watch_gradient(predicted_x0(scheduler,noisy[mask],cfg_prediction(fp_c,fp_u)[mask],t[mask]),'CFG predicted_x0')
        features=watch_gradient(ext.encode_latent(zfp,text_hidden=text[mask]),'CFG semantic features')
        qfp=carrier.coordinates(features)
        sem,raw=calibrated_shift_loss(q0,qfp,carrier.signs,tau,sigma,kappa)
        cycle_z=watch_gradient(ext.cycle(zfp),'CFG VAE cycle output')
        cycle_features=watch_gradient(ext.encode_latent(cycle_z,text_hidden=text[mask]),'CFG cycle features')
        qfp_cycle=carrier.coordinates(cycle_features)
        sem_cycle,cycled=calibrated_shift_loss(q0_cycle,qfp_cycle,carrier.signs,tau,sigma,kappa)
        stats.update(raw);stats.update({'cycle_'+key:value for key,value in cycled.items()})
    fp=sem+CYCLE_WEIGHT*sem_cycle
    loss=preserve*batch_weight+ramp*FP_WEIGHT*fp*semantic_weight
    check_finite(loss)
    values={'loss':loss.detach().item(),'preserve':preserve.detach().item()*batch_weight,
            'sem':sem.detach().item()*semantic_weight,'sem_cycle':sem_cycle.detach().item()*semantic_weight,
            'fp':fp.detach().item()*semantic_weight,**{key:value*semantic_weight for key,value in stats.items()}}
    if diagnostic is not None:diagnostic['microbatch_losses']=values
    backward(loss)
    return values


def threshold_teacher_backward(unet,ext,carrier,scheduler,noisy,text,timesteps,kappa,config,step,backward,diagnostic=None):
    batch=len(noisy);micro=config.get('teacher_micro_batch',1)
    if batch<1 or micro<1:raise ValueError('Teacher batch/microbatch must be positive')
    selected=int((timesteps<=config['semantic_t_max']).sum().item())
    empty=ext.text(['']*batch)
    result={}
    for start in range(0,batch,micro):
        end=min(start+micro,batch)
        if diagnostic is not None:diagnostic.update(microbatch_start=start,microbatch_timesteps=timesteps[start:end].cpu().tolist())
        values=_threshold_microbatch_backward(unet,ext,carrier,scheduler,noisy[start:end],text[start:end],empty[start:end],
              timesteps[start:end],kappa,config,step,batch,selected,backward,diagnostic)
        for key,value in values.items():result[key]=result.get(key,0.)+value
    if selected==0:
        for key in result:
            if key.startswith(('mean_','proxy_','shift_','cycle_')):result[key]=None
    return {**result,'ramp':1.,'semantic_fraction':selected/batch,
            'target_score':config['teacher_calibration']['threshold']+kappa*config['teacher_calibration']['sigma_clean']}


def teacher_backward(unet, ext, carrier, scheduler, noisy, text, timesteps, alpha, config, step, backward, diagnostic=None):
    """Accumulate the original full-batch objective with bounded forward batches.

    Preservation is normalized over all B images; semantic terms over all masked
    images in the effective batch, not separately over each microbatch. The caller
    performs exactly one unscale/clip/optimizer update after this function.
    """
    if is_threshold(config):
        return threshold_teacher_backward(unet,ext,carrier,scheduler,noisy,text,timesteps,alpha,config,step,backward,diagnostic)
    batch = len(noisy)
    micro = config.get('teacher_micro_batch', 1)
    if batch < 1 or micro < 1:
        raise ValueError('Teacher batch and microbatch must be positive')
    semantic_count = int((timesteps <= config['semantic_t_max']).sum().item())
    result = dict.fromkeys(('loss', 'preserve', 'code', 'perp', 'cycle'), 0.)
    for start in range(0, batch, micro):
        end = min(start + micro, batch)
        if diagnostic is not None:
            diagnostic.update(microbatch_start=start, microbatch_timesteps=timesteps[start:end].cpu().tolist())
        values = _teacher_microbatch_backward(unet, ext, carrier, scheduler, noisy[start:end], text[start:end],
                    timesteps[start:end], alpha, config, step, batch, semantic_count, backward, diagnostic)
        for name, value in values.items():
            result[name] += value
    return {**result, 'ramp': fingerprint_ramp(step), 'semantic_fraction': semantic_count / batch}


def latest_teacher_adapters(run):
    found=[]
    for candidate in sorted(list((Path(run)/'teacher_fp/candidates').glob('shift_*'))+
                            list((Path(run)/'teacher_fp/candidates').glob('kappa_*'))):
        if (candidate/'trained.json').exists():
            source=candidate
            for name,digest in read_json(candidate/'trained.json')['files'].items():
                if sha256(candidate/name)!=digest: raise RuntimeError(f'Teacher artifact changed: {name}')
        else:
            checkpoints=[p for p in candidate.glob('checkpoint-*')
                         if p.name[11:].isdigit() and (p/'adapter_model.safetensors').exists()
                         and (p/'adapter_config.json').exists()]
            if not checkpoints: continue
            source=max(checkpoints,key=lambda p:int(p.name[11:]))
        found.append((float(candidate.name.split('_',1)[1]),source))
    if not found: raise ValueError('No saved teacher adapter found; a fresh run performs its own early check')
    return found


def check_existing_run(run, precision='bf16', steps=32):
    """Diagnostic only: reuse old data/checkpoints without accepting changed resume hashes."""
    from semantic_sign import is_sign
    if is_sign(read_json(run/'config.json')):
        from semantic_sign_train import check_sign_existing
        return check_sign_existing(run,precision,steps)
    import datetime
    from diffusers import DDPMScheduler
    from semantic_teacher import extractor,pixels,numerical_policy
    from student import load_adapter
    verify_vendor(); numerical_policy()
    if not torch.cuda.is_available(): raise RuntimeError('Teacher model check requires a GPU')
    if precision=='bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('GPU does not support BF16; use --teacher-precision no for an FP32 diagnostic')
    c=read_json(run/'config.json')
    if c.get('watermark_method')!='semantic_fp': raise ValueError('Expected a semantic_fp run')
    c={**c,'teacher_precision':precision}
    if is_threshold(c):c={**c,'teacher_calibration':frozen_calibration(run)}
    targets=latest_teacher_adapters(run)
    assets=read_json(run/'assets.json'); rows=read_json(run/'prompts.json')['train']
    images=run/'images/teacher_clean_train'
    hashes=read_json(images/'image_hashes.json')
    if len(hashes)!=len(rows) or any(sha256(images/r['file_name'])!=r['sha256'] for r in hashes):
        raise RuntimeError('Teacher training image hashes changed')
    output=run/'diagnostics'/datetime.datetime.now(datetime.timezone.utc).strftime('teacher_check_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    print(f'Teacher diagnostic only; reports: {output}',flush=True)
    for alpha,source in targets:
        candidate=source if source.name.startswith(('shift_','kappa_')) else source.parent
        proof=read_json(candidate/'protocol.json')
        for key,path in (('config_sha256',run/'config.json'),('assets_sha256',run/'assets.json'),
                         ('carrier_sha256',run/'carrier/complete.json')):
            if proof[key]!=sha256(path): raise RuntimeError(f'Teacher protocol changed: {key}')
        ext=extractor(run); unet=copy.deepcopy(ext.unet)
        load_adapter(unet,source)
        # Serialized adapters may carry inference_mode; the probe must train LoRA only.
        for name,parameter in unet.named_parameters(): parameter.requires_grad_('lora_' in name)
        unet.train(); unet.enable_gradient_checkpointing()
        carrier=SemanticCarrier.load(run/'carrier','cuda')
        scheduler=DDPMScheduler.from_pretrained(assets['teacher_clean'],subfolder='scheduler',local_files_only=True)
        with torch.no_grad():
            ids=range(c['teacher_batch'])
            z=ext.image_latent(pixels([images/image_name(i) for i in ids]))
            text=ext.text([rows[i]['text'] for i in ids])
        destination=output/candidate.name
        write_json(destination/'source.json',{'run':str(run),'adapter':str(source),
            'adapter_sha256':sha256(source/'adapter_model.safetensors'),
            'adapter_config_sha256':sha256(source/'adapter_config.json'),
            'source_config_sha256':sha256(run/'config.json'),
            'effective_config':c,'steps':steps,'optimizer':'fresh AdamW; original optimizer was not saved',
            'current_code':{name:sha256(Path(__file__).parent/name) for name in
                ('semantic_teacher_train.py','semantic_carrier.py','semantic_teacher.py')},
            'diagnostic_only':True})
        stress_teacher(unet,ext,carrier,scheduler,z,text,alpha,c,destination,steps=steps)
        del unet,ext,carrier,z,text; gc.collect(); torch.cuda.empty_cache()
    print(f'Teacher diagnostic PASSED; reports: {output}. This is not an optimizer resume.',flush=True)


def train_one(run, alpha, folder):
    from peft import LoraConfig
    from diffusers import DDPMScheduler
    from semantic_checkpoint import (latest_full_checkpoint,restore_training_state,save_training_state,advance_sampler,align_loss_log)
    from semantic_teacher import extractor, pixels, numerical_policy
    verify_vendor(); numerical_policy()
    c=read_json(run/'config.json'); assets=read_json(run/'assets.json')
    if is_threshold(c):c={**c,'teacher_calibration':frozen_calibration(run)}
    protocol={'config_sha256':sha256(run/'config.json'),'assets_sha256':sha256(run/'assets.json'),
              'carrier_sha256':sha256(run/'carrier/complete.json'),candidate_parameter(c):alpha,
              'restart_policy':'restore full checkpoint; explicit legacy continuation uses fresh optimizer',
              'training_images_sha256':sha256(run/'images/teacher_clean_train/image_hashes.json'),
              'micro_batch':c.get('teacher_micro_batch',1), 'reference_cycle_checkpointing':True,
              'teacher_precision':teacher_precision(c), 'numerical_check_steps':12}
    if is_threshold(c):
        protocol['objective']=c['teacher_objective']
        protocol['calibration_target_sha256']=sha256(run/'teacher_fp/calibration_target.json')
    folder.mkdir(parents=True,exist_ok=True); freeze(folder/'protocol.json',protocol)
    if (folder/'trained.json').exists():
        old=read_json(folder/'trained.json')
        for name,digest in old['files'].items():
            if sha256(folder/name)!=digest: raise RuntimeError('Teacher adapter changed')
        return
    torch.manual_seed(c['seed']); torch.cuda.manual_seed_all(c['seed'])
    if teacher_precision(c)=='bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('Teacher BF16 requires a supported GPU; choose --teacher-precision no for FP32')
    ext=extractor(run)
    unet=copy.deepcopy(ext.unet).train()
    config=LoraConfig(r=c['teacher_rank'],lora_alpha=c['teacher_rank'],init_lora_weights='gaussian',
                      target_modules=adapter_targets(unet))
    full_checkpoint=latest_full_checkpoint(folder)
    initial=(c.get('continuation',{}).get('checkpoints',{}).get(candidate_tag(alpha,candidate_parameter(c)))
             if full_checkpoint is None else None)
    if initial:
        from student import load_adapter
        source=Path(initial['path'])
        if (sha256(source/'adapter_model.safetensors')!=initial['adapter_sha256'] or
                sha256(source/'adapter_config.json')!=initial['adapter_config_sha256']):
            raise RuntimeError('Continuation adapter changed')
        config=LoraConfig.from_pretrained(str(source)); load_adapter(unet,source)
        for name,parameter in unet.named_parameters():parameter.requires_grad_('lora_' in name)
    else:
        unet.add_adapter(config)
    unet.enable_gradient_checkpointing()
    params=[p for p in unet.parameters() if p.requires_grad]
    if not params: raise ValueError('Teacher has no trainable adapter parameters')
    optimizer=torch.optim.AdamW(params,lr=c['teacher_lr'])
    scaler=torch.amp.GradScaler('cuda') if teacher_precision(c)=='fp16' else None
    scheduler=DDPMScheduler.from_pretrained(assets['teacher_clean'],subfolder='scheduler',local_files_only=True)
    if not 0<=c['semantic_t_max']<scheduler.config.num_train_timesteps:
        raise ValueError('semantic_t_max outside scheduler range')
    carrier=SemanticCarrier.load(run/'carrier','cuda')
    rows=read_json(run/'prompts.json')['train']; images=run/'images/teacher_clean_train'
    expected=read_json(images/'image_hashes.json')
    if len(expected)!=len(rows) or any(sha256(images/r['file_name'])!=r['sha256'] for r in expected):
        raise RuntimeError('Teacher training image hashes changed')
    rng=torch.Generator('cpu').manual_seed(c['seed'])
    start_step=0;order=[];offset=0
    if full_checkpoint:
        state=restore_training_state(full_checkpoint,unet,optimizer,scaler,rng,protocol)
        start_step,order,offset=state['step'],state['order'],state['offset']
        align_loss_log(folder/'losses.jsonl',start_step,expected_hash=state['loss_prefix_sha256'])
        print(f'Resuming teacher exact state at completed step {start_step}',flush=True)
    elif initial:
        start_step=initial['step']
        if sha256(initial['loss_log'])!=initial['loss_log_sha256']:raise RuntimeError('Source teacher loss log changed')
        align_loss_log(folder/'losses.jsonl',start_step,initial['loss_log'])
        order,offset=advance_sampler(rng,len(rows),c['teacher_batch'],(c['teacher_batch'],4,64,64),
                                    scheduler.config.num_train_timesteps,start_step)
        print(f'Continuing teacher weights at step {start_step}; optimizer reset (legacy checkpoint)',flush=True)
    else:
        align_loss_log(folder/'losses.jsonl',0)
    if not 0<=start_step<=c['teacher_train_steps']:raise ValueError('Checkpoint step outside teacher budget')
    with torch.no_grad():
        probe_ids=range(c['teacher_batch'])
        probe_z=ext.image_latent(pixels([images/image_name(i) for i in probe_ids]))
        probe_text=ext.text([rows[i]['text'] for i in probe_ids])
    stress_teacher(unet,ext,carrier,scheduler,probe_z,probe_text,alpha,c,folder/'numerical_check')
    del probe_z,probe_text
    with (folder/'losses.jsonl').open('a') as log:
        for step in range(start_step,c['teacher_train_steps']):
            if offset+c['teacher_batch']>len(order):
                order=torch.randperm(len(rows),generator=rng).tolist(); offset=0
            ids=order[offset:offset+c['teacher_batch']]; offset+=len(ids)
            x=pixels([images/image_name(i) for i in ids]); prompts=[rows[i]['text'] for i in ids]
            with torch.no_grad():
                z=ext.image_latent(x); text=ext.text(prompts)
                noise=torch.randn(z.shape,generator=rng).to('cuda')
                t=torch.randint(0,scheduler.config.num_train_timesteps,(len(ids),),generator=rng).to('cuda')
                noisy=scheduler.add_noise(z,noise,t)
            torch.cuda.reset_peak_memory_stats()
            context={'phase':'train','step':step+1,**{candidate_parameter(c):alpha},'image_ids':ids,
                     'timesteps':t.cpu().tolist(),'teacher_precision':teacher_precision(c)}
            measured=checked_teacher_step([(n,p) for n,p in unet.named_parameters() if p.requires_grad],
                optimizer,scaler,
                lambda backward:teacher_backward(unet,ext,carrier,scheduler,noisy,text,t,alpha,c,step,backward,context),
                folder/'failure.json',context)
            raw={'step':step+1,**measured,'teacher_precision':teacher_precision(c),
                 'effective_batch':len(ids),'micro_batch':c.get('teacher_micro_batch',1),
                 'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3}
            log.write(json.dumps(raw,allow_nan=False)+'\n'); log.flush()
            if (step+1)%25==0: print(f'Teacher {candidate_parameter(c)}={alpha}: {raw}',flush=True)
            if (step+1)%c['teacher_checkpoint_every']==0 or step+1==c['teacher_train_steps']:
                checkpoint=folder/f'checkpoint-{step+1}'
                save_adapter(unet,checkpoint,config)
                save_training_state(checkpoint,unet,optimizer,scaler,rng,order,offset,step+1,protocol)
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


def candidate_tag(alpha,parameter="alpha"):
    """Round-trip float spelling prevents distinct strengths sharing artifacts."""
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('Candidate strength must be positive and finite')
    return f'{"kappa" if parameter=="kappa" else "shift"}_{alpha!r}'


def choose_candidate(candidates):
    for row in candidates:
        parameter='kappa' if 'kappa' in row else 'alpha'
        candidate_tag(row[parameter],parameter)
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
    return max(valid,key=lambda r:(r['metrics']['detection_rate'],r['metrics']['mean_score'],-r['kappa' if 'kappa' in r else 'alpha']))


def train_candidates(run):
    from semantic_sign import is_sign
    if is_sign(read_json(run/'config.json')):
        from semantic_sign_train import train_sign_candidates
        return train_sign_candidates(run)
    from semantic_teacher import generate_set, score
    from semantic_report import detection_metrics, teacher_quality
    from semantic_assets import snapshot_identity
    c=read_json(run/'config.json'); root=run/'teacher_fp'; root.mkdir(exist_ok=True)
    selected_path=root/'selected.json'
    if selected_path.exists():
        selected=read_json(selected_path)
        if snapshot_identity(run/selected['model'])!=selected['model_hashes']: raise RuntimeError('Selected teacher changed')
        if selected['config_sha256']!=sha256(run/'config.json'): raise RuntimeError('Teacher selection config changed')
        parameter='kappa' if 'kappa' in selected else 'alpha'
        chosen=(next(row for row in selected['candidates']
                     if row[parameter]==selected[parameter] and row['model']==selected['model'])
                if selected['downstream_ready'] else None)
        if chosen is not None:
            freeze(run/'result/teacher_quality.json',{'selected':chosen,'candidates':selected['candidates']})
        if not selected['downstream_ready']:
            raise ValueError('Frozen teacher failed the validation detection gate; resume cannot bypass it')
        return
    threshold=read_json(run/'result/threshold_teacher.json')['threshold']
    if is_threshold(c):
        frozen_calibration(run)
        from semantic_candidates import run_workers
        run_workers(run,c['teacher_kappas'])
        candidates=[]
        calibration=read_json(run/'teacher_fp/calibration_target.json')
        for k in c['teacher_kappas']:
            record=read_json(root/'candidates'/candidate_tag(k,'kappa')/'validation.json')
            if (record['kappa']!=k or record['config_sha256']!=sha256(run/'config.json') or record['calibration']!=calibration
                    or record['model_hashes']!=snapshot_identity(run/record['model'])):
                raise RuntimeError('Candidate validation provenance changed')
            candidates.append(record)
        freeze(root/'candidates.json',{'objective':c['teacher_objective'],'candidates':candidates})
        chosen=choose_candidate(candidates)
        selection={**chosen,'candidates':candidates,
                   'selection_rule':'CLIP-drop gate; highest validation TPR, mean score, lower kappa; LPIPS diagnostic only',
                   'config_sha256':sha256(run/'config.json'),'model_hashes':snapshot_identity(run/chosen['model']),
                   'teacher_threshold_sha256':sha256(run/'result/threshold_teacher.json'),
                   'calibration_target_sha256':sha256(run/'teacher_fp/calibration_target.json'),
                   'test_used':False,'downstream_ready':chosen['metrics']['detection_rate']>=c['teacher_min_tpr']}
        freeze(selected_path,selection)
        freeze(run/'result/teacher_quality.json',{'selected':chosen,'candidates':candidates})
        if not selection['downstream_ready']:
            raise ValueError('All teacher kappa candidates below required validation TPR; students stopped. See teacher_fp/candidates.json')
        return
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


def train_threshold_candidate(run,kappa):
    """Independent worker, always initialized from clean SD2.1 unless exact-resumed."""
    from semantic_teacher import generate_set,score
    from semantic_report import detection_metrics,teacher_quality
    from semantic_assets import snapshot_identity
    c=read_json(run/'config.json')
    if not is_threshold(c) or kappa not in c['teacher_kappas']:
        raise ValueError('Worker candidate not in frozen threshold objective configuration')
    calibration=frozen_calibration(run)
    tag=candidate_tag(kappa,'kappa');folder=run/'teacher_fp/candidates'/tag
    train_one(run,kappa,folder);model=merge_teacher(run,folder)
    label=f'candidate_{tag}_validation'
    generate_set(run,model,'validation',label);score(run,[label])
    metrics=detection_metrics(run,label,calibration['threshold'],c['validation_n'])
    record={'kappa':kappa,'target_score':calibration['threshold']+kappa*calibration['sigma_clean'],
            'objective':c['teacher_objective'],'model':str(model.relative_to(run)),
            'metrics':metrics,'quality':teacher_quality(run,label),
            'calibration':calibration,'model_hashes':snapshot_identity(model),
            'config_sha256':sha256(run/'config.json')}
    freeze(folder/'validation.json',record)
