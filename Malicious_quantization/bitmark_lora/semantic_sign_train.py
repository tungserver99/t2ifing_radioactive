"""RSSF decoded-image teacher objective; reuse historical LoRA/checkpoint/merge machinery."""
import copy
import gc
import importlib
import json
import math
from pathlib import Path
import torch
from common import freeze,read_json,sha256,write_json,image_name,verify_vendor
from semantic_carrier import SemanticCarrier,check_finite


def _checkpoint(function,*args):
    if torch.is_grad_enabled() and any(x.requires_grad for x in args if isinstance(x,torch.Tensor)):
        from torch.utils.checkpoint import checkpoint
        return checkpoint(function,*args,use_reentrant=False)
    return function(*args)


def decoded_image(ext,z):
    def decode(value):
        image=ext.vae.decode(value.float()/ext.vae.config.scaling_factor).sample
        check_finite(image)
        return image.clamp(-1,1)
    return _checkpoint(decode,z)


def decoded_coordinates(ext,carrier,image,text):
    # Checkpoint the VAE encoder as well as the already checkpointed extractor.
    def features(value,hidden):
        return ext.encode_latent(ext.image_latent(value),text_hidden=hidden)
    return carrier.coordinates(_checkpoint(features,image,text))


def _predictions(unet,noisy,t,text,empty,precision=None):
    if precision is None:
        return (unet(noisy,t,encoder_hidden_states=text).sample,
                unet(noisy,t,encoder_hidden_states=empty).sample)
    dtype={'bf16':torch.bfloat16,'fp16':torch.float16,'no':torch.float32}[precision]
    with torch.autocast(noisy.device.type,dtype=dtype,enabled=precision!='no'):
        return (unet(noisy,t,encoder_hidden_states=text).sample,
                unet(noisy,t,encoder_hidden_states=empty).sample)


def sign_teacher_backward(unet,ext,carrier,scheduler,perceptual,z,text,noise,t,late_noise,late_t,mu,c,backward,diagnostic=None):
    from semantic_teacher_train import predicted_x0,teacher_precision,watch_gradient
    from semantic_threshold import cfg_prediction
    from semantic_sign import sign_hinge_loss
    batch=len(z);micro=c.get('teacher_micro_batch',1)
    if batch<1 or micro<1:raise ValueError('Positive RSSF batch/microbatch required')
    empty=ext.text(['']*batch)
    result={}
    for start in range(0,batch,micro):
        end=min(start+micro,batch);sl=slice(start,end);weight=(end-start)/batch
        if diagnostic is not None:diagnostic.update(microbatch_start=start)
        # Separate backward frees preservation activations before image loss.
        noisy=scheduler.add_noise(z[sl],noise[sl],t[sl]) if hasattr(scheduler,'add_noise') else _add_noise(scheduler,z[sl],noise[sl],t[sl])
        with torch.no_grad():clean_c,clean_u=_predictions(ext.unet,noisy,t[sl],text[sl],empty[sl])
        fp_c,fp_u=_predictions(unet,noisy,t[sl],text[sl],empty[sl],teacher_precision(c))
        pc=(fp_c.float()-clean_c.float()).square().mean();pu=(fp_u.float()-clean_u.float()).square().mean()
        preserve=.5*(pc+pu);preserve_loss=c['lambda_preserve']*preserve*weight
        check_finite(preserve_loss);backward(preserve_loss)
        stats={'preserve_conditional':pc.detach().item(),'preserve_unconditional':pu.detach().item(),'preserve':preserve.detach().item()}
        del fp_c,fp_u,clean_c,clean_u,pc,pu,preserve,preserve_loss,noisy
        late=scheduler.add_noise(z[sl],late_noise[sl],late_t[sl]) if hasattr(scheduler,'add_noise') else _add_noise(scheduler,z[sl],late_noise[sl],late_t[sl])
        with torch.no_grad():
            cc,cu=_predictions(ext.unet,late,late_t[sl],text[sl],empty[sl])
            clean_z=predicted_x0(scheduler,late,cfg_prediction(cc,cu),late_t[sl])
            clean_image=decoded_image(ext,clean_z).detach()
            clean_r=decoded_coordinates(ext,carrier,clean_image,text[sl]).detach()
        fc,fu=_predictions(unet,late,late_t[sl],text[sl],empty[sl],teacher_precision(c))
        fp_z=watch_gradient(predicted_x0(scheduler,late,cfg_prediction(fc,fu),late_t[sl]),'RSSF late CFG x0')
        image=watch_gradient(decoded_image(ext,fp_z),'RSSF decoded image')
        fp_r=decoded_coordinates(ext,carrier,image,text[sl])
        fp,extra=sign_hinge_loss(clean_r,fp_r,carrier.signs,mu,c['sign_k_train'])
        lp=_checkpoint(lambda a,b:perceptual(a.float(),b.float()).mean(),image,clean_image)
        pixel=(image-clean_image).square().mean()
        quality=c['lambda_lpips']*lp+c['lambda_pixel']*pixel
        late_loss=(c['lambda_fp']*fp+c['lambda_quality']*quality)*weight
        check_finite(late_loss,lp,pixel,fp)
        stats.update(fp=fp.detach().item(),lpips=lp.detach().item(),pixel_mse=pixel.detach().item(),quality=quality.detach().item())
        stats['loss']=c['lambda_preserve']*stats['preserve']+c['lambda_fp']*stats['fp']+c['lambda_quality']*stats['quality']
        if diagnostic is not None:diagnostic['microbatch_losses']=stats
        backward(late_loss)
        for key,value in {**stats,**extra}.items():
            if key in ('selected_indices','selected_mask'):continue
            if isinstance(value,torch.Tensor):value=value.detach().cpu().tolist()
            if isinstance(value,(list,tuple)):
                result[key]=[a+float(b)*weight for a,b in zip(result.get(key,[0.]*len(value)),value)]
            elif isinstance(value,(float,int)):result[key]=result.get(key,0.)+float(value)*weight
    result.update(preservation_timesteps=t.detach().cpu().tolist(),late_timesteps=late_t.detach().cpu().tolist())
    return result


def _add_noise(scheduler,z,noise,t):
    # Same DDPM forward equation; small scheduler doubles can exercise CPU logic.
    a=scheduler.alphas_cumprod.to(z.device)[t].reshape((-1,)+(1,)*(z.ndim-1))
    return a.sqrt()*z+(1-a).sqrt()*noise


def choose_sign_candidate(candidates,c):
    valid=[]
    for row in candidates:
        q=row['quality'];m=row['metrics']
        values=[m['detection_rate'],q['clip_drop'],q['paired_lpips'],q['paired_pixel_mse'],row['coordinate_margin_mu']]
        if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite RSSF validation metric')
        if (m['detection_rate']>=max(.9,c.get('teacher_min_analytic_tpr',.9)) and
            q['clip_drop']<=min(.01,c.get('teacher_clip_drop_max',.01)) and
            q['paired_lpips']<=min(.15,c.get('teacher_lpips_max',.15))):valid.append(row)
    if not valid:raise ValueError('No RSSF teacher passed analytic TPR/CLIP/LPIPS gates; students stopped')
    return min(valid,key=lambda r:(r['quality']['paired_lpips'],r['quality']['paired_pixel_mse'],r['coordinate_margin_mu']))


def sign_schedule(run,c,training_scheduler):
    from semantic_sign import derive_late_timesteps
    assets=read_json(run/'assets.json');model=Path(assets['teacher_clean'])
    library,classname=read_json(model/'model_index.json')['scheduler']
    if library!='diffusers':raise ValueError('Native scheduler must be the saved Diffusers scheduler')
    diffusers=importlib.import_module('diffusers')
    cls=getattr(diffusers,classname)
    native=cls.from_pretrained(model,subfolder='scheduler',local_files_only=True)
    # Use the installed native pipeline's own scheduler normalization (not a
    # second handwritten policy). No model components/weights are constructed.
    native=diffusers.StableDiffusionPipeline(
        vae=None,text_encoder=None,tokenizer=None,unet=None,scheduler=native,
        safety_checker=None,feature_extractor=None,requires_safety_checker=False).scheduler
    late=derive_late_timesteps(native,training_scheduler,c['inference_steps'],c['late_inference_step_count'])
    protocol={'objective':'semantic_sign_v1','native_scheduler':classname,'training_scheduler':type(training_scheduler).__name__,
              'scheduler_config_sha256':sha256(model/'scheduler/scheduler_config.json'),
              'inference_steps':c['inference_steps'],'late_timesteps':late,'guidance_scale':7.5,
              'alphas_cumprod':[float(training_scheduler.alphas_cumprod[i]) for i in late],
              'carrier_sha256':sha256(run/'carrier/complete.json'),'config_sha256':sha256(run/'config.json')}
    freeze(run/'teacher_fp/sign_protocol.json',protocol)
    return late,protocol



def sample_objective_batch(scheduler,z,rng,late_ids):
    device=z.device
    noise=torch.randn(z.shape,generator=rng).to(device)
    t=torch.randint(0,scheduler.config.num_train_timesteps,(len(z),),generator=rng).to(device)
    late_noise=torch.randn(z.shape,generator=rng).to(device)
    indices=torch.randint(0,len(late_ids),(len(z),),generator=rng)
    late_t=torch.tensor(late_ids,dtype=torch.long)[indices].to(device)
    return noise,t,late_noise,late_t


def stress_sign(unet,ext,carrier,scheduler,perceptual,z,text,mu,c,folder,late_ids,steps=12):
    from semantic_teacher_train import checked_teacher_step
    folder.mkdir(parents=True,exist_ok=True)
    named=[(n,p) for n,p in unet.named_parameters() if p.requires_grad]
    saved=[p.detach().cpu().clone() for _,p in named]
    scratch=torch.optim.AdamW([p for _,p in named],lr=c['teacher_lr'])
    scaler=torch.amp.GradScaler('cuda') if c['teacher_precision']=='fp16' and z.is_cuda else None
    devices=[z.device.index if z.device.index is not None else torch.cuda.current_device()] if z.is_cuda else []
    checks=[]
    try:
        with torch.random.fork_rng(devices=devices):
            rng=torch.Generator('cpu').manual_seed(c['seed']+7919)
            for index in range(steps):
                noise,t,late_noise,late_t=sample_objective_batch(scheduler,z,rng,late_ids)
                if index<4:t.fill_([0,scheduler.config.num_train_timesteps//4,scheduler.config.num_train_timesteps//2,scheduler.config.num_train_timesteps-1][index])
                late_t.fill_(late_ids[index%len(late_ids)])
                context={'phase':'teacher_check','probe':index+1,'coordinate_margin_mu':mu,
                         'timesteps':t.cpu().tolist(),'late_timesteps':late_t.cpu().tolist()}
                if z.is_cuda:torch.cuda.reset_peak_memory_stats(z.device)
                stats=checked_teacher_step(named,scratch,scaler,
                    lambda backward:sign_teacher_backward(unet,ext,carrier,scheduler,perceptual,z,text,noise,t,late_noise,late_t,mu,c,backward,context),
                    folder/'failure.json',context)
                row={**context,**stats}
                if z.is_cuda:row['peak_allocated_gib']=torch.cuda.max_memory_allocated()/1024**3
                checks.append(row);print(f'RSSF numerical check {index+1}/{steps}: {row}',flush=True)
        write_json(folder/'check.json',{'passed':True,'checks':checks,'weights_and_rng_restored':True})
    finally:
        with torch.no_grad():
            for (_,parameter),value in zip(named,saved):parameter.copy_(value);parameter.grad=None
        del scratch,scaler;gc.collect()
        if z.is_cuda:torch.cuda.empty_cache()


def train_sign_one(run,mu,folder):
    from diffusers import DDPMScheduler
    from peft import LoraConfig
    import lpips
    from semantic_teacher import extractor,pixels,numerical_policy
    from semantic_teacher_train import adapter_targets,save_adapter,checked_teacher_step
    from semantic_checkpoint import latest_full_checkpoint,restore_training_state,save_training_state,align_loss_log
    from semantic_sign import detection_count
    verify_vendor();numerical_policy()
    c=read_json(run/'config.json');assets=read_json(run/'assets.json')
    if c.get('continuation'):raise ValueError('RSSF candidates must start from clean SD2.1, not continuation weights')
    scheduler=DDPMScheduler.from_pretrained(assets['teacher_clean'],subfolder='scheduler',local_files_only=True)
    late_ids,schedule_proof=sign_schedule(run,c,scheduler)
    cutoff=detection_count(c['num_bits'],c['analytic_alpha'])
    c={**c,'sign_k_train':min(c['num_bits'],cutoff+c['train_bit_count_margin'])}
    if cutoff>c['num_bits']:raise ValueError('No attainable analytic bit threshold')
    protocol={'objective':'semantic_sign_v1','config_sha256':sha256(run/'config.json'),
              'assets_sha256':sha256(run/'assets.json'),'carrier_sha256':sha256(run/'carrier/complete.json'),
              'training_images_sha256':sha256(run/'images/teacher_clean_train/image_hashes.json'),
              'coordinate_margin_mu':mu,'k_detect':cutoff,'k_train':c['sign_k_train'],
              'late_schedule':schedule_proof,'micro_batch':c['teacher_micro_batch'],'teacher_precision':c['teacher_precision'],
              'numerical_check_steps':12,'reference_cycle_checkpointing':True,
              'lpips':'alex, version=0.1; frozen FP32; input [-1,1]',
              'restart_policy':'restore full checkpoint including AdamW, sampler and all RNG states'}
    folder.mkdir(parents=True,exist_ok=True);freeze(folder/'protocol.json',protocol)
    if (folder/'trained.json').exists():
        for name,digest in read_json(folder/'trained.json')['files'].items():
            if sha256(folder/name)!=digest:raise RuntimeError('RSSF trained adapter changed')
        return
    torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
    if c['teacher_precision']=='bf16' and not torch.cuda.is_bf16_supported():raise RuntimeError('Teacher GPU lacks BF16 support')
    ext=extractor(run);unet=copy.deepcopy(ext.unet).train()
    adapter=LoraConfig(r=c['teacher_rank'],lora_alpha=c['teacher_rank'],init_lora_weights='gaussian',target_modules=adapter_targets(unet))
    unet.add_adapter(adapter);unet.enable_gradient_checkpointing()
    named=[(n,p) for n,p in unet.named_parameters() if p.requires_grad]
    if not named:raise RuntimeError('RSSF teacher has no trainable adapter')
    optimizer=torch.optim.AdamW([p for _,p in named],lr=c['teacher_lr'])
    scaler=torch.amp.GradScaler('cuda') if c['teacher_precision']=='fp16' else None
    perceptual=lpips.LPIPS(net='alex',version='0.1').cuda().float().eval().requires_grad_(False)
    carrier=SemanticCarrier.load(run/'carrier','cuda')
    if carrier.carrier_version!='semantic_sign_v1':raise ValueError('RSSF requires a versioned centered carrier')
    rows=read_json(run/'prompts.json')['train'];images=run/'images/teacher_clean_train'
    expected=read_json(images/'image_hashes.json')
    if len(expected)!=len(rows) or any(sha256(images/r['file_name'])!=r['sha256'] for r in expected):raise RuntimeError('RSSF clean training data changed')
    rng=torch.Generator('cpu').manual_seed(c['seed']);order=[];offset=0;start_step=0
    checkpoint=latest_full_checkpoint(folder)
    if checkpoint:
        state=restore_training_state(checkpoint,unet,optimizer,scaler,rng,protocol)
        start_step,order,offset=state['step'],state['order'],state['offset']
        align_loss_log(folder/'losses.jsonl',start_step,expected_hash=state['loss_prefix_sha256'])
        print(f'Resuming RSSF full state at step {start_step}',flush=True)
    else:align_loss_log(folder/'losses.jsonl',0)
    if not 0<=start_step<=c['teacher_train_steps']:raise ValueError('RSSF checkpoint outside configured budget')
    with torch.no_grad():
        probe_z=ext.image_latent(pixels([images/image_name(i) for i in range(c['teacher_batch'])]))
        probe_text=ext.text([rows[i]['text'] for i in range(c['teacher_batch'])])
    # check.json is a fresh diagnostic for each resume, not a frozen scientific result.
    stress_sign(unet,ext,carrier,scheduler,perceptual,probe_z,probe_text,mu,c,folder/'numerical_check',late_ids)
    del probe_z,probe_text
    with (folder/'losses.jsonl').open('a') as log:
        for step in range(start_step,c['teacher_train_steps']):
            if offset+c['teacher_batch']>len(order):order=torch.randperm(len(rows),generator=rng).tolist();offset=0
            ids=order[offset:offset+c['teacher_batch']];offset+=len(ids)
            with torch.no_grad():
                z=ext.image_latent(pixels([images/image_name(i) for i in ids]));text=ext.text([rows[i]['text'] for i in ids])
                noise,t,late_noise,late_t=sample_objective_batch(scheduler,z,rng,late_ids)
            context={'phase':'train','step':step+1,'coordinate_margin_mu':mu,'image_ids':ids,
                     'timesteps':t.cpu().tolist(),'late_timesteps':late_t.cpu().tolist()}
            torch.cuda.reset_peak_memory_stats()
            measured=checked_teacher_step(named,optimizer,scaler,
                lambda backward:sign_teacher_backward(unet,ext,carrier,scheduler,perceptual,z,text,noise,t,late_noise,late_t,mu,c,backward,context),
                folder/'failure.json',context)
            raw={'step':step+1,**measured,'coordinate_margin_mu':mu,'effective_batch':len(ids),
                 'micro_batch':c['teacher_micro_batch'],'teacher_precision':c['teacher_precision'],
                 'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3}
            log.write(json.dumps(raw,allow_nan=False)+'\n');log.flush()
            if (step+1)%25==0:print(f'Teacher mu={mu}: {raw}',flush=True)
            if (step+1)%c['teacher_checkpoint_every']==0 or step+1==c['teacher_train_steps']:
                dest=folder/f'checkpoint-{step+1}'
                save_adapter(unet,dest,adapter);save_training_state(dest,unet,optimizer,scaler,rng,order,offset,step+1,protocol)
    save_adapter(unet,folder,adapter)
    freeze(folder/'trained.json',{'files':{name:sha256(folder/name) for name in ('adapter_model.safetensors','adapter_config.json','losses.jsonl')},'steps':c['teacher_train_steps']})
    del unet,ext,optimizer,perceptual,carrier;gc.collect();torch.cuda.empty_cache()


def sign_candidate_tag(mu):
    if not math.isfinite(mu) or mu<=0:raise ValueError('Positive finite RSSF margin required')
    return f'mu_{float(mu)!r}'


def train_sign_candidate(run,mu):
    from semantic_teacher_train import merge_teacher
    from semantic_teacher import generate_set,score
    from semantic_sign_report import analytic_metrics
    from semantic_report import teacher_quality
    from semantic_assets import snapshot_identity
    c=read_json(run/'config.json')
    if mu not in c['coordinate_margin_mu_candidates']:raise ValueError('Unregistered RSSF margin')
    folder=run/'teacher_fp/candidates'/sign_candidate_tag(mu)
    train_sign_one(run,mu,folder);model=merge_teacher(run,folder)
    label=f'candidate_{sign_candidate_tag(mu)}_validation'
    generate_set(run,model,'validation',label);score(run,[label])
    metrics=analytic_metrics(run,label,c['validation_n']);quality=teacher_quality(run,label)
    from semantic_report import score_rows
    from semantic_sign import sign_hinge_loss,detection_count
    carrier=SemanticCarrier.load(run/'carrier')
    clean_rows=score_rows(run,'teacher_clean_validation',c['validation_n'],carrier)
    fp_rows=score_rows(run,label,c['validation_n'],carrier)
    clean_r=torch.tensor([row['coordinates'] for row in clean_rows]);fp_r=torch.tensor([row['coordinates'] for row in fp_rows])
    count=min(c['num_bits'],detection_count(c['num_bits'],c['analytic_alpha'])+c['train_bit_count_margin'])
    diagnostic_loss,diagnostics=sign_hinge_loss(clean_r,fp_r,carrier.signs,mu,count)
    diagnostics={key:(value.tolist() if isinstance(value,torch.Tensor) else value) for key,value in diagnostics.items()}
    diagnostics.update(fp_loss=diagnostic_loss.item(),source='held-out native PNG images; clean-only TopK; reporting diagnostics')
    freeze(folder/'validation_diagnostics.json',diagnostics)
    record={'coordinate_margin_mu':mu,'metrics':metrics,'quality':quality,'diagnostics':diagnostics,
            'model':str(model.relative_to(run)),'model_hashes':snapshot_identity(model),
            'config_sha256':sha256(run/'config.json'),'carrier_sha256':sha256(run/'carrier/complete.json')}
    freeze(folder/'validation.json',record)


def train_sign_candidates(run):
    from semantic_sign_report import freeze_analytic_verifier,validate_clean_null
    from semantic_candidates import run_workers
    from semantic_assets import snapshot_identity
    c=read_json(run/'config.json');root=run/'teacher_fp';root.mkdir(exist_ok=True)
    freeze_analytic_verifier(run);SemanticCarrier.load(run/'carrier')
    validate_clean_null(run,labels=('teacher_clean_calibration',))
    selected_path=root/'selected.json'
    if selected_path.exists():
        selected=read_json(selected_path)
        if selected['config_sha256']!=sha256(run/'config.json') or selected['carrier_sha256']!=sha256(run/'carrier/complete.json'):
            raise RuntimeError('RSSF selected experiment changed')
        if snapshot_identity(run/selected['model'])!=selected['model_hashes']:raise RuntimeError('RSSF selected model changed')
        chosen=choose_sign_candidate(selected['candidates'],c)
        if chosen['coordinate_margin_mu']!=selected['coordinate_margin_mu']:raise RuntimeError('RSSF selection changed')
        freeze(run/'result/teacher_quality.json',{'selected':chosen,'candidates':selected['candidates']})
        return
    run_workers(run,c['coordinate_margin_mu_candidates'],parameter='coordinate_margin_mu')
    candidates=[]
    for mu in c['coordinate_margin_mu_candidates']:
        row=read_json(root/'candidates'/sign_candidate_tag(mu)/'validation.json')
        if (row['coordinate_margin_mu']!=mu or row['config_sha256']!=sha256(run/'config.json') or
            row['carrier_sha256']!=sha256(run/'carrier/complete.json') or snapshot_identity(run/row['model'])!=row['model_hashes']):
            raise RuntimeError('RSSF candidate validation provenance changed')
        candidates.append(row)
    freeze(root/'candidates.json',{'objective':'semantic_sign_v1','candidates':candidates})
    try:chosen=choose_sign_candidate(candidates,c)
    except ValueError as exc:
        write_json(root/'failure.json',{'reason':str(exc),'candidates':candidates,'students_started':False})
        raise
    selected={**chosen,'candidates':candidates,'downstream_ready':True,
              'selection_rule':'analytic TPR>=.90, CLIP drop<=.01, LPIPS<=.15; min LPIPS then pixel MSE then mu; validation only',
              'test_used':False}
    freeze(selected_path,selected);freeze(run/'result/teacher_quality.json',{'selected':chosen,'candidates':candidates})



def check_sign_existing(run,precision='bf16',steps=32):
    """Standalone numerical diagnostic with fresh scratch optimizer; never training resume."""
    import datetime
    import lpips
    from diffusers import DDPMScheduler
    from student import load_adapter
    from semantic_teacher import extractor,pixels,numerical_policy
    from semantic_sign import detection_count
    from semantic_checkpoint import latest_full_checkpoint
    verify_vendor();numerical_policy()
    if not torch.cuda.is_available():raise RuntimeError('RSSF model diagnostic requires GPU')
    if precision=='bf16' and not torch.cuda.is_bf16_supported():raise RuntimeError('GPU does not support BF16')
    c={**read_json(run/'config.json'),'teacher_precision':precision}
    c['sign_k_train']=min(c['num_bits'],detection_count(c['num_bits'],c['analytic_alpha'])+c['train_bit_count_margin'])
    assets=read_json(run/'assets.json');scheduler=DDPMScheduler.from_pretrained(assets['teacher_clean'],subfolder='scheduler',local_files_only=True)
    late,_=sign_schedule(run,c,scheduler)
    output=run/'diagnostics'/datetime.datetime.now(datetime.timezone.utc).strftime('rssf_check_%Y%m%d_%H%M%S_%f')
    targets=[]
    for candidate in sorted((run/'teacher_fp/candidates').glob('mu_*')):
        source=candidate if (candidate/'trained.json').exists() else latest_full_checkpoint(candidate)
        if source is None:continue
        proof=read_json(candidate/'protocol.json')
        for name,path in (('config_sha256',run/'config.json'),('assets_sha256',run/'assets.json'),('carrier_sha256',run/'carrier/complete.json')):
            if proof[name]!=sha256(path):raise RuntimeError('RSSF diagnostic source provenance changed')
        targets.append((proof['coordinate_margin_mu'],candidate,source))
    if not targets:raise ValueError('No saved RSSF teacher checkpoint; a fresh run performs its own readiness check')
    for mu,candidate,source in targets:
        if (candidate/'trained.json').exists():
            for name,digest in read_json(candidate/'trained.json')['files'].items():
                if sha256(candidate/name)!=digest:raise RuntimeError('RSSF diagnostic adapter changed')
        ext=extractor(run);unet=copy.deepcopy(ext.unet);load_adapter(unet,source)
        for name,p in unet.named_parameters():p.requires_grad_('lora_' in name)
        unet.train();unet.enable_gradient_checkpointing()
        perceptual=lpips.LPIPS(net='alex',version='0.1').cuda().float().eval().requires_grad_(False)
        carrier=SemanticCarrier.load(run/'carrier','cuda');rows=read_json(run/'prompts.json')['train'];images=run/'images/teacher_clean_train'
        if sha256(images/'image_hashes.json')!=read_json(candidate/'protocol.json')['training_images_sha256']:raise RuntimeError('RSSF diagnostic data manifest changed')
        if any(sha256(images/r['file_name'])!=r['sha256'] for r in read_json(images/'image_hashes.json')):raise RuntimeError('RSSF diagnostic training image changed')
        with torch.no_grad():
            z=ext.image_latent(pixels([images/image_name(i) for i in range(c['teacher_batch'])]))
            text=ext.text([rows[i]['text'] for i in range(c['teacher_batch'])])
        dest=output/candidate.name
        write_json(dest/'source.json',{'source':str(source),'adapter_sha256':sha256(source/'adapter_model.safetensors'),
                                    'teacher_precision':precision,'steps':steps,'diagnostic_only':True,'optimizer':'fresh scratch AdamW, not restored training optimizer'})
        stress_sign(unet,ext,carrier,scheduler,perceptual,z,text,mu,c,dest,late,steps)
        del unet,ext,perceptual,carrier,z,text;gc.collect();torch.cuda.empty_cache()
    print(f'RSSF diagnostic passed; reports: {output}; not a resumed training run.',flush=True)
