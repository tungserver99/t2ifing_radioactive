"""V3 native teacher refinement: fresh AdamW, live prefix, ten-call tail."""
import copy
import gc
import json
import random
from pathlib import Path
import torch
from common import read_json, freeze, sha256, image_name, verify_vendor
from semantic_carrier import SemanticCarrier, check_finite
from semantic_native_rollout import (OBJECTIVE, live_rollout, initial_latents, rgb8_ste,
    dynamic_sign_margin_loss, preservation_loss, accumulate_gradients, capped_gradient)


def release():
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()


def training_protocol(run):
    return {'teacher_objective':OBJECTIVE,
            **{name.replace('/','_').replace('.','_')+'_sha256':sha256(run/name) for name in
               ('config.json','assets.json','prompts.json','source_lineage.json','code_manifest.json',
                'carrier/complete.json','carrier/key.json','images/teacher_clean_train/image_hashes.json')},
            'optimizer':{'type':'AdamW','lr':1e-5,'betas':[.9,.999],'eps':1e-8,'weight_decay':.01},
            'effective_batch':4,'tail_calls':10,'gradient_epsilon':1e-12,
            'step_semantics':'300 sampled batches; zero-margin-gradient batches logged and skipped, never fidelity-only'}


def effective_batch(unet,ext,carrier,factory,rows,clean_paths,c,rng,params,audit_tail=False):
    from semantic_sign_train import decoded_image,decoded_coordinates
    from semantic_teacher import pixels, generation_seed
    gm=[torch.zeros_like(p,dtype=torch.float32) for p in params]
    gf=[torch.zeros_like(p,dtype=torch.float32) for p in params]
    samples=[];used=set()
    if len(rows)!=4: raise ValueError('V3 effective batch must remain four')
    for start in range(0,4,c['native_micro_batch']):
        batch=rows[start:start+c['native_micro_batch']]; size=len(batch); weight=size/4
        with torch.no_grad():
            text=ext.text([r['text'] for r in batch]);empty=ext.text(['']*size)
            clean=pixels(clean_paths[start:start+size]).detach()
        scheduler=factory();scheduler.set_timesteps(50,device=text.device)
        probe_index=int(torch.randint(len(scheduler.timesteps)-10,(1,),generator=rng))
        seen=[];reached=set()
        def observe(i,latent,sched):
            tail=i>=len(sched.timesteps)-10
            if tail:
                if not latent.requires_grad:raise RuntimeError('Differentiable tail lost its graph')
                if audit_tail:latent.register_hook(lambda g,index=i:reached.add(index))
            elif latent.requires_grad:
                raise RuntimeError('Prefix unexpectedly retains autograd graph')
            seen.append((i,sched.counter,tail))
        initial=initial_latents(unet,scheduler,batch,c)
        latent,probe,rollout=live_rollout(unet,scheduler,initial,text,empty,probe_index,observer=observe)
        image=rgb8_ste(decoded_image(ext,latent))
        coords=decoded_coordinates(ext,carrier,image,text)
        lm,stats=dynamic_sign_margin_loss(coords,carrier.signs)
        lq=(image-clean).square().mean()
        lp=preservation_loss(unet,ext.unet,scheduler,probe,text,empty)
        check_finite(image,coords,lm,lq,lp)
        used.update(accumulate_gradients(lm,params,gm,weight,retain_graph=True))
        used.update(accumulate_gradients(lp+lq,params,gf,weight))
        if audit_tail and len(reached)!=10:
            raise RuntimeError(f'Expected all ten tail gradients, reached {sorted(reached)}')
        if any(p.grad is not None for model in (ext.unet,ext.vae,ext.text_encoder) for p in model.parameters()):
            raise RuntimeError('Frozen reference received a gradient')
        samples.append({**stats,**rollout,'micro_batch':size,'weight':weight,
                        'L_P':lp.item(),'L_Q':lq.item(),
                        'image_ids':[r['image_id'] for r in batch],
                        'prompt_sha256':[__import__('hashlib').sha256(r['text'].encode()).hexdigest() for r in batch],
                        'seeds':[generation_seed(c,r) for r in batch]})
        del image,coords,lm,lq,lp,latent,probe,initial,text,empty,clean
    return gm,gf,{'microbatches':samples,'used_parameter_indices':sorted(used),
                  **{key:sum(s[key]*s['weight'] for s in samples) for key in ('L_M','L_P','L_Q','mean_K')} }


def clean_rollout_gate(run,ext,factory):
    """Compare real native pipeline latent output and the new loop, clean FP32."""
    from diffusers import StableDiffusionPipeline
    from semantic_teacher import native_generate,generation_seed
    c=read_json(run/'config.json');row=read_json(run/'prompts.json')['train'][0]
    scheduler=factory()
    pipe=StableDiffusionPipeline(vae=ext.vae,text_encoder=ext.text_encoder,tokenizer=ext.tokenizer,
          unet=ext.unet,scheduler=scheduler,safety_checker=None,feature_extractor=None,requires_safety_checker=False)
    pipe.set_progress_bar_config(disable=True)
    _,expected=native_generate(pipe,[row['text']],[torch.Generator('cuda').manual_seed(generation_seed(c,row))],50,return_latents=True)
    with torch.no_grad():
        text=ext.text([row['text']]);empty=ext.text([''])
        manual=factory();manual.set_timesteps(50,device='cuda')
        actual,_,_=live_rollout(ext.unet,manual,initial_latents(ext.unet,manual,[row],c,dtype=torch.float32),text,empty,precision='no')
    torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5)
    return {'clean_native_latent_max_error':(actual-expected).abs().max().item(),
            'atol':2e-5,'rtol':2e-5,'reference':'native_generate clean FP32; same scheduler and Gaussian protocol',
            'training_policy':'BF16 denoiser, FP32 CFG/history/decode/extractor; production FP16 merge screening'}


def prefix_current_weight_gate(unet,ext,factory,rows,c,params):
    text=ext.text([rows[0]['text']]);empty=ext.text([''])
    prefixes=[]
    # Changing the current adapter must change the no-grad prefix, not only the tail.
    target=next((p for n,p in unet.named_parameters() if p.requires_grad and 'lora_B' in n),None)
    if target is None:raise RuntimeError('Missing warm-start LoRA B parameters')
    original=target.detach().clone()
    try:
        for perturb in (False,True):
            if perturb:
                with torch.no_grad():target.add_(.01)
            scheduler=factory();scheduler.set_timesteps(50,device=text.device)
            captures=[]
            def observe(i,latent,sched):
                if i==len(sched.timesteps)-11:captures.append(latent.detach().clone())
            with torch.no_grad():
                live_rollout(unet,scheduler,initial_latents(unet,scheduler,rows[:1],c),text,empty,observer=observe)
            prefixes.append(captures[0])
        difference=(prefixes[1]-prefixes[0]).abs().max().item()
        if difference==0:raise RuntimeError('Current adapter perturbation does not affect prefix')
        return difference
    finally:
        with torch.no_grad():target.copy_(original)


def train(run):
    from peft import LoraConfig,get_peft_model_state_dict
    from safetensors.torch import load_file
    from student import load_adapter
    from semantic_teacher import extractor,numerical_policy
    from semantic_teacher_train import adapter_targets,save_adapter
    from semantic_checkpoint import (latest_full_checkpoint,restore_training_state,save_training_state,
                                      align_loss_log,verified_state)
    from semantic_trajectory import native_scheduler
    verify_vendor();numerical_policy()
    c=read_json(run/'config.json');a=read_json(run/'assets.json')
    from semantic_native_lineage import source_identity
    if source_identity(Path(c['source_run']))!=read_json(run/'source_lineage.json'):
        raise RuntimeError('Warm-start source differs from frozen v3 source lineage')
    if c['teacher_objective']!=OBJECTIVE:raise ValueError('V3 trainer requires the versioned objective')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('V3 requires a BF16-capable CUDA GPU; local CPU training is unsupported')
    folder=run/'teacher_fp/native_margin_refine';folder.mkdir(parents=True,exist_ok=True)
    protocol=training_protocol(run);freeze(folder/'protocol.json',protocol)
    if (folder/'trained.json').exists():
        proof=read_json(folder/'trained.json')
        if proof['protocol']!=protocol or proof['losses_sha256']!=sha256(folder/'losses.jsonl'):
            raise RuntimeError('Completed v3 training changed')
        for step in (50,100,200,300):verified_state(folder/f'checkpoint-{step}',protocol)
        return
    torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed']);random.seed(c['seed'])
    ext=extractor(run);factory=lambda:native_scheduler(a['teacher_clean'])
    checkpoint=latest_full_checkpoint(folder)
    smoke_path=folder/'smoke.json'
    clean_gate=clean_rollout_gate(run,ext,factory) if not smoke_path.exists() else None
    unet=copy.deepcopy(ext.unet)
    expected_targets=adapter_targets(unet)
    source=Path(c['source_run'])/'teacher_fp/trajectory/checkpoint-100'
    adapter=LoraConfig.from_pretrained(source)
    if set(adapter.target_modules)!=set(expected_targets):
        raise RuntimeError('Source attention/non-grouped Conv2d target scope mismatch')
    load_adapter(unet,source)
    loaded=get_peft_model_state_dict(unet);original=load_file(str(source/'adapter_model.safetensors'))
    if set(loaded)!=set(original) or any(not torch.equal(v.detach().cpu(),original[n]) for n,v in loaded.items()):
        raise RuntimeError('Warm-start adapter tensors are not exactly equal to checkpoint-100')
    for n,p in unet.named_parameters():p.requires_grad_('lora_' in n)
    unet.train();unet.enable_gradient_checkpointing()
    named=[(n,p) for n,p in unet.named_parameters() if p.requires_grad];params=[p for _,p in named]
    if not params:raise RuntimeError('No trainable LoRA parameters')
    source_state=verified_state(source)
    if set(source_state['parameters'])!={n for n,_ in named} or any(
            not torch.equal(p.detach().cpu(),source_state['parameters'][n]) for n,p in named):
        raise RuntimeError('Warm-start trainable parameters differ from source checkpoint state')
    del source_state,loaded,original
    optimizer=torch.optim.AdamW(params,lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=.01)
    carrier=SemanticCarrier.load(run/'carrier','cuda');rows=read_json(run/'prompts.json')['train']
    rng=torch.Generator('cpu').manual_seed(c['seed']);order=[];offset=0;start=0
    if smoke_path.exists():
        smoke=read_json(smoke_path)
        if smoke['protocol']!=protocol or not smoke['passed']:raise RuntimeError('Smoke gate identity changed')
    else:
        probe_rng=torch.Generator('cpu').manual_seed(c['seed'])
        prefix_diff=prefix_current_weight_gate(unet,ext,factory,rows,c,params)
        gm,gf,stats=effective_batch(unet,ext,carrier,factory,rows[:4],
                    [run/'images/teacher_clean_train'/image_name(i) for i in range(4)],c,probe_rng,params,audit_tail=True)
        _,cap=capped_gradient(gm,gf)
        freeze(smoke_path,{'protocol':protocol,'passed':True,'clean_rollout':clean_gate,
                          'current_prefix_max_difference':prefix_diff,'tail_gradients':10,
                          'micro_batch':c['native_micro_batch'],'cap':cap,'losses':stats})
        del gm,gf,stats
    if checkpoint:
        state=restore_training_state(checkpoint,unet,optimizer,None,rng,protocol)
        start,order,offset=state['step'],state['order'],state['offset']
        align_loss_log(folder/'losses.jsonl',start,expected_hash=state['loss_prefix_sha256'])
    else:align_loss_log(folder/'losses.jsonl',0)
    if not 0<=start<=300:raise RuntimeError('Resume step outside fixed v3 budget')
    unused_counts=[0]*len(params)
    with (folder/'losses.jsonl').open('a',encoding='utf8') as log:
        for event in range(start,300):
            if offset+4>len(order):order=torch.randperm(5000,generator=rng).tolist();offset=0
            ids=order[offset:offset+4];offset+=4
            torch.cuda.reset_peak_memory_stats()
            optimizer.zero_grad(set_to_none=True)
            gm,gf,stats=effective_batch(unet,ext,carrier,factory,[rows[i] for i in ids],
                    [run/'images/teacher_clean_train'/image_name(i) for i in ids],c,rng,params)
            combined,cap=capped_gradient(gm,gf)
            if cap['update_applied']:
                for p,g in zip(params,combined):p.grad=g
                torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
                check_finite(*[p.grad for p in params]);optimizer.step();check_finite(*params)
            for i in stats['used_parameter_indices']:unused_counts[i]+=1
            entry={'step':event+1,'image_ids':[rows[i]['image_id'] for i in ids],
                   'protocol_sha256':sha256(folder/'protocol.json'),
                   'source_checkpoint_sha256':read_json(run/'source_lineage.json')['source_checkpoint_sha256'],
                   **stats,**cap,'effective_batch':4,'finite':True,
                   'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
                   'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3}
            log.write(json.dumps(entry,allow_nan=False)+'\n');log.flush()
            print(f"V3 event {event+1}/300: M={stats['L_M']:.6g} P={stats['L_P']:.6g} Q={stats['L_Q']:.6g} alpha={cap['alpha_F']:.5g}",flush=True)
            del gm,gf,combined
            # Checkpoint every event: a killed allocation repeats at most its uncommitted event.
            dest=folder/f'checkpoint-{event+1}'
            save_adapter(unet,dest,adapter)
            save_training_state(dest,unet,optimizer,None,rng,order,offset,event+1,protocol)
            # Keep scientific milestones and the most recent full state; remove only own committed internals.
            if event and event not in (50,100,200,300):
                previous=folder/f'checkpoint-{event}'
                if previous.exists():
                    verified_state(previous,protocol)
                    import shutil
                    shutil.rmtree(previous)
    logs=[json.loads(line) for line in (folder/'losses.jsonl').read_text().splitlines()]
    freeze(folder/'trained.json',{'protocol':protocol,'events':300,'applied_updates':sum(r['update_applied'] for r in logs),
             'skipped_zero_margin_gradient':sum(not r['update_applied'] for r in logs),
             'losses_sha256':sha256(folder/'losses.jsonl'),
             'unused_parameters_this_invocation':[n for (n,_),count in zip(named,unused_counts) if count==0]})
    del unet,ext,optimizer,carrier;release()
