"""Teacher-only RSSF v2: native anchors, four-call tail, native checkpoint selection."""
import argparse
import copy
import csv
import gc
import io
import os
import tempfile
import json
import math
import random
import statistics
from pathlib import Path
import torch
from common import freeze,read_json,sha256,write_json,image_name,verify_vendor
from semantic_carrier import SemanticCarrier,check_finite
from semantic_trajectory import (TraceReader,native_scheduler,restore_scheduler,replay_tail,cfg_forward,
                                 verify_replay,NativeTrace,digest)


def teacher_plan(c):
    return {'config':c,'students':[],'baseline':'paired clean pinned SD2.1','gpu_work':False,
            'teacher_objective':'semantic_sign_v2_trajectory',
            'stages':['assets','clean train PNG + native traces','clean validation','clean calibration',
                      'carrier','clean train/calibration/validation scoring','analytic verifier and null',
                      'exact clean tail replay gate','one teacher training run','native milestone screening',
                      'selected teacher merge','full teacher validation','trajectory drift','teacher-only report'],
            'teacher_loss':'guided native-anchor MSE + 0.8*final-tail RSSF hinge + LPIPS + pixel MSE',
            'selection':'TPR >= .90 and CLIP drop <= .01; min paired LPIPS, pixel MSE, earlier step',
            'success':'full native 512 validation: TPR >= .90, CLIP drop <= .01, LPIPS <= .15',
            'execute':'bash run_semantic_fp.sh','downstream_started':False}


def choose_checkpoint(rows):
    eligible=[]
    for row in rows:
        m,q=row['metrics'],row['quality']
        values=[m['detection_rate'],q['clip_drop'],q['paired_lpips'],q['paired_pixel_mse']]
        if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite teacher screening metric')
        if m['detection_rate']>=.9 and q['clip_drop']<=.01:eligible.append(row)
    return min(eligible,key=lambda r:(r['quality']['paired_lpips'],r['quality']['paired_pixel_mse'],r['step'])) if eligible else None


def trajectory_backward(unet,ext,carrier,scheduler_factory,perceptual,records,text,empty,clean_images,clean_r,anchors,c,backward,diagnostic=None):
    from semantic_sign_train import decoded_image,decoded_coordinates,_checkpoint
    from semantic_sign import sign_hinge_loss,image_bits
    # One graph at a time: preservation backward first, then four-call image loss.
    aggregate={};details=[];batch=len(records)
    if len(text)!=batch or clean_images.requires_grad or clean_r.requires_grad:raise ValueError('Detached, paired v2 targets required')
    for i,record in enumerate(records):
        anchor=int(anchors[i]);scheduler=restore_scheduler(scheduler_factory(),record,text.device,torch.float32)
        latent=record['anchor_latents'][anchor:anchor+1].to(text.device,dtype=torch.float32).detach()
        target=record['anchor_predictions'][anchor:anchor+1].to(text.device,dtype=torch.float32).detach()
        t=torch.tensor(record['anchor_timesteps'][anchor],device=text.device)
        guided=cfg_forward(unet,scheduler,latent,t,text[i:i+1],empty[i:i+1],c['teacher_precision'])
        preserve=(guided.float()-target).square().mean()
        row={'anchor_index':record['anchor_indices'][anchor],'anchor_timestep':int(t),
             'preserve':preserve.item(),'guided_fp_norm':guided.float().norm().item(),'guided_clean_norm':target.norm().item()}
        loss=c['lambda_preserve']*preserve/batch;check_finite(loss);backward(loss)
        del loss,guided,preserve,latent,target
        scheduler=restore_scheduler(scheduler_factory(),record,text.device,torch.float32)
        start_counter=scheduler.counter
        final=replay_tail(scheduler,record,lambda z,t:cfg_forward(unet,scheduler,z,t,text[i:i+1],empty[i:i+1],c['teacher_precision']),dtype=torch.float32)
        image=decoded_image(ext,final)
        coords=decoded_coordinates(ext,carrier,image,text[i:i+1]);check_finite(coords)
        fp,extra=sign_hinge_loss(clean_r[i:i+1].detach(),coords,carrier.signs,c['coordinate_margin_mu'],c['sign_k_train'])
        clean=clean_images[i:i+1].detach()
        lp=_checkpoint(lambda a,b:perceptual(a.float(),b.float()).mean(),image,clean)
        pixel=(image-clean).square().mean();quality=c['lambda_lpips']*lp+c['lambda_pixel']*pixel
        loss=(c['lambda_fp']*fp+c['lambda_quality']*quality)/batch
        check_finite(loss,lp,pixel,fp,final)
        # Selection is solely from final clean PNG coordinates, never from tail outputs.
        selected=(clean_r[i:i+1]*carrier.signs).topk(c['sign_k_train'],dim=-1).indices.detach()
        margins=(coords*carrier.signs).gather(-1,selected)
        row.update(boundary_call_index=record['boundary_index'],tail_timesteps=record['protocol']['timesteps'][-4:],
                   scheduler_start_counter=start_counter,scheduler_end_counter=scheduler.counter,final_latent_norm=final.norm().item(),
                   K_train=c['sign_k_train'],mu=c['coordinate_margin_mu'],
                   selected_clean_correct=int((image_bits(clean_r[i:i+1])==carrier.signs).gather(-1,selected).sum()),
                   final_K=int((image_bits(coords)==carrier.signs).sum()),
                   selected_margin_fraction=float((margins>=c['coordinate_margin_mu']).float().mean()),
                   mean_selected_signed_margin=margins.mean().item(),fp=fp.item(),lpips=lp.item(),pixel_mse=pixel.item(),quality=quality.item())
        row['loss']=c['lambda_preserve']*row['preserve']+c['lambda_fp']*row['fp']+c['lambda_quality']*row['quality']
        if diagnostic is not None:diagnostic['microbatch']=row
        backward(loss);details.append(row)
        for key in ('preserve','fp','lpips','pixel_mse','quality','loss'):
            aggregate[key]=aggregate.get(key,0.)+row[key]/batch
        del loss,fp,lp,pixel,quality,final,image,coords,margins
    aggregate['samples']=details
    return aggregate


def training_protocol(run,reader):
    c=read_json(run/'config.json');a=read_json(run/'assets.json')
    return {'teacher_objective':'semantic_sign_v2_trajectory',
            'config_sha256':sha256(run/'config.json'),'assets_sha256':sha256(run/'assets.json'),
            'carrier_sha256':sha256(run/'carrier/complete.json'),'key_sha256':sha256(run/'carrier/key.json'),
            'code_manifest_sha256':sha256(run/'code_manifest.json'),
            'trace_sha256':sha256(run/'teacher_trace/complete.json'),'trace_protocol':reader.protocol,
            'training_images_sha256':sha256(run/'images/teacher_clean_train/image_hashes.json'),
            'clean_scores_sha256':sha256(run/'scores/teacher_clean_train.json'),
            'scheduler_config_sha256':sha256(Path(a['teacher_clean'])/'scheduler/scheduler_config.json'),
            'batch':c['teacher_batch'],'micro_batch':1,'teacher_precision':c['teacher_precision'],
            'milestones':c['teacher_eval_checkpoints'],'K_train':24,'mu':.10}


def train_trajectory(run):
    from peft import LoraConfig
    import lpips
    from semantic_teacher import extractor,pixels,numerical_policy
    from semantic_teacher_train import adapter_targets,save_adapter,checked_teacher_step
    from semantic_checkpoint import latest_full_checkpoint,restore_training_state,save_training_state,align_loss_log,verified_state
    from semantic_report import score_rows
    verify_vendor();numerical_policy();c=read_json(run/'config.json');assets=read_json(run/'assets.json')
    reader=TraceReader(run/'teacher_trace');protocol=training_protocol(run,reader)
    folder=run/'teacher_fp/trajectory';folder.mkdir(parents=True,exist_ok=True);freeze(folder/'protocol.json',protocol)
    if (folder/'trained.json').exists():
        proof=read_json(folder/'trained.json')
        for name,value in proof['files'].items():
            if sha256(folder/name)!=value:raise RuntimeError('Trained trajectory artifact changed')
        for step in c['teacher_eval_checkpoints']:verified_state(folder/f'checkpoint-{step}',protocol)
        return
    if not torch.cuda.is_bf16_supported():raise RuntimeError('Locked rescue requires BF16 training support')
    torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed']);random.seed(c['seed'])
    ext=extractor(run);unet=copy.deepcopy(ext.unet).train()
    adapter=LoraConfig(r=c['teacher_rank'],lora_alpha=c['teacher_rank'],init_lora_weights='gaussian',target_modules=adapter_targets(unet))
    unet.add_adapter(adapter);unet.enable_gradient_checkpointing()
    named=[(n,p) for n,p in unet.named_parameters() if p.requires_grad]
    if not named:raise RuntimeError('No trainable teacher LoRA')
    optimizer=torch.optim.AdamW([p for _,p in named],lr=c['teacher_lr']);scaler=None
    perceptual=lpips.LPIPS(net='alex',version='0.1').cuda().float().eval().requires_grad_(False)
    carrier=SemanticCarrier.load(run/'carrier','cuda')
    if carrier.carrier_version!='semantic_sign_v1':raise ValueError('v2 requires the unchanged RSSF carrier')
    scores=score_rows(run,'teacher_clean_train',c['train_n'],carrier.to('cpu'))
    coordinates=torch.tensor([r['coordinates'] for r in scores],device='cuda').detach()
    rows=read_json(run/'prompts.json')['train'];rng=torch.Generator('cpu').manual_seed(c['seed']);order=[];offset=0;start=0
    checkpoint=latest_full_checkpoint(folder)
    if checkpoint:
        state=restore_training_state(checkpoint,unet,optimizer,scaler,rng,protocol)
        start,order,offset=state['step'],state['order'],state['offset']
        align_loss_log(folder/'losses.jsonl',start,expected_hash=state['loss_prefix_sha256'])
    else:align_loss_log(folder/'losses.jsonl',0)
    if not 0<=start<=c['teacher_train_steps']:raise ValueError('Checkpoint outside v2 training budget')
    factory=lambda:native_scheduler(assets['teacher_clean'])
    config={**c,'sign_k_train':24}
    with (folder/'losses.jsonl').open('a',encoding='utf8') as log:
        for step in range(start,c['teacher_train_steps']):
            if offset+c['teacher_batch']>len(order):order=torch.randperm(len(rows),generator=rng).tolist();offset=0
            ids=order[offset:offset+c['teacher_batch']];offset+=len(ids)
            records=[reader.get(i) for i in ids]
            anchors=[int(torch.randint(len(r['anchor_indices']),(1,),generator=rng)) for r in records]
            with torch.no_grad():
                text=ext.text([rows[i]['text'] for i in ids]);empty=ext.text(['']*len(ids))
                clean=pixels([run/'images/teacher_clean_train'/image_name(i) for i in ids]).detach()
            context={'phase':'trajectory_train','step':step+1,'image_ids':ids}
            torch.cuda.reset_peak_memory_stats()
            stats=checked_teacher_step(named,optimizer,scaler,
                lambda backward:trajectory_backward(unet,ext,carrier,factory,perceptual,records,text,empty,clean,coordinates[ids],anchors,config,backward,context),
                folder/'failure.json',context)
            entry={'step':step+1,**stats,'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,
                   'effective_batch':len(ids),'micro_batch':1,'teacher_precision':c['teacher_precision'],'finite':True}
            log.write(json.dumps(entry,allow_nan=False)+'\n');log.flush()
            if (step+1)%25==0:print(f'Trajectory teacher: {entry}',flush=True)
            if (step+1)%c['teacher_checkpoint_every']==0 or step+1 in c['teacher_eval_checkpoints'] or step+1==c['teacher_train_steps']:
                dest=folder/f'checkpoint-{step+1}';save_adapter(unet,dest,adapter)
                save_training_state(dest,unet,optimizer,scaler,rng,order,offset,step+1,protocol)
    freeze(folder/'trained.json',{'steps':c['teacher_train_steps'],'files':{'losses.jsonl':sha256(folder/'losses.jsonl')},'protocol_hash':digest(protocol)})
    del unet,ext,optimizer,perceptual,carrier;gc.collect();torch.cuda.empty_cache()


def _release():
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()


def screen_checkpoint(run,step):
    """Fuse in memory for screening; retain only the selected final merged model."""
    from semantic_teacher import load_native,generate_set,score
    from semantic_sign_report import analytic_metrics
    from semantic_report import teacher_quality
    from semantic_checkpoint import verified_state
    from semantic_assets import snapshot_identity
    from student import load_adapter
    c=read_json(run/'config.json');assets=read_json(run/'assets.json');folder=run/'teacher_fp/trajectory'/f'checkpoint-{step}'
    protocol=read_json(folder.parent/'protocol.json');verified_state(folder,protocol)
    label=f'trajectory_checkpoint_{step}_validation';target=folder/'screen.json'
    identity={'checkpoint_sha256':sha256(folder/'checkpoint.json'),'protocol_hash':digest(protocol),
              'clean_validation_sha256':sha256(run/'images/teacher_clean_validation/image_hashes.json'),
              'n':c['teacher_eval_n'],'subset':'first 128 frozen validation rows, same seeds'}
    model=assets['teacher_clean']
    # The immutable adapter is part of model identity even though source path is clean.
    model_identity={**assets['teacher_clean_hashes'],'fused_adapter_sha256':sha256(folder/'adapter_model.safetensors'),'fuse_precision':'float32'}
    if target.exists():
        record=read_json(target)
        if record['identity']!=identity:raise RuntimeError('Checkpoint screening identity changed')
        _verify_screen_images(run,label,record);return record
    # generate_set accepts explicit identity for the fused in-memory pipeline.
    pipe=load_native(model,dtype=torch.float32);load_adapter(pipe.unet,folder,pipe);pipe.fuse_lora(safe_fusing=True);pipe.unload_lora_weights();pipe.to(dtype=torch.float16);pipe.unet.eval()
    generate_set(run,model,'validation',label,pipe,limit=c['teacher_eval_n'],model_identity=model_identity)
    del pipe;_release()
    score(run,[label]);_release()
    metrics=analytic_metrics(run,label,c['teacher_eval_n']);quality=teacher_quality(run,label,limit=c['teacher_eval_n']);_release()
    values=read_json(run/'scores'/f'{label}.json');metrics['mean_score']=statistics.mean(r['score'] for r in values)
    record={'step':step,'metrics':metrics,'quality':quality,'identity':identity,
            'scores_sha256':sha256(run/'scores'/f'{label}.json'),
            'image_hashes_sha256':sha256(run/'images'/label/'image_hashes.json')}
    freeze(target,record);return record


def _verify_screen_images(run,label,record):
    from semantic_report import score_rows
    for key,path in [('scores_sha256',run/'scores'/f'{label}.json'),('image_hashes_sha256',run/'images'/label/'image_hashes.json')]:
        if sha256(path)!=record[key]:raise RuntimeError('Completed checkpoint screen changed')
    score_rows(run,label,record['identity']['n'])


def _write_drift_csv(path,table):
    """Recover the deterministic export from committed JSON, without inference."""
    fields=('call_index','mean_drift','median_drift')
    if not table or any(set(row)!=set(fields) or row['call_index']!=i or
                        any(not math.isfinite(row[key]) or row[key]<0 for key in fields[1:])
                        for i,row in enumerate(table)):
        raise ValueError('Invalid committed trajectory drift table')
    output=io.StringIO(newline='')
    writer=csv.DictWriter(output,fieldnames=fields,lineterminator='\n')
    writer.writeheader();writer.writerows(table)
    content=output.getvalue().encode('utf8');path=Path(path)
    if path.exists() and path.read_bytes()==content:return
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(dir=path.parent,prefix='.drift-csv-',suffix='.partial')
    try:
        with os.fdopen(fd,'wb') as stream:stream.write(content)
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


def trajectory_drift(run,model):
    from semantic_teacher import load_native,native_generate
    c=read_json(run/'config.json');assets=read_json(run/'assets.json');rows=read_json(run/'prompts.json')['validation'][:c['trajectory_drift_n']]
    identity={'selected_sha256':sha256(run/'teacher_fp/selected.json'),'prompt_sha256':sha256(run/'prompts.json'),'n':len(rows)}
    target=run/'result/teacher_trajectory_drift.json'
    if target.exists():
        saved=read_json(target)
        if saved['identity']!=identity:raise RuntimeError('Trajectory drift identity changed')
        _write_drift_csv(target.with_suffix('.csv'),saved['per_call'])
        return
    paths=[]
    for source in (assets['teacher_clean'],model):
        pipe=load_native(source);examples=[]
        for row in rows:
            seed=c['seed']*1000000+row['image_id']
            with NativeTrace(pipe.scheduler,all_latents=True) as trace:
                native_generate(pipe,[row['text']],[torch.Generator('cuda').manual_seed(seed)],c['inference_steps'])
            examples.append(torch.stack([v[0] for v in trace.path]).float())
        paths.append(examples);del pipe;_release()
    values=torch.stack([(b-a).flatten(1).square().mean(1).sqrt() for a,b in zip(*paths)])
    check_finite(values)
    table=[{'call_index':i,'mean_drift':values[:,i].mean().item(),'median_drift':statistics.median(values[:,i].tolist())} for i in range(values.shape[1])]
    freeze(target,{'identity':identity,'diagnostic_only':True,'normalization':'L2 / sqrt(numel latent)','per_call':table})
    _write_drift_csv(target.with_suffix('.csv'),table)


def finish_report(run,screen,chosen,full=None,reason=None):
    c=read_json(run/'config.json')
    passed=bool(full and full['metrics']['detection_rate']>=.9 and full['quality']['clip_drop']<=.01 and full['quality']['paired_lpips']<=.15)
    record={'status':'teacher_only_completed' if passed else 'teacher_only_failed','downstream_started':False,
            'teacher_objective':'semantic_sign_v2_trajectory','success':passed,'reason':reason,
            'gates':{'analytic_tpr_min':.9,'clip_drop_max':.01,'paired_lpips_max':.15},
            'screening':screen,'selected':chosen,'full_validation':full,
            'clean_null':read_json(run/'result/fixed_key_clean_fpr.json'),
            'replay':read_json(run/'teacher_trace/replay.json'),
            'config_sha256':sha256(run/'config.json'),'trace_sha256':sha256(run/'teacher_trace/complete.json')}
    freeze(run/'result/teacher_only.json',record)
    lines=['# RSSF v2 teacher-only rescue','',record['status'],'',
           'No downstream jobs started. Selection uses native screening; success uses full 512-image native validation.',
           'Frozen gates: analytic TPR >= 0.90; CLIP drop <= 0.01; paired LPIPS <= 0.15.','',
           '| Step | Analytic TPR | CLIP drop | LPIPS | Pixel MSE |','|---:|---:|---:|---:|---:|']
    for row in screen:
        m,q=row['metrics'],row['quality'];lines.append(f"| {row['step']} | {m['detection_rate']:.4f} | {q['clip_drop']:.6f} | {q['paired_lpips']:.6f} | {q['paired_pixel_mse']:.6f} |")
    if full:
        m,q=full['metrics'],full['quality']
        lines+=['',f"Selected step: {chosen['step']}; full TPR={m['detection_rate']:.4f}, CLIP drop={q['clip_drop']:.6f}, LPIPS={q['paired_lpips']:.6f}, pixel MSE={q['paired_pixel_mse']:.6f}.",
                f"Model K_M={m['K_M']}/32, p_M={m['p_M']}; continuous mean={m['mean_score']:.6f}."]
    if reason:lines+=['',reason]
    lines+=['','See teacher_only.json, teacher_trajectory_drift.json/CSV and teacher_candidates/*_visuals.','']
    (run/'result/README.md').write_text('\n'.join(lines),encoding='utf8')
    freeze(run/'teacher_complete.json',record)


def campaign(run):
    from semantic_sign_report import freeze_analytic_verifier,validate_clean_null,analytic_metrics,paired_pixel_visuals
    from semantic_report import teacher_quality,freeze_null
    from semantic_teacher_train import merge_teacher
    from semantic_teacher import generate_set,score
    from semantic_assets import snapshot_identity
    c=read_json(run/'config.json')
    if c.get('teacher_objective')!='semantic_sign_v2_trajectory' or not c.get('teacher_only'):
        raise ValueError('This campaign is exclusively v2 teacher-only')
    freeze_analytic_verifier(run);freeze_null(run,'teacher');validate_clean_null(run,labels=('teacher_clean_calibration',))
    verify_replay(run);train_trajectory(run)
    screening=[screen_checkpoint(run,s) for s in c['teacher_eval_checkpoints']]
    freeze(run/'teacher_fp/checkpoint_screening.json',{'checkpoints':screening,'subset':'first 128 validation rows'})
    chosen=choose_checkpoint(screening)
    if chosen is None:
        finish_report(run,screening,None,reason='No checkpoint meets analytic TPR and CLIP eligibility. Students stopped.');return
    folder=run/'teacher_fp/trajectory'/f"checkpoint-{chosen['step']}"
    model=merge_teacher(run,folder)
    selected={**chosen,'model':str(model.relative_to(run)),'model_hashes':snapshot_identity(model),
              'teacher_objective':'semantic_sign_v2_trajectory','downstream_ready':False,
              'selection_rule':'TPR>=.90 and CLIP drop<=.01; lowest LPIPS, then pixel MSE, then earlier step',
              'config_sha256':sha256(run/'config.json'),'carrier_sha256':sha256(run/'carrier/complete.json')}
    freeze(run/'teacher_fp/selected.json',selected)
    generate_set(run,model,'validation','teacher_wm_validation');_release()
    score(run,['teacher_wm_validation']);_release()
    m=analytic_metrics(run,'teacher_wm_validation',c['validation_n'])
    m['mean_score']=statistics.mean(r['score'] for r in read_json(run/'scores/teacher_wm_validation.json'))
    q=teacher_quality(run,'teacher_wm_validation');_release()
    visuals=paired_pixel_visuals(run,'teacher_clean_validation','teacher_wm_validation',c['validation_n'],max_visuals=16)
    freeze(run/'result/selected_teacher_visuals.json',visuals)
    full={'metrics':m,'quality':q};freeze(run/'result/teacher_quality.json',full)
    trajectory_drift(run,model)
    reason=None if m['detection_rate']>=.9 and q['clip_drop']<=.01 and q['paired_lpips']<=.15 else 'Selected teacher failed the frozen full-validation utility/detection gate. Students stopped.'
    finish_report(run,screening,selected,full,reason)



def analyze_teacher_only(run):
    """Rebuild the teacher report from verified saved metrics; no model execution."""
    from semantic_report import score_rows
    from semantic_assets import snapshot_identity
    from semantic_sign_report import analytic_metrics
    record=read_json(run/'teacher_complete.json');c=read_json(run/'config.json')
    if record['config_sha256']!=sha256(run/'config.json'):raise RuntimeError('Teacher report config changed')
    TraceReader(run/'teacher_trace');SemanticCarrier.load(run/'carrier')
    if record['trace_sha256']!=sha256(run/'teacher_trace/complete.json'):raise RuntimeError('Teacher report traces changed')
    for split in ('train','calibration','validation'):
        score_rows(run,'teacher_clean_'+split,c[split+'_n'])
    for row in record['screening']:_verify_screen_images(run,f"trajectory_checkpoint_{row['step']}_validation",row)
    selected=record['selected'];full=record['full_validation']
    if selected:
        if snapshot_identity(run/selected['model'])!=selected['model_hashes']:raise RuntimeError('Selected teacher changed')
        actual=analytic_metrics(run,'teacher_wm_validation',c['validation_n'])
        if any(actual[k]!=v for k,v in full['metrics'].items() if k!='mean_score'):raise RuntimeError('Full teacher evidence changed')
    finish_report(run,record['screening'],selected,full,record['reason'])
    print(f"{record['status']}: {run/'result/README.md'}")

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    campaign(parser.parse_args().run.resolve())
