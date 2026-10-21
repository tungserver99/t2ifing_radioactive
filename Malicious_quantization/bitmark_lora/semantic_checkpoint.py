"""Audited teacher state checkpoints and explicit legacy warm-start lineage."""
import datetime
import hashlib
import json
import os
import random
import shutil
from pathlib import Path
import torch
from common import read_json,write_json,sha256


def cpu_tree(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value


def save_training_state(folder,model,optimizer,scaler,rng,order,offset,step,identity):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    state={'format':1,'step':step,'identity':identity,
           'parameters':{n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad},
           'optimizer':cpu_tree(optimizer.state_dict()),'scaler':scaler.state_dict() if scaler else None,
           'data_rng':rng.get_state(),'order':order,'offset':offset,
           'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
           'python_rng':random.getstate(),
           'loss_prefix_sha256':sha256(folder.parent/'losses.jsonl') if (folder.parent/'losses.jsonl').exists() else None}
    target=folder/'training_state.pt';temporary=folder/'training_state.partial'
    torch.save(state,temporary);temporary.replace(target)
    files={'training_state.pt':sha256(target)}
    for name in ('adapter_model.safetensors','adapter_config.json'):
        if (folder/name).exists():files[name]=sha256(folder/name)
    # Written last. Incomplete folders are not considered resumable checkpoints.
    write_json(folder/'checkpoint.json',{'step':step,'identity':identity,'files':files})


def verified_state(folder,identity=None):
    folder=Path(folder);proof=read_json(folder/'checkpoint.json')
    if identity is not None and proof['identity']!=identity:raise RuntimeError('Checkpoint training identity changed')
    for name,digest in proof['files'].items():
        if sha256(folder/name)!=digest:raise RuntimeError(f'Checkpoint artifact changed: {name}')
    state=torch.load(folder/'training_state.pt',map_location='cpu',weights_only=True)
    if folder.name.startswith('checkpoint-') and folder.name[11:].isdigit() and int(folder.name[11:])!=proof['step']:
        raise RuntimeError('Checkpoint directory/step mismatch')
    if state['format']!=1 or state['identity']!=proof['identity'] or state['step']!=proof['step']:
        raise RuntimeError('Checkpoint state/manifest mismatch')
    return state


def restore_training_state(folder,model,optimizer,scaler,rng,identity):
    state=verified_state(folder,identity)
    parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
    if set(parameters)!=set(state['parameters']):raise RuntimeError('Checkpoint trainable parameter names changed')
    if any(parameters[n].shape!=v.shape or not torch.isfinite(v).all() for n,v in state['parameters'].items()):
        raise RuntimeError('Invalid checkpoint parameters')
    if (scaler is None)!=(state['scaler'] is None):raise RuntimeError('Checkpoint precision/scaler changed')
    with torch.no_grad():
        for name,p in parameters.items():p.copy_(state['parameters'][name]);p.grad=None
    optimizer.load_state_dict(state['optimizer'])
    if scaler:scaler.load_state_dict(state['scaler'])
    rng.set_state(state['data_rng']);torch.set_rng_state(state['torch_rng']);random.setstate(state['python_rng'])
    if state['cuda_rng']:
        if not torch.cuda.is_available() or len(state['cuda_rng'])!=torch.cuda.device_count():
            raise RuntimeError('Checkpoint CUDA RNG device topology changed')
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    return {k:state[k] for k in ('step','order','offset','loss_prefix_sha256')}


def latest_full_checkpoint(folder):
    paths=[p for p in Path(folder).glob('checkpoint-*') if p.name[11:].isdigit() and (p/'checkpoint.json').exists()]
    if not paths:return None
    result=max(paths,key=lambda p:int(p.name[11:]))
    verified_state(result)
    return result


def advance_sampler(rng,count,batch,shape,timesteps,steps):
    order=[];offset=0
    for _ in range(steps):
        if offset+batch>len(order):order=torch.randperm(count,generator=rng).tolist();offset=0
        offset+=batch
        torch.randn(shape,generator=rng);torch.randint(0,timesteps,(batch,),generator=rng)
    return order,offset


def align_loss_log(path,step,source=None,expected_hash=None):
    path=Path(path);source=Path(source) if source else path
    if not source.exists():
        if step:raise RuntimeError('Missing loss log for saved teacher checkpoint')
        return
    original=source.read_bytes();kept=[]
    for line in original.splitlines(keepends=True):
        try:row=json.loads(line)
        except json.JSONDecodeError:break
        if row['step']<=step:kept.append(line)
    if [json.loads(v)['step'] for v in kept]!=list(range(1,step+1)):
        raise RuntimeError('Teacher loss log is not contiguous up to checkpoint')
    text=b''.join(kept)
    if expected_hash is not None and hashlib.sha256(text).hexdigest()!=expected_hash:
        raise RuntimeError('Checkpoint loss log prefix changed')
    if path==source and original!=text:
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        shutil.copy2(path,path.with_name(f'losses.before_resume.{stamp}.jsonl'))
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.partial');temporary.write_bytes(text);temporary.replace(path)


def continuation_config(source,precision='bf16'):
    from semantic_teacher_train import latest_teacher_adapters,candidate_tag
    from semantic_threshold import candidate_parameter
    source=Path(source).resolve();c=read_json(source/'config.json')
    if c.get('watermark_method')!='semantic_fp':raise ValueError('Expected a semantic_fp source run')
    if (source/'teacher_fp/selected.json').exists():raise ValueError('Source teacher is already selected; use its original code to resume')
    targets={}
    for alpha,path in latest_teacher_adapters(source):
        if not path.name.startswith('checkpoint-'):raise ValueError('Continuation expects an unfinished teacher with numbered checkpoints')
        candidate=path.parent;proof=read_json(candidate/'protocol.json')
        for key,name in (('config_sha256','config.json'),('assets_sha256','assets.json'),('carrier_sha256','carrier/complete.json')):
            if proof[key]!=sha256(source/name):raise RuntimeError(f'Source protocol changed: {key}')
        if (path/'checkpoint.json').exists():verified_state(path)
        targets[candidate_tag(alpha,candidate_parameter(c))]={'path':str(path),'step':int(path.name[11:]),
            'adapter_sha256':sha256(path/'adapter_model.safetensors'),
            'adapter_config_sha256':sha256(path/'adapter_config.json'),
            'loss_log':str(candidate/'losses.jsonl'),'loss_log_sha256':sha256(candidate/'losses.jsonl'),
            'optimizer_restored':False}
    return {**c,'teacher_precision':precision,'teacher_checkpoint_every':min(c['teacher_checkpoint_every'],100),
            'continuation':{'source':str(source),'source_config_sha256':sha256(source/'config.json'),
                            'checkpoints':targets,'mode':'weights-only warm start; fresh optimizer; original step schedule'}}


def prepare_continuation(run,config):
    """Fork prep artifacts; immutable data uses hard links, with copy fallback."""
    from semantic_carrier import SemanticCarrier
    from semantic_report import freeze_null
    if config.get('teacher_objective')=='semantic_sign_v1':
        return prepare_sign_reuse(run,config)
    lineage=config.get('preparation_reuse') or config['continuation']
    source=Path(lineage['source']);run=Path(run)
    if sha256(source/'config.json')!=lineage['source_config_sha256']:
        raise RuntimeError('Continuation source config changed')
    def copy_file(src,dst):
        src,dst=Path(src),Path(dst)
        if dst.exists():
            if sha256(src)!=sha256(dst):raise RuntimeError(f'Continuation artifact changed: {dst}')
            return
        dst.parent.mkdir(parents=True,exist_ok=True)
        try:os.link(src,dst)
        except OSError:shutil.copy2(src,dst)
    for name in ('assets.json','prompts.json','carrier','scores/teacher_clean_calibration.json','result/threshold_teacher.json',
                 'images/teacher_clean_train','images/teacher_clean_validation','images/teacher_clean_calibration'):
        src=source/name;dst=run/name
        if src.is_dir():shutil.copytree(src,dst,copy_function=copy_file,dirs_exist_ok=True)
        else:copy_file(src,dst)
    assets=read_json(run/'assets.json')
    if sha256(run/'prompts.json')!=assets['prompt_sha256']:raise RuntimeError('Source prompt hash changed')
    for label in ('teacher_clean_train','teacher_clean_validation','teacher_clean_calibration'):
        directory=run/'images'/label
        for row in read_json(directory/'image_hashes.json'):
            if sha256(directory/row['file_name'])!=row['sha256']:raise RuntimeError(f'Source image changed: {label}')
    SemanticCarrier.load(run/'carrier');freeze_null(run,'teacher')
    write_json(run/('preparation_reuse.json' if config.get('preparation_reuse') else 'continuation.json'),lineage)


def reuse_preparation_config(config,source):
    """Only reuse clean data/carrier/null; never import old fingerprint weights or losses."""
    source=Path(source).resolve();prior=read_json(source/'config.json')
    if prior.get('watermark_method')!='semantic_fp':raise ValueError('Expected semantic_fp preparation source')
    if prior.get('teacher_batch',4)!=config['teacher_batch']:
        raise ValueError('Preparation generation batch differs from new teacher effective batch; use fresh preparation')
    if config.get('teacher_objective')=='semantic_sign_v1':
        fields=('seed','key_seed','train_n','validation_n','calibration_n','test_n','fpr',
                'teacher_snapshot','student_snapshot','cache','inference_steps','score_batch')
        result={**config,**{k:prior[k] for k in fields if k in prior},
                'preparation_reuse':{'source':str(source),'source_config_sha256':sha256(source/'config.json'),
                                     'mode':'verified clean images/prompts only; rebuild 32-bit carrier/key/medians/scores; clean SD2.1 teacher initialization'}}
        if result['inference_steps']!=config['inference_steps']:raise ValueError('Clean preparation inference schedule differs from RSSF configuration')
        if prior.get('carrier_pc_min',8)!=8 or prior.get('carrier_pc_max',64)!=64:raise ValueError('Clean preparation must use the same carrier PC range')
        return result
    # These define the actual reused clean data, carrier, key and null.
    fields=('seed','key_seed','num_bits','carrier_pc_min','carrier_pc_max','stable_candidate_count',
            'train_n','validation_n','calibration_n','test_n','fpr','teacher_snapshot','student_snapshot',
            'cache','inference_steps','score_batch')
    result={**config,**{k:prior[k] for k in fields if k in prior},
            'preparation_reuse':{'source':str(source),'source_config_sha256':sha256(source/'config.json'),
                                 'mode':'clean preparation only; all fingerprint teachers start from clean SD2.1'}}
    if result['fpr']!=.01:raise ValueError('Threshold objective requires the clean 1%-FPR calibration')
    if result['num_bits']!=8:raise ValueError('Threshold rescue protocol requires the existing 8-bit carrier')
    return result



def prepare_sign_reuse(run,config):
    """Only immutable clean images/prompts/assets; old keys, scores and weights never copied."""
    from common import image_name
    source=Path(config['preparation_reuse']['source']);run=Path(run)
    if sha256(source/'config.json')!=config['preparation_reuse']['source_config_sha256']:
        raise RuntimeError('RSSF preparation source config changed')
    def copy_file(src,dst):
        src,dst=Path(src),Path(dst)
        if dst.exists():
            if sha256(src)!=sha256(dst):raise RuntimeError(f'RSSF reused clean artifact changed: {dst}')
            return str(dst)
        dst.parent.mkdir(parents=True,exist_ok=True)
        try:os.link(src,dst)
        except OSError:shutil.copy2(src,dst)
        return str(dst)
    for name in ('assets.json','prompts.json','images/teacher_clean_train','images/teacher_clean_validation','images/teacher_clean_calibration'):
        src=source/name;dst=run/name
        if src.is_dir():shutil.copytree(src,dst,copy_function=copy_file,dirs_exist_ok=True)
        else:copy_file(src,dst)
    assets=read_json(run/'assets.json');prompts=read_json(run/'prompts.json')
    if sha256(run/'prompts.json')!=assets['prompt_sha256']:raise RuntimeError('RSSF reused prompt hash changed')
    for split in ('train','validation','calibration'):
        directory=run/'images'/f'teacher_clean_{split}'
        expected=read_json(directory/'image_hashes.json');rows=prompts[split]
        if len(expected)!=len(rows) or len(rows)!=config[split+'_n']:
            raise RuntimeError('RSSF reused clean image count mismatch')
        if [r['file_name'] for r in expected]!=[image_name(i) for i in range(len(rows))]:raise RuntimeError('RSSF reused clean image ordering changed')
        if any(sha256(directory/r['file_name'])!=r['sha256'] for r in expected):raise RuntimeError('RSSF reused clean image changed')
        generation=read_json(directory/'generation.json')
        if generation['cfg']!=7.5 or generation['resolution']!=512 or generation['steps']!=config['inference_steps']:
            raise RuntimeError('RSSF reused preprocessing/generation schedule mismatch')
    write_json(run/'preparation_reuse.json',config['preparation_reuse'])
