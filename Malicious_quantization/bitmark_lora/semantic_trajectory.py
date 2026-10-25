"""Observational native PNDM traces and exact differentiable four-call replay.

Only declarative JSON + safetensors are stored. The native scheduler performs
all denoising algebra; this module never substitutes a diffusion equation.
"""
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from common import freeze, read_json, sha256, write_json
from semantic_carrier import check_finite

STATE_FIELDS = ('counter','ets','cur_model_output','cur_sample','num_inference_steps',
                'timesteps','_timesteps','prk_timesteps','plms_timesteps')
REPLAY_TOLERANCES = dict(latent_atol=2e-3, latent_rtol=2e-3,
                         png_pixel_max=2/255+2e-3, lpips_max=1e-4,
                         anchor_atol=2e-3, anchor_rtol=2e-3)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def config_dict(scheduler):
    # Diffusers uses FrozenDict; private constructor hints do not affect step().
    return {k:v for k,v in dict(scheduler.config).items() if not k.startswith('_')}


def validate_pndm(scheduler):
    from diffusers import PNDMScheduler
    if type(scheduler) is not PNDMScheduler:
        raise ValueError('Trajectory rescue supports only the pinned PNDMScheduler')
    if not scheduler.config.skip_prk_steps or scheduler.config.prediction_type!='epsilon':
        raise ValueError('Trajectory rescue requires skip_prk_steps=true and prediction_type=epsilon')
    if any(not hasattr(scheduler,k) for k in STATE_FIELDS):
        raise ValueError('Installed PNDM mutable state is incompatible with audited serializer')
    # A changed implementation must be audited, not silently approximated.
    import diffusers
    if diffusers.__version__!='0.35.1':
        raise ValueError('Trajectory serializer audited for diffusers==0.35.1 only')


def clone_tree(value,device='cpu',dtype=None):
    if isinstance(value,torch.Tensor):
        check_finite(value)
        return value.detach().to(device=device,dtype=dtype if value.is_floating_point() and dtype else value.dtype).clone()
    if isinstance(value,np.ndarray):return value.copy()
    if isinstance(value,list):return [clone_tree(v,device,dtype) for v in value]
    if isinstance(value,dict):return {k:clone_tree(v,device,dtype) for k,v in value.items()}
    if value is None or isinstance(value,(int,float,bool,str)):return value
    raise TypeError(f'Unsupported scheduler state type: {type(value).__name__}')


def scheduler_state(scheduler):
    validate_pndm(scheduler)
    return {k:clone_tree(getattr(scheduler,k)) for k in STATE_FIELDS}


def scheduler_protocol(scheduler):
    validate_pndm(scheduler)
    return {'scheduler_class':type(scheduler).__name__,'scheduler_config':config_dict(scheduler),
            'scheduler_config_hash':digest(config_dict(scheduler)),
            'timesteps':[int(t) for t in scheduler.timesteps],
            'num_inference_steps':scheduler.num_inference_steps}


def cfg_forward(unet,scheduler,latent,t,text,empty,precision=None):
    duplicated=torch.cat([latent,latent])
    model_input=scheduler.scale_model_input(duplicated,t)
    hidden=torch.cat([empty,text])
    enabled=precision in {'bf16','fp16'}
    dtype=torch.bfloat16 if precision=='bf16' else torch.float16
    with torch.autocast(latent.device.type,dtype=dtype,enabled=enabled):
        result=unet(model_input,t,encoder_hidden_states=hidden).sample
    u,c=result.chunk(2)
    guided=u+7.5*(c-u)
    check_finite(guided)
    return guided


def _sample_state(state,index,batch):
    # Scheduler history tensors have sample layout; timestep arrays are protocol.
    result=clone_tree(state)
    for field in ('ets','cur_sample','cur_model_output'):
        def split(v):
            if isinstance(v,list):return [split(x) for x in v]
            if isinstance(v,torch.Tensor):
                if v.ndim<1 or v.shape[0]!=batch:raise ValueError('Unexpected PNDM history tensor shape')
                return v[index:index+1].clone()
            return v
        result[field]=split(state[field])
    return result


class NativeTrace:
    """Wrap step once, copy inputs before the call, return its original output."""
    def __init__(self,scheduler,tail_calls=4,anchor_count=5,all_latents=False):
        validate_pndm(scheduler)
        if tail_calls!=4 or anchor_count!=5:raise ValueError('Locked trace uses four tail calls and five anchors')
        self.scheduler=scheduler;self.tail_calls=tail_calls;self.all_latents=all_latents
        self.calls=0;self.anchors=[];self.tail_outputs=[];self.path=[]

    def __enter__(self):
        self.original=self.scheduler.step
        self.had_instance_step='step' in self.scheduler.__dict__
        self.instance_step=self.scheduler.__dict__.get('step')
        self.scheduler.step=self.step
        return self

    def __exit__(self,*exc):
        if self.had_instance_step:self.scheduler.step=self.instance_step
        else:del self.scheduler.__dict__['step']

    def step(self,model_output,timestep,sample,*args,**kwargs):
        i=self.calls
        if i==0:
            self.protocol=scheduler_protocol(self.scheduler)
            self.count=len(self.scheduler.timesteps);self.boundary=self.count-self.tail_calls
            if self.boundary<4:raise ValueError('PNDM trajectory too short for populated PLMS boundary')
            self.anchor_ids=list(dict.fromkeys(round(f*(self.boundary-1)) for f in (0,.25,.5,.75,1)))
        if int(timestep)!=self.protocol['timesteps'][i]:raise ValueError('Native scheduler call sequence changed')
        if self.all_latents:self.path.append(clone_tree(sample))
        if i in self.anchor_ids:
            self.anchors.append((i,int(timestep),clone_tree(sample),clone_tree(model_output)))
        if i==self.boundary:
            self.boundary_latent=clone_tree(sample);self.boundary_state=scheduler_state(self.scheduler)
        if i>=self.boundary:self.tail_outputs.append(clone_tree(model_output))
        result=self.original(model_output,timestep,sample,*args,**kwargs)
        self.calls+=1
        if self.calls==self.count:
            final=result.prev_sample if hasattr(result,'prev_sample') else result[0]
            self.final_latent=clone_tree(final);self.final_state=scheduler_state(self.scheduler)
        return result

    def records(self):
        if self.calls!=self.count:raise ValueError('Incomplete native trace')
        batch=len(self.boundary_latent);rows=[]
        for j in range(batch):
            rows.append({'protocol':self.protocol,'call_count':self.count,'boundary_index':self.boundary,
                         'anchor_indices':self.anchor_ids,'anchor_timesteps':[a[1] for a in self.anchors],
                         'anchor_latents':torch.stack([a[2][j] for a in self.anchors]),
                         'anchor_predictions':torch.stack([a[3][j] for a in self.anchors]),
                         'boundary_latent':self.boundary_latent[j:j+1].clone(),
                         'boundary_state':_sample_state(self.boundary_state,j,batch),
                         'tail_outputs':torch.stack([v[j] for v in self.tail_outputs]),
                         'final_latent':self.final_latent[j:j+1].clone(),
                         'final_state':_sample_state(self.final_state,j,batch)})
        return rows


def restore_scheduler(scheduler,record,device,dtype=None):
    validate_pndm(scheduler)
    p=record['protocol']
    scheduler.set_timesteps(p['num_inference_steps'],device=device)
    if scheduler_protocol(scheduler)!=p:raise ValueError('Trace scheduler config/timestep protocol mismatch')
    state=record['boundary_state'];boundary=record['boundary_index']
    if set(state)!=set(STATE_FIELDS) or state['counter']!=boundary or boundary!=len(p['timesteps'])-4:
        raise ValueError('Trace boundary counter/state mismatch')
    if len(state['ets'])<3 or not scheduler.config.skip_prk_steps or boundary<len(scheduler.prk_timesteps):
        raise ValueError('Trace boundary is not populated PLMS history')
    latent=record['boundary_latent']
    for history in state['ets']:
        if history.shape!=latent.shape or history.dtype!=latent.dtype:raise ValueError('PNDM history shape/dtype mismatch')
    if state['timesteps'].tolist()!=p['timesteps'] or state['num_inference_steps']!=p['num_inference_steps']:
        raise ValueError('Serialized scheduler timestep sequence changed')
    for name,value in clone_tree(state,device,dtype).items():setattr(scheduler,name,value)
    return scheduler


def replay_tail(scheduler,record,predict,dtype=None):
    # Only cached constants are detached. New predictions/history stay in graph.
    device=scheduler.timesteps.device
    latent=clone_tree(record['boundary_latent'],device,dtype)
    for t in scheduler.timesteps[record['boundary_index']:]:
        output=predict(latent,t);check_finite(output)
        latent=scheduler.step(output,t,latent).prev_sample
        check_finite(latent)
    if scheduler.counter!=record['call_count']:raise ValueError('Tail scheduler counter progression changed')
    return latent


def pack_tree(value,tensors,prefix):
    if isinstance(value,torch.Tensor):
        tensors[prefix]=value.detach().cpu().contiguous().clone()
        return {'tensor':prefix,'shape':list(value.shape),'dtype':str(value.dtype)}
    if isinstance(value,np.ndarray):
        return {'ndarray':pack_tree(torch.from_numpy(value.copy()),tensors,prefix)}
    if isinstance(value,dict):return {'dict':{k:pack_tree(v,tensors,prefix+'.'+k) for k,v in value.items()}}
    if isinstance(value,list):return {'list':[pack_tree(v,tensors,prefix+f'.{i}') for i,v in enumerate(value)]}
    if value is None or isinstance(value,(int,float,bool,str)):return {'scalar':value}
    raise TypeError(type(value))


def unpack_tree(value,tensors):
    if 'tensor' in value:
        t=tensors[value['tensor']]
        if list(t.shape)!=value['shape'] or str(t.dtype)!=value['dtype']:raise ValueError('Trace tensor schema mismatch')
        check_finite(t);return t
    if 'ndarray' in value:return unpack_tree(value['ndarray'],tensors).numpy().copy()
    if 'dict' in value:return {k:unpack_tree(v,tensors) for k,v in value['dict'].items()}
    if 'list' in value:return [unpack_tree(v,tensors) for v in value['list']]
    return value['scalar']


class TraceWriter:
    def __init__(self,folder,protocol,shard_size=64):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        freeze(self.folder/'protocol.json',protocol);self.shard_size=shard_size;self.pending=[]
        self.index={};self.shards=[]
        for path in sorted(self.folder.glob('shard_*.json')):
            proof=read_json(path);tensor_path=path.with_suffix('.safetensors')
            if sha256(tensor_path)!=proof['tensor_sha256']:raise RuntimeError('Committed trace shard changed')
            if proof['protocol_sha256']!=sha256(self.folder/'protocol.json'):raise RuntimeError('Trace shard protocol changed')
            for key in proof['records']:
                if key in self.index:raise ValueError('Duplicate trace image ID')
                self.index[key]=path.name
            self.shards.append(path.name)

    def has(self,image_id):return str(image_id) in self.index

    def append(self,image_id,record,metadata):
        if self.has(image_id):raise ValueError('Trace image already committed')
        self.pending.append((str(image_id),record,metadata))
        if len(self.pending)>=self.shard_size:self.flush()

    def flush(self):
        if not self.pending:return
        from safetensors.torch import save_file
        stem=f'shard_{len(self.shards):05d}';tensors={};records={}
        for key,record,metadata in self.pending:
            records[key]={'state':pack_tree(record,tensors,'row_'+key),'metadata':metadata}
        target=self.folder/(stem+'.safetensors');tmp=self.folder/(stem+'.partial')
        save_file(tensors,str(tmp));tmp.replace(target)
        proof={'tensor_sha256':sha256(target),'protocol_sha256':sha256(self.folder/'protocol.json'),'records':records}
        freeze(self.folder/(stem+'.json'),proof)  # commit marker, written last
        for key in records:self.index[key]=stem+'.json'
        self.shards.append(stem+'.json');self.pending=[]

    def finish(self,count):
        self.flush()
        if set(self.index)!={str(i) for i in range(count)}:raise ValueError('Incomplete trace ID coverage')
        freeze(self.folder/'index.json',self.index)
        files=['protocol.json','index.json']+(['instrumentation.json'] if (self.folder/'instrumentation.json').exists() else [])+[n for s in self.shards for n in (s,s.replace('.json','.safetensors'))]
        freeze(self.folder/'complete.json',{'count':count,'files':{n:sha256(self.folder/n) for n in files}})


class TraceReader:
    def __init__(self,folder):
        self.folder=Path(folder);proof=read_json(self.folder/'complete.json')
        for name,expected in proof['files'].items():
            if sha256(self.folder/name)!=expected:raise RuntimeError(f'Trace artifact changed: {name}')
        self.index=read_json(self.folder/'index.json');self.protocol=read_json(self.folder/'protocol.json');self.cache=OrderedDict()
        if set(self.index)!={str(i) for i in range(proof['count'])}:raise ValueError('Trace index coverage changed')

    def get(self,image_id):
        name=self.index[str(image_id)]
        if name not in self.cache:
            from safetensors.torch import load_file
            meta=read_json(self.folder/name)
            self.cache[name]=(meta,load_file(str(self.folder/name.replace('.json','.safetensors'))))
            while len(self.cache)>2:self.cache.popitem(last=False)
        meta,tensors=self.cache[name];self.cache.move_to_end(name)
        return unpack_tree(meta['records'][str(image_id)]['state'],tensors)


def native_scheduler(model):
    from diffusers import PNDMScheduler,StableDiffusionPipeline
    index=read_json(Path(model)/'model_index.json')
    if index['scheduler']!=['diffusers','PNDMScheduler']:raise ValueError('Saved native scheduler must be Diffusers PNDM')
    scheduler=PNDMScheduler.from_pretrained(model,subfolder='scheduler',local_files_only=True)
    # Mirror the installed pipeline's config normalization, without model weights.
    scheduler=StableDiffusionPipeline(vae=None,text_encoder=None,tokenizer=None,unet=None,scheduler=scheduler,
                    safety_checker=None,feature_extractor=None,requires_safety_checker=False).scheduler
    validate_pndm(scheduler)
    return scheduler


def verify_replay(run):
    """Real FP16 native clean replay gate. Must pass before any optimization."""
    import gc
    import lpips
    from semantic_teacher import load_native,pixels
    from semantic_replay_diagnostics import comparison_check,image_diagnostics
    c=read_json(run/'config.json');assets=read_json(run/'assets.json');reader=TraceReader(run/'teacher_trace')
    identity={'trace_sha256':sha256(run/'teacher_trace/complete.json'),
              'clean_images_sha256':sha256(run/'images/teacher_clean_train/image_hashes.json'),
              'tolerances':REPLAY_TOLERANCES,'sample_ids':[0,1,2,3]}
    target=run/'teacher_trace/replay.json'
    if target.exists():
        old=read_json(target)
        if old['identity']!=identity or not old['passed']:raise RuntimeError('Clean replay proof changed')
        return True
    pipe=load_native(assets['teacher_clean']);perceptual=None
    try:
        return _measure_replay(run,c,assets,reader,identity,target,pipe,lpips,pixels,
                               comparison_check,image_diagnostics)
    finally:
        del pipe,perceptual;gc.collect();torch.cuda.empty_cache()


def _measure_replay(run,c,assets,reader,identity,target,pipe,lpips,pixels,comparison_check,image_diagnostics,device='cuda'):
    from semantic_replay_diagnostics import tensor_comparison
    perceptual=lpips.LPIPS(net='alex',version='0.1').cuda().eval().requires_grad_(False)
    prompts=read_json(run/'prompts.json')['train'];results=[]
    ids=identity['sample_ids']
    record=batch_records([reader.get(i) for i in ids])
    with torch.inference_mode():
        text=pipe._encode_prompt([prompts[i]['text'] for i in ids],device,1,True)
        empty,cond=text.chunk(2)
        scheduler=restore_scheduler(native_scheduler(assets['teacher_clean']),record,device)
        anchor_errors=[];checks=[]
        for j,t in enumerate(record['anchor_timesteps']):
            z=record['anchor_latents'][j].to(device);expected=record['anchor_predictions'][j].to(device)
            actual=cfg_forward(pipe.unet,scheduler,z,torch.tensor(t,device=device),cond,empty)
            checks.append(tensor_comparison(f'anchor_{j}',actual,expected,
                atol=REPLAY_TOLERANCES['anchor_atol'],rtol=REPLAY_TOLERANCES['anchor_rtol']))
            anchor_errors.append((actual-expected).abs().max().item())
        final=replay_tail(scheduler,record,lambda z,t:cfg_forward(pipe.unet,scheduler,z,t,cond,empty))
        expected=record['final_latent'].to(device)
        checks.append(tensor_comparison('final_latent',final,expected,
            atol=REPLAY_TOLERANCES['latent_atol'],rtol=REPLAY_TOLERANCES['latent_rtol']))
        check_finite(final)
        cached=restore_scheduler(native_scheduler(assets['teacher_clean']),record,device);counter=[0]
        def cached_prediction(z,t):
            out=record['tail_outputs'][counter[0]].to(device);counter[0]+=1;return out
        cached_final=replay_tail(cached,record,cached_prediction)
        checks.append(tensor_comparison('cached_output_latent',cached_final,expected,atol=0,rtol=0))
        checks.append(comparison_check('cached_scheduler_state',lambda:assert_state_close(scheduler_state(cached),record['final_state'],atol=0,rtol=0)))
        checks.append(comparison_check('replayed_scheduler_state',lambda:assert_state_close(scheduler_state(scheduler),record['final_state'],
                           atol=REPLAY_TOLERANCES['latent_atol'],rtol=REPLAY_TOLERANCES['latent_rtol'])))
        image=pipe.vae.decode(final/pipe.vae.config.scaling_factor).sample.clamp(-1,1).float()
        reference=pipe.vae.decode(expected/pipe.vae.config.scaling_factor).sample.clamp(-1,1).float()
        from common import image_name
        clean=pixels([run/'images/teacher_clean_train'/image_name(i) for i in ids])
        check_finite(image)
        results=image_diagnostics(image,clean,reference,perceptual,ids,REPLAY_TOLERANCES)
        for j,row in enumerate(results):
            row.update({'latent_max_abs_error':(final[j]-expected[j]).abs().max().item(),
                            'cached_output_latent_error':(cached_final[j]-expected[j]).abs().max().item(),
                            'anchor_max_abs_errors':anchor_errors,
                            'call_count':record['call_count'],'final_counter':scheduler.counter})
            print(f"Replay image {row['image_id']}: passed={row['passed']}, pixel={row['pixel_max_abs_error']}, LPIPS={row['lpips']}",flush=True)
    passed=all(row['passed'] for row in results) and all(check['passed'] for check in checks)
    diagnostic={'identity':identity,'passed':passed,'checks':checks,'samples':results,
                'reference_definition':'FP16 VAE decode of original cached final latent; diagnostic only'}
    write_json(run/'teacher_trace/replay_diagnostics.json',diagnostic)
    if passed:freeze(target,{'identity':identity,'passed':True,'samples':results})
    return passed


def generate_traced_train(run,model,pipe,model_hashes):
    """Generate PNG and trace together; only uncommitted shards may be retried."""
    from common import image_name
    from semantic_teacher import native_generate
    c=read_json(run/'config.json');rows=read_json(run/'prompts.json')['train'];out=run/'images/teacher_clean_train'
    out.mkdir(parents=True,exist_ok=True);pipe.scheduler.set_timesteps(c['inference_steps'],device='cuda')
    protocol={'version':1,'teacher_objective':'semantic_sign_v2_trajectory',
              'native':scheduler_protocol(pipe.scheduler),'saved_scheduler_sha256':sha256(Path(model)/'scheduler/scheduler_config.json'),
              'clean_model_hash':digest(model_hashes),'config_sha256':sha256(run/'config.json'),
              'prompts_sha256':sha256(run/'prompts.json'),'dtype':'torch.float16','guidance_scale':7.5,
              'tail_unroll_calls':4,'trajectory_anchor_count':5,'shard_size':64,'replay_tolerances':REPLAY_TOLERANCES}
    writer=TraceWriter(run/'teacher_trace',protocol,c['trajectory_shard_size'])
    seeds=[c['seed']*1000000+r['image_id'] for r in rows]
    freeze(out/'generation.json',{'model_hashes':model_hashes,'split':'train','prompts_sha256':sha256(run/'prompts.json'),
        'seeds':seeds,'steps':c['inference_steps'],'cfg':7.5,'resolution':512,'batch_size':c['teacher_batch'],
        'dtype':'float16','scheduler':type(pipe.scheduler).__name__,'native':True})
    # Validate committed PNGs even before the full dataset manifest exists.
    for name in writer.shards:
        for key,entry in read_json(writer.folder/name)['records'].items():
            if sha256(out/image_name(int(key)))!=entry['metadata']['png_sha256']:
                raise RuntimeError('Clean PNG associated with committed trace changed')
    if (writer.folder/'complete.json').exists():
        TraceReader(writer.folder)
        if not (out/'image_hashes.json').exists():_freeze_train_images(out,rows,seeds)
        expected=read_json(out/'image_hashes.json')
        if len(expected)!=len(rows) or any(sha256(out/r['file_name'])!=r['sha256'] for r in expected):
            raise RuntimeError('Clean traced training image manifest changed')
        return
    for start in range(0,len(rows),c['teacher_batch']):
        ids=list(range(start,min(start+c['teacher_batch'],len(rows))))
        if all(writer.has(i) for i in ids):continue
        if any(writer.has(i) for i in ids):raise ValueError('Trace shard splits a native generation batch')
        def generators():return [torch.Generator('cuda').manual_seed(seeds[i]) for i in ids]
        with NativeTrace(pipe.scheduler,4,5) as trace:
            images,latent=native_generate(pipe,[rows[i]['text'] for i in ids],generators(),c['inference_steps'],return_latents=True)
        records=trace.records()
        # Prove observational instrumentation on the first generation batch.
        integrity=writer.folder/'instrumentation.json'
        if not integrity.exists():
            ordinary,ordinary_latent=native_generate(pipe,[rows[i]['text'] for i in ids],generators(),c['inference_steps'],return_latents=True)
            torch.testing.assert_close(ordinary_latent,latent,atol=0,rtol=0)
            if any(not np.array_equal(np.asarray(a),np.asarray(b)) for a,b in zip(images,ordinary)):
                raise ValueError('Trace instrumentation changed native PNG pixels')
            freeze(integrity,{'passed':True,'image_ids':ids,'latent_max_error':0.,'pixel_max_error':0.,'protocol_hash':digest(protocol)})
        for i,image,record in zip(ids,images,records):
            path=out/image_name(i);tmp=path.with_suffix('.partial.png');image.save(tmp);tmp.replace(path)
            writer.append(i,record,{'prompt_image_id':rows[i]['image_id'],'seed':seeds[i],'png_sha256':sha256(path),'dtype':'torch.float16'})
        print(f'Native PNG + trace: {start+len(ids)}/{len(rows)}',flush=True)
    writer.finish(len(rows));_freeze_train_images(out,rows,seeds)


def _freeze_train_images(out,rows,seeds):
    from common import image_name
    entries=[{'file_name':image_name(i),'text':r['text'],'seed':seeds[i],'sha256':sha256(out/image_name(i))} for i,r in enumerate(rows)]
    with (out/'metadata.jsonl').open('w',encoding='utf8') as f:
        for row in entries:f.write(json.dumps({k:row[k] for k in ('file_name','text')})+'\n')
    freeze(out/'image_hashes.json',entries)


def batch_records(records):
    """Reassemble a native batch for FP16 replay with the original kernel shapes."""
    result=clone_tree(records[0])
    for row in records[1:]:
        for field in ('protocol','call_count','boundary_index','anchor_indices','anchor_timesteps'):
            if row[field]!=result[field]:raise ValueError('Cannot combine differing trajectory protocols')
    for field in ('boundary_latent','final_latent'):
        result[field]=torch.cat([r[field] for r in records])
    for field in ('anchor_latents','anchor_predictions','tail_outputs'):
        result[field]=torch.stack([r[field] for r in records],dim=1)
    for kind in ('boundary_state','final_state'):
        for field in ('ets','cur_sample','cur_model_output'):
            values=[r[kind][field] for r in records]
            if isinstance(values[0],torch.Tensor):result[kind][field]=torch.cat(values)
            elif isinstance(values[0],list):result[kind][field]=[torch.cat([v[i] for v in values]) for i in range(len(values[0]))]
            elif any(v!=values[0] for v in values):raise ValueError('Scalar scheduler state differs within native batch')
    return result


def assert_state_close(actual,expected,atol,rtol):
    if isinstance(actual,dict):
        if set(actual)!=set(expected):raise ValueError('Replay state fields changed')
        for key in actual:assert_state_close(actual[key],expected[key],atol,rtol)
    elif isinstance(actual,list):
        if len(actual)!=len(expected):raise ValueError('Replay history length changed')
        for a,b in zip(actual,expected):assert_state_close(a,b,atol,rtol)
    elif isinstance(actual,torch.Tensor):
        if actual.shape!=expected.shape or actual.dtype!=expected.dtype:
            raise ValueError('Replay scheduler tensor shape/dtype changed')
        torch.testing.assert_close(actual.cpu(),expected.cpu(),atol=atol,rtol=rtol)
    elif isinstance(actual,np.ndarray):
        if not np.array_equal(actual,expected):raise ValueError('Replay scheduler sequence changed')
    elif actual!=expected:raise ValueError('Replay scheduler scalar progression changed')
