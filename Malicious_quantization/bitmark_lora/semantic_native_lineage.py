"""Fail-closed v2 checkpoint-100 lineage and immutable v3 preparation."""
import importlib.metadata
import shutil
import re
from pathlib import Path
import torch
from common import HERE, read_json, freeze, sha256, image_name, write_json
from semantic_checkpoint import verified_state
from semantic_carrier import SemanticCarrier
from semantic_native_rollout import OBJECTIVE

LOCKED = dict(teacher_objective=OBJECTIVE, teacher_only=False, teacher_rank=32,
              teacher_scope='attention_conv',teacher_train_steps=300,teacher_batch=4,
              teacher_lr=1e-5,teacher_precision='bf16',inference_steps=50,
              tail_unroll_calls=10,sign_k_train=26,coordinate_margin_mu=.02,
              fidelity_cap_ratio=.5,global_grad_clip=1.,teacher_eval_checkpoints=[50,100,200,300],
              teacher_eval_n=256,teacher_checkpoint_every=25,analytic_alpha=.01,num_bits=32,
              teacher_clip_drop_max=.01,teacher_lpips_max=.15,
              ranks=[32],adapter_scopes=['attention_conv'],dora=False,lora_only=True,
              paper_full_ema=False,train_clean_controls=True,batch_size=4,learning_rate=1e-4,
              precision='fp16',max_train_steps=7500,epochs=6,checkpoint_steps=[2500,5000,7500],
              skip_utility=False,watermark_method='semantic_fp')


def verify_images(folder, rows, config):
    from semantic_teacher import generation_seed
    entries = read_json(folder/'image_hashes.json')
    import json
    captions=[json.loads(line) for line in (folder/'metadata.jsonl').read_text(encoding='utf8').splitlines()]
    expected_captions=[{'file_name':image_name(i),'text':row['text']} for i,row in enumerate(rows)]
    if captions!=expected_captions:
        raise RuntimeError('RGB-caption metadata differs from frozen prompts or contains extra fields')
    if sorted(p.name for p in folder.glob('sample_*.png'))!=[image_name(i) for i in range(len(rows))]:
        raise RuntimeError('Unexpected/incomplete PNG dataset')
    generation = read_json(folder/'generation.json')
    if len(entries)!=len(rows) or generation['steps']!=50 or generation['cfg']!=7.5 or generation['resolution']!=512:
        raise RuntimeError(f'Wrong frozen dataset/protocol: {folder}')
    if generation['scheduler']!='PNDMScheduler' or generation.get('native') is not True:
        raise RuntimeError('Source images must be ordinary native PNDM50')
    seeds = [generation_seed(config,row) for row in rows]
    if generation['seeds']!=seeds:
        raise RuntimeError('Source image seed lineage changed')
    for i,(entry,row) in enumerate(zip(entries,rows)):
        if entry['file_name']!=image_name(i) or entry['text']!=row['text'] or entry['seed']!=seeds[i]:
            raise RuntimeError('Frozen clean image/prompt alignment changed')
        if sha256(folder/entry['file_name'])!=entry['sha256']:
            raise RuntimeError(f'Frozen clean PNG changed: {folder/entry["file_name"]}')


def source_identity(source):
    from semantic_assets import snapshot_identity, TEACHER_REV
    source = Path(source)
    c,a = read_json(source/'config.json'),read_json(source/'assets.json')
    checkpoint = source/'teacher_fp/trajectory/checkpoint-100'
    protocol = read_json(checkpoint.parent/'protocol.json')
    if c.get('teacher_objective')!='semantic_sign_v2_trajectory' or protocol.get('teacher_objective')!='semantic_sign_v2_trajectory':
        raise ValueError('Warm start requires a v2 trajectory source run')
    if c['teacher_rank']!=32 or c['train_n']!=5000 or c['inference_steps']!=50 or c['teacher_batch']!=4:
        raise ValueError('Source rank/train-pool/sampling protocol is incompatible')
    required={'training_state.pt','adapter_model.safetensors','adapter_config.json'}
    if not required.issubset(read_json(checkpoint/'checkpoint.json')['files']):
        raise RuntimeError('Source checkpoint manifest does not commit all required files')
    state = verified_state(checkpoint,protocol)
    if state['step']!=100: raise ValueError('Warm-start checkpoint must be exactly 100')
    for field,name in [('config_sha256','config.json'),('assets_sha256','assets.json'),
                       ('carrier_sha256','carrier/complete.json'),('key_sha256','carrier/key.json'),
                       ('code_manifest_sha256','code_manifest.json'),
                       ('training_images_sha256','images/teacher_clean_train/image_hashes.json')]:
        if protocol[field]!=sha256(source/name):
            raise RuntimeError(f'Source lineage changed: {field}')
    trace=protocol['trace_protocol']
    if trace['prompts_sha256']!=sha256(source/'prompts.json') or trace['guidance_scale']!=7.5:
        raise RuntimeError('Source checkpoint prompt/CFG lineage mismatch')
    if protocol['trace_sha256']!=sha256(source/'teacher_trace/complete.json'):
        raise RuntimeError('Source committed trace manifest changed')
    if a['prompt_sha256']!=sha256(source/'prompts.json'):
        raise RuntimeError('Source frozen prompts changed')
    if snapshot_identity(a['teacher_clean'])!=a['teacher_clean_hashes']:
        raise RuntimeError('Source clean SD2.1 snapshot changed')
    model=Path(a['teacher_clean'])
    if protocol['scheduler_config_sha256']!=sha256(model/'scheduler/scheduler_config.json'):
        raise RuntimeError('Source scheduler changed')
    teacher_source=a['teacher_source']
    if 'repository' in teacher_source and teacher_source.get('revision')!=TEACHER_REV:
        raise RuntimeError('Wrong pinned teacher revision')
    if read_json(model/'unet/config.json').get('cross_attention_dim')!=1024:
        raise RuntimeError('Expected SD2.1 teacher')
    from semantic_trajectory import native_scheduler, scheduler_protocol
    scheduler=native_scheduler(model); scheduler.set_timesteps(50)
    production=scheduler_protocol(scheduler)
    if protocol['trace_protocol']['native']!=production:
        raise RuntimeError('Source native scheduler/call protocol differs')
    adapter=read_json(checkpoint/'adapter_config.json')
    if adapter['r']!=32 or adapter.get('lora_alpha')!=32 or adapter.get('use_dora',False):
        raise ValueError('Source must be the rank-32 ordinary LoRA')
    carrier=SemanticCarrier.load(source/'carrier')
    if c.get('analytic_alpha',.01)!=.01 or c.get('num_bits')!=32:
        raise ValueError('Source analytic verifier differs from locked RSSF protocol')
    if carrier.carrier_version!='semantic_sign_v1' or len(carrier.signs)!=32:
        raise ValueError('Source must use the frozen 32-bit RSSF carrier')
    prompts=read_json(source/'prompts.json'); seen_ids=set(); seen_text=set()
    for split in ('train','validation','test','calibration'):
        rows=prompts[split]
        if len(rows)!=c[split+'_n'] or not rows:
            raise ValueError('Source split count mismatch')
        for row in rows:
            if row['image_id'] in seen_ids or row['text'] in seen_text:
                raise ValueError('Source split membership is not disjoint')
            seen_ids.add(row['image_id']); seen_text.add(row['text'])
    if len(prompts['validation'])<256:
        raise ValueError('At least 256 source validation rows are required; no test substitution')
    for split in ('train','validation','calibration'):
        folder=source/'images'/('teacher_clean_'+split)
        verify_images(folder,prompts[split],c)
        if read_json(folder/'generation.json')['prompts_sha256']!=a['prompt_sha256']:
            raise RuntimeError('Source clean PNG prompt hash changed')
    return {'source_run':str(source.resolve()), 'warm_start_step':100,
            'source_config_sha256':sha256(source/'config.json'),
            'source_checkpoint_sha256':sha256(checkpoint/'checkpoint.json'),
            'source_adapter_sha256':sha256(checkpoint/'adapter_model.safetensors'),
            'source_protocol_sha256':sha256(checkpoint.parent/'protocol.json'),
            'prompt_sha256':a['prompt_sha256'],'carrier_sha256':sha256(source/'carrier/complete.json'),
            'key_sha256':sha256(source/'carrier/key.json'),
            'clean_image_manifests':{s:sha256(source/f'images/teacher_clean_{s}/image_hashes.json') for s in ('train','validation','calibration')},
            'scheduler_protocol':production,'optimizer':'fresh AdamW, no source moments',
            'teacher_clean_hashes':a['teacher_clean_hashes']}


def copy_immutable(source,target):
    source,target=Path(source),Path(target)
    if source.is_dir():
        for path in source.rglob('*'):
            if path.is_file():copy_immutable(path,target/path.relative_to(source))
    else:
        if target.exists():
            if sha256(source)!=sha256(target): raise RuntimeError(f'Reused artifact changed: {target}')
        else:
            target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_name(target.name+'.partial')
            shutil.copyfile(source,temporary); temporary.replace(target)


def repository_head(root):
    """Best-effort provenance from Git metadata; never invoke a Git executable.

    Copied/exported PBS deployments may have no metadata at all. In that case
    content hashes remain authoritative and the unavailable commit is JSON null.
    """
    try:
        git=Path(root)/'.git'
        if git.is_file():
            pointer=git.read_text(encoding='utf8').strip()
            if not pointer.startswith('gitdir: '):return None
            git=(Path(root)/pointer[8:]).resolve()
        head=(git/'HEAD').read_text(encoding='utf8').strip()
        if head.startswith('ref: '):
            reference=head[5:]
            if not reference.startswith('refs/') or '..' in reference.split('/'):return None
            common=git
            if (git/'commondir').is_file():
                common=(git/(git/'commondir').read_text(encoding='utf8').strip()).resolve()
            for directory in dict.fromkeys((git,common)):
                loose=directory/reference
                if loose.is_file():
                    head=loose.read_text(encoding='utf8').strip()
                    break
                packed=directory/'packed-refs'
                if packed.is_file():
                    match=next((line.split()[0] for line in packed.read_text(encoding='utf8').splitlines()
                                if len(line.split())==2 and line.split()[1]==reference),None)
                    if match:
                        head=match
                        break
        return head if re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}',head) else None
    except (OSError,UnicodeError,ValueError):
        return None


def code_identity():
    root=HERE.parent.parent
    head=repository_head(root)
    return {'git_head':head,'files':{p.relative_to(root).as_posix():sha256(p)
             for p in sorted(HERE.glob('*.py'))},
            'launcher_sha256':sha256(root/'run_rssf_native_margin.sh'),
            'spec_sha256':sha256(root/'GPT_guide/rssf_native_margin_refinement_student_distill_codex_spec.md'),
            'vendor_manifest_sha256':sha256(HERE/'vendor/provenance.json'),
            'entrypoint_sha256':sha256(root/'main_native_margin.py')}


def freeze_code_identity(run):
    """Audit initialization-only bugfixes; never replace a trained run's identity."""
    from semantic_sign_native_margin_refine import training_protocol
    from semantic_trajectory import digest
    target=run/'code_manifest.json';current=code_identity()
    prior=read_json(target) if target.exists() else None
    folder=run/'teacher_fp/native_margin_refine';protocol_path=folder/'protocol.json'
    changed=prior is not None and prior!=current
    old_protocol=read_json(protocol_path) if protocol_path.exists() else None
    # Compare all non-code inputs before touching either committed file.
    if old_protocol is not None:
        expected=training_protocol(run)
        key='code_manifest_json_sha256'
        if {k:v for k,v in old_protocol.items() if k!=key}!={k:v for k,v in expected.items() if k!=key}:
            raise RuntimeError('Initialization protocol inputs changed beyond code identity')
    pending=old_protocol is not None and old_protocol!=training_protocol(run)
    if changed or pending:
        log=folder/'losses.jsonl'
        artifacts=(any(folder.glob('checkpoint-*')) or (log.exists() and log.stat().st_size>0)
                   or any((folder/name).exists() for name in ('smoke.json','trained.json'))
                   or any((run/name).exists() for name in ('teacher_fp/selected.json','student_data.json','complete.json'))
                   or any((run/'students').glob('*'))
                   or (run/'result/teacher_screening.json').exists())
        if artifacts:
            raise RuntimeError('Code identity changed after training artifacts exist; use a separate run')
        history=run/'code_updates'/digest(prior)
        freeze(history/'code_manifest.json',prior)
        if old_protocol is not None:freeze(history/'protocol.json',old_protocol)
        freeze(history/'upgrade.json',{'reason':'initialization-only code update; no smoke/checkpoint/optimizer events',
                                       'old_identity':digest(prior),'new_identity':digest(current)})
        write_json(target,current)
        if old_protocol is not None:write_json(protocol_path,training_protocol(run))
    else:
        freeze(target,current)


def prepare(run,source,micro_batch=2,student_snapshot=None):
    from semantic_assets import snapshot_identity,verified_student_snapshot,STUDENT_REV
    lineage=source_identity(source)
    old=read_json(source/'config.json')
    c={**old,**LOCKED,'source_run':str(source),'warm_start_step':100,'native_micro_batch':micro_batch,
       'validation_generation_n':old['validation_n'],'validation_draws':old.get('validation_draws',2)}
    for obsolete in ('continuation','preparation_reuse'):c.pop(obsolete,None)
    if micro_batch not in (1,2):raise ValueError('Only explicit microbatch 1 or 2 is supported')
    freeze(run/'config.json',c); freeze(run/'source_lineage.json',lineage)
    for name in ('prompts.json','carrier','images/teacher_clean_train','images/teacher_clean_validation','images/teacher_clean_calibration'):
        copy_immutable(source/name,run/name)
    for name,key in [('prompts.json','prompt_sha256'),('carrier/complete.json','carrier_sha256'),('carrier/key.json','key_sha256')]:
        if sha256(run/name)!=lineage[key]:raise RuntimeError('Copied immutable identity mismatch')
    assets=read_json(source/'assets.json')
    if (run/'assets.json').exists():
        saved=read_json(run/'assets.json')
        if snapshot_identity(saved['student'])!=saved['student_hashes']:
            raise RuntimeError('Pinned student changed')
        if student_snapshot and str(Path(student_snapshot).resolve())!=saved['student']:
            raise ValueError('Student snapshot cannot change on resume')
        assets=saved
    else:
        student=student_snapshot or assets.get('student')
        if student:
            student=verified_student_snapshot(student)
        else:
            from assets import hub_snapshot
            def keep(n):
                return n=='model_index.json' or (n.split('/')[0] in {'unet','vae','text_encoder','tokenizer','scheduler','feature_extractor'} and n.endswith(('.json','.txt','.safetensors')) and '.fp16.' not in n and '.non_ema.' not in n)
            student=hub_snapshot('CompVis/stable-diffusion-v1-4',STUDENT_REV,Path(c['cache']),keep)
        assets={**assets,'student':str(student),'student_hashes':snapshot_identity(student),'student_revision':STUDENT_REV}
        freeze(run/'assets.json',assets)
    if assets['student_revision']!=STUDENT_REV or read_json(Path(assets['student'])/'unet/config.json').get('cross_attention_dim')!=768:
        raise RuntimeError('Wrong pinned SD1.4 student')
    if assets['teacher_clean_hashes']!=lineage['teacher_clean_hashes'] or assets['prompt_sha256']!=lineage['prompt_sha256']:
        raise RuntimeError('Destination reference identity differs from source')
    freeze_code_identity(run)
    freeze(run/'environment.json',{'packages':{n:importlib.metadata.version(n) for n in ('torch','diffusers','peft','transformers','accelerate','lpips')},
                                 'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None})
    return c
