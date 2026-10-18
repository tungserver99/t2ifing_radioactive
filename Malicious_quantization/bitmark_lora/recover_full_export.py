"""Explicit recovery of completed full training after pipeline-export failure.

Never trains, never chooses the final step instead of validation selection. Records
code changes separately without rewriting the historical run code manifest.
"""
import argparse
import datetime
import fcntl
import hashlib
import shutil
import sys
from pathlib import Path

from common import HERE, freeze, read_json, sha256, verify_vendor, write_json, runtime_root
from student import branches, checkpoint_hashes, training_command


def validate_snapshot(folder, initial_shapes):
    import torch
    from safetensors import safe_open
    weights = folder / 'unet/diffusion_pytorch_model.safetensors'
    if not (folder / 'unet/config.json').is_file():
        raise RuntimeError(f'Missing snapshot config: {folder}')
    with safe_open(str(weights), framework='pt', device='cpu') as state:
        if set(state.keys()) != set(initial_shapes):
            raise RuntimeError(f'Snapshot parameter keys mismatch: {folder}')
        for key in state.keys():
            tensor = state.get_tensor(key)
            if tuple(tensor.shape) != initial_shapes[key] or not torch.isfinite(tensor).all():
                raise RuntimeError(f'Invalid snapshot tensor: {folder}/{key}')
    return checkpoint_hashes(folder)


def recover_branch(run, name):
    import torch
    from safetensors import safe_open
    c, assets = read_json(run/'config.json'), read_json(run/'assets.json')
    branch = next(b for b in branches(c) if b['name']==name)
    if branch['kind'] != 'full': raise ValueError('Export recovery supports full UNet only')
    out = run/'students'/name
    if (out/'complete.json').exists():
        if read_json(out/'complete.json')['checkpoints'] != checkpoint_hashes(out):
            raise RuntimeError(f'Completed branch was modified: {name}')
        return
    final = c['max_train_steps']
    steps = c['checkpoint_steps']
    if final not in steps: raise RuntimeError('No final-step snapshot declared; cannot recover completed training')
    protocol = read_json(out/'protocol.json')
    command = training_command(c,assets,run,branch)
    # Interpreter may change when moving the same environment; all training arguments must match.
    if protocol['branch'] != branch or protocol['command'][1:] != command[1:]:
        raise RuntimeError('Training protocol mismatch; refusing to recover')
    images = read_json(run/'images'/f"teacher_{branch['condition']}_train"/'image_hashes.json')
    if images != protocol['training_images'] or len(images) != c['train_n']:
        raise RuntimeError('Training image manifest mismatch')
    for row in images:
        if sha256(run/'images'/f"teacher_{branch['condition']}_train"/row['file_name']) != row['sha256']:
            raise RuntimeError('Training images changed')
    initial = Path(assets['student'])/'unet/diffusion_pytorch_model.safetensors'
    with safe_open(str(initial),framework='pt',device='cpu') as f:
        shapes={key:tuple(f.get_slice(key).get_shape()) for key in f.keys()}
    snapshots={str(step):validate_snapshot(out/f'snapshot-{step}',shapes) for step in steps}
    source = out/f'snapshot-{final}'/'unet'
    with safe_open(str(initial),framework='pt',device='cpu') as base, \
         safe_open(str(source/'diffusion_pytorch_model.safetensors'),framework='pt',device='cpu') as trained:
        if not any(not torch.equal(base.get_tensor(k),trained.get_tensor(k)) for k in base.keys()):
            raise RuntimeError('Final snapshot equals base model; adaptation not established')
    audit={'source_step':final,'snapshots':snapshots,'protocol_sha256':sha256(out/'protocol.json'),
           'reason':'Final full pipeline export failed; recover inference UNet only, no optimizer restart',
           'selection':'Held-out validation will select among all recorded milestones, not necessarily final step'}
    freeze(out/'export_recovery.json',audit)
    target=out/'unet'; target.mkdir(exist_ok=True)
    for filename in ('config.json','diffusion_pytorch_model.safetensors'):
        dst=target/filename
        if dst.exists():
            if sha256(dst)!=sha256(source/filename):
                raise RuntimeError(f'Existing final export differs from final snapshot: {dst}')
        else:
            temporary=dst.with_suffix(dst.suffix+'.partial')
            shutil.copyfile(source/filename,temporary); temporary.replace(dst)
    write_json(out/'complete.json',{'checkpoints':checkpoint_hashes(out),'finite_weights':True,
        'actual_updates_verified':True,'recovered_from_snapshot':final})
    print(f'Recovered {name} from step {final}; no training repeated.',flush=True)


def main(run):
    from run import stage, persistent_log
    from validation import validation_label
    run=Path(run).resolve(); verify_vendor()
    c=read_json(run/'config.json')
    if c.get('watermark_method')!='serum': raise ValueError('This recovery entrypoint is for SERUM runs')
    lock_dir=runtime_root()/'locks'; lock_dir.mkdir(parents=True,exist_ok=True)
    with (lock_dir/(hashlib.sha256(str(run).encode()).hexdigest()+'.lock')).open('w') as lock, persistent_log(run):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Preserve historical provenance; record a separate explicitly requested repair event.
        audit={'old_manifest_sha256':sha256(run/'code_manifest.json'),
               'current_source':{str(p.relative_to(HERE)):sha256(p) for p in HERE.glob('*.py')},
               'current_vendor_sha256':sha256(HERE/'vendor/provenance.json'),
               'scope':'Recover final full export and finish evaluation; no train stage is invoked',
               'timestamp':datetime.datetime.now().astimezone().isoformat()}
        recovery_id=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        write_json(run/'recovery'/f'{recovery_id}.json',audit)
        all_branches=branches(c)
        # Verify every branch is complete/recoverable before any expensive evaluation.
        for b in all_branches:
            out=run/'students'/b['name']
            if b['kind']=='full': recover_branch(run,b['name'])
            if not (out/'complete.json').exists():
                raise RuntimeError(f"Branch {b['name']} has not finished training; export-only recovery cannot continue")
            if read_json(out/'complete.json')['checkpoints']!=checkpoint_hashes(out):
                raise RuntimeError(f"Branch weights changed: {b['name']}")
        def call(file,label,*args):
            stage(run,'recovery_'+label,[sys.executable,str(HERE/file),'--run',str(run),*args])
        # Revalidate immutable cache, then rebuild only missing milestone images/scores/utility.
        call('validation.py','cache','--stage','cache')
        for b in all_branches:
            name=b['name']
            call('validation.py',f'validate_{name}','--stage','evaluate','--branch',name)
            labels=[validation_label(name,step) for step in c['checkpoint_steps']]
            call('serum_teacher.py',f'scores_{name}','--stage','score','--directories',*labels)
            if not c['skip_utility']: call('utility.py',f'quality_{name}','--split','validation','--labels',*labels)
            call('validation.py',f'select_{name}','--stage','select','--branch',name)
        for name in ['unadapted']+[b['name'] for b in all_branches]:
            call('student.py',f'generate_{name}','--stage','generate','--branch',name)
        call('serum_teacher.py','test_scores','--stage','score','--directories','unadapted',*[b['name'] for b in all_branches])
        call('report.py','radioactivity')
        if not c['skip_utility']:
            call('utility.py','utility'); call('report.py','final_report')
        write_json(run/'complete.json',{'status':'completed','result':str(run/'result'),'recovery':recovery_id})
        print(f"Finished recovery. Results: {run/'result/README.md'}",flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True)
    main(p.parse_args().run)
