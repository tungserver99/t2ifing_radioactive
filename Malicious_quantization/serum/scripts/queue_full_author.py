"""GPU0 full-UNet C11/C10 queue, isolated from the frozen LoRA queues."""
import os,json,time,subprocess,fcntl,hashlib,shutil
from pathlib import Path
R=Path('/root/workspace/serum');M=R/'metrics/serum_author';PY=str(R/'.venv-student/bin/python')
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    tmp=p.with_suffix('.tmp.json');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)
def status(stage,state,**kw):save(M/'full_queue_status.json',dict(stage=stage,state=state,gpu=0,updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**kw))
def run(stage,cmd):
    with (R/f'logs/full_author_{stage}.log').open('w') as log:
        p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        while p.poll() is None:status(stage,'running',pid=p.pid);time.sleep(15)
        if p.returncode:raise RuntimeError(f'{stage} failed ({p.returncode}); inspect log')
    status(stage,'completed')
def traincmd(dataset,dest,steps,interval):
    return [PY,'-m','accelerate.commands.launch','--num_processes','1','scripts/train_full_sd14.py',
      '--pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4',f'--train_data_dir={dataset}',
      '--image_column=image','--caption_column=text','--resolution=512','--train_batch_size=8',
      '--gradient_accumulation_steps=1','--learning_rate=5e-6','--lr_scheduler=constant',
      '--lr_warmup_steps=0',f'--max_train_steps={steps}',f'--checkpointing_steps={interval}',
      '--mixed_precision=fp16','--seed=0','--max_grad_norm=1.0',f'--output_dir={dest}']
def main():
    os.chdir(R);lock=(R/'assets/serum_author/full_queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    frozen={p:sha(R/p) for p in ['scripts/train_full_sd14.py','scripts/select_full_checkpoint.py','scripts/generate_full_student.py','scripts/audit_full_student.py','scripts/student_utility.py','metrics/serum_author/threshold_1pct.json']}
    save(R/'manifests/full_author_plan.json',dict(conditions=['C10_full_seed0_author','C11_full_step1.00_seed0_author'],adaptation='all UNet parameters; frozen VAE/text encoder; no EMA',steps=20000,batch=8,lr=5e-6,seed=0,checkpoint_interval=1000,checkpoint_selection='minimum unseen denoising MSE; all 20 snapshots plus final; no detector feedback',storage='retain all UNets; optimizer only at latest checkpoint; older checkpoints cannot resume training',frozen=frozen))
    while True:
        dep=json.loads((M/'C11_parallel_validation_status.json').read_text())
        if dep['state']=='failed':raise RuntimeError('GPU0 validation/utility helper failed')
        if dep['state']=='completed' and dep['stage']=='C11_validation_and_C10_utility':break
        status('GPU0_dependency','waiting',dependency=dep['stage']);time.sleep(20)
    # Prioritize C11, including the smoke test, once its audited dataset exists.
    while not (M/'wm_strong_val_inventory.json').exists() or not (R/'data/serum_author/hf/C11_step1.00_train/metadata.jsonl').exists():
        dep=json.loads((M/'strong_queue_status.json').read_text())
        if dep['state']=='failed':raise RuntimeError('C11 dataset queue failed')
        status('C11_dataset_dependency','waiting');time.sleep(20)
    plan=json.loads((R/'manifests/full_author_plan.json').read_text())
    plan['conditions']=['C11_full_step1.00_seed0_author','C10_full_seed0_author']
    plan['execution_order']='C11 strong, then C10 raw; smoke uses C11 data'
    save(R/'manifests/full_author_plan.json',plan)
    dataset=R/'data/serum_author/hf/C11_step1.00_train'
    smoke=R/'students/full_author_smoke';assert not smoke.exists()
    assert shutil.disk_usage(R).free>120*1024**3,'Insufficient space for both full runs'
    run('smoke_train',traincmd(dataset,smoke,4,4))
    # A real optimizer/save/reload check. No evaluation prompts are used here.
    import torch
    from safetensors.torch import load_file
    weights=load_file(str(smoke/'unet/diffusion_pytorch_model.safetensors'))
    assert weights and all(torch.isfinite(w).all() for w in weights.values())
    from diffusers import UNet2DConditionModel
    base=UNet2DConditionModel.from_pretrained('CompVis/stable-diffusion-v1-4',subfolder='unet')
    assert any(not torch.equal(w,base.state_dict()[k]) for k,w in weights.items()),'Full UNet did not update'
    trained=UNet2DConditionModel.from_pretrained(str(smoke),subfolder='unet')
    assert all(torch.equal(trained.state_dict()[k],w) for k,w in weights.items())
    del base,trained,weights
    save(M/'full_smoke.json',dict(status='passed',steps=4,batch=8,finite_weights=True,parameters_updated=True,unet_reload=True))
    assert smoke.resolve().parent==(R/'students').resolve();shutil.rmtree(smoke)
    for name,kind in [('C11_full_step1.00_seed0_author','strong'),('C10_full_seed0_author','raw')]:
        for p,h in frozen.items():assert sha(R/p)==h,p
        if kind=='strong':
            while not (M/'wm_strong_val_inventory.json').exists() or not (R/'data/serum_author/hf/C11_step1.00_train/metadata.jsonl').exists():
                dep=json.loads((M/'strong_queue_status.json').read_text())
                if dep['state']=='failed':raise RuntimeError('C11 dataset queue failed')
                status('C11_dataset_dependency','waiting');time.sleep(20)
        dataset=R/('data/serum_author/hf/C10_train' if kind=='raw' else 'data/serum_author/hf/C11_step1.00_train')
        val=R/('data/serum_author/raw/wm_val' if kind=='raw' else 'data/serum_author/laundered/wm_step_1.00_val')
        dest=R/'students'/name;assert not dest.exists(),'Partial full run needs explicit recovery'
        assert shutil.disk_usage(R).free>65*1024**3,'Insufficient space before full condition'
        cmd=traincmd(dataset,dest,20000,1000)
        save(R/'manifests'/f'{name}.json',dict(command=cmd,metadata_sha256=sha(dataset/'metadata.jsonl'),threshold_sha256=frozen['metrics/serum_author/threshold_1pct.json']))
        run(name+'_train',cmd)
        assert len(list(dest.glob('checkpoint-*')))==20
        run(name+'_validation',[PY,'scripts/select_full_checkpoint.py','--student',str(dest),'--images',str(val)])
        selection=json.loads((dest/'validation_selection.json').read_text());assert len(selection['results'])==21
        assert all(__import__('math').isfinite(x['mse']) for x in selection['results'])
        best=(dest/'BEST_UNET_PATH.txt').read_text().strip()
        run(name+'_generation',[PY,'scripts/generate_full_student.py','--unet',best,'--prompts','prompts/student_eval.txt','--out',f'student_outputs/{name}'])
        assert sha(M/'threshold_1pct.json')==frozen['metrics/serum_author/threshold_1pct.json']
        run(name+'_audit',[str(R/'.venv/bin/python'),'scripts/audit_full_student.py','--student',name])
        run(name+'_utility',[PY,'scripts/student_utility.py','--student',name])
    results={name:dict(audit=json.loads((M/f'{name}_audit.json').read_text()),utility=json.loads((R/f'metrics/{name}_utility.json').read_text())) for name in ['C10_full_seed0_author','C11_full_step1.00_seed0_author']}
    save(M/'full_author_summary.json',dict(conditions=results,scope='one seed; C10/C11 only; no full-finetune clean controls; 1000-step selection cadence differs from LoRA'))
    status('all','completed')
if __name__=='__main__':
    try:main()
    except BaseException as e:status('queue','failed',error=str(e));raise
