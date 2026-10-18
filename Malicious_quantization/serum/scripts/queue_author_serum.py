"""Queue author-artifact raw controls after the current corrected core finishes."""
import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from author_serum_stage0 import sha,save_json

R=Path('/root/workspace/serum');A=R/'assets/serum_author';M=R/'metrics/serum_author'
PY=str(R/'.venv-student/bin/python');SERUM=str(R/'.venv/bin/python')
STATUS=M/'queue_status.json'
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
           HF_DATASETS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')

def status(stage,state,**kw):
    save_json(STATUS,{'stage':stage,'state':state,'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),**kw})

def run(stage,cmd):
    print(stage,cmd,flush=True)
    with (R/'logs'/f'serum_author_{stage}.log').open('w') as log:
        proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while proc.poll() is None:
                status(stage,'running',pid=proc.pid);time.sleep(20)
            if proc.returncode:raise RuntimeError(f'{stage}: exit {proc.returncode}')
        finally:
            if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM)
    status(stage,'completed')

def main():
    os.chdir(R);M.mkdir(parents=True,exist_ok=True)
    lock=(A/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while not (A/'SHA256SUMS').exists():
        live=subprocess.run(['tmux','has-session','-t','quant-author-assets'],capture_output=True).returncode==0
        if not live:raise RuntimeError('Author asset download failed; inspect logs/author_assets_download.log')
        status('download','waiting');time.sleep(20)
    if not (A/'manifest.json').exists() or json.loads((A/'manifest.json').read_text()).get('status')!='verified':
        run('asset_validation',[SERUM,'scripts/prepare_author_serum.py'])
    manifest=json.loads((A/'manifest.json').read_text());assert manifest['status']=='verified'
    trainer=R/'third_party/diffusers/examples/text_to_image/train_text_to_image_lora.py'
    fix=json.loads((R/'manifests/lora_gradient_clipping_fix.json').read_text())
    assert sha(trainer)==fix['after_sha256']
    frozen={str(p):sha(p) for p in [trainer,R/'scripts/author_serum_stage0.py',R/'scripts/select_lora_checkpoint.py',R/'scripts/generate_student.py',A/'manifest.json']}
    plan={'branch':'author_SD21_key','conditions':['C00_seed0_author','C10_seed0_author'],
        'raw_controls_only':True,'mild_jobs':0,'dependency':'none; user cancelled previous core and switched artifacts',
        'source_files':frozen,'prompts':{p.name:sha(p) for p in (R/'prompts').glob('*.txt')},
        'teacher':'authors SD2.1 cache + epoch50 key/detector; alpha0.5; 50steps CFG7.5; prompt-paired legacy clean reuse after exact backbone equivalence',
        'teacher_images':22000,'new_watermarked_images':11000,'reused_clean_images':11000,
        'threshold':'separate author detector calibration on reused equivalent-backbone 5000 clean images; 1% FPR; fixed thereafter',
        'training':'SD1.4 LoRA rank4, LR1e-5, batch8, seed0, 20000 total steps, corrected clipping; C00 resumes corrected clean checkpoint, C10 starts fresh',
        'selection':'minimum denoising MSE on 1000 condition-specific validation images',
        'evaluation':'5000 outputs per control, shared prompts and seeds; author detector+VAE+fixed author threshold',
        'next_gate':'review raw C10 versus C00 before author-key strong C01/C11 or additional seeds',
        'recipe_limit':'author artifacts supplied; exact paper LoRA recipe remains unavailable'}
    save_json(R/'manifests/serum_author_plan.json',plan)
    for p,h in frozen.items():assert sha(p)==h,p
    run('smoke',[SERUM,'scripts/author_serum_stage0.py','--smoke'])
    if not (M/'clean_reuse.json').exists():run('clean_reuse',[PY,'scripts/reuse_author_clean.py'])
    processes=[];logs=[]
    try:
        for gpu in [0,1]:
            worker_env=env.copy();worker_env['CUDA_VISIBLE_DEVICES']=str(gpu)
            log=(R/'logs'/f'serum_author_teacher_gpu{gpu}.log').open('w');logs.append(log)
            processes.append(subprocess.Popen([SERUM,'scripts/author_serum_stage0.py','--worker',str(gpu)],
                env=worker_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):raise RuntimeError('Author teacher worker failed')
            status('teacher_generation','running',worker_pids=[p.pid for p in processes if p.poll() is None]);time.sleep(20)
        assert all(p.returncode==0 for p in processes)
    finally:
        for p in processes:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for log in logs:log.close()
    run('stage0_audit',[SERUM,'scripts/author_serum_stage0.py','--audit-only'])
    threshold_hash=sha(M/'threshold_1pct.json')
    for base,kind in [('C00','clean'),('C10','wm')]:
        name=base+'_seed0_author';dest=R/'students'/name
        resume=None
        if base=='C00':
            provenance=json.loads((M/'C00_resume_provenance.json').read_text());resume=provenance['resume_checkpoint']
            assert dest.exists() and Path(resume).is_dir()
        else:assert not dest.exists(),'Partial author run requires explicit recovery'
        dataset=R/'data/serum_author/hf'/f'{base}_train'
        run(base+'_dataset',[SERUM,'scripts/make_hf_imagefolder.py','--images',f'data/serum_author/raw/{kind}_train',
            '--prompts','prompts/adapt_train.txt','--out',str(dataset)])
        cmd=[PY,'-m','accelerate.commands.launch','--num_processes','1',str(trainer),
             '--pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4',f'--train_data_dir={dataset}',
             '--image_column=image','--caption_column=text','--resolution=512','--train_batch_size=8',
             '--gradient_accumulation_steps=1','--learning_rate=1e-5','--rank=4','--max_train_steps=20000',
             '--checkpointing_steps=500','--mixed_precision=fp16','--seed=0',f'--output_dir={dest}']
        if resume:
            assert sha(dataset/'metadata.jsonl')==provenance['source_train_metadata_sha256']
            cmd.append(f'--resume_from_checkpoint={resume}')
        save_json(R/'manifests'/f'{name}.json',{'plan_sha256':sha(R/'manifests/serum_author_plan.json'),
            'train_command':cmd,'threshold_sha256':threshold_hash,'metadata_sha256':sha(dataset/'metadata.jsonl')})
        run(base+'_train',cmd)
        from safetensors.torch import load_file
        import torch
        candidates=[*dest.glob('checkpoint-*'),dest];assert len(candidates)==41
        for c in candidates:
            weights=load_file(str(c/'pytorch_lora_weights.safetensors'))
            assert weights and all(torch.isfinite(t).all() for t in weights.values())
        run(base+'_validation',[PY,'scripts/select_lora_checkpoint.py','--student',str(dest),
            '--images',f'data/serum_author/raw/{kind}_val'])
        best=(dest/'BEST_LORA_PATH.txt').read_text().strip()
        run(base+'_generation',[PY,'scripts/generate_student.py','--lora',best,
            '--prompts','prompts/student_eval.txt','--out',f'student_outputs/{name}'])
        assert sha(M/'threshold_1pct.json')==threshold_hash
        run(base+'_audit',[SERUM,'scripts/audit_author_student.py','--student',name])
    run('summary',[SERUM,'scripts/audit_author_student.py','--summary'])
    status('all','completed',summary='metrics/serum_author/raw_baseline_summary.json')

if __name__=='__main__':
    try:main()
    except BaseException as e:
        status('queue','failed',error=str(e));raise
