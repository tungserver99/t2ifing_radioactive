"""Run author-key strong conditions on GPU1 after teacher calibration completes."""
import csv
import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from author_serum_stage0 import ROOT as R,A,M,sha,save_json

PY=str(R/'.venv-student/bin/python');SERUM=str(R/'.venv/bin/python');CTRL=str(R/'.venv-ctrlregen/bin/python')
STATUS=M/'strong_queue_status.json'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
    HF_DATASETS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')

def status(stage,state,**kw):
    save_json(STATUS,{'stage':stage,'state':state,'gpu':1,'mild_jobs':0,'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),**kw})

def run(stage,cmd):
    print(stage,cmd,flush=True)
    with (R/'logs'/f'serum_author_strong_{stage}.log').open('w') as log:
        p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while p.poll() is None:status(stage,'running',pid=p.pid);time.sleep(20)
            if p.returncode:raise RuntimeError(f'{stage}: exit {p.returncode}')
        finally:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
    status(stage,'completed')

def audit_dataset(kind,split,folder):
    n,seed=(5000,10000) if split=='train' else (1000,20000)
    rawdir=R/f'data/raw/clean_{split}' if kind=='clean' else R/f'data/serum_author/raw/wm_{split}'
    raw=sorted(rawdir.glob('*.png'));files=sorted(folder.glob('*.png'))
    prompts=(R/f'prompts/adapt_{split}.txt').read_text().splitlines()
    assert len(raw)==len(files)==len(prompts)==n
    with (folder/'laundering_manifest.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==n
    inventory=json.loads((R/'manifests/clipfix_core_inventory.json').read_text()) if kind=='clean' else None
    result=[]
    for i,(src,dst,row) in enumerate(zip(raw,files,rows)):
        assert src.name==dst.name and int(row['index'])==i
        assert Path(row['source']).resolve()==src.resolve() and Path(row['output']).resolve()==dst.resolve()
        assert int(row['seed'])==seed+i and float(row['step'])==1.0
        if inventory is not None:assert sha(dst)==inventory[f'C01_step1.00_{split}'][i]['sha256']
        with Image.open(dst) as im:im.load();assert im.size==(512,512) and im.mode=='RGB'
        result.append({'source_id':i,'source':str(src),'source_sha256':sha(src),'output':str(dst),'output_sha256':sha(dst),'seed':seed+i})
    save_json(M/f'{kind}_strong_{split}_inventory.json',result)

def main():
    os.chdir(R);M.mkdir(parents=True,exist_ok=True)
    lock=(A/'strong_queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    trainer=R/'third_party/diffusers/examples/text_to_image/train_text_to_image_lora.py'
    fixed=json.loads((R/'manifests/lora_gradient_clipping_fix.json').read_text())['after_sha256']
    assert sha(trainer)==fixed
    frozen={str(p):sha(p) for p in [trainer,R/'scripts/launder_ctrlregen.py',R/'scripts/select_lora_checkpoint.py',
        R/'scripts/generate_student.py',R/'scripts/audit_author_student.py',R/'scripts/student_utility.py',A/'manifest.json']}
    plan={'conditions':['C01_step1.00_seed0_author','C11_step1.00_seed0_author'],'gpu':1,'mild_jobs':0,
        'dependency':'verified author teacher_detection.json + frozen threshold, then run alongside raw controls on GPU0',
        'C01_data':'reuse audited existing clean strong regeneration: identical clean sources/backbone',
        'C11_data':'fresh CtrlRegen+ on author-key WM train/val; strength1.00; train seed10000+i, val seed20000+i',
        'training':'fresh SD1.4 LoRA rank4 LR1e-5 batch8 seed0 20000steps; corrected clipping',
        'evaluation':'1000 condition-specific validation MSE; 5000 outputs; common fixed author threshold; CLIP/FID/KID',
        'authorization':'user explicitly requested C01/C11 queued before baseline review; exploratory if C10 remains weak',
        'selection':'strong1.00 retained from prior preservation-based policy; no downstream detector tuning',
        'frozen_sources':frozen}
    save_json(R/'manifests/serum_author_strong_plan.json',plan)
    while not (M/'teacher_detection.json').exists():
        dep=json.loads((M/'queue_status.json').read_text())
        if dep['state']=='failed':raise RuntimeError('Author raw queue failed before teacher readiness')
        status('teacher_dependency','waiting',dependency=dep['stage']);time.sleep(20)
    assert sha(A/'manifest.json')==json.loads((M/'threshold_1pct.json').read_text())['author_manifest_sha256']
    threshold_hash=sha(M/'threshold_1pct.json')
    for base,kind in [('C01','clean'),('C11','wm')]:
        for p,h in frozen.items():assert sha(p)==h,p
        dirs={}
        for split,seed in [('train',10000),('val',20000)]:
            if kind=='clean':folder=R/f'data/laundered/clean_step_1.00_{split}'
            else:
                folder=R/f'data/serum_author/laundered/wm_step_1.00_{split}'
                run(f'{base}_regen_{split}',[CTRL,'scripts/launder_ctrlregen.py','--ctrlregen-repo','third_party/CtrlRegen',
                    '--input-dir',f'data/serum_author/raw/wm_{split}','--output-dir',str(folder),'--step','1.00',
                    '--base-seed',str(seed),'--resume','--spatial-control','third_party/CtrlRegen/public_checkpoints/spatialnet_ckp/spatial_control_ckp_14000',
                    '--semantic-control','third_party/CtrlRegen/public_checkpoints/semanticnet_ckp'])
            audit_dataset(kind,split,folder);dirs[split]=folder
        name=base+'_step1.00_seed0_author';dest=R/'students'/name;assert not dest.exists(),'Partial training requires review'
        dataset=R/f'data/serum_author/hf/{base}_step1.00_train'
        run(base+'_dataset',[SERUM,'scripts/make_hf_imagefolder.py','--images',str(dirs['train']),
            '--prompts','prompts/adapt_train.txt','--out',str(dataset)])
        cmd=[PY,'-m','accelerate.commands.launch','--num_processes','1',str(trainer),
            '--pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4',f'--train_data_dir={dataset}',
            '--image_column=image','--caption_column=text','--resolution=512','--train_batch_size=8',
            '--gradient_accumulation_steps=1','--learning_rate=1e-5','--rank=4','--max_train_steps=20000',
            '--checkpointing_steps=500','--mixed_precision=fp16','--seed=0',f'--output_dir={dest}']
        save_json(R/'manifests'/f'{name}.json',{'train_command':cmd,'threshold_sha256':threshold_hash,
            'metadata_sha256':sha(dataset/'metadata.jsonl'),'strong_plan_sha256':sha(R/'manifests/serum_author_strong_plan.json')})
        run(base+'_train',cmd)
        expected={f'checkpoint-{s}' for s in range(500,20001,500)}
        assert {p.name for p in dest.glob('checkpoint-*')}==expected
        for c in [*dest.glob('checkpoint-*'),dest]:
            weights=load_file(str(c/'pytorch_lora_weights.safetensors'));assert weights and all(torch.isfinite(t).all() for t in weights.values())
        run(base+'_validation',[PY,'scripts/select_lora_checkpoint.py','--student',str(dest),'--images',str(dirs['val'])])
        selection=json.loads((dest/'validation_selection.json').read_text());assert len(selection['results'])==41
        assert all(np.isfinite(r['mse']) for r in selection['results'])
        best=(dest/'BEST_LORA_PATH.txt').read_text().strip()
        run(base+'_generation',[PY,'scripts/generate_student.py','--lora',best,'--prompts','prompts/student_eval.txt','--out',f'student_outputs/{name}'])
        assert sha(M/'threshold_1pct.json')==threshold_hash
        run(base+'_audit',[SERUM,'scripts/audit_author_student.py','--student',name])
        run(base+'_utility',[PY,'scripts/student_utility.py','--student',name])
    while True:
        dep=json.loads((M/'queue_status.json').read_text())
        if dep['state']=='failed':raise RuntimeError('Raw controls failed; full summary withheld')
        if dep['stage']=='all' and dep['state']=='completed':break
        status('raw_dependency','waiting',dependency=dep['stage']);time.sleep(20)
    for base in ['C00','C10']:run(base+'_utility',[PY,'scripts/student_utility.py','--student',base+'_seed0_author'])
    results={}
    for base in ['C00','C10','C01','C11']:
        name=base+('_step1.00' if base in ['C01','C11'] else '')+'_seed0_author'
        results[base]={'audit':json.loads((M/f'{name}_audit.json').read_text()),'utility':json.loads((R/f'metrics/{name}_utility.json').read_text())}
    save_json(M/'core_strong_summary.json',{'conditions':results,'threshold_sha256':threshold_hash,
        'scope':'single seed, strong only; report baseline validity before attack interpretation'})
    status('all','completed',summary='metrics/serum_author/core_strong_summary.json')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    try:
        if args.preflight:
            for split in ['train','val']:audit_dataset('clean',split,R/f'data/laundered/clean_step_1.00_{split}')
            print('C01 reused strong dataset: full decode/hash/pairing/seed audit passed',flush=True)
        else:main()
    except BaseException as e:status('queue','failed',error=str(e));raise
