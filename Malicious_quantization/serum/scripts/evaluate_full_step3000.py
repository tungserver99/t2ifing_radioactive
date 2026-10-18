"""Evaluate full C10/C11 at fixed step 3000 in parallel, without retraining."""
import argparse,os,json,time,subprocess,fcntl,hashlib
from pathlib import Path
R=Path('/root/workspace/serum');M=R/'metrics/serum_author'
PY=str(R/'.venv-student/bin/python');SERUM=str(R/'.venv/bin/python')
NAMES={'C10':'C10_full_step3000_seed0_author','C11':'C11_full_step1.00_step3000_seed0_author'}
SOURCES={'C10':'C10_full_seed0_author','C11':'C11_full_step1.00_seed0_author'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    tmp=p.with_suffix('.tmp.json');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)
def status(condition,stage,state,**kw):save(M/f'full_step3000_{condition}_status.json',dict(condition=condition,stage=stage,state=state,updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**kw))
def worker(condition,gpu):
    os.chdir(R);name=NAMES[condition];checkpoint=R/'students'/SOURCES[condition]/'checkpoint-3000'
    lock=(R/f'assets/serum_author/step3000_{condition}.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    try:
        assert (checkpoint/'unet/diffusion_pytorch_model.safetensors').is_file()
        frozen={p:sha(R/p) for p in ['scripts/generate_full_student.py','scripts/audit_full_checkpoint.py','scripts/student_utility.py','scripts/author_serum_stage0.py','assets/serum_author/manifest.json','metrics/serum_author/threshold_1pct.json','prompts/student_eval.txt']}
        status(condition,'preflight','running',gpu=gpu)
        model_hash=sha(checkpoint/'unet/diffusion_pytorch_model.safetensors')
        save(R/f'manifests/{name}.json',dict(condition=condition,checkpoint=str(checkpoint),checkpoint_step=3000,checkpoint_sha256=model_hash,gpu=gpu,
            selection='fixed step 3000 requested by user; original validation MSE reused',n=5000,batch=4,steps=50,cfg=7.5,base_seed=50000,frozen=frozen))
        commands=[('generation',[PY,'scripts/generate_full_student.py','--unet',str(checkpoint),'--prompts','prompts/student_eval.txt','--out',f'student_outputs/{name}','--resume']),
                  ('audit',[SERUM,'scripts/audit_full_checkpoint.py','--student',name,'--checkpoint',str(checkpoint)]),
                  ('utility',[PY,'scripts/student_utility.py','--student',name])]
        for stage,cmd in commands:
            for p,h in frozen.items():assert sha(R/p)==h,p
            with (R/f'logs/full_step3000_{condition}_{stage}.log').open('w') as log:
                proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
                while proc.poll() is None:status(condition,stage,'running',gpu=gpu,pid=proc.pid);time.sleep(15)
                if proc.returncode:raise RuntimeError(f'{stage} exited {proc.returncode}; inspect full_step3000 log')
        assert sha(checkpoint/'unet/diffusion_pytorch_model.safetensors')==model_hash
        status(condition,'all','completed',gpu=gpu,student=name)
    except BaseException as e:status(condition,'queue','failed',gpu=gpu,error=str(e));raise
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--condition',choices=['C10','C11']);parser.add_argument('--gpu',type=int,choices=[0,1]);a=parser.parse_args()
    if a.condition:
        assert a.gpu is not None;worker(a.condition,a.gpu);return
    os.chdir(R);lock=(R/'assets/serum_author/step3000_queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    processes=[];logs=[]
    for condition,gpu in [('C10',0),('C11',1)]:
        log=(R/f'logs/full_step3000_{condition}_worker.log').open('w');logs.append(log)
        processes.append(subprocess.Popen([PY,'scripts/evaluate_full_step3000.py','--condition',condition,'--gpu',str(gpu)],stdout=log,stderr=subprocess.STDOUT))
    while any(p.poll() is None for p in processes):time.sleep(15)
    assert all(p.returncode==0 for p in processes),'A worker failed; see per-condition status/logs'
    results={c:dict(audit=json.loads((M/f'{n}_audit.json').read_text()),utility=json.loads((R/f'metrics/{n}_utility.json').read_text())) for c,n in NAMES.items()}
    save(M/'full_step3000_summary.json',dict(conditions=results,scope='one seed; fixed checkpoint 3000, not minimum-MSE selection within a budget; unchanged frozen threshold and shared evaluation prompts/seeds'))
    for log in logs:log.close()
if __name__=='__main__':main()
