"""One-command native UNet watermark training and data-mediated LoRA evaluation."""
import argparse
import datetime
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import paths
from paths import ROOT, HERE, SHARED, freeze, read_json, sha256, runtime_root, results_root
from common import verify_vendor
from run import stage, persistent_log, preflight
from student import branches


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path); p.add_argument('--resume',type=Path)
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--cache',type=Path,default=runtime_root()/'cache')
    p.add_argument('--seed',type=int,default=0); p.add_argument('--key-seed',type=int,default=1729)
    p.add_argument('--ranks',type=int,nargs='+',default=[16,32])
    p.add_argument('--student-scope',choices=['attention','attention_conv'],default='attention_conv')
    p.add_argument('--teacher-checkpoints',type=int,nargs='+',default=[250,500,1000])
    p.add_argument('--checkpoint-steps',type=int,nargs='+',default=[2500,5000,7500,10000])
    p.add_argument('--teacher-rank',type=int,default=32); p.add_argument('--teacher-lr',type=float,default=1e-4)
    p.add_argument('--learning-rate',type=float,default=1e-4); p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--precision',choices=['fp16','bf16','no'],default='fp16')
    for name,n in [('design-fit',256),('teacher-train',512),('meta-probe',128),('calibration',1000),
                   ('pilot-train',512),('pilot-eval',256),('train',5000),('validation',256),('test',1000)]:
        p.add_argument('--'+name+'-n',type=int,default=n)
    p.add_argument('--validation-generation-n',type=int,default=128)
    p.add_argument('--validation-draws',type=int,default=2)
    p.add_argument('--inference-steps',type=int,default=50); p.add_argument('--inference-batch',type=int,default=4)
    p.add_argument('--pilot-steps',type=int,default=500); p.add_argument('--pilot-rank',type=int,default=16)
    p.add_argument('--meta-rank',type=int,default=16); p.add_argument('--meta-steps',type=int,default=2)
    p.add_argument('--meta-warmup',type=int,default=250); p.add_argument('--meta-every',type=int,default=10)
    p.add_argument('--meta-weight',type=float,default=.25,help='0 runs the teacher-only ablation with the same budget')
    p.add_argument('--transfer-margin',type=float,default=.5); p.add_argument('--score-margin',type=float,default=1.)
    p.add_argument('--teacher-watermark-weight',type=float,default=5.)
    p.add_argument('--timestep-min',type=int,default=50); p.add_argument('--timestep-max',type=int,default=700)
    p.add_argument('--fpr',type=float,default=.01)
    p.add_argument('--min-teacher-tpr',type=float,default=.95)
    p.add_argument('--min-teacher-ci-low',type=float,default=.90)
    p.add_argument('--min-student-tpr',type=float,default=.20)
    p.add_argument('--min-psnr',type=float,default=25.)
    p.add_argument('--max-clip-drop',type=float,default=.03)
    p.add_argument('--max-clean-fpr',type=float,default=.03)
    p.add_argument('--skip-utility',action='store_true')
    p.add_argument('--min-free-gb',type=float,default=100.)
    a=p.parse_args(argv)
    if a.resume and a.output: p.error('Choose output or resume')
    positive=[v for k,v in vars(a).items() if k.endswith('_n')]+[a.teacher_rank,a.meta_rank,a.meta_steps,a.meta_every,
        a.pilot_rank,a.pilot_steps,a.batch_size,a.inference_batch,a.inference_steps,a.validation_draws,*a.ranks,*a.checkpoint_steps,*a.teacher_checkpoints]
    if min(positive)<1 or min(a.seed,a.key_seed,a.meta_warmup)<0: p.error('Invalid counts/seeds')
    for name in ('ranks','checkpoint_steps','teacher_checkpoints'):
        if len(set(getattr(a,name)))!=len(getattr(a,name)): p.error('Duplicate '+name)
        setattr(a,name,sorted(getattr(a,name)))
    if not 0<=a.timestep_min<a.timestep_max<1000: p.error('Invalid timestep range')
    if not 0<a.fpr<1 or a.calibration_n*a.fpr<1: p.error('Calibration needs at least 1/FPR images')
    if a.design_fit_n<4 or a.test_n<2 or not 2<=a.validation_generation_n<=a.validation_n: p.error('Insufficient fit/test/validation images')
    if not 0<a.min_teacher_ci_low<=a.min_teacher_tpr<=1 or not 0<a.min_student_tpr<=1 or not 0<a.max_clean_fpr<1:
        p.error('Invalid detection gates')
    for name in ('teacher_lr','learning_rate','score_margin','transfer_margin','teacher_watermark_weight'):
        if not math.isfinite(getattr(a,name)) or getattr(a,name)<=0: p.error('Invalid '+name)
    for name in ('meta_weight','min_free_gb','min_psnr','max_clip_drop'):
        if not math.isfinite(getattr(a,name)) or getattr(a,name)<0: p.error('Invalid '+name)
    return a


def config(a):
    c={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(a).items() if k not in ('output','resume','plan_only','min_free_gb')}
    c.update(watermark_method='new_watermark',max_train_steps=max(a.checkpoint_steps),
        loss_weights={'watermark':a.teacher_watermark_weight,'prior':1.,'pixels':.1,'semantic':.1},
        gates={k:c[k] for k in ('min_teacher_tpr','min_teacher_ci_low','min_student_tpr','min_psnr','max_clip_drop','max_clean_fpr')},
        lora_only=True,paper_full_ema=False,train_clean_controls=False)
    c['student_branches']=[{'name':f'lora_r{r}_{cond}','kind':'lora','rank':r,'scope':a.student_scope,'condition':cond}
                           for r in a.ranks for cond in ('wm',)]
    return c


def code_manifest():
    files=list(HERE.glob('*.py'))+list(SHARED.glob('*.py'))+[SHARED/'vendor/provenance.json',ROOT/'run_new.sh']
    return {str(p.relative_to(ROOT)):sha256(p) for p in sorted(files)}


def execute(run,c):
    def own(stage_name,**kwargs):
        args=[sys.executable,'-u',str(HERE/'stages.py'),'--run',str(run),stage_name]
        for k,v in kwargs.items(): args+=['--'+k,str(v)]
        label=stage_name+''.join('_'+str(v) for v in kwargs.values())
        stage(run,label,args)
    def shared(file,name,*args,subrun=None):
        stage(run,name,[sys.executable,'-u',str(SHARED/file),'--run',str(subrun or run),*args])
    for name in ['assets','clean_design','detector','encode','teacher_train']: own(name)
    for step in c['teacher_checkpoints']:
        own('export',step=step); own('teacher_design',step=step)
    own('setup_pilots')
    for ident in ['clean']+[str(s) for s in c['teacher_checkpoints']]:
        name='pilot_clean' if ident=='clean' else 'pilot_wm'
        shared('student.py','train_pilot_'+ident,'--stage','train','--branch',name,subrun=run/'pilots'/ident)
        own('pilot_generate',name=ident)
    own('select_teacher')
    for condition in ('clean','wm'): own('final_teacher',name=condition)
    shared('validation.py','validation_cache','--stage','cache')
    for b in branches(c):
        name=b['name']
        shared('student.py','train_'+name,'--stage','train','--branch',name)
        shared('validation.py','validate_'+name,'--stage','evaluate','--branch',name)
        own('select_student',name=name)
    own('final_student',name='unadapted')
    for b in branches(c): own('final_student',name=b['name'])
    if not c['skip_utility']:
        shared('utility.py','utility','--labels','teacher_clean_test','teacher_wm_test','unadapted',*[b['name'] for b in branches(c)])
    own('report')
    print(f'Finished. Results: {run}/result/README.md',flush=True)


def main(argv=None):
    argv=sys.argv[1:] if argv is None else argv
    a=arguments(argv)
    if a.resume:
        # Config is immutable; overrides must use a separate run.
        allowed={'--resume','--plan-only','--min-free-gb'}
        if any(x.startswith('--') and x.split('=')[0] not in allowed for x in argv):
            raise ValueError('Resume uses frozen config; do not supply experiment overrides')
        run=a.resume.expanduser().resolve(); c=read_json(run/'config.json')
    else:
        c=config(a)
        stamp=datetime.datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
        run=(a.output or results_root()/f'run_new_{stamp}_{os.getpid()}').expanduser().resolve()
    if a.plan_only:
        print(json.dumps({'run':str(run),'config':c,'execution':'without --plan-only all stages run, including student evaluation'},indent=2)); return
    if not a.resume: run.mkdir(parents=True,exist_ok=False)
    lockdir=runtime_root()/'locks'; lockdir.mkdir(parents=True,exist_ok=True)
    lockname=hashlib.sha256(str(run).encode()).hexdigest()+'.lock'
    with (lockdir/lockname).open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with persistent_log(run):
            print(f'Persistent run: {run}\nPython: {sys.executable}',flush=True)
            verify_vendor(); freeze(run/'config.json',c); freeze(run/'code_manifest.json',code_manifest())
            # Reuse the no-FlashAttention CUDA/import/disk preflight from SERUM.
            check=argparse.Namespace(cache=Path(c['cache']),output=run,min_free_gb=a.min_free_gb,watermark_method='serum')
            env=preflight(check)
            env['packages']={n:importlib.metadata.version(n) for n in ('torch','torchvision','diffusers','transformers','peft','accelerate','safetensors')}
            freeze(run/'environment.json',env)
            if not c['skip_utility']: __import__('torch_fidelity')
            execute(run,c)


if __name__=='__main__': main()
