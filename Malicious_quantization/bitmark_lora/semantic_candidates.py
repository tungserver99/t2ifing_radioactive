"""One isolated candidate per visible GPU; serialize when only one GPU exists."""
import os
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from common import HERE


def schedule_candidates(values,slots,work):
    if not slots or len(set(slots))!=len(slots):raise ValueError('Need unique GPU slots')
    if len(slots)==1:return [work(value,slots[0]) for value in values]
    available=queue.Queue()
    for slot in slots:available.put(slot)
    def execute(value):
        slot=available.get()
        try:return work(value,slot)
        finally:available.put(slot)
    with ThreadPoolExecutor(max_workers=min(len(values),len(slots))) as pool:
        return list(pool.map(execute,values))


def visible_slots():
    import torch
    count=torch.cuda.device_count()
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    slots=[v.strip() for v in visible.split(',') if v.strip()] if visible is not None else [str(i) for i in range(count)]
    if count<1:raise RuntimeError('Teacher candidate training requires a GPU')
    return slots[:count]


def run_workers(run,values,parameter='kappa'):
    if parameter not in ('kappa','coordinate_margin_mu'):raise ValueError('Unknown teacher candidate parameter')
    flag='--kappa' if parameter=='kappa' else '--coordinate-margin-mu'
    slots=visible_slots()[:len(values)]
    print(f'Teacher candidates: {len(values)}; exclusive GPU slots: {slots}. '
          'One candidate per GPU; a single GPU executes candidates sequentially.',flush=True)
    def work(value,slot):
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=slot;env['PYTHONUNBUFFERED']='1'
        log=run/'logs'/f'teacher_{parameter}_{value!r}.log';log.parent.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,str(HERE/'semantic_teacher.py'),'--run',str(run),'--stage','candidate',flag,str(value)]
        with log.open('a') as stream:
            process=subprocess.Popen(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            try:
                for line in process.stdout:
                    stream.write(line);stream.flush();print(f'[{parameter}={value}] {line}',end='',flush=True)
                code=process.wait()
                if code:raise RuntimeError(f'Teacher {parameter}={value} failed (exit {code}); see {log}')
            finally:
                if process.poll() is None:process.terminate();process.wait()
                process.stdout.close()
        return value
    return schedule_candidates(values,slots,work)
