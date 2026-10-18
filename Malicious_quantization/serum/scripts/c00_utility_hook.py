"""Insert C00 utility evaluation after its audit, before the raw queue advances."""
import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path

R=Path('/root/workspace/serum');M=R/'metrics/serum_author';PARENT=15851
STATUS=M/'C00_utility_hook_status.json'

def status(state,**kw):
    obj={'state':state,'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),**kw}
    tmp=STATUS.with_suffix('.tmp.json');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(STATUS)

def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False

def main():
    os.chdir(R)
    lock=(M/'C00_utility_hook.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    deadline=time.monotonic()+8*3600
    paused=False
    try:
        while True:
            dep=json.loads((M/'queue_status.json').read_text())
            if dep['state']=='failed':raise RuntimeError('Raw queue failed before C00 utility')
            if dep['stage']=='C00_audit':
                cmd=Path(f'/proc/{PARENT}/cmdline').read_bytes().decode().replace('\x00',' ')
                assert 'queue_author_serum.py' in cmd,cmd
                os.kill(PARENT,signal.SIGSTOP);paused=True
                child=dep.get('pid')
                status('waiting_for_C00_audit',queue_parent_paused=True,audit_pid=child)
                stop=time.monotonic()+300
                while child and alive(child):
                    if time.monotonic()>stop:raise TimeoutError('C00 audit did not finish')
                    time.sleep(2)
                audit=json.loads((M/'C00_seed0_author_audit.json').read_text())
                assert audit['n']==5000
                break
            if dep['stage'].startswith('C10') or dep['stage']=='all':
                raise RuntimeError('C00 insertion point passed; review before launching concurrent GPU work')
            if time.monotonic()>deadline:raise TimeoutError('C00 utility scheduling deadline exceeded')
            status('waiting_for_C00',dependency=dep['stage']);time.sleep(2)
        assert len(list((R/'student_outputs/C00_seed0_author').glob('*.png')))==5000
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            HF_DATASETS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
        with (R/'logs/serum_author_C00_utility_immediate.log').open('w') as log:
            p=subprocess.Popen([str(R/'.venv-student/bin/python'),'scripts/student_utility.py','--student','C00_seed0_author'],
                env=env,stdout=log,stderr=subprocess.STDOUT)
            try:
                stop=time.monotonic()+1800
                while p.poll() is None:
                    status('running',pid=p.pid,queue_parent_paused=True)
                    if time.monotonic()>stop:raise TimeoutError('C00 utility exceeded 30 minutes')
                    time.sleep(10)
                if p.returncode:raise RuntimeError(f'C00 utility exited {p.returncode}')
            finally:
                if p.poll() is None:
                    p.terminate()
                    try:p.wait(timeout=20)
                    except subprocess.TimeoutExpired:p.kill();p.wait()
        result=json.loads((R/'metrics/C00_seed0_author_utility.json').read_text())
        assert result['num_images']==5000
        status('completed',result='metrics/C00_seed0_author_utility.json',utility=result)
    finally:
        if paused:os.kill(PARENT,signal.SIGCONT)

if __name__=='__main__':
    try:main()
    except BaseException as e:status('failed',error=str(e));raise
