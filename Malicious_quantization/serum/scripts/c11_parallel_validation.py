"""Precompute C11 validation on idle GPU0 without changing frozen queue sources."""
import os
import json
import subprocess
import time
import csv
from pathlib import Path
from PIL import Image

R=Path('/root/workspace/serum')
M=R/'metrics/serum_author'
def status(state, **kwargs):
    data=dict(state=state,gpu=0,updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**kwargs)
    p=M/'C11_parallel_validation_status.json'
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(p)
def main():
    os.chdir(R)
    raw=json.loads((M/'queue_status.json').read_text())
    assert raw['stage']=='all' and raw['state']=='completed'
    strong=json.loads((M/'strong_queue_status.json').read_text())
    assert strong['stage']=='C11_regen_train' and strong['state']=='running'
    folder=R/'data/serum_author/laundered/wm_step_1.00_val'
    assert not folder.exists(), 'Existing validation job/data requires review'
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
               HF_DATASETS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    cmd=[str(R/'.venv-ctrlregen/bin/python'),'scripts/launder_ctrlregen.py',
         '--ctrlregen-repo','third_party/CtrlRegen','--input-dir','data/serum_author/raw/wm_val',
         '--output-dir',str(folder),'--step','1.00','--base-seed','20000','--resume',
         '--spatial-control','third_party/CtrlRegen/public_checkpoints/spatialnet_ckp/spatial_control_ckp_14000',
         '--semantic-control','third_party/CtrlRegen/public_checkpoints/semanticnet_ckp']
    status('running',stage='C11_regen_val',command=cmd)
    with (R/'logs/C11_parallel_validation.log').open('w') as log:
        subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    sources=sorted((R/'data/serum_author/raw/wm_val').glob('*.png'))
    outputs=sorted(folder.glob('*.png'))
    with (folder/'laundering_manifest.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(sources)==len(outputs)==len(rows)==1000
    for i,(src,dst,row) in enumerate(zip(sources,outputs,rows)):
        assert src.name==dst.name and int(row['index'])==i and int(row['seed'])==20000+i
        assert float(row['step'])==1 and Path(row['source']).resolve()==src.resolve()
        assert Path(row['output']).resolve()==dst.resolve()
        with Image.open(dst) as im:im.load();assert im.size==(512,512) and im.mode=='RGB'
    status('completed',stage='C11_regen_val',images=1000,pairing_seed_decode_audit='passed')
    # The existing queue performs its full hash inventory audit before training.
    with (R/'logs/C10_parallel_utility.log').open('w') as log:
        subprocess.run([str(R/'.venv-student/bin/python'),'scripts/student_utility.py',
                        '--student','C10_seed0_author'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    status('completed',stage='C11_validation_and_C10_utility',images=1000)
if __name__=='__main__':
    try:main()
    except BaseException as e:status('failed',error=str(e));raise
