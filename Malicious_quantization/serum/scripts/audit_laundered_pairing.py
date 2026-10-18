import argparse,csv,json
from pathlib import Path
from PIL import Image

p=argparse.ArgumentParser(); p.add_argument('--kind',required=True); p.add_argument('--step',required=True)
a=p.parse_args(); r=Path('/root/workspace/serum'); reports={}
for split,n,seed in [('train',5000,10000),('val',1000,20000)]:
    raw=sorted((r/f'data/raw/{a.kind}_{split}').glob('*.png'))
    out=r/f'data/laundered/{a.kind}_step_{a.step}_{split}'
    files=sorted(out.glob('*.png'))
    prompts=(r/f'prompts/adapt_{split}.txt').read_text().splitlines()
    assert len(raw)==len(files)==len(prompts)==n
    assert [x.name for x in raw]==[x.name for x in files]
    with (out/'laundering_manifest.csv').open() as f: rows=list(csv.DictReader(f))
    assert len(rows)==n
    for i,(src,dst,row) in enumerate(zip(raw,files,rows)):
        assert Path(row['source']).resolve()==src.resolve() and Path(row['output']).resolve()==dst.resolve()
        assert int(row['seed'])==seed+i and float(row['step'])==float(a.step)
        with Image.open(dst) as im:
            im.load(); assert im.size==(512,512) and im.mode=='RGB'
    reports[split]={'count':n,'pairing_ok':True,'decoded_all':True}
(r/f'manifests/laundered_{a.kind}_step{a.step}_audit.json').write_text(json.dumps(reports,indent=2))
print(json.dumps(reports),flush=True)
