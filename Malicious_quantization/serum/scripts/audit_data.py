import hashlib
import json
from pathlib import Path
from PIL import Image

ROOT = Path('/root/workspace/serum')
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(s):
    return ' '.join(s.split())

def main():
    specs = [('adapt_train', 5000, 'clean_train', 'wm_train'),
             ('adapt_val', 1000, 'clean_val', 'wm_val'),
             ('detector_calib', 5000, 'calib_clean', 'calib_wm_unused')]
    report = {'ok': True, 'errors': [], 'counts': {}, 'split_overlap': {}}
    prompts = {}
    for split, n in [('adapt_train',5000),('adapt_val',1000),('student_eval',5000),('detector_calib',5000)]:
        p = ROOT / 'prompts' / (split+'.txt')
        lines = p.read_text().splitlines()
        prompts[split] = lines
        vals = [norm(s) for s in lines]
        if len(vals)!=n or any(not s for s in vals) or len(set(vals))!=n:
            report['errors'].append('Invalid/duplicate prompts: '+split)
    keys = list(prompts)
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            overlap = set(map(norm,prompts[a])) & set(map(norm,prompts[b]))
            report['split_overlap'][a+'/'+b] = len(overlap)
            if overlap: report['errors'].append('Overlapping splits: '+a+'/'+b)
    rows = []
    for split,n,clean,wm in specs:
        half = n//2
        for shard in (0,1):
            sp = ROOT/'prompts/shards'/f'{split}_{shard}.txt'
            if sp.read_text().splitlines()!=prompts[split][shard*half:(shard+1)*half]:
                report['errors'].append('Prompt shard order mismatch: '+str(sp))
        for folder,prefix,kind in [(clean,'clean','clean'),(wm,'watermarked','wm')]:
            dstroot=ROOT/'data/raw'/folder
            expected={f'{prefix}_{i:06d}.png' for i in range(n)}
            actual={p.name for p in dstroot.iterdir() if p.is_file()}
            report['counts'][folder]=len(actual)
            if actual!=expected: report['errors'].append('Unexpected file set: '+folder)
            for i in range(n):
                dst=dstroot/f'{prefix}_{i:06d}.png'
                src=ROOT/'data/shards'/f'{split}_{kind}_{i//half}'/f'{prefix}_{i%half:06d}.png'
                try:
                    with Image.open(dst) as im:
                        im.load()
                        if im.mode!='RGB' or im.size!=(512,512): raise ValueError('Invalid format/size')
                    digest=sha(dst)
                    if digest!=sha(src): raise ValueError('Source/merge hash mismatch')
                    rows.append({'split':split,'kind':kind,'index':i,'prompt':prompts[split][i],
                                 'prompt_sha256':hashlib.sha256(prompts[split][i].encode()).hexdigest(),
                                 'source':str(src),'image':str(dst),'sha256':digest})
                except Exception as e: report['errors'].append(str(dst)+': '+repr(e))
    report['ok']=not report['errors']
    out=ROOT/'manifests'
    (out/'teacher_pairing.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (out/'teacher_integrity_v2.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    if not report['ok']: raise SystemExit(1)

if __name__=='__main__': main()
