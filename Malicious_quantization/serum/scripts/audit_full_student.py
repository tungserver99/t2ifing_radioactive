"""Audit author-key students with their own frozen threshold, never the old key."""
import argparse
import json
from pathlib import Path
import numpy as np
from author_serum_stage0 import ROOT,A,M,sha,save_json,load_models,score_directory

def main():
    p=argparse.ArgumentParser();p.add_argument('--student');p.add_argument('--summary',action='store_true');args=p.parse_args()
    threshold=json.loads((M/'threshold_1pct.json').read_text())
    assert threshold['author_manifest_sha256']==sha(A/'manifest.json')
    if args.summary:
        results={c:json.loads((M/f'{c}_seed0_author_audit.json').read_text()) for c in ['C00','C10']}
        save_json(M/'raw_baseline_summary.json',{'conditions':results,
            'C10_minus_C00_detection_rate':results['C10']['rate']-results['C00']['rate'],
            'scope':'one seed; authors key+detector+cache, locally implemented SD1.4 adaptation; review before attack branch'})
        return
    assert args.student in ['C10_full_seed0_author','C11_full_step1.00_seed0_author']
    manifest,config,model,wm,vae=load_models()
    images=ROOT/'student_outputs'/args.student
    assert len(list(images.glob('*.png')))==5000
    s=score_directory(images,M/f'{args.student}_scores.npy',model,vae)
    detections=int((s>=np.float32(threshold['threshold'])).sum())
    result={'student':args.student,'n':len(s),'detections':detections,'rate':detections/len(s),
        'threshold':threshold,'checkpoint_selection':json.loads((ROOT/'students'/args.student/'validation_selection.json').read_text())['best']}
    save_json(M/f'{args.student}_audit.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
