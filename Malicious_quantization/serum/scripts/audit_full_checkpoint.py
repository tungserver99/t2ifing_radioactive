"""Author-detector audit of explicitly requested full checkpoints."""
import argparse,json
from pathlib import Path
import numpy as np
from author_serum_stage0 import ROOT,A,M,sha,save_json,load_models,score_directory

def main():
    p=argparse.ArgumentParser();p.add_argument('--student',required=True);p.add_argument('--checkpoint',type=Path,required=True);a=p.parse_args()
    assert Path(a.student).name==a.student
    checkpoint=a.checkpoint.resolve();assert checkpoint.is_relative_to(ROOT/'students')
    images=ROOT/'student_outputs'/a.student
    generation=json.loads((images/'generation_manifest.json').read_text())
    assert Path(generation['unet']).resolve()==checkpoint
    assert generation['num_images']==5000 and generation['prompt_sha256']==sha(ROOT/'prompts/student_eval.txt')
    files=sorted(images.glob('*.png'));assert len(files)==5000
    assert [p.name for p in files]==[f'sample_{i:06d}.png' for i in range(5000)]
    threshold=json.loads((M/'threshold_1pct.json').read_text())
    assert threshold['author_manifest_sha256']==sha(A/'manifest.json')
    original=json.loads((checkpoint.parent/'validation_selection.json').read_text())
    match=[x for x in original['results'] if Path(x['checkpoint']).resolve()==checkpoint];assert len(match)==1
    _,_,model,wm,vae=load_models()
    scores=score_directory(images,M/f'{a.student}_scores.npy',model,vae)
    positives=int((scores>=np.float32(threshold['threshold'])).sum())
    record=dict(student=a.student,n=5000,detections=positives,rate=positives/5000,threshold=threshold,
        checkpoint_selection=dict(checkpoint=str(checkpoint),mse=match[0]['mse'],criterion='fixed step explicitly requested; not selected on detector scores'),
        generation_manifest_sha256=sha(images/'generation_manifest.json'))
    save_json(M/f'{a.student}_audit.json',record);print(json.dumps(record,indent=2),flush=True)
if __name__=='__main__':main()
