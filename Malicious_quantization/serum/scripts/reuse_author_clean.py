"""Reuse audited clean images and a corrected clean-control training checkpoint."""
import json,os,shutil
from pathlib import Path
from PIL import Image
from safetensors.torch import load_file
import torch
from author_serum_stage0 import ROOT,A,D,M,sha,save_json

def main():
    comparison=json.loads((M/'backbone_comparison.json').read_text())
    assert all(comparison[c]['tensors_equal'] for c in ['unet','vae','text_encoder'])
    assert all(comparison['config_files_equal'].values())
    inventory=json.loads((ROOT/'manifests/clipfix_core_inventory.json').read_text())
    reused={}
    for srcname,dstname,count,promptfile in [('clean_train','clean_train',5000,'adapt_train.txt'),
        ('clean_val','clean_val',1000,'adapt_val.txt'),('calib_clean','clean_calib',5000,'detector_calib.txt')]:
        src=ROOT/'data/raw'/srcname;dst=D/dstname;dst.mkdir(parents=True,exist_ok=True)
        files=sorted(src.glob('*.png'));assert len(files)==count
        expected=inventory.get('C00_train' if srcname=='clean_train' else 'C00_val' if srcname=='clean_val' else '')
        assert len((ROOT/'prompts'/promptfile).read_text().splitlines())==count
        records=[]
        for i,p in enumerate(files):
            with Image.open(p) as im:im.load();assert im.size==(512,512) and im.mode=='RGB'
            target=dst/f'{i:05d}.png'
            if expected is not None:assert sha(p)==expected[i]['sha256'],str(p)
            if target.exists():target.unlink()
            os.link(p,target)
            records.append({'source':str(p),'destination':str(target),'source_id':i,'sha256':sha(p)})
        reused[dstname]=records
    save_json(M/'clean_reuse.json',{'backbone_comparison_sha256':sha(M/'backbone_comparison.json'),
        'images':reused,'pairing':'same prompt IDs; legacy clean noise is not claimed equal to freshly generated WM noise',
        'utility_reference_reused':str(ROOT/'data/utility_teacher_clean')})
    src=ROOT/'students/C00_seed0_clipfix';dst=ROOT/'students/C00_seed0_author'
    old=json.loads((ROOT/'manifests/C00_seed0_clipfix.json').read_text())
    fix=json.loads((ROOT/'manifests/lora_gradient_clipping_fix.json').read_text())
    assert old['trainer_sha256']==fix['after_sha256']
    candidates=sorted(src.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
    valid=[]
    for candidate in candidates:
        weights=load_file(str(candidate/'pytorch_lora_weights.safetensors'))
        assert weights and all(torch.isfinite(t).all() for t in weights.values())
        assert (candidate/'optimizer.bin').exists() and (candidate/'random_states_0.pkl').exists()
        valid.append(candidate)
    assert valid
    assert not dst.exists(),'Do not overwrite an author student run'
    dst.mkdir()
    for candidate in valid:shutil.copytree(candidate,dst/candidate.name,copy_function=os.link)
    save_json(M/'C00_resume_provenance.json',{'source':str(src),'destination':str(dst),
        'resume_checkpoint':str(dst/valid[-1].name),'resume_step':int(valid[-1].name.split('-')[-1]),
        'source_manifest_sha256':sha(ROOT/'manifests/C00_seed0_clipfix.json'),
        'source_train_metadata_sha256':sha(ROOT/'data/hf/C00_train/metadata.jsonl'),
        'teacher_equivalence':'all model tensors and tokenizer/scheduler/model configs exactly match author cache',
        'scope':'clean control only; no old-key C10/C11 adapters reused'})
    print('Reused 11000 clean images and C00 checkpoint',valid[-1].name,flush=True)

if __name__=='__main__':main()
