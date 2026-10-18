"""One common CLIPScore definition and teacher reference for all students."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel,CLIPProcessor
import torch_fidelity
torch.serialization.add_safe_globals([
    np.ndarray, np.dtype, type(np.dtype('float64')), type(np.dtype('float32')),
    np._core.multiarray._reconstruct, np._core.multiarray.scalar,
])

p=argparse.ArgumentParser(); p.add_argument('--student',required=True); a=p.parse_args()
r=Path('/root/workspace/serum'); images=r/'student_outputs'/a.student
ref=r/'data/utility_teacher_clean'; files=sorted(images.glob('*.png'))
prompts=(r/'prompts/student_eval.txt').read_text().splitlines()
assert len(files)==len(prompts)==len(list(ref.glob('*.png')))==5000
torch.set_num_threads(4)
model=CLIPModel.from_pretrained('openai/clip-vit-base-patch32').eval().to('cuda')
processor=CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
scores=[]
with torch.no_grad():
    for start in range(0,5000,32):
        batch=[]
        for f in files[start:start+32]:
            with Image.open(f) as im: batch.append(im.convert('RGB').copy())
        x=processor(images=batch,return_tensors='pt').to('cuda')
        text=processor(text=prompts[start:start+32],padding=True,truncation=True,max_length=77,return_tensors='pt').to('cuda')
        vi=model.get_image_features(**x); vt=model.get_text_features(**text)
        if not isinstance(vi,torch.Tensor): vi=vi.pooler_output
        if not isinstance(vt,torch.Tensor): vt=vt.pooler_output
        vi=torch.nn.functional.normalize(vi,dim=-1); vt=torch.nn.functional.normalize(vt,dim=-1)
        scores.extend((2.5*(vi*vt).sum(-1).clamp(min=0)).cpu().tolist())
        if start%640==0: print('CLIP scored',min(start+32,5000),flush=True)
np.save(r/f'metrics/{a.student}_clipscore.npy',np.array(scores))
del model
torch.cuda.empty_cache()
distribution=torch_fidelity.calculate_metrics(input1=str(images),input2=str(ref),cuda=True,
    fid=True,kid=True,isc=False,verbose=True,kid_subset_size=1000,kid_subsets=100,
    rng_seed=42000,input1_cache_name=a.student,input2_cache_name='utility_teacher_clean',cache_root=str(r/'data/utility_feature_cache'))
result={'student':a.student,'num_images':5000,'clipscore_mean':float(np.mean(scores)),
    'clipscore_definition':'2.5 * max(cosine, 0), OpenAI CLIP ViT-B/32',
    'reference':str(ref),'reference_definition':'5000 clean SD2.1 Base outputs on student_eval prompts; teacher-distribution similarity, not COCO FID',
    'kid_subset_size':1000,'kid_subsets':100,'rng_seed':42000,
    'distribution_metrics':{k:float(v) for k,v in distribution.items()}}
(r/f'metrics/{a.student}_utility.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
