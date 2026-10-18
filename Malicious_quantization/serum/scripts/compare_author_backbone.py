"""Compare original-cache model tensors/configs with the previously used mirror."""
import json
from pathlib import Path
import torch
from safetensors import safe_open
from author_serum_stage0 import ROOT,A,M,sha,save_json

original=(A/'sd21_snapshot').resolve();mirror=ROOT/'models/sd21-base'
assert original.is_dir()
report={}
for component,filename in [('unet','diffusion_pytorch_model.safetensors'),('vae','diffusion_pytorch_model.safetensors'),('text_encoder','model.safetensors')]:
    p=original/component/filename;q=mirror/component/filename
    with safe_open(p,framework='pt',device='cpu') as x,safe_open(q,framework='pt',device='cpu') as y:
        assert set(x.keys())==set(y.keys()),component
        exact=True;fp16=True
        for k in x.keys():
            a=x.get_tensor(k);b=y.get_tensor(k)
            exact=exact and torch.equal(a,b)
            fp16=fp16 and torch.equal(a.half(),b.half())
    report[component]={'tensors_equal':exact,'fp16_tensors_equal':fp16,'original_sha256':sha(p),'mirror_sha256':sha(q)}
configs=['model_index.json','scheduler/scheduler_config.json','unet/config.json','vae/config.json',
         'text_encoder/config.json','tokenizer/tokenizer_config.json','tokenizer/vocab.json',
         'tokenizer/merges.txt','tokenizer/special_tokens_map.json']
report['config_files_equal']={f:sha(original/f)==sha(mirror/f) for f in configs}
report['reuse_policy']='Reuse prompt splits, code, environment and SD1.4/CLIP/CtrlRegen assets. Old-key WM images/adapters/scores cannot serve as author-key primary results. Existing clean utility reference may be retained only when backbone/tokenizer/scheduler equivalence is established; teacher pairs here are freshly generated with common noise.'
save_json(M/'backbone_comparison.json',report);print(json.dumps(report,indent=2))
