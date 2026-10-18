import json
import torch
from author_serum_stage0 import ROOT,A,M,save_json
frozen=json.loads((ROOT/'manifests/serum_frozen.json').read_text())
old=torch.load(frozen['watermark']['path'],weights_only=True,map_location='cpu')['grid']
new=torch.load(A/'watermark.pt',weights_only=True,map_location='cpu')['grid']
oldn=(old-old.mean())/old.std();newn=(new-new.mean())/new.std()
result={'grid_equal':torch.equal(old,new),'normalized_grid_equal':torch.equal(oldn,newn),
        'normalized_grid_correlation':float(torch.nn.functional.cosine_similarity(oldn.flatten(),newn.flatten(),dim=0))}
save_json(M/'key_comparison.json',result);print(result)
