import tempfile
import random
from collections import Counter
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
import torch
from common import image_name,sha256,write_json
import semantic_sign_report as report
class SignReportTests(unittest.TestCase):
 def test_auc(self):
  self.assertEqual(report.roc_auc([1,1],[1,1])['auc'],.5)
  self.assertEqual(report.roc_auc([2,3],[0,1])['auc'],1.)
  self.assertEqual(report.roc_auc([0],[1])['auc'],0.)
 def test_random_keys(self):
  r=torch.tensor([[0.]*32,list(range(-16,16))]); a=report.random_key_validation(r,32,.01,seed=19)
  self.assertEqual(a,report.random_key_validation(r,32,.01,seed=19));self.assertEqual(a['num_fake_keys'],1000)
  self.assertEqual(sum(a['histogram'].values()),2000);self.assertAlmostEqual(a['mean_K'],16,delta=.5)
  self.assertAlmostEqual(a['variance_K'],8,delta=1.)
 def test_vectorized_null_matches_manual_random_keys(self):
  r=torch.tensor([[0.,-1.,2.,-3.],[1.,0.,-1.,4.]])
  result=report.random_key_validation(r,4,.1,seed=7,num_keys=1000)
  bits=report.image_bits(r).tolist();rng=random.Random(7);hist=Counter()
  for _ in range(1000):
   signs=[rng.choice((-1,1)) for _ in range(4)]
   for row in bits:hist[sum(b==s for b,s in zip(row,signs))]+=1
  self.assertEqual(result['histogram'],{str(k):hist[k] for k in range(5)})

 def test_query_budgets(self):
  r=torch.ones(23,32);a=report.query_budget_metrics(r,torch.ones(32),.01,seed=4,repeats=5)
  self.assertEqual(a,report.query_budget_metrics(r,torch.ones(32),.01,seed=4,repeats=5))
  self.assertEqual([x['query_n'] for x in a],[10,20]);self.assertEqual(a[0]['median_K_M'],32)
  self.assertEqual(a[0]['detection_fraction'],1.)
 def test_saved_evidence(self):
  from semantic_sign import image_evidence
  with tempfile.TemporaryDirectory() as tmp:
   run=Path(tmp);write_json(run/'config.json',{'analytic_alpha':.01});r=torch.ones(2,32);rows=[]
   for i,e in enumerate(image_evidence(r,torch.ones(32),.01)):
    p=run/'images/test'/image_name(i);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixture')
    rows.append(dict(file=p.name,sha256=sha256(p),coordinates=r[i].tolist(),**e))
   write_json(run/'scores/test.json',rows);carrier=type('Carrier',(),{'signs':torch.ones(32)})()
   a=report.analytic_metrics(run,'test',2,carrier=carrier);self.assertEqual(a['K_M'],32);self.assertEqual(a['p_M'],2**-32)
   rows[0]['K']=0;write_json(run/'scores/test.json',rows)
   with self.assertRaisesRegex(ValueError,'evidence'):report.analytic_metrics(run,'test',2,carrier=carrier)
 def test_summary_artifacts_and_frozen_secondary_null(self):
  from semantic_sign import image_evidence
  from common import read_json
  with tempfile.TemporaryDirectory() as tmp:
   run=Path(tmp);config={'teacher_objective':'semantic_sign_v1','num_bits':32,'analytic_alpha':.01,'fpr':.01,'test_n':20,'calibration_n':20,'score_batch':1}
   write_json(run/'config.json',config);write_json(run/'carrier/complete.json',{'fixture':True})
   write_json(run/'prompts.json',{'fixture':True});write_json(run/'assets.json',{'fixture':True})
   for label in ('teacher_clean_calibration','unadapted_calibration','teacher_clean_test','teacher_wm_test','unadapted'):
    positive=label=='teacher_wm_test';r=torch.ones(20,32) if positive else -torch.ones(20,32);rows=[]
    for i,e in enumerate(image_evidence(r,torch.ones(32),.01)):
     path=run/'images'/label/image_name(i);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixture')
     rows.append(dict(file=path.name,sha256=sha256(path),coordinates=r[i].tolist(),**e))
    write_json(run/'scores'/f'{label}.json',rows)
    write_json(run/'scores'/f'{label}.provenance.json',{'carrier_sha256':sha256(run/'carrier/complete.json'),
     'prompt_sha256':sha256(run/'prompts.json'),'reference_assets_sha256':sha256(run/'assets.json'),
     'batch_size':1,'dtype':'float32','probe_timestep':0,'posterior':'mean'})
   q={'clean_clip':.7,'fp_clip':.699,'clip_drop':.001,'paired_lpips':.1,'paired_pixel_mse':.01,'passes':True}
   candidate={'coordinate_margin_mu':.1,'quality':q,'metrics':{'detection_rate':1.,'K_M':32,'p_M':2**-32}}
   write_json(run/'result/teacher_quality.json',{'selected':candidate,'candidates':[candidate]})
   carrier=type('Carrier',(),{'signs':torch.ones(32)})()
   with patch.object(report.SemanticCarrier,'load',return_value=carrier),patch('student.branches',return_value=[]):
    report.summarize(run)
    with patch.object(report,'random_key_validation',side_effect=AssertionError('must reuse verified cache')):
     report.summarize(run)
   for name in ('analytic_verifier.json','random_key_null_validation.json','fixed_key_clean_fpr.json','model_level_verification.json','query_budget.json','query_budget.csv','teacher_quality.csv','README.md'):
    self.assertTrue((run/'result'/name).exists(),name)
   value=read_json(run/'result/radioactivity.json');self.assertEqual(value['primary'],'analytic_binomial')
   self.assertEqual(value['table'][1]['detection_rate'],1.)
   self.assertEqual(value['table'][1]['secondary_continuous']['auc'],1.)
   self.assertEqual(read_json(run/'result/threshold.json')['role'],'secondary_continuous_comparison')
   with patch.object(report.SemanticCarrier,'load',return_value=carrier):
    write_json(run/'assets.json',{'fixture':'changed'})
    with self.assertRaisesRegex(RuntimeError,'provenance'):
     report.validate_clean_null(run)
    write_json(run/'assets.json',{'fixture':True})
    (run/'images/teacher_clean_calibration'/image_name(0)).write_bytes(b'changed image')
    with self.assertRaisesRegex(RuntimeError,'Scored image changed'):
     report.validate_clean_null(run)

 def test_teacher_quality_sign_gates_and_historical_cache_shape(self):
  import semantic_report
  from PIL import Image
  class Frozen:
   @classmethod
   def from_pretrained(cls,*args,**kwargs):return cls()
   def cuda(self):return self
   def eval(self):return self
   def requires_grad_(self,*args):return self
  class Clip(Frozen):
   def __call__(self,**kwargs):
    n=len(kwargs['images']);return SimpleNamespace(image_embeds=torch.ones(n,1),text_embeds=torch.ones(n,1))
  class Batch(dict):
   def to(self,*args):return self
  class Processor(Frozen):
   def __call__(self,**kwargs):return Batch(images=kwargs['images'])
  class LPIPS(Frozen):
   def __init__(self,*args,**kwargs):pass
   def __call__(self,a,b):return torch.full((len(a),1,1,1),self.value)
  modules={'transformers':SimpleNamespace(CLIPModel=Clip,CLIPProcessor=Processor),
           'lpips':SimpleNamespace(LPIPS=LPIPS),'utility':SimpleNamespace(CLIP_REV='fixture'),
           'semantic_teacher':SimpleNamespace(pixels=lambda paths:torch.zeros(len(paths),3,4,4))}
  with tempfile.TemporaryDirectory() as tmp,patch.dict('sys.modules',modules):
   for objective,limit,value in (('semantic_sign_v1',9.,.2),('threshold_cfg_v1',9.,.2),('semantic_sign_v1',.1,.12)):
    run=Path(tmp)/(objective+str(limit));LPIPS.value=value
    write_json(run/'config.json',{'teacher_objective':objective,'teacher_clip_drop_max':.01,'teacher_lpips_max':limit,'score_batch':1})
    write_json(run/'prompts.json',{'validation':[{'text':'paired fixture'}]})
    for label in ('teacher_clean_validation','candidate'):
     d=run/'images'/label;d.mkdir(parents=True);Image.new('RGB',(4,4),'white').save(d/image_name(0))
     write_json(d/'generation.json',{'seeds':[1],'prompts_sha256':'same'})
    a=semantic_report.teacher_quality(run,'candidate');self.assertEqual(a,semantic_report.teacher_quality(run,'candidate'))
    if objective=='semantic_sign_v1':
     self.assertFalse(a['passes']);self.assertEqual(a['paired_pixel_mse'],0.)
     self.assertEqual(a['identity']['lpips_max'],min(.15,limit));self.assertTrue(a['visual_comparisons'])
    else:
     self.assertTrue(a['passes']);self.assertNotIn('paired_pixel_mse',a)
     self.assertNotIn('pixel_mse_range',a['identity'])

 def test_pixel_visuals(self):
  from PIL import Image
  with tempfile.TemporaryDirectory() as tmp:
   run=Path(tmp)
   for label,color in [('clean',(0,0,0)),('fp',(255,255,255))]:
    d=run/'images'/label;d.mkdir(parents=True);Image.new('RGB',(4,3),color).save(d/image_name(0))
   a=report.paired_pixel_visuals(run,'clean','fp',1);self.assertEqual(a['paired_pixel_mse'],4.)
   self.assertTrue((run/a['visual_comparisons'][0]).exists())
if __name__=='__main__':unittest.main()
