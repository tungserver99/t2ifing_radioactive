"""Audit regressions for native pipeline scheduler configuration, without weights."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from common import write_json


class NativeSchedulerAuditTests(unittest.TestCase):
    def test_native_pipeline_normalization_precedes_late_id_selection(self):
        from semantic_sign_train import sign_schedule
        for offset in (0,1):
            pipeline_calls=[]
            class Scheduler:
                @classmethod
                def from_pretrained(cls,*args,**kwargs):return cls()
                def __init__(self):
                    self.config=SimpleNamespace(steps_offset=offset,clip_sample=True)
                    self.alphas_cumprod=torch.linspace(.99,.01,1000)
                def set_timesteps(self,steps):
                    self.timesteps=torch.tensor([8,6,4,2,0])+self.config.steps_offset
            class Pipeline:
                def __init__(self,**kwargs):
                    pipeline_calls.append(kwargs)
                    assert all(kwargs[name] is None for name in ('vae','unet','text_encoder','tokenizer','safety_checker','feature_extractor'))
                    self.scheduler=kwargs['scheduler']
                    self.scheduler.config.steps_offset=1
                    self.scheduler.config.clip_sample=False
            module=SimpleNamespace(PNDMScheduler=Scheduler,StableDiffusionPipeline=Pipeline)
            with tempfile.TemporaryDirectory() as d:
                run=Path(d);model=run/'snapshot'
                write_json(model/'model_index.json',{'scheduler':['diffusers','PNDMScheduler']})
                write_json(model/'scheduler/scheduler_config.json',{'steps_offset':offset,'clip_sample':True})
                write_json(run/'assets.json',{'teacher_clean':str(model)})
                c={'inference_steps':50,'late_inference_step_count':3}
                write_json(run/'config.json',c);write_json(run/'carrier/complete.json',{})
                training=SimpleNamespace(alphas_cumprod=torch.linspace(.99,.01,1000))
                with patch('semantic_sign_train.importlib.import_module',return_value=module):
                    late,proof=sign_schedule(run,c,training)
                self.assertEqual(late,[5,3,1])
                self.assertEqual(len(pipeline_calls),1)
                self.assertEqual(proof['late_timesteps'],[5,3,1])

    @unittest.skipUnless(importlib.util.find_spec('diffusers'),'Optional Diffusers absent; no installation in CPU audit')
    def test_real_pndm_and_pipeline_constructor_without_model_components(self):
        from diffusers import PNDMScheduler,DDPMScheduler,StableDiffusionPipeline
        from semantic_sign import derive_late_timesteps
        config=dict(num_train_timesteps=1000,beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear',
                    skip_prk_steps=True,steps_offset=0,set_alpha_to_one=False,prediction_type='epsilon')
        native=PNDMScheduler(**config)
        # Only a scheduler and empty pipeline shell, no weights, GPU or downloads.
        shell=StableDiffusionPipeline(vae=None,text_encoder=None,tokenizer=None,unet=None,scheduler=native,
                                      safety_checker=None,feature_extractor=None,requires_safety_checker=False)
        training=DDPMScheduler.from_config(native.config)
        late=derive_late_timesteps(shell.scheduler,training,50,5)
        self.assertEqual(late,[81,61,41,21,1])
