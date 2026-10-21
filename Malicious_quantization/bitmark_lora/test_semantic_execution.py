"""CPU proofs for feature-only execution and narrowly scoped resume upgrades."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from semantic_carrier import SemanticExtractor
from test_semantic_memory import tiny_extractor


class FeatureExecutionTests(unittest.TestCase):
    def test_skip_decoder_preserves_features_gradients_rng_and_hooks(self):
        class UNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.down=torch.nn.Conv2d(2,3,1)
                self.mid_block=torch.nn.Sequential(torch.nn.SiLU(),torch.nn.Conv2d(3,3,1))
                self.up=torch.nn.Conv2d(3,2,1)
                self.tail_calls=0
            def forward(self,z,t,encoder_hidden_states):
                h=self.mid_block(self.down(z)+encoder_hidden_states.mean()*0.1)
                self.tail_calls+=1
                return SimpleNamespace(sample=self.up(h))
        torch.manual_seed(31)
        base=UNet()
        for checkpointing in (False,True):
            ext=tiny_extractor(checkpointing)
            ext.unet=copy.deepcopy(base).eval().requires_grad_(False)
            reference=copy.deepcopy(base).eval().requires_grad_(False)
            z=torch.randn(2,2,5,5,requires_grad=True)
            x=z.detach().clone().requires_grad_(True)
            hidden=torch.randn(2,2)
            captured=[]
            handle=reference.mid_block.register_forward_hook(lambda m,a,o:captured.append(o))
            reference(x,torch.zeros(2,dtype=torch.long),encoder_hidden_states=hidden)
            handle.remove()
            expected=captured[0].mean((-2,-1))
            rng=torch.get_rng_state().clone()
            actual=ext.encode_latent(z,text_hidden=hidden)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            actual.square().sum().backward()
            expected.square().sum().backward()
            torch.testing.assert_close(z.grad,x.grad,rtol=0,atol=0)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(ext.unet.tail_calls,0)
            self.assertEqual(len(ext.unet.mid_block._forward_hooks),0)
            with torch.no_grad():
                torch.testing.assert_close(ext.encode_latent(z,text_hidden=hidden),expected,rtol=0,atol=0)
            self.assertEqual(ext.unet.tail_calls,0)
            # Ordinary noise prediction must still execute the complete UNet.
            ext.unet(z,torch.zeros(2,dtype=torch.long),encoder_hidden_states=hidden)
            self.assertEqual(ext.unet.tail_calls,1)

    def test_genuine_error_propagates_and_removes_hook(self):
        ext=tiny_extractor(False)
        class Broken(torch.nn.Module):
            def forward(self,x):raise RuntimeError('real mid-block failure')
        ext.unet.mid_block=Broken()
        with self.assertRaisesRegex(RuntimeError,'real mid-block failure'):
            ext.encode_latent(torch.randn(1,2,4,4),text_hidden=torch.ones(1,2))
        self.assertEqual(len(ext.unet.mid_block._forward_hooks),0)


class ExecutionUpgradeTests(unittest.TestCase):
    def test_upgrade_is_explicit_narrow_audited_and_preserves_training_state(self):
        from semantic_execution import upgrade_execution_manifest,BASELINE_HASHES,TARGET_HASHES
        from common import write_json,read_json,sha256,HERE
        with tempfile.TemporaryDirectory() as d:
            run=Path(d)
            old={name:values[0] for name,values in BASELINE_HASHES.items()}
            old['other.py']='unchanged'
            new={**old,'semantic_carrier.py':TARGET_HASHES['semantic_carrier.py'],'run.py':TARGET_HASHES['run.py'],
                 'semantic_execution.py':'newpolicy','test_semantic_execution.py':'newtests'}
            write_json(run/'code_manifest.json',old)
            write_json(run/'config.json',{'watermark_method':'semantic_fp','teacher_objective':'threshold_cfg_v1'})
            checkpoint=run/'teacher_fp/candidates/kappa_1.0/checkpoint-100/training_state.pt'
            checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b'optimizer + RNG must stay exact')
            original=(run/'config.json').read_bytes(),checkpoint.read_bytes()
            # Unrelated source changes must never pass this upgrade.
            for bad in ({**new,'semantic_carrier.py':'unreviewed optimization'}, {**new,'other.py':'changed'}, {k:v for k,v in new.items() if k!='other.py'}):
                with self.assertRaisesRegex(RuntimeError,'Unapproved'):
                    upgrade_execution_manifest(run,bad)
                self.assertEqual(read_json(run/'code_manifest.json'),old)
            upgrade_execution_manifest(run,new)
            self.assertEqual(read_json(run/'code_manifest.json'),new)
            audit=read_json(run/'execution_upgrades/midblock_features_v1.json')
            self.assertEqual(audit['before'],old);self.assertEqual(audit['after'],new)
            self.assertEqual(original,((run/'config.json').read_bytes(),checkpoint.read_bytes()))
            upgrade_execution_manifest(run,new)  # Idempotent after successful migration.
            write_json(run/'code_manifest.json',{**old,'semantic_carrier.py':'unknown previous revision'})
            with self.assertRaisesRegex(RuntimeError,'Unapproved'):
                upgrade_execution_manifest(run,new)

    def test_upgrade_requires_resume_and_does_not_change_experiment_config(self):
        from run import parse_args,config_from_args
        with self.assertRaises(SystemExit):parse_args(['--upgrade-execution'],method='semantic_fp')
        a=parse_args(['--resume','old','--upgrade-execution'],method='semantic_fp')
        self.assertTrue(a.upgrade_execution)
        self.assertNotIn('upgrade_execution',config_from_args(a))
