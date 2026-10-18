"""Second-pass specification audit: CPU algebra, gradients, integrity and edge cases."""
import ast
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from common import HERE, image_name, read_json, sha256, write_json
from semantic_carrier import SemanticCarrier, SemanticExtractor, save_carrier
from semantic_teacher_train import predicted_x0, semantic_losses


class MathematicalAuditTests(unittest.TestCase):
    def test_nonzero_losses_and_gradients_match_hand_calculation(self):
        carrier=SemanticCarrier(torch.tensor([10.,20.,30.]),torch.eye(3)[:,:2],
                                torch.tensor([2.,4.]),torch.tensor([1.,-1.]),[0,1])
        clean=torch.tensor([[7.,-8.,9.]])
        delta=torch.tensor([[2.,4.,6.]],requires_grad=True)
        code,perp=semantic_losses(clean,clean+delta,carrier,.5)
        self.assertAlmostEqual(code.item(),2.5)
        self.assertAlmostEqual(perp.item(),12.)
        (code+perp).backward()
        torch.testing.assert_close(delta.grad,torch.tensor([[.5,.75,4.]]))
        shifted_clean=clean+torch.tensor([[100.,-50.,900.]])
        other=semantic_losses(shifted_clean,shifted_clean+delta.detach(),carrier,.5)
        torch.testing.assert_close(torch.stack(other),torch.stack([code,perp]).detach())

    def test_x0_matches_vendored_ddpm_prediction_branch(self):
        # Execute the upstream branch verbatim, without importing/installing Diffusers.
        source=HERE.parent/'serum/third_party/diffusers/src/diffusers/schedulers/scheduling_ddpm.py'
        tree=ast.parse(source.read_text(encoding='utf8'))
        step=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='step')
        branch=next(n for n in step.body if isinstance(n,ast.If)
                    and isinstance(n.test,ast.Compare)
                    and ast.unparse(n.test.left)=='self.config.prediction_type')
        block=ast.Module(body=[copy.deepcopy(branch)],type_ignores=[])
        compiled=compile(ast.fix_missing_locations(block),str(source),'exec')
        z=torch.randn(3,4,4,4,generator=torch.Generator().manual_seed(31))
        prediction=torch.randn(z.shape,generator=torch.Generator().manual_seed(32))
        scheduler=SimpleNamespace(alphas_cumprod=torch.tensor([.999,.5,.0001]),config=SimpleNamespace())
        alpha=scheduler.alphas_cumprod.reshape(3,1,1,1)
        for kind in ('epsilon','v_prediction','sample'):
            scheduler.config.prediction_type=kind
            namespace={'self':scheduler,'sample':z,'model_output':prediction,
                       'alpha_prod_t':alpha,'beta_prod_t':1-alpha}
            exec(compiled,namespace)
            torch.testing.assert_close(predicted_x0(scheduler,z,prediction,torch.arange(3)),namespace['pred_original_sample'])

    def test_cycle_loss_updates_only_teacher_through_frozen_reference(self):
        class VAE(torch.nn.Module):
            config=SimpleNamespace(scaling_factor=2.)
            def __init__(self): super().__init__();self.gain=torch.nn.Parameter(torch.tensor(.75))
            def encode(self,x): return SimpleNamespace(latent_dist=SimpleNamespace(mean=x*self.gain))
            def decode(self,z): return SimpleNamespace(sample=z*.5)
        class Reference(torch.nn.Module):
            def __init__(self):
                super().__init__();self.mid_block=torch.nn.Conv2d(2,2,1,bias=False)
                with torch.no_grad():self.mid_block.weight.copy_(torch.eye(2).reshape(2,2,1,1))
            def forward(self,z,t,encoder_hidden_states):return SimpleNamespace(sample=self.mid_block(z))
        class Text(torch.nn.Module):
            def forward(self,x):return (x.float(),)
        ext=SemanticExtractor(SimpleNamespace(vae=VAE(),unet=Reference(),text_encoder=Text(),tokenizer=None),'cpu')
        key=SemanticCarrier(torch.zeros(2),torch.eye(2),torch.ones(2),torch.tensor([1.,-1.]),[0,1])
        scheduler=SimpleNamespace(alphas_cumprod=torch.tensor([.5]),config=SimpleNamespace(prediction_type='epsilon'))
        noisy=torch.full((2,2,4,4),.1);t=torch.zeros(2,dtype=torch.long);text=torch.zeros(2,2)
        prediction=torch.nn.Parameter(torch.zeros_like(noisy));optimizer=torch.optim.SGD([prediction],lr=.1)
        with torch.no_grad():
            clean_z=predicted_x0(scheduler,noisy,torch.zeros_like(noisy),t)
            clean_f=ext.encode_latent(clean_z,text_hidden=text)
            clean_cycle=ext.encode_latent(ext.cycle(clean_z),text_hidden=text)
        def loss():
            fp_z=predicted_x0(scheduler,noisy,prediction,t)
            fp_f=ext.encode_latent(fp_z,text_hidden=text)
            fp_cycle=ext.encode_latent(ext.cycle(fp_z),text_hidden=text)
            code,perp=semantic_losses(clean_f,fp_f,key,.5)
            cycle,_=semantic_losses(clean_cycle,fp_cycle,key,.5)
            return code+perp+cycle
        before=loss();before.backward()
        self.assertGreater(prediction.grad.abs().sum().item(),0.)
        for module in (ext.vae,ext.unet,ext.text_encoder):
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in module.parameters()))
        optimizer.step()
        self.assertLess(loss().item(),before.item())
        self.assertEqual(len(ext.unet.mid_block._forward_hooks),0)


class IntegrityAuditTests(unittest.TestCase):
    def test_distinct_valid_strengths_have_distinct_artifact_paths(self):
        from semantic_teacher_train import candidate_tag
        self.assertNotEqual(candidate_tag(.5000001),candidate_tag(.5000002))
        self.assertEqual(candidate_tag(.5),candidate_tag(.50))
        for value in (float('nan'),float('inf'),0.,-.5):
            with self.assertRaises(ValueError):candidate_tag(value)

    def test_calibration_rejects_score_coordinate_disagreement(self):
        from semantic_report import freeze_null
        c={'carrier_pc_min':0,'carrier_pc_max':15,'stable_candidate_count':16,'num_bits':8,
           'key_seed':0,'calibration_n':100,'fpr':.01}
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);write_json(run/'config.json',c)
            features=torch.randn(100,20,generator=torch.Generator().manual_seed(4))
            save_carrier(run/'carrier',features,features,c,{'train_prompt_sha256':'clean train'})
            rows=[]
            for i in range(100):
                p=run/'images/teacher_clean_calibration'/image_name(i);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'image')
                rows.append({'file':p.name,'sha256':sha256(p),'score':100.,'coordinates':[0.]*8})
            write_json(run/'scores/teacher_clean_calibration.json',rows)
            with self.assertRaisesRegex(ValueError,'[Ss]core'):
                freeze_null(run,'teacher')
            self.assertFalse((run/'result/threshold_teacher.json').exists())

    def test_candidate_quality_gate_cannot_accept_nonfinite_cached_metrics(self):
        from semantic_teacher_train import choose_candidate
        for invalid in (float('nan'),float('inf')):
            rows=[{'alpha':.5,'quality':{'passes':True},
                   'metrics':{'detection_rate':.95,'mean_score':invalid}}]
            with self.assertRaises(ValueError):choose_candidate(rows)

if __name__=='__main__':unittest.main()
