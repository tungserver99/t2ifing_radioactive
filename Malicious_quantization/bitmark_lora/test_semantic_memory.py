"""CPU regressions for bounded-memory teacher training and project-root outputs."""
import contextlib
import copy
import io
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from semantic_carrier import SemanticCarrier, SemanticExtractor


def tiny_extractor(checkpointing):
    class VAE(torch.nn.Module):
        config=SimpleNamespace(scaling_factor=2.)
        def __init__(self):
            super().__init__();self.encoder=torch.nn.Conv2d(2,2,1);self.decoder=torch.nn.Conv2d(2,2,1)
        def encode(self,x):return SimpleNamespace(latent_dist=SimpleNamespace(mean=self.encoder(x).tanh()))
        def decode(self,z):return SimpleNamespace(sample=self.decoder(z).tanh())
    class UNet(torch.nn.Module):
        def __init__(self):
            super().__init__();self.mid_block=torch.nn.Sequential(torch.nn.Conv2d(2,2,1),torch.nn.SiLU())
        def forward(self,z,t,encoder_hidden_states):return SimpleNamespace(sample=self.mid_block(z))
    class Text(torch.nn.Module):
        def forward(self,tokens):return (tokens.float(),)
    pipe=SimpleNamespace(vae=VAE(),unet=UNet(),text_encoder=Text(),tokenizer=None)
    return SemanticExtractor(pipe,'cpu',gradient_checkpointing=checkpointing)


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_preserves_features_cycle_and_input_gradient(self):
        torch.manual_seed(7);ordinary=tiny_extractor(False)
        torch.manual_seed(7);saved=tiny_extractor(True)
        x=torch.randn(2,2,8,8,requires_grad=True);y=x.detach().clone().requires_grad_(True)
        hidden=torch.ones(2,2)
        def result(ext,z):return ext.encode_latent(z,text_hidden=hidden)+ext.encode_latent(ext.cycle(z),text_hidden=hidden)
        a=result(ordinary,x);b=result(saved,y)
        torch.testing.assert_close(a,b)
        a.square().sum().backward();b.square().sum().backward()
        torch.testing.assert_close(x.grad,y.grad)
        self.assertEqual(len(saved.unet.mid_block._forward_hooks),0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for m in (saved.vae,saved.unet) for p in m.parameters()))

    def test_checkpoint_retains_fewer_activation_elements(self):
        torch.manual_seed(9);ordinary=tiny_extractor(False)
        torch.manual_seed(9);saved=tiny_extractor(True)
        def retained(ext):
            sizes=[]
            with torch.autograd.graph.saved_tensors_hooks(lambda t:(sizes.append(t.numel()) or t),lambda t:t):
                z=torch.randn(2,2,16,16,requires_grad=True)
                f=ext.encode_latent(ext.cycle(z),text_hidden=torch.ones(2,2))
            return sum(sizes)
        self.assertLess(retained(saved),retained(ordinary))


class MicrobatchTests(unittest.TestCase):
    def test_effective_batch_loss_and_gradients_match_full_batch(self):
        from semantic_teacher_train import teacher_backward
        torch.manual_seed(11);ext=tiny_extractor(False)
        class Teacher(torch.nn.Module):
            def __init__(self):super().__init__();self.layer=torch.nn.Conv2d(2,2,1);self.seen=[]
            def forward(self,z,t,encoder_hidden_states):
                self.seen.append(len(z));return SimpleNamespace(sample=self.layer(z).tanh())
        base=Teacher();carrier=SemanticCarrier(torch.zeros(2),torch.eye(2),torch.tensor([.7,1.2]),torch.tensor([1.,-1.]),[0,1])
        scheduler=SimpleNamespace(alphas_cumprod=torch.linspace(.99,.1,1000),config=SimpleNamespace(prediction_type='epsilon'))
        noisy=torch.randn(5,2,8,8)*.1;text=torch.ones(5,2)
        base_c={'precision':'no','semantic_t_max':500,'cycle_every':4,'lambda_preserve':1.,
                'lambda_code':.1,'lambda_perp':.01,'lambda_cycle':.05}
        for step,t in ((249,torch.tensor([1,600,2,800,3])),(251,torch.tensor([1,600,2,800,3])),
                       (752,torch.tensor([1,600,2,800,3])),(752,torch.tensor([600,700,800,900,999]))):
            results=[]
            for micro in (5,1,2):
                model=copy.deepcopy(base);c={**base_c,'teacher_micro_batch':micro}
                values=teacher_backward(model,ext,carrier,scheduler,noisy,text,t,.5,c,step,lambda loss:loss.backward())
                results.append((values,[p.grad.clone() for p in model.parameters()]))
                self.assertLessEqual(max(model.seen),micro)
            for values,gradients in results[1:]:
                for name in ('loss','preserve','code','perp','cycle','semantic_fraction','ramp'):
                    self.assertAlmostEqual(values[name],results[0][0][name],places=6,msg=name)
                for a,b in zip(gradients,results[0][1]):torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)


class OutputLocationTests(unittest.TestCase):
    def test_semantic_outputs_beside_main_keep_legacy_location(self):
        import common
        with patch.dict(os.environ,{},clear=True):
            self.assertEqual(common.results_root('semantic_fp'),common.HERE.parent.parent)
            self.assertEqual(common.results_root('serum'),common.HERE.parent)
            self.assertEqual(common.results_root(),common.HERE.parent)
        with patch.dict(os.environ,{'BITMARK_RESULTS_ROOT':str(common.HERE.parent.parent/'explicit_results')}):
            self.assertEqual(common.results_root('semantic_fp'),(common.HERE.parent.parent/'explicit_results').resolve())

    def test_cli_default_microbatch_one_and_plan_output_root(self):
        import common
        from run import main,parse_args
        c=parse_args([],method='semantic_fp')
        self.assertEqual((c.teacher_batch,c.teacher_micro_batch),(4,1))
        stream=io.StringIO()
        with patch.dict(os.environ,{},clear=True),contextlib.redirect_stdout(stream):main(['--plan-only'],method='semantic_fp')
        self.assertEqual(Path(json.loads(stream.getvalue())['results_root']),common.HERE.parent.parent)

if __name__=='__main__':unittest.main()
