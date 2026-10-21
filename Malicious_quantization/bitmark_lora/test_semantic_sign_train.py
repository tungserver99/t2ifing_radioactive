"""RSSF training objective with tiny CPU components, never pretrained models."""
import copy
import unittest
from types import SimpleNamespace
import torch
from test_semantic_memory import tiny_extractor
from semantic_carrier import SemanticCarrier

class SignTrainingTests(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(47)
        ext=tiny_extractor(True);ext.text=lambda prompts:torch.zeros(len(prompts),2)
        class Teacher(torch.nn.Module):
            def __init__(self):super().__init__();self.layer=torch.nn.Conv2d(2,2,1)
            def forward(self,z,t,encoder_hidden_states):
                return SimpleNamespace(sample=self.layer(z)+encoder_hidden_states.mean(-1).reshape(-1,1,1,1)*.1)
        class LPIPS(torch.nn.Module):
            def __init__(self):super().__init__();self.scale=torch.nn.Parameter(torch.tensor(1.),requires_grad=False)
            def forward(self,a,b):return ((a-b)*self.scale).square().mean((1,2,3),keepdim=True)
        model=Teacher();carrier=SemanticCarrier(torch.zeros(2),torch.eye(2),torch.ones(2),torch.tensor([1.,-1.]),[0,1])
        scheduler=SimpleNamespace(alphas_cumprod=torch.linspace(.99,.1,1000),config=SimpleNamespace(prediction_type='v_prediction',num_train_timesteps=1000))
        cfg={'teacher_precision':'no','precision':'no','lambda_preserve':1.,'lambda_fp':.8,'lambda_quality':1.,
             'lambda_lpips':1.,'lambda_pixel':1.,'sign_k_train':1,'analytic_alpha':.3}
        z=torch.randn(3,2,4,4)*.1;text=torch.ones(3,2)
        return ext,model,carrier,scheduler,LPIPS(),cfg,z,text

    def test_loss_gradient_microbatch_equivalence_and_frozen_image_path(self):
        from semantic_sign_train import sign_teacher_backward
        ext,base,carrier,scheduler,lpips,c,z,text=self.fixture()
        noise=torch.randn_like(z);late_noise=torch.randn_like(z)
        t=torch.tensor([0,500,999]);late_t=torch.tensor([1,21,41])
        values=[]
        for micro in (3,1,2):
            model=copy.deepcopy(base)
            result=sign_teacher_backward(model,ext,carrier,scheduler,lpips,z,text,noise,t,late_noise,late_t,.2,
                                         {**c,'teacher_micro_batch':micro},lambda loss:loss.backward())
            self.assertAlmostEqual(result['loss'],result['preserve']+.8*result['fp']+result['quality'],places=6)
            self.assertAlmostEqual(result['quality'],result['lpips']+result['pixel_mse'],places=6)
            self.assertGreater(result['pixel_mse'],0.)
            values.append((result,[p.grad.clone() for p in model.parameters()]))
        for result,grads in values[1:]:
            for key in ('loss','preserve','fp','quality','lpips','pixel_mse'):
                self.assertAlmostEqual(result[key],values[0][0][key],places=5)
            for a,b in zip(grads,values[0][1]):torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
        for module in (ext.vae,ext.unet,ext.text_encoder,lpips):
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in module.parameters()))
        self.assertGreater(sum(g.abs().sum() for g in values[0][1]),0)

    def test_image_quality_alone_propagates_to_teacher(self):
        from semantic_sign_train import sign_teacher_backward
        ext,model,carrier,scheduler,lpips,c,z,text=self.fixture()
        c.update(lambda_preserve=0.,lambda_fp=0.,lambda_quality=1.,teacher_micro_batch=1)
        noise=torch.randn_like(z);t=torch.tensor([0,500,999]);late_t=torch.tensor([1,21,41])
        sign_teacher_backward(model,ext,carrier,scheduler,lpips,z,text,noise,t,noise,late_t,.2,c,lambda l:l.backward())
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()))

    def test_selection_gates_then_lowest_distortion_not_strongest_tpr(self):
        from semantic_sign_train import choose_sign_candidate
        def row(mu,tpr,lp,pixel,clip=.005):
            return {'coordinate_margin_mu':mu,'metrics':{'detection_rate':tpr},
                    'quality':{'paired_lpips':lp,'paired_pixel_mse':pixel,'clip_drop':clip}}
        rows=[row(.1,.91,.10,.02),row(.2,1.,.12,.01),row(.3,1.,.3,.01)]
        self.assertEqual(choose_sign_candidate(rows,{})['coordinate_margin_mu'],.1)
        tied=[row(.2,.91,.1,.03),row(.3,.99,.1,.02),row(.1,.91,.1,.02)]
        self.assertEqual(choose_sign_candidate(tied,{})['coordinate_margin_mu'],.1)
        self.assertEqual(choose_sign_candidate([row(.1,1.,.01,.01,clip=.02),row(.2,.91,.1,.1)],{})['coordinate_margin_mu'],.2)
        rows[0]['metrics']['detection_rate']=.89
        self.assertEqual(choose_sign_candidate(rows,{})['coordinate_margin_mu'],.2)
        with self.assertRaisesRegex(ValueError,'No RSSF'):
            choose_sign_candidate([row(.1,1.,.2,.1)],{})


    def test_each_image_loss_independently_has_teacher_gradient(self):
        from semantic_sign_train import sign_teacher_backward
        for path in ('semantic','lpips','pixel'):
            ext,model,carrier,scheduler,lpips,c,z,text=self.fixture()
            c.update(lambda_preserve=0.,lambda_fp=float(path=='semantic'),lambda_quality=float(path!='semantic'),
                     lambda_lpips=float(path=='lpips'),lambda_pixel=float(path=='pixel'),teacher_micro_batch=1)
            noise=torch.randn_like(z);t=torch.tensor([0,500,999]);late_t=torch.tensor([1,21,41])
            sign_teacher_backward(model,ext,carrier,scheduler,lpips,z,text,noise,t,noise,late_t,.2,c,lambda l:l.backward())
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()),path)
            for frozen in (ext.vae,ext.unet,ext.text_encoder,lpips):
                self.assertTrue(all(p.grad is None for p in frozen.parameters()),path)

    def test_full_rssf_state_resume_and_probe_restoration(self):
        import tempfile
        from pathlib import Path
        from semantic_sign_train import sign_teacher_backward,sample_objective_batch,stress_sign
        from semantic_teacher_train import checked_teacher_step
        from semantic_checkpoint import save_training_state,restore_training_state
        ext,model,carrier,scheduler,lpips,c,z,text=self.fixture()
        c.update(seed=7,teacher_lr=.001,teacher_micro_batch=1)
        opt=torch.optim.AdamW(model.parameters(),lr=c['teacher_lr']);rng=torch.Generator().manual_seed(18)
        late=[1,21,41,61,81]
        def update(net,optimizer,data_rng):
            batch=sample_objective_batch(scheduler,z,data_rng,late)
            checked_teacher_step(net.named_parameters(),optimizer,None,
                lambda back:sign_teacher_backward(net,ext,carrier,scheduler,lpips,z,text,*batch,.2,c,back),None,{})
            return [value.clone() for value in batch]
        update(model,opt,rng)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);identity={'objective':'semantic_sign_v1'}
            save_training_state(root/'checkpoint-1',model,opt,None,rng,[2,0,1],3,1,identity)
            expected_batch=update(model,opt,rng)
            expected=copy.deepcopy(model.state_dict());expected_opt=copy.deepcopy(opt.state_dict())
            restored=copy.deepcopy(model);other=torch.optim.AdamW(restored.parameters(),lr=c['teacher_lr']);g=torch.Generator()
            state=restore_training_state(root/'checkpoint-1',restored,other,None,g,identity)
            self.assertEqual((state['step'],state['order'],state['offset']),(1,[2,0,1],3))
            before=copy.deepcopy(restored.state_dict());global_rng=torch.get_rng_state().clone();sampler=g.get_state().clone()
            stress_sign(restored,ext,carrier,scheduler,lpips,z,text,.2,c,root/'probe',late,steps=4)
            for key,value in before.items():torch.testing.assert_close(value,restored.state_dict()[key],rtol=0,atol=0)
            self.assertTrue(torch.equal(global_rng,torch.get_rng_state()));self.assertTrue(torch.equal(sampler,g.get_state()))
            actual_batch=update(restored,other,g)
            for a,b in zip(expected_batch,actual_batch):torch.testing.assert_close(a,b,rtol=0,atol=0)
            for key,value in expected.items():torch.testing.assert_close(value,restored.state_dict()[key],rtol=0,atol=0)
            for index,row in expected_opt['state'].items():
                for key,value in row.items():torch.testing.assert_close(value,other.state_dict()['state'][index][key],rtol=0,atol=0)
            # Failure after a real scratch update must still restore the original state.
            class Broken(torch.nn.Module):
                def __init__(self):super().__init__();self.calls=0
                def forward(self,a,b):
                    self.calls+=1
                    if self.calls>6:raise RuntimeError('probe failure')
                    return (a-b).square().mean().reshape(1)
            before=copy.deepcopy(restored.state_dict());global_rng=torch.get_rng_state().clone()
            with self.assertRaisesRegex(RuntimeError,'probe failure'):
                stress_sign(restored,ext,carrier,scheduler,Broken(),z,text,.2,c,root/'failed',late,steps=4)
            for key,value in before.items():torch.testing.assert_close(value,restored.state_dict()[key],rtol=0,atol=0)
            self.assertTrue(torch.equal(global_rng,torch.get_rng_state()))
