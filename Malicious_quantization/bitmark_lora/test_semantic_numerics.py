"""CPU contracts for teacher precision, numerical failure and isolated stress updates."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from test_semantic_memory import tiny_extractor
from semantic_carrier import SemanticCarrier

class NumericsTests(unittest.TestCase):
    def test_teacher_precision_does_not_change_student_command(self):
        from run import parse_args, config_from_args
        from student import training_command
        from semantic_teacher_train import teacher_precision
        c=config_from_args(parse_args([],method='semantic_fp'))
        self.assertEqual(teacher_precision(c),'bf16')
        cmd=training_command(c,{'student':'base'},Path('run'),{'kind':'full','name':'full_wm','condition':'wm'})
        self.assertEqual(cmd[cmd.index('--mixed_precision')+1],'fp16')
        self.assertEqual(teacher_precision({'precision':'fp16'}),'fp16')

    def test_nonfinite_gradient_rejected_before_weights_change_with_report(self):
        from semantic_teacher_train import checked_teacher_step
        p=torch.nn.Parameter(torch.tensor(1.)); opt=torch.optim.AdamW([p],lr=.1)
        def bad(backward):
            backward(p*torch.tensor(float('inf')))
            return {'loss':1.}
        with tempfile.TemporaryDirectory() as d:
            report=Path(d)/'failure.json'
            with self.assertRaisesRegex(RuntimeError,'gradient'):
                checked_teacher_step([('adapter',p)],opt,None,bad,report,{'step':12})
            self.assertEqual(p.item(),1.)
            import json
            data=json.loads(report.read_text())
            self.assertEqual(data['step'],12)
            self.assertIn('adapter',data['nonfinite_parameters'])
            self.assertFalse(opt.state)

    def test_large_finite_gradients_clip_without_float32_norm_overflow(self):
        from semantic_teacher_train import checked_teacher_step
        p=torch.nn.Parameter(torch.ones(2)); opt=torch.optim.SGD([p],lr=.1)
        def large(backward):
            backward((p*1e20).sum()); return {'loss':2e20}
        result=checked_teacher_step([('adapter',p)],opt,None,large,None,{'step':1})
        self.assertTrue(torch.isfinite(p).all())
        self.assertGreater(result['grad_norm'],1e20)
        torch.testing.assert_close(p,torch.full((2,),1.-.1/(2.**.5)))

    def test_finite_forward_fp16_overflow_is_rejected_bf16_update_is_finite(self):
        from semantic_teacher_train import checked_teacher_step
        for dtype in (torch.float16,torch.bfloat16):
            p=torch.nn.Parameter(torch.zeros(1));opt=torch.optim.SGD([p],lr=.1)
            def compute(backward):
                loss=(p.to(dtype)*1e5).float().sum()
                self.assertTrue(torch.isfinite(loss))
                backward(loss);return {'loss':loss.item()}
            if dtype==torch.float16:
                with self.assertRaisesRegex(RuntimeError,'gradient'):
                    checked_teacher_step([('adapter',p)],opt,None,compute,None,{})
                self.assertEqual(p.item(),0.)
            else:
                checked_teacher_step([('adapter',p)],opt,None,compute,None,{})
                self.assertAlmostEqual(p.item(),-.1,places=5)

    def test_boundary_hook_reports_nonfinite_backward(self):
        from semantic_teacher_train import watch_gradient,checked_teacher_step
        p=torch.nn.Parameter(torch.tensor(1.));opt=torch.optim.SGD([p],lr=.1)
        def compute(backward):
            intermediate=watch_gradient(p*2.,'cycle features')
            backward(intermediate*float('inf'));return {}
        with tempfile.TemporaryDirectory() as d:
            report=Path(d)/'failure.json'
            with self.assertRaisesRegex(RuntimeError,'cycle features'):
                checked_teacher_step([('adapter',p)],opt,None,compute,report,{'step':5})
            self.assertEqual(p.item(),1.)
            self.assertTrue(report.exists())

    def test_checkpoint_selection_is_numeric_and_diagnostic_does_not_mutate_files(self):
        from semantic_teacher_train import latest_teacher_adapters
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);candidate=root/'teacher_fp/candidates/shift_0.5'
            for step in (500,1000,150):
                path=candidate/f'checkpoint-{step}';path.mkdir(parents=True)
                (path/'adapter_model.safetensors').write_bytes(b'weights')
                (path/'adapter_config.json').write_text('{}')
            before={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
            self.assertEqual(latest_teacher_adapters(root),[(.5,candidate/'checkpoint-1000')])
            self.assertEqual(before,{str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()})

    def test_stress_updates_restore_parameters_and_rng_and_cover_cycle_boundaries(self):
        from semantic_teacher_train import stress_teacher
        torch.manual_seed(13); ext=tiny_extractor(True)
        class Teacher(torch.nn.Module):
            def __init__(self):super().__init__();self.layer=torch.nn.Conv2d(2,2,1)
            def forward(self,z,t,encoder_hidden_states):return SimpleNamespace(sample=self.layer(z).tanh())
        model=Teacher();before=copy.deepcopy(model.state_dict());rng=torch.get_rng_state().clone()
        carrier=SemanticCarrier(torch.zeros(2),torch.eye(2),torch.ones(2),torch.tensor([1.,-1.]),[0,1])
        scheduler=SimpleNamespace(alphas_cumprod=torch.linspace(.99,.1,1000),config=SimpleNamespace(prediction_type='epsilon',num_train_timesteps=1000),
                                  add_noise=lambda z,n,t:z+n*.1)
        c={'precision':'fp16','teacher_precision':'no','teacher_micro_batch':1,'teacher_lr':1e-4,
           'semantic_t_max':500,'cycle_every':4,'lambda_preserve':1.,'lambda_code':.1,'lambda_perp':.01,'lambda_cycle':.05,'seed':0}
        z=torch.zeros(2,2,8,8);text=torch.ones(2,2)
        with tempfile.TemporaryDirectory() as d:
            result=stress_teacher(model,ext,carrier,scheduler,z,text,.5,c,Path(d),steps=4)
        self.assertTrue(result['passed'])
        self.assertEqual([r['timestep'] for r in result['checks']],[0,250,500,501])
        self.assertTrue(all(r['ramp']==1. for r in result['checks']))
        self.assertGreater(result['checks'][0]['cycle'],0.)
        for k,v in before.items():torch.testing.assert_close(model.state_dict()[k],v,rtol=0,atol=0)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        class Broken(Teacher):
            def forward(self,z,t,encoder_hidden_states):
                result=super().forward(z,t,encoder_hidden_states)
                result.sample=result.sample*float('nan')
                return result
        broken=Broken();saved=copy.deepcopy(broken.state_dict());rng=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                stress_teacher(broken,ext,carrier,scheduler,z,text,.5,c,Path(d),steps=4)
            self.assertTrue((Path(d)/'failure.json').exists())
        for k,v in saved.items():torch.testing.assert_close(broken.state_dict()[k],v,rtol=0,atol=0)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))

if __name__=='__main__':unittest.main()
