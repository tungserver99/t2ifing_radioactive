"""CPU checks of actual gradients, merge/reload, gates and launch protocol."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import paths
from new_watermark.method import make_detector, score_features, choose_teacher, predict_x0
from new_watermark.launch import arguments, config


class Protocol(unittest.TestCase):
    def test_full_defaults_and_teacher_guard(self):
        c=config(arguments([]))
        self.assertGreater(c['meta_weight'],0)
        self.assertEqual(c['gates']['min_teacher_tpr'],.95)
        self.assertEqual(c['gates']['min_teacher_ci_low'],.90)
        self.assertEqual(c['checkpoint_steps'],[2500,5000,7500,10000])
        self.assertEqual(len(c['student_branches']),2)
        self.assertTrue(all(b['condition']=='wm' for b in c['student_branches']))

    def test_final_training_commands_use_same_lr_and_full_horizon(self):
        from student import training_command, training_conditions
        c=config(arguments([]))
        self.assertEqual(training_conditions(c),('wm',))
        self.assertFalse(c['train_clean_controls'])
        for branch in c['student_branches']:
            cmd=training_command(c,{'student':'/base'},Path('/run'),branch)
            for flag,value in [('--learning_rate','0.0001'),('--max_train_steps','10000'),
                               ('--lr_scheduler','constant'),('--train_batch_size','4')]:
                self.assertEqual(cmd[cmd.index(flag)+1],value)
            self.assertNotIn('--scale_lr',cmd)

    def test_strong_teacher_required_even_if_pilot_perfect(self):
        gates=config(arguments([]))['gates']
        weak=dict(step=250,teacher_tpr=.6,teacher_ci_low=.54,psnr=30.,clip_drop=0.,pilot_tpr=1.,clean_pilot_fpr=.01,pilot_clip_drop=0.)
        strong={**weak,'step':1000,'teacher_tpr':.99,'teacher_ci_low':.96,'pilot_tpr':.4}
        self.assertEqual(choose_teacher([weak,strong],gates)['selected_step'],1000)
        self.assertFalse(choose_teacher([weak],gates)['valid_design_winner'])
        self.assertFalse(choose_teacher([{**strong,'pilot_tpr':.03}],gates)['valid_design_winner'])
        self.assertFalse(choose_teacher([{**strong,'teacher_ci_low':.85}],gates)['valid_design_winner'])

    def test_detector_is_fixed_and_score_matches_torch(self):
        import numpy as np
        import torch
        from new_watermark.method import torch_scores
        x=np.random.default_rng(0).normal(size=(100,16))
        d=make_detector(x,9)
        np.testing.assert_allclose(score_features(x,d)[:,0],torch_scores(torch.tensor(x),d).numpy())
        self.assertEqual(d,make_detector(x,9))

    def test_split_disjoint(self):
        from assets import split_prompts
        splits=split_prompts([{'image_id':i,'id':i,'caption':f'photo {i}'} for i in range(100)],0,
                            {'teacher_train':20,'meta_probe':20,'train':20,'validation':20,'test':20})
        self.assertEqual(len({r['image_id'] for rows in splits.values() for r in rows}),100)


class Gradients(unittest.TestCase):
    def test_adamw_matches_torch_first_and_second_step(self):
        import torch
        from new_watermark.meta import adamw_step
        p=torch.tensor([.3,-.4],dtype=torch.float64,requires_grad=True)
        ref=torch.nn.Parameter(p.detach().clone())
        opt=torch.optim.AdamW([ref],lr=.001,weight_decay=.01)
        state={}; ps={'x':p}
        for step in (1,2):
            g=torch.tensor([.15,-.25],dtype=torch.float64)
            ref.grad=g.clone(); torch.nn.utils.clip_grad_norm_([ref],1.); opt.step(); opt.zero_grad()
            ps,state=adamw_step(ps,{'x':g},state,step,.001)
            torch.testing.assert_close(ps['x'],ref,atol=1e-12,rtol=1e-10)

    def test_actual_unet_inner_gradient_and_finite_difference(self):
        import torch
        from diffusers import UNet2DConditionModel, DDPMScheduler
        from diffusers.models.attention_processor import AttnProcessor
        from peft import LoraConfig
        from new_watermark.meta import fresh_parameters, unroll, functional_prediction
        torch.set_num_threads(2); torch.manual_seed(3)
        model=UNet2DConditionModel(sample_size=4,in_channels=4,out_channels=4,
             down_block_types=('CrossAttnDownBlock2D',),up_block_types=('CrossAttnUpBlock2D',),
             block_out_channels=(8,),layers_per_block=1,norm_num_groups=4,cross_attention_dim=8,attention_head_dim=2).double()
        model.requires_grad_(False)
        model.add_adapter(LoraConfig(r=2,lora_alpha=2,target_modules=['to_q','to_v'],init_lora_weights='gaussian'))
        model.double().requires_grad_(False).eval(); model.set_attn_processor(AttnProcessor())
        scheduler=DDPMScheduler(num_train_timesteps=20)
        z=torch.randn(1,4,4,4,dtype=torch.float64); text=torch.randn(1,3,8,dtype=torch.float64)
        direction=torch.randn_like(z); probe=torch.randn_like(z); t=torch.tensor([5])
        draws=[(torch.randn_like(z),torch.tensor([7])),(torch.randn_like(z),torch.tensor([12]))]
        def objective(amount,detach=False):
            initial={k:p.double().detach().requires_grad_(True) for k,p in fresh_parameters(model,17).items()}
            data=z+amount*direction
            if detach: data=data.detach()
            params=unroll(model,initial,data,text,scheduler,draws,.001,True,True)
            return functional_prediction(model,params,probe,t,text,True).square().mean()
        theta=torch.tensor(.05,dtype=torch.float64,requires_grad=True)
        value=objective(theta); grad=torch.autograd.grad(value,theta)[0]
        self.assertTrue(torch.isfinite(grad)); self.assertGreater(abs(float(grad)),1e-10)
        h=1e-4
        fd=(objective(torch.tensor(.05+h,dtype=torch.float64))-objective(torch.tensor(.05-h,dtype=torch.float64)))/(2*h)
        torch.testing.assert_close(grad,fd,atol=2e-6,rtol=.02)
        blocked=torch.autograd.grad(objective(theta,True),theta,allow_unused=True)[0]
        self.assertIsNone(blocked)
        # The functional rollout must not mutate the actual clean base adapter.
        self.assertTrue(all(torch.count_nonzero(p)==0 for n,p in model.named_parameters() if '.lora_B.' in n))

    def test_epsilon_and_v_x0(self):
        import torch
        z=torch.randn(1,4,3,3); noise=torch.randn_like(z); a=torch.tensor(.7)
        noisy=a.sqrt()*z+(1-a).sqrt()*noise
        torch.testing.assert_close(predict_x0(noisy,noise,a,'epsilon'),z)
        velocity=a.sqrt()*noise-(1-a).sqrt()*z
        torch.testing.assert_close(predict_x0(noisy,velocity,a,'v_prediction'),z)



class Reporting(unittest.TestCase):
    def test_weak_teacher_cannot_produce_successful_final_claim(self):
        from unittest.mock import patch
        from common import write_json, read_json, sha256
        from new_watermark.stages import summarize
        c=config(arguments(['--ranks','16','--test-n','10','--validation-n','2',
                            '--validation-generation-n','2','--checkpoint-steps','2500','--skip-utility']))
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp)
            write_json(run/'config.json',c)
            rows=[{'image_id':i,'text':f'image {i}'} for i in range(10)]
            write_json(run/'prompts.json',{'test':rows,'validation':rows[:2]})
            detector={'directions':[[1.],[-1.]],'center':[0.,0.],'scale':[1.,1.],'thresholds':[.5,.5]}
            write_json(run/'result/detector.json',detector)
            write_json(run/'result/teacher_selection.json',{'detector_sha256':sha256(run/'result/detector.json'),
                'selected_step':1000,'valid_design_winner':True,'warning':None})
            for b in c['student_branches']:
                name=b['name']
                write_json(run/'students'/name/'complete.json',{'optimizer_update_counts':{'applied':2500}})
                write_json(run/'result/selection'/f'{name}.json',{'selected_step':2500})
                write_json(run/'validation'/f'validation_{name}_step2500.json',{'denoising_mse':.1})
            class FakeFeatures:
                def __init__(self,run): pass
                def read(self,label,rows):
                    values=[0.]*len(rows) if 'clean' in label or label=='unadapted' else [1.]*len(rows)
                    if label=='teacher_wm_test': values=[1.]*5+[0.]*5
                    return {'features':[[x] for x in values],'clip_mean':.8,
                            'identity':{'images':['test_hash']*len(rows)}}
            with patch('new_watermark.stages.FeatureReader',FakeFeatures),patch('new_watermark.stages.paired_psnr',return_value=30.):
                summarize(run)
            report=read_json(run/'result/report.json')
            self.assertFalse(report['teacher_valid'])
            self.assertFalse(report['clean_trained_controls'])
            self.assertFalse(report['causal_inheritance_established'])
            self.assertIn('r16_wm_minus_unadapted',report['effects'])
            self.assertFalse(report['claim_checks']['16']['declared_gates_passed'])
            self.assertTrue(report['claim_checks']['16']['student_detection_floor_passed'])
            self.assertFalse(read_json(run/'result/teacher_strength.json')['valid'])

    def test_clean_reference_does_not_generate_unused_training_data(self):
        from unittest.mock import patch
        from new_watermark.stages import final_teacher_images
        c=config(arguments([]))
        selection={'detector_sha256':'hash','selected_step':1000,'selected_model_export_sha256':'export'}
        def read(path):
            return {'config.json':c,'teacher_selection.json':selection,'assets.json':{'base_revision':'base'}}[path.name]
        with patch('new_watermark.stages.read_json',side_effect=read),              patch('new_watermark.stages.sha256',return_value='hash'),              patch('new_watermark.stages.verify_export',return_value='export'),              patch('new_watermark.stages.load_base'),              patch('new_watermark.stages.generate') as generate:
            final_teacher_images(Path('/run'),'clean')
            self.assertEqual([x.args[2] for x in generate.call_args_list],['teacher_clean_test'])
            generate.reset_mock()
            final_teacher_images(Path('/run'),'wm')
            self.assertEqual([x.args[2] for x in generate.call_args_list],
                             ['teacher_wm_train','teacher_wm_validation','teacher_wm_test'])

    def test_launcher_includes_all_training_and_evaluation_stages(self):
        from unittest.mock import patch
        from new_watermark.launch import execute
        calls=[]
        with patch('new_watermark.launch.stage',side_effect=lambda run,label,cmd:calls.append((label,cmd))):
            execute(Path('/synthetic_run'),config(arguments([])))
        labels=[x[0] for x in calls]
        self.assertEqual(labels[-1],'report')
        self.assertIn('utility',labels)
        self.assertNotIn('final_teacher_fixed',labels)
        self.assertFalse(any(x.endswith('_clean') or x.endswith('_fixed') for x in labels if x.startswith('train_lora_')))
        self.assertIn('select_teacher',labels)
        self.assertEqual(len([x for x in labels if x.startswith('train_lora_')]),2)
        self.assertLess(labels.index('select_teacher'),labels.index('final_teacher_wm'))
        self.assertTrue(all(Path(cmd[2]).is_file() for _,cmd in calls))


if __name__=='__main__': unittest.main()
