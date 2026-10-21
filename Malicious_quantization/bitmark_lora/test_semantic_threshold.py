"""Threshold-CFG objective: CPU hand calculations, accumulation and isolation."""
import copy
import json
import math
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from semantic_carrier import SemanticCarrier
from test_semantic_memory import tiny_extractor

class ThresholdObjectiveTests(unittest.TestCase):
    def test_target_detached_clipped_and_mean_over_bits(self):
        from semantic_threshold import calibrated_shift_loss
        signs=torch.tensor([1.,-1.]);q0=torch.tensor([[0.,0.],[2.,-2.],[-3.,3.]],requires_grad=True)
        # tau=1, sigma=.5, kappa=1 -> T=1.5, shifts=[1.5,0,2.5].
        qfp=q0.detach().clone().requires_grad_(True)
        loss,stats=calibrated_shift_loss(q0,qfp,signs,1.,.5,1.)
        self.assertAlmostEqual(loss.item(),(2.25+0+6.25)/3,places=6)
        self.assertAlmostEqual(stats['mean_required_shift'],4/3,places=6)
        loss.backward();self.assertIsNone(q0.grad)
        target=q0.detach()+torch.tensor([[1.5,-1.5],[0.,0.],[2.5,-2.5]])
        zero,_=calibrated_shift_loss(q0,target,signs,1.,.5,1.)
        self.assertAlmostEqual(zero.item(),0.)

    def test_cfg_exact_combination_and_prediction_types(self):
        from semantic_threshold import cfg_prediction
        from semantic_teacher_train import predicted_x0
        cond=torch.tensor([[[[2.]]]]);uncond=torch.tensor([[[[1.]]]])
        prediction=cfg_prediction(cond,uncond)
        self.assertEqual(prediction.item(),8.5)
        scheduler=SimpleNamespace(alphas_cumprod=torch.tensor([.25]),config=SimpleNamespace())
        noisy=torch.tensor([[[[3.]]]]);t=torch.tensor([0])
        for kind,expected in (('epsilon',(3.-math.sqrt(.75)*8.5)/.5),
                              ('v_prediction',.5*3.-math.sqrt(.75)*8.5),('sample',8.5)):
            scheduler.config.prediction_type=kind
            self.assertAlmostEqual(predicted_x0(scheduler,noisy,prediction,t).item(),expected,places=5)

    def test_cfg_backward_microbatch_equivalence_cycle_every_semantic_step(self):
        from semantic_teacher_train import teacher_backward
        torch.manual_seed(23);ext=tiny_extractor(True)
        ext.text=lambda prompts:torch.zeros(len(prompts),2)
        class Teacher(torch.nn.Module):
            def __init__(self):super().__init__();self.layer=torch.nn.Conv2d(2,2,1);self.calls=[]
            def forward(self,z,t,encoder_hidden_states):
                self.calls.append(encoder_hidden_states.detach().clone())
                h=encoder_hidden_states.mean(-1).reshape(-1,1,1,1)
                return SimpleNamespace(sample=self.layer(z)+h*.1)
        base=Teacher();carrier=SemanticCarrier(torch.zeros(2),torch.eye(2),torch.ones(2),torch.tensor([1.,-1.]),[0,1])
        scheduler=SimpleNamespace(alphas_cumprod=torch.linspace(.99,.1,1000),config=SimpleNamespace(prediction_type='v_prediction'))
        noisy=torch.randn(5,2,4,4)*.1;text=torch.ones(5,2)
        c={'teacher_objective':'threshold_cfg_v1','precision':'no','teacher_precision':'no',
           'semantic_t_max':500,'cycle_every':4,'teacher_calibration':{'threshold':.8,'sigma_clean':.3}}
        for step,t in ((249,torch.tensor([0,600,250,700,500])),(753,torch.tensor([0,600,250,700,500])),
                       (753,torch.tensor([600,700,800,900,999]))):
            results=[]
            for micro in (5,1,2):
                model=copy.deepcopy(base)
                values=teacher_backward(model,ext,carrier,scheduler,noisy,text,t,1.5,{**c,'teacher_micro_batch':micro},step,lambda loss:loss.backward())
                self.assertEqual(values['ramp'],1.)
                self.assertAlmostEqual(values['loss'],values['preserve']+.8*(values['sem']+.5*values['sem_cycle']),places=6)
                results.append((values,[p.grad.clone() for p in model.parameters()]))
                self.assertTrue(any(torch.all(h==0) for h in model.calls))
                self.assertTrue(any(torch.all(h==1) for h in model.calls))
                self.assertLessEqual(max(len(h) for h in model.calls),micro)
            if step==753 and (t<=500).any():self.assertGreater(results[0][0]['sem_cycle'],0.)
            for values,grads in results[1:]:
                for name,value in results[0][0].items():
                    if value is None:self.assertIsNone(values[name])
                    else:self.assertAlmostEqual(values[name],value,places=5,msg=name)
                for a,b in zip(grads,results[0][1]):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)

    def test_new_bf16_backward_updates_only_teacher_with_finite_cycle_gradients(self):
        from semantic_teacher_train import teacher_backward,checked_teacher_step
        torch.manual_seed(29);ext=tiny_extractor(True)
        ext.text=lambda prompts:torch.zeros(len(prompts),2)
        class Teacher(torch.nn.Module):
            def __init__(self):super().__init__();self.layer=torch.nn.Conv2d(2,2,1)
            def forward(self,z,t,encoder_hidden_states):return SimpleNamespace(sample=self.layer(z))
        model=Teacher();initial=copy.deepcopy(model.state_dict())
        carrier=SemanticCarrier(torch.zeros(2),torch.eye(2),torch.ones(2),torch.tensor([1.,-1.]),[0,1])
        scheduler=SimpleNamespace(alphas_cumprod=torch.linspace(.99,.1,1000),config=SimpleNamespace(prediction_type='epsilon'))
        c={'teacher_objective':'threshold_cfg_v1','precision':'fp16','teacher_precision':'bf16',
           'semantic_t_max':500,'teacher_micro_batch':1,'teacher_calibration':{'threshold':.8,'sigma_clean':.3}}
        noisy=torch.randn(2,2,4,4)*.1;text=torch.ones(2,2);t=torch.tensor([0,500])
        opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
        values=checked_teacher_step(model.named_parameters(),opt,None,
            lambda backward:teacher_backward(model,ext,carrier,scheduler,noisy,text,t,1.5,c,0,backward),None,{'step':1})
        self.assertTrue(math.isfinite(values['loss']))
        self.assertGreater(values['sem_cycle'],0.)
        self.assertTrue(any(not torch.equal(model.state_dict()[k],v) for k,v in initial.items()))
        for module in (ext.vae,ext.unet,ext.text_encoder):
            self.assertTrue(all(p.grad is None and not p.requires_grad and p.dtype==torch.float32 for p in module.parameters()))

    def test_lpips_diagnostic_only_for_new_objective(self):
        from semantic_threshold import quality_passes
        self.assertTrue(quality_passes({'teacher_objective':'threshold_cfg_v1','teacher_clip_drop_max':.01},.005,.32))
        self.assertFalse(quality_passes({'teacher_objective':'threshold_cfg_v1','teacher_clip_drop_max':.01},.02,.05))
        self.assertFalse(quality_passes({'teacher_objective':'threshold_cfg_v1','teacher_clip_drop_max':.1},.05,.05))
        self.assertFalse(quality_passes({'teacher_clip_drop_max':.01,'teacher_lpips_max':.15},.005,.32))

    def test_gpu_slots_are_exclusive_and_work_is_actually_parallel_when_available(self):
        from semantic_candidates import schedule_candidates
        active=set();lock=threading.Lock();overlap=[];barrier=threading.Barrier(2)
        def work(value,gpu):
            with lock:
                self.assertNotIn(gpu,active);active.add(gpu);overlap.append(len(active))
            barrier.wait(timeout=5)
            with lock:active.remove(gpu)
            return value*2
        self.assertEqual(schedule_candidates([1.,1.5],['gpu0','gpu1'],work),[2.,3.])
        self.assertIn(2,overlap)
        self.assertEqual(schedule_candidates([1.,1.5,2.],['gpu0'],lambda value,gpu:value),[1.,1.5,2.])

    def test_calibration_is_clean_only_recomputed_and_has_provenance(self):
        import statistics
        from common import write_json,sha256,image_name
        from semantic_report import freeze_null
        from semantic_threshold import frozen_calibration
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);c={'calibration_n':100,'fpr':.01,'teacher_kappas':[1.,1.5,2.]}
            write_json(run/'config.json',c)
            carrier=run/'carrier';carrier.mkdir()
            torch.save({'mean':torch.zeros(2),'basis':torch.eye(2)},carrier/'pca.pt')
            write_json(carrier/'key.json',{'selected_pc_indices':[0,1],'sigmas':[1.,1.],'signs':[1.,-1.],
                                         'basis_sha256':sha256(carrier/'pca.pt')})
            write_json(carrier/'complete.json',{n:sha256(carrier/n) for n in ('pca.pt','key.json')})
            rows=[]
            for i in range(100):
                path=run/'images/teacher_clean_calibration'/image_name(i)
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'image')
                value=i/100.;rows.append({'file':path.name,'sha256':sha256(path),'score':value,'coordinates':[value,-value]})
            write_json(run/'scores/teacher_clean_calibration.json',rows);freeze_null(run,'teacher')
            # These intentionally unusable files must never influence target fitting.
            (run/'validation.json').write_text('not JSON')
            (run/'test.json').write_text('not JSON')
            result=frozen_calibration(run)
            self.assertEqual(result['false_positives'],1)
            self.assertAlmostEqual(result['sigma_clean'],statistics.pstdev(i/100. for i in range(100)))
            self.assertFalse(result['validation_used']);self.assertFalse(result['test_used'])
            self.assertEqual(result['n'],100)
            self.assertAlmostEqual(result['targets'][1]['target_score'],math.nextafter(.98,math.inf)+1.5*result['sigma_clean'])
            rows[0]['score']=10.
            write_json(run/'scores/teacher_clean_calibration.json',rows)
            with self.assertRaises(ValueError):frozen_calibration(run)

    def test_candidate_campaign_gate_and_selected_resume_preserve_full_record(self):
        from unittest.mock import patch
        from common import write_json,read_json,sha256,image_name
        from semantic_threshold import frozen_calibration,quality_passes
        from semantic_report import freeze_null
        from semantic_teacher_train import train_candidates,candidate_tag
        for passes_tpr in (True,False):
            with self.subTest(passes_tpr=passes_tpr),tempfile.TemporaryDirectory() as d:
                run=Path(d);c={'teacher_objective':'threshold_cfg_v1','calibration_n':100,'fpr':.01,
                              'teacher_kappas':[1.,1.5,2.],'teacher_min_tpr':.9,'teacher_clip_drop_max':.01,'teacher_lpips_max':.15}
                write_json(run/'config.json',c);carrier=run/'carrier';carrier.mkdir()
                torch.save({'mean':torch.zeros(2),'basis':torch.eye(2)},carrier/'pca.pt')
                write_json(carrier/'key.json',{'selected_pc_indices':[0,1],'sigmas':[1.,1.],'signs':[1.,-1.],
                                             'basis_sha256':sha256(carrier/'pca.pt')})
                write_json(carrier/'complete.json',{n:sha256(carrier/n) for n in ('pca.pt','key.json')})
                rows=[]
                for i in range(100):
                    path=run/'images/teacher_clean_calibration'/image_name(i);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'image')
                    rows.append({'file':path.name,'sha256':sha256(path),'score':i/100.,'coordinates':[i/100.,-i/100.]})
                write_json(run/'scores/teacher_clean_calibration.json',rows);freeze_null(run,'teacher')
                def workers(run_arg,kappas):
                    cal=frozen_calibration(run_arg)
                    for k in kappas:
                        tpr=({1.:.91,1.5:.95,2.:.99} if passes_tpr else {1.:.89,1.5:.88,2.:.99})[k]
                        drop=.02 if k==2. else .005
                        record={'kappa':k,'target_score':cal['threshold']+k*cal['sigma_clean'],
                            'model':f'model_{k}','model_hashes':{},'calibration':cal,'config_sha256':sha256(run/'config.json'),
                            'metrics':{'detection_rate':tpr,'mean_score':1.2,'bit_accuracy':1.},
                            'quality':{'passes':quality_passes(c,drop,.4),'clip_drop':drop,'paired_lpips':.4}}
                        write_json(run/'teacher_fp/candidates'/candidate_tag(k,'kappa')/'validation.json',record)
                    return kappas
                with patch('semantic_candidates.run_workers',side_effect=workers),patch('semantic_assets.snapshot_identity',return_value={}):
                    if passes_tpr:
                        train_candidates(run)
                        chosen=read_json(run/'teacher_fp/selected.json')
                        self.assertEqual(chosen['kappa'],1.5)
                        self.assertTrue(chosen['downstream_ready'])
                        expected=read_json(run/'result/teacher_quality.json')
                        (run/'result/teacher_quality.json').unlink()
                        train_candidates(run)
                        self.assertEqual(read_json(run/'result/teacher_quality.json'),expected)
                    else:
                        with self.assertRaisesRegex(ValueError,'below required'):train_candidates(run)
                        self.assertFalse(read_json(run/'teacher_fp/selected.json')['downstream_ready'])
                        with self.assertRaisesRegex(ValueError,'cannot bypass'):train_candidates(run)

    def test_reuse_cannot_silently_replace_effective_teacher_batch(self):
        from common import write_json
        from semantic_checkpoint import reuse_preparation_config
        with tempfile.TemporaryDirectory() as d:
            source=Path(d);write_json(source/'config.json',{'watermark_method':'semantic_fp','teacher_batch':8})
            with self.assertRaisesRegex(ValueError,'generation batch differs'):
                reuse_preparation_config({'teacher_batch':4},source)

    def test_new_cli_plan_and_student_protocol_unchanged(self):
        from run import parse_args,config_from_args
        from student import branches,training_command
        c=config_from_args(parse_args(['--teacher-objective','threshold_cfg_v1'],method='semantic_fp'))
        self.assertEqual(c['teacher_objective'],'threshold_cfg_v1')
        self.assertEqual(c['teacher_kappas'],[1.,1.5,2.])
        self.assertEqual((c['teacher_batch'],c['teacher_micro_batch'],c['teacher_train_steps']),(4,1,3000))
        self.assertEqual(c['cycle_every'],1)
        for b in branches(c):
            cmd=training_command(c,{'student':'base'},Path('run'),b)
            self.assertEqual(cmd[cmd.index('--mixed_precision')+1],'fp16')
            self.assertEqual(cmd[cmd.index('--train_batch_size')+1],'8')

if __name__=='__main__':unittest.main()
