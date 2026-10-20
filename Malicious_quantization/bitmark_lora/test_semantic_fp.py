"""CPU-only contracts; no model downloads or GPU experiments."""
import unittest
from types import SimpleNamespace
import torch
from semantic_carrier import SemanticCarrier, fit_pca, choose_key, check_finite
from semantic_teacher_train import predicted_x0, semantic_losses, fingerprint_ramp
from semantic_teacher import native_generate


class CarrierTests(unittest.TestCase):
    def test_projection_score_bits_and_relative_loss(self):
        c = SemanticCarrier(torch.tensor([1., 2., 3.]), torch.eye(3)[:, :2],
                            torch.tensor([2., 4.]), torch.tensor([1., -1.]), [0, 1])
        f = torch.tensor([[3., -2., 3.], [5., -6., 3.]])
        torch.testing.assert_close(c.coordinates(f), torch.tensor([[1., -1.], [2., -2.]]))
        torch.testing.assert_close(c.scalar_score(f), torch.tensor([1., 2.]))
        self.assertEqual(c.bit_metrics(c.coordinates(f))['bit_accuracy'], 1.)
        clean = torch.tensor([[2., 3., 4.], [8., 1., 7.]])
        shifted = clean + torch.tensor([1., -2., 0.])
        code, perp = semantic_losses(clean, shifted, c, .5)
        self.assertAlmostEqual(code.item(), 0.)
        self.assertAlmostEqual(perp.item(), 0.)

    def test_pca_key_determinism(self):
        f = torch.randn(100, 20, generator=torch.Generator().manual_seed(9))
        mean, basis, variance = fit_pca(f)
        torch.testing.assert_close(basis.T @ basis, torch.eye(20), atol=2e-5, rtol=2e-5)
        a = choose_key(list(range(16)), 8, 0)
        self.assertEqual(a, choose_key(list(range(16)), 8, 0))
        self.assertNotEqual(a, choose_key(list(range(16)), 8, 1))
        self.assertEqual(len(set(a[0])), 8)
        with self.assertRaises(ValueError): choose_key([0, 1], 8, 0)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError): check_finite(torch.tensor([float('nan')]))
        with self.assertRaises(ValueError): fit_pca(torch.ones(8, 4))

    def test_prediction_types(self):
        z = torch.tensor([[[[2.]]], [[[3.]]]])
        e = torch.ones_like(z)
        t = torch.tensor([0, 1])
        scheduler = SimpleNamespace(alphas_cumprod=torch.tensor([.25, .64]),
                                    config=SimpleNamespace(prediction_type='epsilon'))
        a = torch.tensor([.25, .64]).reshape(2, 1, 1, 1)
        torch.testing.assert_close(predicted_x0(scheduler, z, e, t), (z-(1-a).sqrt()*e)/a.sqrt())
        scheduler.config.prediction_type = 'v_prediction'
        torch.testing.assert_close(predicted_x0(scheduler, z, e, t), a.sqrt()*z-(1-a).sqrt()*e)
        scheduler.config.prediction_type = 'sample'
        torch.testing.assert_close(predicted_x0(scheduler, z, e, t), e)
        scheduler.config.prediction_type = 'unknown'
        with self.assertRaises(ValueError): predicted_x0(scheduler, z, e, t)

    def test_ramp(self):
        self.assertEqual(fingerprint_ramp(249), 0.)
        self.assertEqual(fingerprint_ramp(250), 0.)
        self.assertAlmostEqual(fingerprint_ramp(500), .5)
        self.assertEqual(fingerprint_ramp(750), 1.)

    def test_native_generation(self):
        calls = []
        def pipeline(prompts, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(images=torch.zeros(1,4,2,2))
        pipeline.vae=SimpleNamespace(config=SimpleNamespace(scaling_factor=2.),decode=lambda z:SimpleNamespace(sample=z))
        pipeline.image_processor=SimpleNamespace(postprocess=lambda x,**kw:['image'])
        self.assertEqual(native_generate(pipeline, ['caption'], [object()], 50), ['image'])
        self.assertEqual(set(calls[0]), {'generator', 'num_inference_steps', 'guidance_scale', 'height', 'width', 'output_type'})
        self.assertNotIn('latents', calls[0])


class ProtocolIntegrationTests(unittest.TestCase):
    def test_frozen_defaults_and_student_isolation(self):
        from run import parse_args, config_from_args
        from student import branches, training_command
        from pathlib import Path
        c=config_from_args(parse_args(['--setting','full'],method='semantic_fp'))
        self.assertEqual((c['train_n'],c['validation_n'],c['calibration_n'],c['test_n']),(5000,1000,5000,1000))
        self.assertEqual((c['batch_size'],c['learning_rate'],c['full_learning_rate']),(8,1e-5,5e-6))
        self.assertEqual(c['checkpoint_steps'],[2500,5000,7500])
        self.assertEqual(len(branches(c)),6)
        for b in branches(c):
            cmd=training_command(c,{'student':'clean_sd14'},Path('/run'),b)
            self.assertFalse(any(any(x in part for x in ('semantic','carrier','key','verifier')) for part in cmd))
            self.assertEqual(cmd[cmd.index('--max_train_steps')+1],'7500')
        pilot=config_from_args(parse_args([],method='semantic_fp'))
        self.assertEqual([b['name'] for b in branches(pilot)],['lora_r32_conv_wm','full_wm'])

    def test_candidate_selection_quality_and_no_test_dependency(self):
        from semantic_teacher_train import choose_candidate
        rows=[{'alpha':.25,'quality':{'passes':True},'metrics':{'detection_rate':.95,'mean_score':.4}},
              {'alpha':.5,'quality':{'passes':False},'metrics':{'detection_rate':1.,'mean_score':1.}}]
        self.assertEqual(choose_candidate(rows)['alpha'],.25)
        with self.assertRaises(ValueError): choose_candidate(rows[1:])

    def test_scheduler_against_installed_ddpm_if_available(self):
        try:
            from diffusers import DDPMScheduler
        except ImportError:
            self.skipTest('Diffusers absent; algebra tested without model packages')
        for kind in ('epsilon','v_prediction','sample'):
            scheduler=DDPMScheduler(prediction_type=kind,clip_sample=False)
            z=torch.randn(2,4,2,2); p=torch.randn_like(z)
            expected=scheduler.step(p,123,z).pred_original_sample
            torch.testing.assert_close(predicted_x0(scheduler,z,p,torch.tensor([123,123])),expected)

class ExtractorAndIntegrityTests(unittest.TestCase):
    def test_fixed_extractor_mean_determinism_and_input_gradients(self):
        from semantic_carrier import SemanticExtractor
        class VAE(torch.nn.Module):
            config=SimpleNamespace(scaling_factor=2.)
            def __init__(self):
                super().__init__(); self.weight=torch.nn.Parameter(torch.ones(()))
            def encode(self,x):
                return SimpleNamespace(latent_dist=SimpleNamespace(mean=x*self.weight))
            def decode(self,z): return SimpleNamespace(sample=z*self.weight)
        class UNet(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.mid_block=torch.nn.Conv2d(2,3,1,bias=False)
            def forward(self,z,t,encoder_hidden_states):
                return SimpleNamespace(sample=self.mid_block(z))
        class Text(torch.nn.Module):
            def forward(self,t): return (t.float(),)
        class Tokenizer:
            model_max_length=2
            def __call__(self,prompts,**kw): return SimpleNamespace(input_ids=torch.ones(len(prompts),2,dtype=torch.long))
        ext=SemanticExtractor(SimpleNamespace(vae=VAE(),unet=UNet(),text_encoder=Text(),tokenizer=Tokenizer()),device='cpu')
        x=torch.randn(2,2,4,4,requires_grad=True)
        f=ext.encode_image(x,['a','b'])
        torch.testing.assert_close(f,ext.encode_image(x,['a','b']))
        torch.testing.assert_close(ext.image_latent(x),x*2)
        f.sum().backward()
        self.assertGreater(x.grad.abs().sum().item(),0)
        self.assertTrue(all(not p.requires_grad and p.grad is None for m in (ext.vae,ext.unet,ext.text_encoder) for p in m.parameters()))
        self.assertEqual(len(ext.unet.mid_block._forward_hooks),0)

    def test_carrier_serialization_and_tampering(self):
        import tempfile
        from pathlib import Path
        from semantic_carrier import save_carrier
        x=torch.randn(100,20,generator=torch.Generator().manual_seed(6))
        c={'carrier_pc_min':0,'carrier_pc_max':15,'stable_candidate_count':16,'num_bits':8,'key_seed':0}
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            save_carrier(folder,x,x,c,{'train_prompt_sha256':'train-only'})
            a=SemanticCarrier.load(folder)
            torch.testing.assert_close(a.basis.T@a.basis,torch.eye(8),atol=2e-5,rtol=2e-5)
            with (folder/'key.json').open('a') as f: f.write(' ')
            with self.assertRaises(RuntimeError): SemanticCarrier.load(folder)

    def test_stability_clean_only_and_finite(self):
        from semantic_carrier import stability_metrics
        x=torch.randn(100,10,generator=torch.Generator().manual_seed(6))
        mean,basis,var=fit_pca(x)
        rows=stability_metrics(x,x,mean,basis,var,0,9)
        self.assertEqual({r['pc'] for r in rows},set(range(10)))
        self.assertTrue(all(abs(r['correlation']-1)<1e-5 and r['normalized_mse']==0 for r in rows))

    def test_teacher_gate_survives_resume(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from common import write_json, sha256
        from semantic_teacher_train import train_candidates
        with tempfile.TemporaryDirectory() as d:
            run=Path(d); write_json(run/'config.json',{})
            write_json(run/'teacher_fp/selected.json',{'model':'model','model_hashes':{},
                'config_sha256':sha256(run/'config.json'),'downstream_ready':False})
            with patch('semantic_assets.snapshot_identity',return_value={}):
                with self.assertRaises(ValueError): train_candidates(run)

    def test_orchestration_order_and_no_test_selection(self):
        import tempfile
        import contextlib
        import io
        from pathlib import Path
        from unittest.mock import patch
        import run as runner
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            root=Path(d); calls=[]
            with patch.object(runner,'preflight',return_value={'gpu':'mock'}), \
                 patch.object(runner,'runtime_root',return_value=root/'local'), \
                 patch.object(runner.importlib.metadata,'version',return_value='test'), \
                 patch.object(runner,'stage',side_effect=lambda run,label,cmd:calls.append((label,cmd))):
                runner.main(['--output',str(root/'run'),'--skip-utility'],method='semantic_fp')
            names=[x[0] for x in calls]
            self.assertLess(names.index('01d_teacher_null'),names.index('01e_train_native_teacher'))
            self.assertLess(names.index('01e_train_native_teacher'),names.index('02_teacher'))
            self.assertLess(names.index('03c_student_calibration_scores'),names.index('04_freeze_threshold'))
            self.assertLess(names.index('04_freeze_threshold'),names.index('train_lora_r32_conv_wm'))
            self.assertLess(max(i for i,n in enumerate(names) if n.startswith('select_')),
                            min(i for i,n in enumerate(names) if n.startswith('generate_')))

class ReportingTests(unittest.TestCase):
    def test_two_frozen_nulls_and_model_bits(self):
        import tempfile
        from pathlib import Path
        from common import write_json, sha256, image_name
        from semantic_carrier import save_carrier
        from semantic_report import freeze_thresholds, detection_metrics
        c={'carrier_pc_min':0,'carrier_pc_max':15,'stable_candidate_count':16,'num_bits':8,'key_seed':0,
           'calibration_n':100,'fpr':.01}
        with tempfile.TemporaryDirectory() as d:
            run=Path(d); write_json(run/'config.json',c)
            x=torch.randn(100,20,generator=torch.Generator().manual_seed(5))
            save_carrier(run/'carrier',x,x,c,{'train_prompt_sha256':'train'})
            carrier=SemanticCarrier.load(run/'carrier')
            def dataset(label,values):
                rows=[]
                for i,value in enumerate(values):
                    p=run/'images'/label/image_name(i);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixture')
                    q=carrier.signs*value
                    rows.append({'file':p.name,'sha256':sha256(p),'score':value,'coordinates':q.tolist()})
                write_json(run/'scores'/f'{label}.json',rows)
            dataset('teacher_clean_calibration',[0.]*100)
            dataset('unadapted_calibration',[10.]*100)
            threshold=freeze_thresholds(run)
            self.assertLess(threshold['teacher']['threshold'],threshold['student']['threshold'])
            self.assertEqual(threshold['student']['empirical_fpr'],0.)
            dataset('student',[5.]*2)
            metrics=detection_metrics(run,'student',threshold['student']['threshold'],2)
            self.assertEqual(metrics['detection_rate'],0.)
            self.assertEqual(metrics['bit_accuracy'],1.)
            dataset('unadapted_calibration',[20.]*100)
            with self.assertRaises(RuntimeError): freeze_thresholds(run)

    def test_semantic_validation_selects_mse_and_uses_student_threshold(self):
        from validation import choose_checkpoint
        rows=[{'step':2500,'denoising_mse':.2,'score':100},
              {'step':5000,'denoising_mse':.1,'score':-100},
              {'step':7500,'denoising_mse':.1,'score':200}]
        self.assertEqual(choose_checkpoint(rows)['step'],5000)
        from report import detector_value
        self.assertEqual(detector_value({'score':-3.},'semantic_score'),-3.)

class ResumeArtifactTests(unittest.TestCase):
    def test_selected_teacher_recovers_quality_manifest_after_interruption(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from common import read_json, write_json, sha256
        from semantic_teacher_train import train_candidates
        with tempfile.TemporaryDirectory() as d:
            run=Path(d); write_json(run/'config.json',{})
            chosen={'alpha':.5,'model':'model','metrics':{},'quality':{'passes':True}}
            write_json(run/'teacher_fp/selected.json',{**chosen,'model_hashes':{},
                'candidates':[chosen],'config_sha256':sha256(run/'config.json'),'downstream_ready':True})
            with patch('semantic_assets.snapshot_identity',return_value={}): train_candidates(run)
            self.assertEqual(read_json(run/'result/teacher_quality.json')['selected'],chosen)


class NativeDatasetTests(unittest.TestCase):
    def test_production_dataset_needs_no_key_and_pairs_normal_seeds(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from PIL import Image
        from common import write_json, read_json
        from semantic_teacher import generate_set
        class Pipeline:
            scheduler=SimpleNamespace()
            def __init__(self):
                self.calls=[]
                self.vae=SimpleNamespace(config=SimpleNamespace(scaling_factor=2.),decode=lambda z:SimpleNamespace(sample=z))
                self.image_processor=SimpleNamespace(postprocess=lambda x,**kw:[Image.new('RGB',(512,512)) for _ in x])
            def __call__(self,prompts,**kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(images=torch.zeros(len(prompts),4,2,2))
        with tempfile.TemporaryDirectory() as d:
            run=Path(d)
            write_json(run/'config.json',{'seed':0,'teacher_batch':2,'inference_steps':50})
            write_json(run/'prompts.json',{'train':[{'image_id':31,'text':'a'},{'image_id':57,'text':'b'}]})
            pipe=Pipeline()
            cpu_generator=torch.Generator
            with patch('torch.Generator',side_effect=lambda device:cpu_generator('cpu')), \
                 patch('semantic_assets.snapshot_identity',return_value={'weights':'digest'}), \
                 patch('semantic_carrier.SemanticCarrier.load',side_effect=AssertionError('key requested at inference')):
                generate_set(run,'ordinary_model','train','teacher_clean_train',pipe)
                generate_set(run,'ordinary_model','train','teacher_wm_train',pipe)
            self.assertFalse((run/'carrier').exists())
            self.assertEqual(len(pipe.calls),2)
            for call in pipe.calls:
                self.assertNotIn('latents',call)
                self.assertEqual([g.initial_seed() for g in call['generator']],[31,57])
            a=read_json(run/'images/teacher_clean_train/generation.json')
            b=read_json(run/'images/teacher_wm_train/generation.json')
            self.assertEqual(a['seeds'],b['seeds'])


class NativeFiniteTests(unittest.TestCase):
    def test_no_inference_hooks_and_nonfinite_decode_fails(self):
        import ast
        import inspect
        from semantic_teacher import load_native
        tree=ast.parse(inspect.getsource(load_native))
        self.assertFalse(any(isinstance(n,ast.Attribute) and n.attr=='register_forward_hook' for n in ast.walk(tree)))
        class Pipeline:
            vae=SimpleNamespace(config=SimpleNamespace(scaling_factor=2.),
                                decode=lambda z:SimpleNamespace(sample=torch.full((1,3,2,2),float('inf'))))
            image_processor=SimpleNamespace(postprocess=lambda *a,**k: self.fail('Nonfinite image was postprocessed'))
            def __call__(self,prompts,**kwargs): return SimpleNamespace(images=torch.zeros(1,4,2,2))
        with self.assertRaises(ValueError): native_generate(Pipeline(),['a'],[object()],50)


class StudentGenerationFiniteTests(unittest.TestCase):
    def test_nonfinite_student_decode_cannot_be_saved_as_black_png(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from PIL import Image
        from common import write_json
        from student import generate_images
        class Pipeline:
            scheduler=SimpleNamespace()
            vae=SimpleNamespace(config=SimpleNamespace(scaling_factor=2.),
                 decode=lambda z:SimpleNamespace(sample=torch.full((len(z),3,2,2),float('inf'))))
            image_processor=SimpleNamespace(postprocess=lambda x,**kw:[Image.new('RGB',(512,512)) for _ in x])
            def __call__(self,prompts,**kwargs):
                return SimpleNamespace(images=torch.zeros(len(prompts),4,2,2) if kwargs.get('output_type')=='latent'
                                       else [Image.new('RGB',(512,512)) for _ in prompts])
        with tempfile.TemporaryDirectory() as d:
            run=Path(d)
            write_json(run/'config.json',{'watermark_method':'semantic_fp','seed':0,'inference_batch':2,'inference_steps':50})
            write_json(run/'prompts.json',{'test':[{'image_id':1,'text':'a'},{'image_id':2,'text':'b'}]})
            factory=torch.Generator
            with patch('torch.Generator',side_effect=lambda device:factory('cpu')):
                with self.assertRaisesRegex(ValueError,'Nonfinite'):
                    generate_images(run,'unadapted',Pipeline(),None)


class AssetPinningTests(unittest.TestCase):
    def test_local_student_checks_manifest_contents_not_just_revision_label(self):
        import tempfile
        from pathlib import Path
        from common import write_json
        from semantic_assets import snapshot_identity, verified_student_snapshot, STUDENT_REV
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            for name in ('model_index.json','unet/config.json','vae/config.json','scheduler/scheduler_config.json',
                         'text_encoder/config.json','tokenizer/tokenizer_config.json'):
                write_json(folder/name,{})
            for component in ('unet','vae','text_encoder'):
                (folder/component/'model.safetensors').write_bytes(b'original')
            hashes=snapshot_identity(folder)
            write_json(folder/'verified.json',{'repository':'CompVis/stable-diffusion-v1-4','revision':STUDENT_REV,
                'files':[{'file':name,'sha256':digest} for name,digest in hashes.items()]})
            self.assertEqual(verified_student_snapshot(folder),str(folder.resolve()))
            (folder/'unet/model.safetensors').write_bytes(b'changed but revision label unchanged')
            with self.assertRaises(RuntimeError): verified_student_snapshot(folder)

class NativeAutogradTests(unittest.TestCase):
    def test_native_helper_disables_gradients_even_for_direct_callers(self):
        class Pipeline:
            vae=SimpleNamespace(config=SimpleNamespace(scaling_factor=1.))
            image_processor=SimpleNamespace(postprocess=lambda x,**kw:['image'])
            def __call__(self,prompts,**kwargs): return SimpleNamespace(images=torch.zeros(1,4,2,2))
        pipe=Pipeline()
        def decode(z):
            self.assertFalse(torch.is_grad_enabled())
            return SimpleNamespace(sample=z)
        pipe.vae.decode=decode
        self.assertEqual(native_generate(pipe,['a'],[object()],50),['image'])


class FinalReportTests(unittest.TestCase):
    def test_final_table_joins_utility_and_paired_teacher_quality(self):
        import csv
        import tempfile
        from pathlib import Path
        from common import write_json, image_name, sha256
        from semantic_carrier import save_carrier
        from semantic_report import summarize
        from student import branches
        c={'watermark_method':'semantic_fp','carrier_pc_min':0,'carrier_pc_max':15,'stable_candidate_count':16,
           'num_bits':8,'key_seed':0,'calibration_n':100,'fpr':.01,'test_n':2,'ranks':[32],
           'adapter_scopes':['attention_conv'],'train_clean_controls':False,'lora_only':False,'paper_full_ema':False}
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);write_json(run/'config.json',c)
            x=torch.randn(100,20,generator=torch.Generator().manual_seed(4))
            save_carrier(run/'carrier',x,x,c,{'train_prompt_sha256':'train'})
            carrier=SemanticCarrier.load(run/'carrier')
            def dataset(label,n,value):
                rows=[]
                for i in range(n):
                    p=run/'images'/label/image_name(i);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixture')
                    rows.append({'file':p.name,'sha256':sha256(p),'score':value,'coordinates':(carrier.signs*value).tolist()})
                write_json(run/'scores'/f'{label}.json',rows)
            dataset('teacher_clean_calibration',100,0.)
            dataset('unadapted_calibration',100,1.)
            labels=['teacher_clean_test','teacher_wm_test','unadapted']+[b['name'] for b in branches(c)]
            for label in labels:dataset(label,2,2.)
            for b in branches(c):write_json(run/'result/selection'/f"{b['name']}.json",{'selected_step':5000})
            chosen={'alpha':.5,'model':'model','metrics':{'detection_rate':.95,'mean_score':.5,'bit_accuracy':1.},
                    'quality':{'clean_clip':1.,'fp_clip':.995,'clip_drop':.005,'paired_lpips':.1,'passes':True}}
            write_json(run/'result/teacher_quality.json',{'selected':chosen,'candidates':[chosen]})
            write_json(run/'result/utility.json',{'branches':{label:{'clip_score':.7,'frechet_inception_distance':3.,
                       'kernel_inception_distance_mean':.02} for label in labels}})
            summarize(run)
            with (run/'result/detection_semantic.csv').open() as f:table=list(csv.DictReader(f))
            self.assertEqual(len(table),5)
            self.assertTrue(all(float(r['clip'])==.7 for r in table))
            with (run/'result/teacher_quality.csv').open() as f:quality=list(csv.DictReader(f))
            self.assertEqual(float(quality[0]['paired_lpips']),.1)
            text=(run/'result/README.md').read_text(encoding='utf8')
            self.assertIn('Clean CLIP',text)
            self.assertIn('lora_r32_conv_wm',text)
            chosen={k:v for k,v in chosen.items() if k!='alpha'};chosen['kappa']=1.5
            write_json(run/'result/teacher_quality.json',{'selected':chosen,'candidates':[chosen]})
            summarize(run)
            with (run/'result/teacher_quality.csv').open() as f:quality=list(csv.DictReader(f))
            self.assertEqual(float(quality[0]['kappa']),1.5)
            self.assertIn('| Kappa |',(run/'result/README.md').read_text(encoding='utf8'))

if __name__ == '__main__': unittest.main()
