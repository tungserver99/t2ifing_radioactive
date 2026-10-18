"""SERUM protocol, independent calibration, and artifact integrity regression tests."""
import contextlib
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from common import image_name, read_json, sha256, write_json
from report import freeze_threshold, summarize
from run import parse_args, config_from_args
from serum_assets import safe_extract, verified_file, teacher_snapshot
from student import branches, training_command
from validation import select, validation_label


class SerumTests(unittest.TestCase):
    def test_diffusion_defaults_and_training(self):
        c = config_from_args(parse_args([], method='serum'))
        self.assertEqual(c['ranks'], [16,32])
        self.assertEqual(c['checkpoint_steps'], [2500,5000,7500])
        self.assertEqual(c['max_train_steps'], 7500)
        self.assertEqual(c['train_n'], 5000)
        self.assertEqual(len(branches(c)), 6)
        for b in branches(c):
            cmd = training_command(c, {'student':'/sd14'}, Path('/run'), b)
            self.assertEqual(cmd[cmd.index('--max_train_steps')+1], '7500')
            self.assertEqual(cmd[cmd.index('--learning_rate')+1], '5e-06' if b['kind']=='full' else '1e-05')
            self.assertFalse(any('scores' in x or 'test' in x for x in cmd))
            self.assertNotEqual(b.get('rank'), 4)
            self.assertEqual('--use_dora' in cmd, b.get('use_dora', False))
        self.assertEqual(branches(c)[-2]['name'], 'dora_r32_conv_wm')
        old = dict(c); old.pop('dora')
        self.assertFalse(any(b.get('use_dora') for b in branches(old)))
        self.assertEqual(len(branches(config_from_args(parse_args(['--no-dora'], method='serum')))), 5)

    def test_orchestration_uses_diffusion_and_calibrates_before_training(self):
        import run as runner
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root=Path(d); calls=[]
            with patch.object(runner,'preflight',return_value={'gpu':'mock'}), \
                 patch.object(runner,'runtime_root',return_value=root/'local'), \
                 patch.object(runner.importlib.metadata,'version',return_value='test'), \
                 patch.object(runner,'stage',side_effect=lambda run,label,cmd:calls.append((label,cmd))):
                runner.main(['--output',str(root/'run'),'--skip-utility'],method='serum')
            names=[label for label,_ in calls]
            self.assertLess(names.index('03c_student_calibration_scores'), names.index('04_freeze_threshold'))
            self.assertLess(names.index('04_freeze_threshold'), min(i for i,n in enumerate(names) if n.startswith('train_')))
            self.assertLess(max(i for i,n in enumerate(names) if n.startswith('select_')),
                            min(i for i,n in enumerate(names) if n.startswith('generate_')))
            sources={Path(cmd[1]).name for _,cmd in calls}
            self.assertIn('serum_teacher.py',sources); self.assertIn('serum_assets.py',sources)
            self.assertNotIn('teacher.py',sources); self.assertNotIn('assets.py',sources)
            env=read_json(root/'run/environment.json')
            self.assertNotIn('flash-attn',env['packages'])

    def write_scores(self,run,label,values):
        rows=[]
        for i,value in enumerate(values):
            p=run/'images'/label/image_name(i); p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(f'{label}-{i}'.encode())
            rows.append({'file':p.name,'sha256':sha256(p),'score':value})
        write_json(run/'scores'/f'{label}.json',rows)

    def test_two_domains_report_and_selection_ignore_detector(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            run=Path(d)
            c=config_from_args(parse_args(['--ranks','16','--adapter-scopes','attention','--lora-only',
                '--no-dora','--test-n','10','--calibration-n','100','--max-train-steps','2','--checkpoint-steps','1','2',
                '--validation-generation-n','2','--skip-utility'],method='serum'))
            write_json(run/'config.json',c)
            for label,values in {'teacher_clean_calibration':[.1]*100,'unadapted_calibration':[.4]*100,
                    'teacher_wm_test':[.95]*10,'teacher_clean_test':[.1]*10,'unadapted':[.3]*10,
                    'lora_r16_wm':[.8]*8+[.3]*2}.items():
                self.write_scores(run,label,values)
            threshold=freeze_threshold(run)
            self.assertGreater(threshold['serum_score_student_calibrated'],threshold['serum_score'])
            for step,loss,value in [(1,.1,.1),(2,.2,.99)]:
                label=validation_label('lora_r16_wm',step)
                write_json(run/'validation'/f'{label}.json',{'step':step,'denoising_mse':loss,
                    'checkpoint_hashes':{'weight':'abc'},'validation_cache_sha256':'abc'})
                self.write_scores(run,label,[value]*2)
            select(run,'lora_r16_wm'); summarize(run)
            r=read_json(run/'result/radioactivity.json')
            self.assertEqual(r['selected_checkpoints']['lora_r16_wm']['selected_step'],1)
            primary={x['branch']:x for x in r['tables']['serum_score']}
            secondary={x['branch']:x for x in r['tables']['serum_score_student_calibrated']}
            self.assertEqual(primary['unadapted']['detected'],10)
            self.assertEqual(secondary['unadapted']['detected'],0)
            self.assertEqual(secondary['lora_r16_wm']['detected'],8)
            self.assertTrue((run/'result/checkpoint_metrics.csv').exists())
            # A change in final test cannot silently move either calibration threshold.
            self.write_scores(run,'teacher_wm_test',[.01]*10)
            self.assertEqual(freeze_threshold(run),threshold)
            self.write_scores(run,'unadapted_calibration',[.6]*100)
            with self.assertRaises(RuntimeError): freeze_threshold(run)

    def test_dora_paired_report_under_both_thresholds(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            run = Path(d)
            c = config_from_args(parse_args(['--ranks','32','--adapter-scopes','attention_conv',
                '--lora-only','--test-n','10','--calibration-n','100','--skip-utility'],method='serum'))
            c.pop('checkpoint_steps')
            write_json(run/'config.json',c)
            for label,values in {'teacher_clean_calibration':[.1]*100,'unadapted_calibration':[.2]*100,
                    'teacher_wm_test':[.95]*10,'teacher_clean_test':[.1]*10,'unadapted':[.1]*10,
                    'lora_r32_conv_wm':[.9]*3+[.1]*7,'dora_r32_conv_wm':[.9]*8+[.1]*2}.items():
                self.write_scores(run,label,values)
            summarize(run)
            report = read_json(run/'result/radioactivity.json')
            for mode in ('serum_score','serum_score_student_calibrated'):
                self.assertAlmostEqual(report['paired_effects'][mode]['dora_minus_lora_conv_r32']['estimate_pp'],50.)
            self.assertIn('dora_r32_conv_wm',(run/'result/detection_serum_score.csv').read_text())

    def test_reject_bad_artifacts_and_zip_traversal(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); p=root/'weight'; p.write_bytes(b'bad')
            with self.assertRaises(RuntimeError): verified_file(p,'0'*64)
            for name in ('../escape','/absolute','a\\b'):
                archive=root/'test.zip'
                with zipfile.ZipFile(archive,'w') as z: z.writestr(name,b'bad')
                with self.assertRaises(ValueError): safe_extract(archive,root/'out')
            archive=root/'link.zip'
            with zipfile.ZipFile(archive,'w') as z:
                info=zipfile.ZipInfo('link'); info.external_attr=0o120777<<16; z.writestr(info,'elsewhere')
            with self.assertRaises(ValueError): safe_extract(archive,root/'out')

    def test_snapshot_publish_reuse_and_tamper(self):
        import serum_assets as assets
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); archive=root/'cache.zip'
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr(f'models--sd21/snapshots/{assets.TEACHER_REV}/model_index.json','{}')
            with patch.object(assets,'CACHE_SHA256',sha256(archive)):
                snapshot=Path(teacher_snapshot(archive,root))
                self.assertTrue((snapshot/'model_index.json').exists())
                archive.unlink()
                self.assertEqual(teacher_snapshot(archive,root),str(snapshot))
                (snapshot/'model_index.json').write_text('changed')
                with self.assertRaises(RuntimeError): teacher_snapshot(archive,root)


if __name__=='__main__': unittest.main()
