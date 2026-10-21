import json
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr
import io

class SignProtocolTests(unittest.TestCase):
    def test_new_defaults_and_students_unchanged(self):
        from run import parse_args,config_from_args
        from student import branches
        a=parse_args([],method='semantic_fp');c=config_from_args(a)
        self.assertEqual(c['teacher_objective'],'semantic_sign_v1')
        self.assertEqual((c['num_bits'],c['stable_candidate_count']),(32,48))
        self.assertEqual(c['coordinate_margin_mu_candidates'],[.1,.2,.3])
        self.assertEqual((c['analytic_alpha'],c['train_bit_count_margin'],c['late_inference_step_count']),(.01,2,5))
        self.assertEqual((c['teacher_batch'],c['teacher_micro_batch'],c['teacher_train_steps'],c['teacher_precision']),(4,1,3000,'bf16'))
        self.assertEqual((c['max_train_steps'],c['batch_size'],c['precision']),(7500,8,'fp16'))
        self.assertEqual([b['name'] for b in branches(c)],['lora_r32_conv_wm','full_wm'])
        self.assertNotIn('teacher_kappas',c);self.assertNotIn('semantic_t_max',c)

    def test_reuse_keeps_new_carrier_definition_and_never_old_key(self):
        from run import parse_args,config_from_args
        from semantic_checkpoint import reuse_preparation_config
        from common import write_json
        c=config_from_args(parse_args([],method='semantic_fp'))
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)
            old={**c,'teacher_objective':'threshold_cfg_v1','num_bits':8,'stable_candidate_count':16,'seed':23,'key_seed':17}
            write_json(source/'config.json',old)
            new=reuse_preparation_config(c,source)
            self.assertEqual((new['num_bits'],new['stable_candidate_count']),(32,48))
            self.assertEqual((new['seed'],new['key_seed']),(23,17))
            self.assertIn('rebuild',new['preparation_reuse']['mode'])

    def test_sign_worker_uses_mu_cli_and_separate_artifacts(self):
        from semantic_candidates import run_workers
        from unittest.mock import patch,MagicMock
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);calls=[]
            def child(command,**kwargs):
                calls.append(command);p=MagicMock();p.stdout=iter([]);p.wait.return_value=0;p.poll.return_value=0
                p.stdout=io.StringIO('done\n');return p
            with patch('semantic_candidates.visible_slots',return_value=['0']),patch('semantic_candidates.subprocess.Popen',side_effect=child):
                run_workers(run,[.1,.2,.3],parameter='coordinate_margin_mu')
            self.assertEqual([c[-2:] for c in calls],[['--coordinate-margin-mu','0.1'],['--coordinate-margin-mu','0.2'],['--coordinate-margin-mu','0.3']])
            self.assertTrue((run/'logs/teacher_coordinate_margin_mu_0.1.log').exists())


    def test_reuse_copies_clean_data_only_and_checks_image_hashes(self):
        from semantic_checkpoint import reuse_preparation_config,prepare_continuation
        from common import write_json,read_json,sha256,image_name
        from run import parse_args,config_from_args
        from PIL import Image
        c=config_from_args(parse_args([],method='semantic_fp'))
        c.update(train_n=2,validation_n=2,calibration_n=2,test_n=2)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'old';source.mkdir();dest=root/'new';dest.mkdir()
            old={**c,'teacher_objective':'threshold_cfg_v1','num_bits':8,'stable_candidate_count':16}
            write_json(source/'config.json',old)
            prompts={split:[{'image_id':i,'text':'flowers'} for i in range(2)] for split in ('train','validation','calibration','test')}
            write_json(source/'prompts.json',prompts)
            write_json(source/'assets.json',{'prompt_sha256':sha256(source/'prompts.json')})
            write_json(source/'carrier/key.json',{'old_key':'must not copy'})
            write_json(source/'scores/teacher_clean_calibration.json',{'old_scores':'must not copy'})
            write_json(source/'teacher_fp/selected.json',{'old_weights':'must not copy'})
            for split in ('train','validation','calibration'):
                directory=source/'images'/f'teacher_clean_{split}';directory.mkdir(parents=True)
                hashes=[]
                for i in range(2):
                    path=directory/image_name(i);Image.new('RGB',(512,512),(20+i,40,60)).save(path)
                    hashes.append({'file_name':path.name,'sha256':sha256(path)})
                write_json(directory/'image_hashes.json',hashes)
                write_json(directory/'generation.json',{'cfg':7.5,'resolution':512,'steps':50})
            new=reuse_preparation_config(c,source)
            before=sha256(source/'images/teacher_clean_train'/image_name(0))
            prepare_continuation(dest,new)
            self.assertEqual(sha256(dest/'images/teacher_clean_train'/image_name(0)),before)
            self.assertEqual(sha256(source/'images/teacher_clean_train'/image_name(0)),before)
            for name in ('carrier','scores','teacher_fp'):
                self.assertFalse((dest/name).exists(),name)
            self.assertTrue((dest/'preparation_reuse.json').exists())
            # A corrupt clean image cannot be accepted as preparation on a later fork.
            (source/'images/teacher_clean_train'/image_name(0)).write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError,'clean image changed'):
                prepare_continuation(root/'bad',new)


    def test_analytic_student_checkpoint_diagnostics_keep_mse_selection(self):
        from common import write_json,read_json
        from run import parse_args,config_from_args
        from validation import select,validation_label
        from semantic_report import freeze_thresholds
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);c=config_from_args(parse_args([],method='semantic_fp'))
            c.update(max_train_steps=2,checkpoint_steps=[1,2],validation_generation_n=2,skip_utility=True)
            write_json(run/'config.json',c)
            with patch('semantic_report.freeze_null',return_value={'threshold':.5}):
                threshold=freeze_thresholds(run)
            self.assertEqual(threshold['analytic_binomial'],24)
            for step,loss in ((1,.1),(2,.2)):
                write_json(run/'validation'/f"{validation_label('lora_r32_conv_wm',step)}.json",
                           {'step':step,'denoising_mse':loss,'checkpoint_hashes':{},'validation_cache_sha256':'same'})
            with patch('report.scores',side_effect=lambda run,label,n:[{'K':23,'score':10.},{'K':25,'score':10.}]):
                select(run,'lora_r32_conv_wm')
            selected=read_json(run/'result/selection/lora_r32_conv_wm.json')
            self.assertEqual(selected['selected_step'],1)
            self.assertFalse(selected['detector_used_for_selection'])
            report=read_json(run/'result/validation/lora_r32_conv_wm.json')
            self.assertTrue(all(row['detection_rate']==.5 for row in report['milestones']))


    def test_reject_inactive_objective_knobs(self):
        from run import parse_args
        for flags in (['--teacher-kappas','1','1.5','2'],['--semantic-t-max','500'],
                      ['--teacher-objective','threshold_cfg_v1','--lambda-fp','.8']):
            with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                parse_args(flags,method='semantic_fp')
