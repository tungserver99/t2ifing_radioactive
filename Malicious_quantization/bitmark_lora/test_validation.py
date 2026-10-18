"""Protocol regression tests for held-out selection and the rank/scope experiment."""
import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import common
from common import read_json, write_json, sha256, image_name
from run import parse_args, config_from_args
from student import branches, training_command
from validation import choose_checkpoint, select, validation_label


class ValidationTests(unittest.TestCase):
    def test_factorial_budget(self):
        c = config_from_args(parse_args([]))
        bs = branches(c)
        lora = [b for b in bs if b['kind'] == 'lora']
        self.assertEqual({(b['rank'], b['scope']) for b in lora},
                         {(r,s) for r in [16,32] for s in ['attention','attention_conv']})
        for b in bs:
            cmd = training_command(c, {'student': '/base'}, Path('/run'), b)
            self.assertEqual(cmd[cmd.index('--max_train_steps')+1], '7500')
            self.assertEqual(cmd[cmd.index('--snapshot_steps')+1:cmd.index('--snapshot_steps')+4], ['2500','5000','7500'])
        self.assertEqual(c['validation_draws'], 2)

    def test_budget_overrides_and_final_snapshot(self):
        self.assertEqual(parse_args(['--max-train-steps','2600']).checkpoint_steps, [2500,2600])
        self.assertEqual(parse_args(['--epochs','5']).max_train_steps, 1250)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(['--epochs','5','--max-train-steps','5000'])

    def test_drive_results_local_environment(self):
        with patch.dict(os.environ, {'COLAB_RELEASE_TAG':'test'}, clear=True), \
             patch.object(common, 'HERE', Path('/content/drive/MyDrive/repo/bitmark_lora')):
            self.assertEqual(common.results_root(), Path('/content/drive/MyDrive/repo'))
            self.assertEqual(common.runtime_root(), Path('/content/bitmark_runtime'))

    def test_selection_ignores_detection_and_quality(self):
        rows = [{'step':500, 'denoising_mse':.2, 'tpr':1., 'fid':1.},
                {'step':1000, 'denoising_mse':.1, 'tpr':0., 'fid':100.},
                {'step':2000, 'denoising_mse':.1, 'tpr':1., 'fid':0.}]
        self.assertEqual(choose_checkpoint(rows)['step'], 1000)
        with self.assertRaises(ValueError):
            choose_checkpoint([{'step':1, 'denoising_mse':float('nan')}])

    def test_orchestrator_freezes_all_selections_before_student_test(self):
        import run as runner
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            root = Path(d)
            output = root/'persistent_run'
            calls = []
            with patch.object(runner, 'preflight', return_value={'gpu':'mock'}), \
                 patch.object(runner, 'runtime_root', return_value=root/'local'), \
                 patch.object(runner.importlib.metadata, 'version', return_value='test'), \
                 patch.object(runner, 'stage', side_effect=lambda run, label, cmd: calls.append((label, cmd))):
                runner.main(['--output',str(output),'--ranks','4','--adapter-scopes','attention','attention_conv',
                             '--max-train-steps','2','--checkpoint-steps','1','2','--skip-utility'])
            selected = [i for i,(label,_) in enumerate(calls) if label.startswith('select_')]
            tests = [i for i,(label,_) in enumerate(calls) if label.startswith('generate_')]
            self.assertEqual(len(selected), 3)
            self.assertLess(max(selected), min(tests))
            self.assertTrue((output/'run.log').exists())
            self.assertTrue((output/'complete.json').exists())

    def test_persisted_selection_and_milestone_report(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            run = Path(d)
            c = config_from_args(parse_args(['--ranks','4','--adapter-scopes','attention','--lora-only',
                '--max-train-steps','2','--checkpoint-steps','1','2', '--validation-generation-n','2','--skip-utility']))
            write_json(run/'config.json', c)
            write_json(run/'result/threshold.json', {'absolute_z':5.})
            for step, loss, z in [(1,.1,0.), (2,.2,10.)]:
                label = validation_label('lora_r4_wm', step)
                write_json(run/'validation'/f'{label}.json', {'step':step,'denoising_mse':loss,
                    'checkpoint_hashes':{'weights':'abc'},'validation_cache_sha256':'def'})
                rows = []
                for i in range(2):
                    path = run/'images'/label/image_name(i)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b'fixture')
                    rows.append({'file':path.name,'sha256':sha256(path),'z_score':z,'green_fraction':.5})
                write_json(run/'scores'/f'{label}.json', rows)
            select(run, 'lora_r4_wm')
            selection = read_json(run/'result/selection/lora_r4_wm.json')
            self.assertEqual(selection['selected_step'], 1)
            self.assertFalse(selection['test_used_for_selection'])
            self.assertFalse(selection['detector_used_for_selection'])
            self.assertTrue((run/'result/checkpoint_metrics.csv').exists())
            select(run, 'lora_r4_wm')  # resumable without test data
            record = run/'validation'/f"{validation_label('lora_r4_wm',2)}.json"
            row = read_json(record); row['denoising_mse'] = .01; write_json(record,row)
            with self.assertRaises(RuntimeError):
                select(run, 'lora_r4_wm')


if __name__ == '__main__':
    unittest.main()
