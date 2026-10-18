"""CPU-only protocol/integrity tests: python3 -m unittest discover -s bitmark_lora."""
import contextlib
import io
import math
import tempfile
import unittest
from pathlib import Path

from assets import split_prompts
from common import freeze, image_name, read_json, sha256, verify_vendor, write_json
from report import calibrate, paired_ci, scores, summarize, wilson
from run import config_from_args, parse_args, stage
from student import branches, training_command, training_conditions


class ProtocolTests(unittest.TestCase):
    def config(self, **overrides):
        c = config_from_args(parse_args(["--ranks", "4", "--adapter-scopes", "attention"]))
        c.update(overrides)
        return c

    def test_vendor_integrity(self):
        verify_vendor()

    def test_matched_training_budget(self):
        c = self.config()
        students = branches(c)
        self.assertEqual([b['name'] for b in students], ['lora_r4_wm', 'full_wm'])
        self.assertEqual(training_conditions(c), ('wm',))
        for b in students:
            cmd = training_command(c, {'student': '/pinned/model'}, Path('/run'), b)
            self.assertEqual(cmd[cmd.index('--learning_rate')+1], '0.0001')
            self.assertEqual(cmd[cmd.index('--num_train_epochs')+1], '5')
            self.assertEqual(cmd[cmd.index('--max_train_steps')+1], '7500')
            self.assertNotIn('--use_ema', cmd)
            self.assertFalse(any('scores' in x or 'test' in x for x in cmd))
            self.assertNotIn('--push_to_hub', cmd)

    def test_ema_is_separate_control(self):
        c = self.config(paper_full_ema=True)
        b = branches(c)[-1]
        self.assertEqual(b['name'], 'paper_full_wm_ema')
        self.assertIn('--use_ema', training_command(c, {'student': '/model'}, Path('/run'), b))

    def test_optional_clean_controls(self):
        c = config_from_args(parse_args(['--train-clean-controls', '--ranks', '4', '--adapter-scopes', 'attention']))
        self.assertEqual(training_conditions(c), ('clean', 'wm'))
        self.assertEqual([b['name'] for b in branches(c)],
                         ['lora_r4_clean', 'lora_r4_wm', 'full_clean', 'full_wm'])

    def test_split_no_image_or_caption_leak(self):
        anns = [{'image_id': i//2, 'id': i, 'caption': f'image {i//2}'} for i in range(200)]
        a = split_prompts(anns, 0, {'train': 20, 'calibration': 30, 'test': 40})
        rows = sum(a.values(), [])
        self.assertEqual(len({r['image_id'] for r in rows}), 90)
        self.assertEqual(len({r['text'] for r in rows}), 90)
        self.assertEqual(a, split_prompts(anns, 0, {'train': 20, 'calibration': 30, 'test': 40}))

    def test_tied_calibration_never_exceeds_target(self):
        for values in ([0.]*100, list(range(100)), [1.]*99+[2.]):
            t = calibrate(values, .01)
            self.assertLessEqual(sum(v >= t for v in values), 1)

    def test_bad_scores_rejected(self):
        for values in ([], [float('nan')], [float('inf')]):
            with self.assertRaises(ValueError):
                calibrate(values, .01)

    def test_wilson_extremes(self):
        self.assertAlmostEqual(wilson(0, 100)[0], 0)
        self.assertGreater(wilson(0, 100)[1], 0)
        self.assertLess(wilson(100, 100)[0], 1)
        self.assertAlmostEqual(wilson(100, 100)[1], 1)

    def test_paired_zero_difference(self):
        self.assertEqual(paired_ci([0]*50), [0., 0.])

    def test_resume_rejects_changed_settings(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'manifest.json'
            freeze(p, {'rank': 4})
            freeze(p, {'rank': 4})
            with self.assertRaises(RuntimeError):
                freeze(p, {'rank': 16})

    def test_stage_failure_not_success(self):
        import sys
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):
                stage(Path(d), 'failure', [sys.executable, '-c', 'raise SystemExit(7)'])
            self.assertIn('"exit_code": 7', (Path(d)/'stages.jsonl').read_text())

    def test_complete_report_clean_adjustment_and_tamper(self):
        self.check_report(legacy=True)

    def test_report_without_clean_training_files(self):
        self.check_report(legacy=False)

    def check_report(self, legacy):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            run = Path(d)
            c = self.config(test_n=10, calibration_n=100)
            c.pop('checkpoint_steps')
            if legacy:
                del c['train_clean_controls']
            write_json(run/'config.json', c)
            data = {'teacher_clean_calibration': [0.]*100, 'teacher_clean_test': [0.]*10,
                    'teacher_wm_test': [10.]*10, 'unadapted': [0.]*10,
                    'lora_r4_wm': [10.]*2+[0.]*8, 'full_wm': [10.]*9+[0.]}
            if legacy:
                data.update(lora_r4_clean=[0.]*10, full_clean=[0.]*10)
            for label, values in data.items():
                rows = []
                out = run/'images'/label
                out.mkdir(parents=True)
                for i, z in enumerate(values):
                    p = out/image_name(i)
                    p.write_bytes(f'fixture-{label}-{i}'.encode())
                    rows.append({'file': p.name, 'sha256': sha256(p), 'z_score': z, 'green_fraction': .5})
                write_json(run/'scores'/f'{label}.json', rows)
            summarize(run)
            r = read_json(run/'result/radioactivity.json')
            key = 'full_minus_lora_r4' + ('_clean_adjusted' if legacy else '')
            self.assertAlmostEqual(r['paired_effects']['absolute_z'][key]['estimate_pp'], 70.)
            if not legacy:
                self.assertFalse(any('clean_adjusted' in k for k in r['paired_effects']['absolute_z']))
                self.assertEqual(len(r['tables']['absolute_z']), 5)
            self.assertTrue((run/'result/README.md').exists())
            (run/'images/unadapted'/image_name(0)).write_bytes(b'changed')
            with self.assertRaises(RuntimeError):
                scores(run, 'unadapted', 10)


if __name__ == '__main__':
    unittest.main()
