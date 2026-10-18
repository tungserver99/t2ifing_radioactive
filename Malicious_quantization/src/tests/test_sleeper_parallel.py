import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from wmq_sleeper_parallel import split_arguments, automatic_parallelism, aggregate


class SleeperParallelTests(unittest.TestCase):
    def test_split_defaults_and_explicit_methods(self):
        methods, engine, options = split_arguments(['--steps', '7', '--parallel-branches', '2'])
        self.assertIn('coherent_probe_qat', methods)
        self.assertNotIn('conditional_rollout_qat', methods)
        self.assertEqual(engine, ['--steps', '7'])
        self.assertEqual(options.parallel_branches, 2)
        methods, engine, _ = split_arguments(['--methods', 'fixed_ptq', 'equivariance_qat',
                                               '--test-n', '3'])
        self.assertEqual(methods, ['fixed_ptq', 'equivariance_qat'])
        self.assertEqual(engine, ['--test-n', '3'])
        with self.assertRaises(ValueError):
            split_arguments(['--methods', 'fixed_ptq', 'fixed_ptq'])
        for value in ('nan', '0', '-1'):
            with self.assertRaises(ValueError):
                split_arguments(['--parallel-vram-per-process-gib', value])

    @patch('torch.cuda.mem_get_info', return_value=(80*2**30, 96*2**30))
    def test_parallelism_uses_free_memory_not_gpu_name(self, _):
        workers, free, total, reserve = automatic_parallelism(9, 12, 8)
        self.assertEqual(workers, 6)
        self.assertEqual((free, total, reserve), (80., 96., 8.))

    def test_aggregate_one_attack_row_and_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = Path(tmp)
            result = suite/'fixed_ptq'/'sleepermark_1'
            result.mkdir(parents=True)
            with (result/'watermark_retention.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=['label', 'tpr'])
                writer.writeheader()
                writer.writerows([{'label': 'marked_reference_test', 'tpr': '1'},
                                  {'label': 'fixed_ptq_w4', 'tpr': '.9'}])
            for file, label in [('fid.csv', 'fixed_ptq_w4'), ('ordinary_fid.csv', 'fixed_ptq_w4_ordinary')]:
                with (result/file).open('w', newline='', encoding='utf-8') as handle:
                    writer = csv.DictWriter(handle, fieldnames=['label', 'fid_to_marked_reference'])
                    writer.writeheader(); writer.writerow({'label': label, 'fid_to_marked_reference': '12.3'})
            jobs = [{'method': 'fixed_ptq', 'returncode': 0, 'log': suite/'fixed.log'},
                    {'method': 'bad', 'returncode': 4, 'log': suite/'bad.log'}]
            failures = aggregate(suite, jobs, {'parallel_branches': 2})
            self.assertEqual(len(failures), 1)
            with (suite/'parallel_summary.csv').open(encoding='utf-8') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]['label'], 'fixed_ptq_w4')
            self.assertEqual(rows[0]['fid_status'], 'complete')
            self.assertEqual(rows[0]['fid_ordinary_to_marked_reference'], '12.3')
            self.assertEqual(rows[0]['baseline_tpr'], '1')
            self.assertEqual(json.loads((suite/'parallel_summary.json').read_text())['status'], 'partial_failure')


if __name__ == '__main__':
    unittest.main()
