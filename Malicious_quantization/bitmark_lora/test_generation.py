"""Regression checks for generation control flow; no GPU/model downloads."""
import ast
import contextlib
import io
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from common import HERE, read_json, write_json


class GenerationTests(unittest.TestCase):
    def test_optional_statistics_and_scale_reset(self):
        source = HERE / 'vendor/bitmark/Infinity/infinity/models/infinity.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == 'autoregressive_infer_cfg')
        loop = next(n for n in method.body if isinstance(n, ast.For)
                    and isinstance(n.target, ast.Tuple)
                    and [x.id for x in n.target.elts] == ['si', 'pn'])
        reset = next(n for n in loop.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'scale_bit_flips'
                             for t in n.targets))
        stats = next(n for n in loop.body if isinstance(n, ast.If)
                     and isinstance(n.test, ast.Name) and n.test.id == 'get_stat_data')
        reset_code = compile(ast.Module(body=[reset], type_ignores=[]), str(source), 'exec')
        stats_code = compile(ast.Module(body=[stats], type_ignores=[]), str(source), 'exec')
        env = dict(get_stat_data=True, batch_size=2, ret={0: {'stat_data': {}}, 1: {'stat_data': {}}},
                   scale_entropy=[.2, .3], token_entropy=[.4], bit_n_of_tokens_entropy=[.5])
        # Disabled statistics, enabled statistics, then a scale with no measurement.
        for si, flips in enumerate((None, [3, 4], None)):
            exec(reset_code, env)
            env['si'] = si
            if flips is not None:
                env.update(scale_bit_flips=flips, token_bit_flips=[1], bit_n_of_tokens_flips=[2])
            exec(stats_code, env)
            for i in range(2):
                actual = env['ret'][i]['stat_data'][f'scale_{si}']
                self.assertEqual(actual['scale']['entropy'], env['scale_entropy'][i])
                self.assertEqual('bit_flips' in actual['scale'], flips is not None)
                if flips is not None:
                    self.assertEqual(actual['scale']['bit_flips'], flips[i])

    def test_teacher_splits_pairing_and_resume(self):
        import numpy as np
        import teacher
        calls = []
        args = types.SimpleNamespace(seed=0)
        vae = types.SimpleNamespace(tgt_h=8, tgt_w=8, vae=object())

        class ImageTensor:
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return np.zeros((8, 8, 3), dtype=np.uint8)

        def gen_img(prompts, vae, watermark):
            calls.append((prompts, args.seed, args.watermark_delta))
            return {}, [], ImageTensor()

        model = types.SimpleNamespace(args=args, gen_img=gen_img)
        official = types.SimpleNamespace(WatermarkInference=lambda args, vae: object())
        torch_stub = types.SimpleNamespace(manual_seed=lambda seed: None,
                                          inference_mode=contextlib.nullcontext)
        with tempfile.TemporaryDirectory() as tmp, patch.dict('sys.modules', {'torch': torch_stub}), \
             patch.object(teacher, 'initialize', return_value=(args, vae, model, official)), \
             contextlib.redirect_stdout(io.StringIO()):
            run = Path(tmp)
            write_json(run/'config.json', {'seed': 7, 'train_clean_controls': True})
            prompts = {split: [{'image_id': offset+i, 'text': f'{split}-{i}'} for i in range(n)]
                       for split, offset, n in [('train', 0, 3), ('calibration', 10, 2), ('test', 20, 2)]}
            write_json(run/'prompts.json', prompts)
            teacher.generate(run)
            self.assertEqual(len(calls), 12)
            for split in ('train', 'test'):
                n = len(prompts[split])
                rows = [r for r in calls if r[0][0].startswith(split)]
                self.assertEqual([r[1] for r in rows[:n]], [r[1] for r in rows[n:]])
                self.assertEqual([r[2] for r in rows], [0.]*n+[2.]*n)
            for folder in (run/'images').iterdir():
                records = read_json(folder/'image_hashes.json')
                self.assertEqual(len(records), len(list(folder.glob('sample_*.png'))))
                self.assertEqual(len((folder/'metadata.jsonl').read_text().splitlines()), len(records))
            teacher.generate(run)
            self.assertEqual(len(calls), 12, 'Resume must reuse completed valid images')


if __name__ == '__main__':
    unittest.main()
