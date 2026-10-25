"""CPU-only scientific reporting contracts for the privileged control oracle."""
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from common import image_name, read_json
from semantic_sign import image_evidence, model_evidence
from semantic_oracle_report import aggregate_metrics, write_reports


SIGNS = torch.ones(32)


def row(i, k=24, lpips=.1, clip_drop=.005, float_feasible=True):
    coordinates = [1.] * k + [-1.] * (32-k)
    clean = [1.] * 16 + [-1.] * 16
    evidence = image_evidence(torch.tensor([coordinates]), SIGNS)[0]
    clean_evidence = image_evidence(torch.tensor([clean]), SIGNS)[0]
    return dict(image_id=i, prompt='synthetic prompt', clean_K=16,
                clean_p_img=clean_evidence['p_img'], clean_continuous_score=0.,
                clean_coordinates=clean, coordinates=coordinates,
                float_feasible=float_feasible, best_step=20, w_det_at_best=1.,
                K_float=24, min_top24_margin_float=.02, float_lpips=.09,
                float_pixel_mse=.001, first_feasible_step=10 if float_feasible else None,
                K_png=k, p_img_png=evidence['p_img'], analytic_detected_png=k>=24,
                png_lpips=lpips, png_pixel_mse=.002, clean_clip=.3,
                oracle_clip=.3-clip_drop, clip_drop=clip_drop,
                relative_delta_energy_total=.1,
                relative_delta_energy_per_call=[.1]*4,
                delta_rms_per_call=[.2]*4, cached_output_rms_per_call=[.6]*4)


class OracleReportTests(unittest.TestCase):
    def test_canonical_float32_model_mean_tie_and_saved_clean_score_tolerance(self):
        rows = [row(i, 32) for i in range(3)]
        for item, first_coordinate in zip(rows, [1.e8, -1., -1.e8]):
            item['coordinates'][0] = first_coordinate
            evidence = image_evidence(torch.tensor([item['coordinates']], dtype=torch.float32), SIGNS)[0]
            item.update(K_png=evidence['K'], p_img_png=evidence['p_img'],
                        analytic_detected_png=evidence['analytic_detected'])
            # Canonical score_rows accepts CPU/CUDA float32 reduction differences.
            item['clean_continuous_score'] += 9.e-7
        coordinates = torch.tensor([r['coordinates'] for r in rows], dtype=torch.float32)
        expected = model_evidence(coordinates, SIGNS)
        self.assertEqual(expected['K_M'], 32)
        self.assertEqual(model_evidence(coordinates.double(), SIGNS)['K_M'], 31)
        report = aggregate_metrics(rows, SIGNS, {'oracle_n': 3})
        self.assertEqual(report['model_evidence'], expected)

    def test_png_is_authoritative_in_both_float_disagreement_directions(self):
        report = aggregate_metrics([row(0, 23), row(1, 24, float_feasible=False)], SIGNS, {'oracle_n': 2})
        self.assertEqual(report['oracle_feasible_count'], 1)
        self.assertEqual(report['float_feasible_rate'], .5)
        self.assertEqual(report['png_after_float_feasible_survival_rate'], 0.)
        self.assertEqual([r['oracle_feasible'] for r in report['per_image']], [False, True])
        self.assertEqual(report['per_image'][1]['bit_flips_toward_secret'], 8)

    def test_exact_primary_gate_and_quality_gate(self):
        rows = [row(i) for i in range(10)]
        rows[-1] = row(9, lpips=.151)
        report = aggregate_metrics(rows, SIGNS, {'oracle_n': 10})
        self.assertEqual(report['status'], 'oracle_pass')
        self.assertEqual(report['oracle_feasible_rate'], .9)
        self.assertEqual(report['png_analytic_tpr'], 1.)
        from report import wilson
        self.assertEqual(report['oracle_feasible_ci95'], wilson(9, 10))
        rows[0] = row(0, clip_drop=.1)
        self.assertEqual(aggregate_metrics(rows, SIGNS, {'oracle_n': 10})['status'], 'oracle_fail')

    def test_negative_secret_sign_ties_and_away_flips(self):
        signs = -SIGNS
        r = row(0)
        clean = [-1.]*16 + [1.]*16
        final = [0.] + [-1.]*15 + [-1.]*9 + [1.]*7
        r['clean_coordinates'], r['coordinates'] = clean, final
        clean_ev = image_evidence(torch.tensor([clean]), signs)[0]
        final_ev = image_evidence(torch.tensor([final]), signs)[0]
        r.update(clean_K=clean_ev['K'], clean_p_img=clean_ev['p_img'],
                 clean_continuous_score=clean_ev['score'], K_png=final_ev['K'],
                 p_img_png=final_ev['p_img'], analytic_detected_png=final_ev['analytic_detected'])
        report = aggregate_metrics([r], signs, {'oracle_n': 1})
        self.assertEqual(report['per_image'][0]['bit_flips_away_from_secret'], 1)
        self.assertEqual(report['per_image'][0]['bit_flips_toward_secret'], 9)
        self.assertEqual(report['per_image'][0]['K_change'], 8)

    def test_empty_float_survival_and_nested_nonfinite(self):
        r = row(0, float_feasible=False)
        report = aggregate_metrics([r], SIGNS, {'oracle_n': 1})
        self.assertIsNone(report['png_after_float_feasible_survival_rate'])
        r['delta_rms_per_call'][2] = float('inf')
        with self.assertRaises(ValueError):
            aggregate_metrics([r], SIGNS, {'oracle_n': 1})

    def test_frontier_duplicate_and_partial_steps_reject(self):
        from semantic_oracle_report import _frontier
        entry = {'step': 10, 'images': [{'image_id': 0, 'K_float': 24,
                 'float_lpips': .1, 'relative_delta_energy_total': .2}]}
        with self.assertRaises(ValueError):
            _frontier([entry, entry], {0})
        with self.assertRaises(ValueError):
            _frontier([entry], {0, 1})

    def test_visual_selection_includes_categories_and_is_order_independent(self):
        from semantic_oracle_report import _representatives
        rows = [row(i) for i in range(30)]
        rows[29]['K_png'] = 23
        rows[28]['relative_delta_energy_total'] = 100.
        rows[27]['png_lpips'] = .0001
        for r in rows:
            r['oracle_feasible'] = r['K_png'] >= 24
        selected = _representatives(rows, 16)
        self.assertEqual(len(selected), 16)
        self.assertTrue({27, 28, 29}.issubset(selected))
        self.assertEqual(selected, _representatives(list(reversed(rows)), 16))

    def test_stale_evidence_nonfinite_and_duplicate_ids_reject(self):
        for key, value in [('K_png', 25), ('p_img_png', .1), ('clean_K', 15), ('clip_drop', float('nan'))]:
            bad = row(0)
            bad[key] = value
            with self.assertRaises(ValueError):
                aggregate_metrics([bad], SIGNS, {'oracle_n': 1})
        with self.assertRaises(ValueError):
            aggregate_metrics([row(0), row(0)], SIGNS, {'oracle_n': 2})
        bad = row(0, 32)
        bad['p_img_png'] = 0.
        with self.assertRaises(ValueError):
            aggregate_metrics([bad], SIGNS, {'oracle_n': 1})

    def test_report_bins_frontier_and_representative_visuals(self):
        rows = [row(i, 23 if i == 4 else 24) for i in range(5)]
        rows[1]['relative_delta_energy_total'] = 5.
        frontier = [[{'step': 10, 'images': [dict(image_id=r['image_id'], K_float=r['K_float'],
                     float_lpips=r['float_lpips'], relative_delta_energy_total=r['relative_delta_energy_total'])
                    for r in rows[:2]]}], [{'step': 10, 'images': [dict(image_id=r['image_id'],
                    K_float=r['K_float'], float_lpips=r['float_lpips'],
                    relative_delta_energy_total=r['relative_delta_energy_total']) for r in rows[2:]]}]]
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp)/'source', Path(tmp)/'oracle'
            for r in rows:
                for directory in [source/'images/teacher_clean_train', output/'oracle_images']:
                    directory.mkdir(parents=True, exist_ok=True)
                    Image.new('RGB', (8, 8), (128, 128, 128)).save(directory/image_name(r['image_id']))
            result = write_reports(output, rows, frontier, {'files': {'key': 'abc'}},
                                   {'version': 1}, {'oracle_n': 5}, source, SIGNS)
            self.assertEqual(result['frontier'][0]['sample_count'], 5)
            bins = read_json(output/'metrics/clean_k_bins.json')
            self.assertIsNone(bins[0]['oracle_feasible_rate'])
            self.assertEqual(bins[1]['sample_count'], 5)
            self.assertEqual(len(list((output/'result/sample_pairs').glob('sample_*.png'))), 5)
            self.assertEqual(read_json(output/'result/sample_pairs/manifest.json')['selected_image_ids'], [0, 1, 2, 3, 4])
            self.assertIn('not a deployable fingerprinting method', (output/'result/README.md').read_text())
            self.assertEqual(len(result['oracle_protocol_sha256']), 64)


if __name__ == '__main__':
    unittest.main()
