"""PNG-authoritative reporting for the privileged per-image control oracle.

The optimization margin is deliberately absent from the final verifier. Strong
model evidence never substitutes for acceptable per-image PNG quality/detection.
"""
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from statistics import mean, median

import torch

from common import image_name, sha256, write_json
from semantic_sign import image_evidence, model_evidence


def _finite(value, path='report'):
    if isinstance(value, dict):
        for key, item in value.items():
            _finite(item, f'{path}.{key}')
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _finite(item, f'{path}[{i}]')
    elif isinstance(value, Real) and not math.isfinite(value):
        raise ValueError(f'Nonfinite numeric value in {path}')


def _equal(actual, expected, name):
    if isinstance(expected, bool) or isinstance(expected, int) or name in ('p_img_png', 'clean_p_img'):
        valid = actual == expected
    elif name == 'clean_continuous_score':
        # Match semantic_report.score_rows' canonical saved-score validation.
        valid = math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-6)
    else:
        # Source scores may have been reduced in FP32 before JSON serialization.
        valid = math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-7)
    if not valid:
        raise ValueError(f'Stale derived evidence: {name}: {actual} != {expected}')


def _validated_rows(rows, signs, config):
    _finite(rows)
    if signs.device.type != 'cpu' or signs.shape != (32,):
        raise ValueError('Expected the locked 32 CPU secret signs')
    if not rows or len(rows) != config.get('oracle_n', 256):
        raise ValueError('Report requires all oracle_n images')
    if len({r['image_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate report image IDs')
    # Match semantic_sign_report.coordinates: mean-sign ties depend on FP32
    # reduction, so higher precision would silently change model evidence.
    coords = torch.tensor([r['coordinates'] for r in rows], dtype=torch.float32)
    clean = torch.tensor([r['clean_coordinates'] for r in rows], dtype=torch.float32)
    if coords.shape != (len(rows), 32) or clean.shape != coords.shape:
        raise ValueError('Expected exactly 32 clean and PNG coordinates per image')
    png_evidence, clean_evidence = image_evidence(coords, signs), image_evidence(clean, signs)
    result = []
    for row, png, original in zip(rows, png_evidence, clean_evidence):
        for field, expected in [('K_png', png['K']), ('p_img_png', png['p_img']),
                                ('analytic_detected_png', png['analytic_detected']),
                                ('clean_K', original['K']), ('clean_p_img', original['p_img']),
                                ('clean_continuous_score', original['score'])]:
            _equal(row[field], expected, field)
        _equal(row['clip_drop'], row['clean_clip']-row['oracle_clip'], 'clip_drop')
        for key in ('relative_delta_energy_per_call', 'delta_rms_per_call', 'cached_output_rms_per_call'):
            if len(row[key]) != 4:
                raise ValueError(f'Expected four tail-call values: {key}')
        result.append({**row, 'oracle_feasible': png['K'] >= 24 and row['png_lpips'] <= .15,
                       'K_change': png['K']-original['K'], 'initially_correct_bits': original['K'],
                       'bit_flips_toward_secret': sum(not a and b for a, b in zip(original['matches'], png['matches'])),
                       'bit_flips_away_from_secret': sum(a and not b for a, b in zip(original['matches'], png['matches'])),
                       'continuous_score_png': png['score'],
                       'clip_decreased': row['clip_drop'] > 0})
    return sorted(result, key=lambda r: r['image_id']), model_evidence(coords, signs)


def _bins(rows):
    definitions = [('<=14', 0, 14), ('15-16', 15, 16), ('17-18', 17, 18),
                   ('19-20', 19, 20), ('21-22', 21, 22), ('>=23', 23, 32)]
    result = []
    for label, low, high in definitions:
        selected = [r for r in rows if low <= r['clean_K'] <= high]
        first = [r['first_feasible_step'] for r in selected if r['first_feasible_step'] is not None]
        result.append({'clean_K_bin': label, 'sample_count': len(selected),
                       'oracle_feasible_rate': mean(r['oracle_feasible'] for r in selected) if selected else None,
                       'median_png_lpips': median(r['png_lpips'] for r in selected) if selected else None,
                       'median_relative_delta_energy': median(r['relative_delta_energy_total'] for r in selected) if selected else None,
                       'median_first_feasible_step': median(first) if first else None,
                       'first_feasibility_definition': 'first float K>=24 with min top24 margin>=0.02; excludes never-feasible images'})
    return result


def aggregate_metrics(rows, signs, config):
    """Validate serialized evidence, then apply only the locked PNG/CLIP gates."""
    from report import wilson

    _finite(config)
    for key, expected in [('final_png_lpips_max', .15), ('aggregate_clip_drop_max', .01),
                          ('oracle_feasible_rate_min', .90), ('K_detect', 24)]:
        if key in config and config[key] != expected:
            raise ValueError(f'Locked report setting changed: {key}')
    rows, model = _validated_rows(rows, signs, config)
    n = len(rows)
    feasible = sum(r['oracle_feasible'] for r in rows)
    floats = [r for r in rows if r['float_feasible']]
    clip_drop = mean(r['clip_drop'] for r in rows)
    result = {'status': 'oracle_pass' if feasible/n >= .90 and clip_drop <= .01 else 'oracle_fail',
              'oracle_n': n, 'oracle_feasible_count': feasible, 'oracle_feasible_rate': feasible/n,
              'oracle_feasible_ci95': wilson(feasible, n),
              'aggregate_clean_clip': mean(r['clean_clip'] for r in rows),
              'aggregate_oracle_clip': mean(r['oracle_clip'] for r in rows),
              'aggregate_clip_drop': clip_drop,
              'png_analytic_tpr': mean(r['analytic_detected_png'] for r in rows),
              'model_K_M': model['K_M'], 'model_p_M': model['p_M'],
              'model_bit_accuracy': model['bit_accuracy'], 'mean_continuous_score': model['mean_score'],
              'model_evidence': model,
              'float_feasible_rate': len(floats)/n,
              'png_after_float_feasible_survival_rate': mean(r['analytic_detected_png'] for r in floats) if floats else None,
              'png_after_float_feasible_survival_definition': 'PNG K>=24 among selected float-feasible candidates; no PNG quality gate',
              'png_quality_feasible_after_float_feasible_survival_rate': mean(r['oracle_feasible'] for r in floats) if floats else None,
              'median_best_step': median(r['best_step'] for r in rows),
              'median_relative_residual_energy': median(r['relative_delta_energy_total'] for r in rows),
              'per_image': rows, 'clean_k_bins': _bins(rows),
              'scientific_scope': 'Privileged per-image control oracle; not a deployable fingerprinting method. Oracle success does not establish teacher success.',
              'teacher_training_started': False}
    for name in ('png_lpips', 'png_pixel_mse'):
        result[f'mean_{name}'] = mean(r[name] for r in rows)
        result[f'median_{name}'] = median(r[name] for r in rows)
    diagnostics=[r['zero_control'] for r in rows
                 if r.get('zero_control',{}).get('fp32_graph_drift_is_diagnostic')]
    result['zero_control_precision_drift']={
        'policy':'Native replay/PNG/RSSF gates retained; FP32/native drift diagnostic only',
        'images_with_diagnostics':len(diagnostics),
        'images_outside_reference_tolerance':sum(d['fp32_graph_latent_outside_reference_tolerance']>0 for d in diagnostics),
        'elements_outside_reference_tolerance':sum(d['fp32_graph_latent_outside_reference_tolerance'] for d in diagnostics),
        'latent_elements':sum(d['fp32_graph_latent_elements'] for d in diagnostics),
        'max_latent_absolute_error':max((d['fp32_graph_latent_max_error'] for d in diagnostics),default=None),
        'max_latent_rms_error':max((d['fp32_graph_latent_rms_error'] for d in diagnostics),default=None)}
    _finite(result)
    return result


def _frontier(records, expected_ids):
    grouped = {}
    def visit(value):
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        step = value['step']
        if not isinstance(step, int) or step < 0:
            raise ValueError('Invalid frontier step')
        bucket = grouped.setdefault(step, {})
        for row in value['images']:
            ident = row['image_id']
            if ident in bucket or ident not in expected_ids:
                raise ValueError('Duplicate or unexpected frontier image ID')
            bucket[ident] = row
    _finite(records)
    visit(records)
    result = []
    for step, bucket in sorted(grouped.items()):
        if set(bucket) != expected_ids:
            raise ValueError('Frontier step does not cover all oracle images')
        images = list(bucket.values())
        result.append({'step': step, 'sample_count': len(images),
                       'mean_K_float': mean(r['K_float'] for r in images),
                       'fraction_K_float_ge24': mean(r['K_float'] >= 24 for r in images),
                       'mean_float_lpips': mean(r['float_lpips'] for r in images),
                       'median_float_lpips': median(r['float_lpips'] for r in images),
                       'mean_relative_delta_energy': mean(r['relative_delta_energy_total'] for r in images)})
    return result


def _representatives(rows, count):
    selected = set()
    successes = [r for r in rows if r['oracle_feasible']]
    failures = [r for r in rows if not r['oracle_feasible']]
    if successes:
        selected.add(min(successes, key=lambda r: (r['png_lpips'], r['image_id']))['image_id'])
        selected.add(min(successes, key=lambda r: (-r['relative_delta_energy_total'], r['image_id']))['image_id'])
    if failures:
        selected.add(min(failures, key=lambda r: r['image_id'])['image_id'])
    # Metric quantiles, stable ID tie breaks, with no aesthetic selection.
    for field in ('png_lpips', 'relative_delta_energy_total', 'image_id'):
        ordered = sorted(rows, key=lambda r: (r[field], r['image_id']))
        for i in range(count):
            if len(selected) >= count:
                break
            selected.add(ordered[round(i*(len(ordered)-1)/max(1, count-1))]['image_id'])
    for row in rows:
        if len(selected) >= count:
            break
        selected.add(row['image_id'])
    return sorted(selected)


def write_reports(output, rows, frontier_records, source_manifest, protocol, config, source_run, signs):
    """Write audit JSON and representative clean/oracle/absolute-difference panels."""
    from PIL import Image, ImageChops, ImageDraw

    output, source_run = Path(output).resolve(), Path(source_run).resolve()
    if output == source_run or source_run in output.parents or output in source_run.parents:
        raise ValueError('Report output must be disjoint from source run')
    report = aggregate_metrics(rows, signs, config)
    report['frontier'] = _frontier(frontier_records, {r['image_id'] for r in rows})
    _finite(source_manifest)
    _finite(protocol)
    report['source_hashes'] = source_manifest
    report['oracle_protocol_sha256'] = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    pairs = output/'result/sample_pairs'
    pairs.mkdir(parents=True, exist_ok=True)
    selected = _representatives(report['per_image'], min(len(rows), max(16, config.get('visual_count', 16))))
    manifests = []
    for ident in selected:
        clean_path = source_run/'images/teacher_clean_train'/image_name(ident)
        oracle_path = output/'oracle_images'/image_name(ident)
        with Image.open(clean_path) as im:
            clean = im.convert('RGB')
        with Image.open(oracle_path) as im:
            oracle = im.convert('RGB')
        if clean.size != oracle.size:
            raise ValueError('Clean/oracle pair dimensions differ')
        width, height = clean.size
        panel = Image.new('RGB', (3*width, height+24), 'white')
        draw = ImageDraw.Draw(panel)
        for column, (label, picture) in enumerate([('Clean', clean), ('Oracle', oracle), ('Absolute difference', ImageChops.difference(clean, oracle))]):
            panel.paste(picture, (column*width, 24))
            draw.text((column*width, 4), label, fill='black')
        path = pairs/image_name(ident)
        panel.save(path)
        manifests.append({'image_id': ident, 'clean_sha256': sha256(clean_path),
                          'oracle_sha256': sha256(oracle_path), 'panel_sha256': sha256(path)})
    write_json(pairs/'manifest.json', {'selection_rule': 'low LPIPS success, high residual-energy success, failure, then stable metric/ID quantiles',
                                      'difference_display': 'unscaled absolute RGB8 difference',
                                      'selected_image_ids': selected, 'files': manifests})
    for name in ('per_image', 'frontier', 'clean_k_bins'):
        write_json(output/'metrics'/f'{name}.json', report[name])
    aggregate = {k: v for k, v in report.items() if k not in ('per_image', 'frontier', 'clean_k_bins')}
    write_json(output/'metrics/aggregate.json', aggregate)
    write_json(output/'result/oracle_feasibility.json', aggregate)
    conclusion = ('The result supports proceeding to a separate teacher-imprinting experiment.' if report['status'] == 'oracle_pass' else
                  'The result indicates that the current RSSF per-image carrier should be reconsidered before further teacher optimization.')
    drift=report['zero_control_precision_drift']
    (output/'result/README.md').write_text(
        '# RSSF control oracle feasibility\n\n'
        'This is a privileged per-image control oracle and not a deployable fingerprinting method.\n\n'
        'Zero-control policy: native-dtype replay, canonical PNG and stored RSSF coordinate checks remain mandatory. '
        'FP32/native latent drift is diagnostic only; the FP32 graph is not corrected to match FP16. '
        'Quality remains relative to the original clean image.\n\n'
        f"FP32 drift diagnostics: {drift['images_with_diagnostics']}/{len(rows)} images recorded; "
        f"{drift['images_outside_reference_tolerance']} images outside the unchanged reference tolerance; "
        f"maximum absolute latent drift: {drift['max_latent_absolute_error']}; "
        f"maximum per-image latent RMS drift: {drift['max_latent_rms_error']}.\n\n"
        f"Status: **{report['status']}**. Actual PNG oracle feasibility: {report['oracle_feasible_count']}/{len(rows)} "
        f"({report['oracle_feasible_rate']:.2%}); aggregate CLIP drop: {report['aggregate_clip_drop']:.6f}.\n\n"
        'Float optimization feasibility requires K>=24 and the optimization-only top24 margin of 0.02. '
        'Actual PNG detector feasibility uses the unchanged RSSF verifier at K>=24. Paired image quality '
        'requires PNG LPIPS<=0.15. The primary gate requires PNG quality-and-detection feasibility>=90% '
        'and mean CLIP drop<=0.01. Model-level evidence is secondary.\n\n'
        f'{conclusion}\n\nOracle success does not establish teacher success. No teacher training starts automatically.\n', encoding='utf-8')
    _finite(report)
    return report
