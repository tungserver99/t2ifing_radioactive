"""RSSF analytic verification and reporting, independent of empirical score thresholds.

The binomial null randomizes precommitted secret signs, conditional on a fixed
key-independent image/model. Fixed deployed-key image FPR is an empirical check;
image dependence is not used to claim a binomial sampling law across images.
"""
import csv
import math
import random
import statistics
from collections import Counter
import torch
from common import freeze, image_name, read_json, sha256, write_json
from semantic_carrier import SemanticCarrier
from semantic_sign import binomial_tail, detection_count, image_bits, image_evidence, model_evidence

REPORT_SEED = 271828
QUERY_BUDGETS = (10, 20, 50, 100, 200, 500, 1000)
NULL_DEFINITION = ('The evaluated image/model is independent of the secret key generated before '
                   'fingerprint training. Conditional on any fixed semantic vector, independently '
                   'randomized secret signs give K ~ Binomial(m, 0.5).')
NULL_CAVEAT = ('This is a key-randomization null, not a guarantee that images under one fixed '
               'deployed key are independent Bernoulli trials or have exactly nominal FPR. '
               'Report fixed-key clean FPR without rerolling or selecting the key.')


def _alpha(config):
    return config.get('analytic_alpha', config.get('alpha_test', .01))


def _coordinates(rows):
    return torch.tensor([row['coordinates'] for row in rows], dtype=torch.float32)


def validate_image_evidence(rows, carrier, alpha):
    """Reject stale or edited saved bits/counts/p-values rather than trust JSON."""
    expected = image_evidence(_coordinates(rows), carrier.signs, alpha)
    for row, evidence in zip(rows, expected):
        for field in ('centered_coordinates', 'bits', 'matches', 'K', 'p_img', 'analytic_detected'):
            if field not in row or row[field] != evidence[field]:
                raise ValueError(f'Saved semantic sign evidence mismatch: {field}')
        if 'centered_coordinates' in row and row['centered_coordinates'] != row['coordinates']:
            raise ValueError('Saved centered coordinates do not match semantic coordinates')


def analytic_metrics(run, label, n, carrier=None):
    from report import wilson
    from semantic_report import score_rows
    carrier = carrier if carrier is not None else SemanticCarrier.load(run/'carrier')
    c = read_json(run/'config.json'); alpha = _alpha(c)
    rows = score_rows(run, label, n, carrier=carrier)
    validate_image_evidence(rows, carrier, alpha)
    evidence = image_evidence(_coordinates(rows), carrier.signs, alpha)
    counts = [r['K'] for r in evidence]; ps = [r['p_img'] for r in evidence]
    detected = sum(r['analytic_detected'] for r in evidence); ci = wilson(detected, n)
    return {'branch': label, 'detected': detected, 'n': n, 'detection_rate': detected/n,
            'ci95_low': ci[0], 'ci95_high': ci[1], 'mean_K': statistics.mean(counts),
            'median_K': statistics.median(counts), 'K_distribution': _histogram(counts, len(carrier.signs)),
            'p_img_distribution': {'values': ps, 'min': min(ps), 'median': statistics.median(ps),
                                   'mean': statistics.mean(ps), 'max': max(ps)},
            'alpha_test': alpha, 'k_detect': detection_count(len(carrier.signs), alpha),
            **model_evidence(_coordinates(rows), carrier.signs, alpha)}


def _histogram(values, m):
    count = Counter(values)
    return {str(k): count[k] for k in range(m+1)}


def freeze_analytic_verifier(run):
    c = read_json(run/'config.json'); carrier = SemanticCarrier.load(run/'carrier')
    m = len(carrier.signs); alpha = _alpha(c)
    result = {'num_bits': m, 'alpha_test': alpha, 'k_detect': detection_count(m, alpha),
              'binomial_tail_table': {str(k): binomial_tail(k, m) for k in range(m+1)},
              'bit_tie_rule': 'r >= 0 -> +1; r < 0 -> -1; compare bits == secret signs',
              'analytic_null': 'key_randomization_binomial', 'null_definition': NULL_DEFINITION,
              'fixed_key_caveat': NULL_CAVEAT, 'carrier_sha256': sha256(run/'carrier/complete.json'),
              'selected_pc_indices': getattr(carrier, 'selected_pc_indices', None)}
    freeze(run/'result/analytic_verifier.json', result)
    return result


def random_key_validation(r, m, alpha, seed=REPORT_SEED, num_keys=1000):
    """Fake signs use the deployed selected PC set; coordinates never change."""
    bits = image_bits(r).detach().cpu().float()
    if bits.ndim != 2 or not len(bits) or bits.shape[1] != m or num_keys < 1:
        raise ValueError('Clean coordinates must be a nonempty N by m matrix')
    rng = random.Random(seed)
    keys = torch.tensor([[rng.choice((-1, 1)) for _ in range(m)] for _ in range(num_keys)], dtype=torch.float32)
    # +/-1 dot products are exact for practical carrier sizes; bound temporary
    # allocation by 128 fake keys and 4096 images rather than N*1000 matrices.
    histogram_tensor = torch.zeros(m+1, dtype=torch.int64)
    for image_start in range(0, len(bits), 4096):
        for key_start in range(0, num_keys, 128):
            dot = bits[image_start:image_start+4096] @ keys[key_start:key_start+128].T
            counts = ((m+dot)/2).to(torch.int64)
            histogram_tensor += torch.bincount(counts.flatten(), minlength=m+1)
    histogram = histogram_tensor.tolist()
    total = len(bits)*num_keys; cutoff = detection_count(m, alpha)
    mean = sum(k*v for k, v in enumerate(histogram))/total
    variance = sum((k-mean)**2*v for k, v in enumerate(histogram))/total
    return {'num_fake_keys': num_keys, 'num_images': len(bits), 'seed': seed,
            'selected_pc_set': 'same as deployed key; independently randomized signs only',
            'histogram': {str(k): value for k, value in enumerate(histogram)},
            'theoretical_probabilities': {str(k): math.comb(m,k)/2**m for k in range(m+1)},
            'empirical_tail_probability': sum(histogram[cutoff:])/total,
            'theoretical_tail_probability': binomial_tail(cutoff, m),
            'k_detect': cutoff, 'mean_K': mean, 'variance_K': variance,
            'theoretical_mean_K': m/2, 'theoretical_variance_K': m/4,
            'null_definition': NULL_DEFINITION, 'fixed_key_caveat': NULL_CAVEAT,
            'used_for_key_selection': False}


def validate_clean_null(run, labels=('teacher_clean_calibration', 'unadapted_calibration')):
    from semantic_report import score_rows
    c = read_json(run/'config.json'); carrier = SemanticCarrier.load(run/'carrier')
    random_results = {}; fixed = {}
    random_path = run/'result/random_key_null_validation.json'
    random_cache = read_json(random_path)['branches'] if random_path.exists() else {}
    carrier_hash = sha256(run/'carrier/complete.json')
    for label in labels:
        n = c['calibration_n'] if label.endswith('calibration') else c['test_n']
        rows = score_rows(run, label, n, carrier=carrier)
        proof = {'score_sha256': sha256(run/'scores'/f'{label}.json'),
                 'carrier_sha256': carrier_hash,
                 'selected_pc_indices': getattr(carrier, 'selected_pc_indices', None),
                 'num_fake_keys': 1000, 'num_images': n, 'seed': REPORT_SEED,
                 'k_detect': detection_count(len(carrier.signs), _alpha(c))}
        if label in random_cache:
            if any(random_cache[label].get(key) != value for key, value in proof.items()):
                raise RuntimeError(f'Clean null validation provenance changed: {label}')
            # score_rows above still checks every image hash and saved evidence.
            random_results[label] = random_cache[label]
        else:
            random_results[label] = random_key_validation(_coordinates(rows), len(carrier.signs), _alpha(c))
            random_results[label].update(proof)
        fixed[label] = analytic_metrics(run, label, n, carrier=carrier)
        fixed[label]['empirical_fixed_key_fpr'] = fixed[label]['detection_rate']
    # Extend previous calibration checks when final test reports become available.
    for name, entries in [('random_key_null_validation', random_results), ('fixed_key_clean_fpr', fixed)]:
        path = run/'result'/f'{name}.json'
        if path.exists():
            old = read_json(path)
            for label, value in entries.items():
                if label in old['branches'] and old['branches'][label] != value:
                    raise RuntimeError(f'Clean null validation changed: {label}')
            entries = {**old['branches'], **entries}
        write_json(path, {'branches': entries, 'null_definition': NULL_DEFINITION,
                          'fixed_key_caveat': NULL_CAVEAT, 'used_for_key_selection': False})
    return fixed


def query_budget_metrics(r, signs, alpha, seed=REPORT_SEED, repeats=100):
    if repeats < 2:
        raise ValueError('Query budget report requires multiple deterministic subsets')
    rng = random.Random(seed); result = []
    for n in QUERY_BUDGETS:
        if n > len(r):
            continue
        samples = [model_evidence(r[rng.sample(range(len(r)), n)], signs, alpha) for _ in range(repeats)]
        ks = [x['K_M'] for x in samples]; ps = [x['p_M'] for x in samples]
        result.append({'query_n': n, 'repeats': repeats, 'seed': seed,
                       'mean_K_M': statistics.mean(ks), 'median_K_M': statistics.median(ks),
                       'median_p_M': statistics.median(ps),
                       'detection_fraction': sum(p <= alpha for p in ps)/repeats,
                       'subset_K_M': ks, 'subset_p_M': ps, 'reporting_only': True})
    return result


def roc_auc(positive, negative):
    """Empirical ROC including tied scores; no dependency beyond the standard library."""
    if not positive or not negative:
        raise ValueError('ROC requires both positive and clean observations')
    groups = {}
    for values, index in ((positive, 0), (negative, 1)):
        for value in values:
            if not math.isfinite(value):
                raise ValueError('Nonfinite ROC score')
            groups.setdefault(value, [0, 0])[index] += 1
    tp = fp = 0; points = [{'fpr': 0., 'tpr': 0., 'threshold': None}]; auc = 0.
    for threshold, (p, n) in sorted(groups.items(), reverse=True):
        previous = points[-1]; tp += p; fp += n
        point = {'fpr': fp/len(negative), 'tpr': tp/len(positive), 'threshold': threshold}
        auc += (point['fpr']-previous['fpr'])*(point['tpr']+previous['tpr'])/2
        points.append(point)
    return {'auc': auc, 'roc': points, 'positive_n': len(positive), 'negative_n': len(negative)}


def paired_pixel_visuals(run, clean_label, fp_label, n, max_visuals=8):
    """MSE in the same [-1,1] image range as LPIPS and teacher quality loss."""
    from PIL import Image, ImageChops, ImageDraw
    import numpy as np
    values = []; outputs = []
    directory = run/'result/teacher_candidates'/f'{fp_label}_visuals'
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        with Image.open(run/'images'/clean_label/image_name(i)) as image:
            clean = image.convert('RGB')
        with Image.open(run/'images'/fp_label/image_name(i)) as image:
            fp = image.convert('RGB')
        if clean.size != fp.size:
            raise ValueError('Paired quality image sizes differ')
        a = np.asarray(clean, dtype=np.float64)/127.5-1
        b = np.asarray(fp, dtype=np.float64)/127.5-1
        values.append(float(((a-b)**2).mean()))
        if i < max_visuals:
            width, height = clean.size
            canvas = Image.new('RGB', (3*width, height+24), 'white')
            for j, (title, image) in enumerate((('Clean', clean), ('Fingerprint', fp),
                                               ('Absolute difference', ImageChops.difference(clean, fp)))):
                canvas.paste(image, (j*width, 24)); ImageDraw.Draw(canvas).text((j*width, 4), title, fill='black')
            path = directory/image_name(i); canvas.save(path)
            outputs.append(path.relative_to(run).as_posix())
    if not values:
        raise ValueError('No paired quality images')
    return {'paired_pixel_mse': statistics.mean(values), 'pixel_mse_per_image': values,
            'pixel_mse_range': '[-1,1]', 'visual_comparisons': outputs}


def _csv(path, rows, fields=None):
    if not rows and fields is None:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]), extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def summarize(run):
    from student import branches
    from semantic_report import freeze_thresholds, score_rows
    c = read_json(run/'config.json'); alpha = _alpha(c)
    carrier = SemanticCarrier.load(run/'carrier'); verifier = freeze_analytic_verifier(run)
    threshold = freeze_thresholds(run)
    labels = ['teacher_clean_test', 'teacher_wm_test', 'unadapted']+[b['name'] for b in branches(c)]
    rows = {label: score_rows(run, label, c['test_n'], carrier=carrier) for label in labels}
    utility = read_json(run/'result/utility.json') if (run/'result/utility.json').exists() else None
    validate_clean_null(run, ('teacher_clean_calibration', 'unadapted_calibration',
                                          'teacher_clean_test', 'unadapted'))
    table = []; models = {}; queries = {}; query_rows = []
    for label in labels:
        family = 'teacher' if label.startswith('teacher_') else 'student'
        metrics = analytic_metrics(run, label, c['test_n'], carrier=carrier)
        values = [r['score'] for r in rows[label]]; t = threshold[family]['threshold']
        clean_label = 'teacher_clean_test' if family == 'teacher' else 'unadapted'
        continuous = {'threshold': t, 'threshold_family': family, 'calibration_source': threshold[family]['source'],
                      'target_fpr': c['fpr'], 'detection_rate': sum(v >= t for v in values)/len(values),
                      'scores': values, **roc_auc(values, [r['score'] for r in rows[clean_label]])}
        metrics['secondary_continuous'] = continuous
        quality = (utility or {}).get('branches', {}).get(label, {})
        metrics.update(fid=quality.get('frechet_inception_distance'), kid=quality.get('kernel_inception_distance_mean'),
                       clip=quality.get('clip_score'))
        if label in ('teacher_clean_test', 'unadapted'):
            metrics['empirical_fixed_key_fpr'] = metrics['detection_rate']
        table.append(metrics)
        models[label] = model_evidence(_coordinates(rows[label]), carrier.signs, alpha)
        queries[label] = query_budget_metrics(_coordinates(rows[label]), carrier.signs, alpha)
        query_rows += [dict(branch=label, **item) for item in queries[label]]
    write_json(run/'result/model_level_verification.json', {'analytic_verifier': verifier, 'branches': models})
    write_json(run/'result/query_budget.json', {'reporting_only': True, 'branches': queries})
    _csv(run/'result/query_budget.csv', query_rows,
         ['branch', 'query_n', 'repeats', 'seed', 'mean_K_M', 'median_K_M', 'median_p_M', 'detection_fraction', 'reporting_only'])
    selection = {b['name']: read_json(run/'result/selection'/f"{b['name']}.json") for b in branches(c)}
    freeze(run/'result/selection_manifest.json', selection)
    tq = read_json(run/'result/teacher_quality.json'); quality_rows = []
    for candidate in tq['candidates']:
        q = candidate['quality']; m = candidate['metrics']; mu = candidate['coordinate_margin_mu']
        quality_rows.append({'coordinate_margin_mu': mu, 'selected': mu == tq['selected']['coordinate_margin_mu'],
                             'clean_clip': q['clean_clip'], 'fingerprinted_clip': q['fp_clip'], 'clip_drop': q['clip_drop'],
                             'paired_lpips': q['paired_lpips'], 'paired_pixel_mse': q['paired_pixel_mse'],
                             'quality_passes': q['passes'], 'analytic_validation_tpr': m['detection_rate'],
                             'K_M': m['K_M'], 'p_M': m['p_M']})
    _csv(run/'result/teacher_quality.csv', quality_rows)
    _csv(run/'result/detection_semantic.csv', [dict(r, secondary_tpr_at_calibrated_fpr=r['secondary_continuous']['detection_rate'],
            secondary_auc=r['secondary_continuous']['auc']) for r in table],
         ['branch', 'detected', 'n', 'detection_rate', 'ci95_low', 'ci95_high', 'mean_K', 'median_K', 'K_M', 'p_M',
          'bit_accuracy', 'mean_score', 'secondary_tpr_at_calibrated_fpr', 'secondary_auc', 'fid', 'kid', 'clip'])
    warnings = ['One training seed; image confidence intervals do not measure training-seed uncertainty.', NULL_CAVEAT,
                'Teacher selected using held-out validation only; student checkpoints selected solely by denoising MSE.',
                'Final test and deterministic query subsets are reporting only.']
    if utility is None:
        warnings.append('FID/KID/CLIP utility unavailable (explicitly skipped).')
    write_json(run/'result/radioactivity.json', {'method': 'semantic_sign_v1', 'primary': 'analytic_binomial',
               'analytic_verifier': verifier, 'secondary_continuous_thresholds': threshold, 'table': table,
               'selected_checkpoints': selection, 'teacher_quality': tq, 'warnings': warnings, 'utility': utility})
    lines = ['# Redundant semantic sign fingerprint: SD2.1 to SD1.4', '',
             f"Primary image test: K >= {verifier['k_detect']} of {verifier['num_bits']}; exact binomial p <= {alpha:g}.",
             NULL_DEFINITION, NULL_CAVEAT, '',
             '| Model | Analytic image TPR / clean FPR | Mean / median K | Model K | Model p | Bit accuracy | Mean score | Secondary TPR at calibrated FPR | AUC | FID | KID | CLIP |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    fmt = lambda x: 'unavailable' if x is None else f'{x:.5g}'
    for r in table:
        lines.append(f"| {r['branch']} | {r['detection_rate']:.2%} | {r['mean_K']:.3g} / {r['median_K']:.3g} | {r['K_M']}/{verifier['num_bits']} | {r['p_M']:.5g} | {r['bit_accuracy']:.1%} | {r['mean_score']:.5g} | {r['secondary_continuous']['detection_rate']:.2%} | {r['secondary_continuous']['auc']:.5g} | {fmt(r['fid'])} | {fmt(r['kid'])} | {fmt(r['clip'])} |")
    lines += ['', f"Secondary continuous TPR uses frozen disjoint clean {100*c['fpr']:g}% FPR calibration; it does not define the analytic verifier or training loss.",
              '', '## Paired teacher validation', '',
              '| Mu | Selected | Analytic TPR | CLIP drop | LPIPS | Pixel MSE [-1,1] | Quality pass |',
              '|---|---|---:|---:|---:|---:|---|']
    for q in quality_rows:
        lines.append(f"| {q['coordinate_margin_mu']:g} | {q['selected']} | {q['analytic_validation_tpr']:.2%} | {q['clip_drop']:.5g} | {q['paired_lpips']:.5g} | {q['paired_pixel_mse']:.5g} | {q['quality_passes']} |")
    lines += ['', 'Frozen gates: analytic teacher TPR >= 0.90, CLIP drop <= 0.01, LPIPS <= 0.15. Passing candidates ranked by LPIPS, pixel MSE, then mu.',
              'See query_budget.json/CSV, model_level_verification.json, random_key_null_validation.json, fixed_key_clean_fpr.json and paired visual comparisons in teacher_candidates/.', '']
    lines += ['- '+warning for warning in warnings]
    (run/'result/README.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
