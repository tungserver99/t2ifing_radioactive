"""Post-freeze descriptive quality/evasion analysis. Never select attack weights."""
import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
from collections import defaultdict


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def table(path, rows):
    if rows:
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


def frontier(rows):
    """Maximize SSIM and conditional evasion within ONE comparison group."""
    return [r for r in rows if not any(
        s['ssim'] >= r['ssim'] and s['conditional_evasion'] >= r['conditional_evasion'] and
        (s['ssim'] > r['ssim'] or s['conditional_evasion'] > r['conditional_evasion']) for s in rows)]


def quality_views(owner, report, manifest, thresholds):
    reference = next(r for r in owner['rows'] if r['method'] == manifest.get('reference_label', 'marked_reference_test'))
    base = reference['per_image_detected']
    denominator = sum(base)
    valid = owner['reference_validation']['valid'] and denominator > 0
    points, views = [], []
    branches = {b['label']: b for b in manifest['test_branches']}
    for row in owner['rows']:
        quality = report['branch_quality'][row['method']]
        branch = branches[row['method']]
        detected = row['per_image_detected']
        if len(detected) != len(base):
            raise ValueError('Unpaired owner outcomes')
        conditional = sum(a and not b for a, b in zip(base, detected)) / denominator if valid else None
        group = branch.get('comparison_group', row.get('comparison_group'))
        threat = branch.get('threat_model', 'unspecified')
        point = {'method': row['method'], 'role': row.get('role'), 'comparison_group': group,
            'threat_model': threat, 'ssim': row['ssim'], 'psnr': row['psnr'], 'lpips': row.get('lpips'),
            'absolute_evasion': 1 - row['tpr'], 'conditional_evasion': conditional,
            'reference_valid': valid, 'baseline_detected': denominator,
            'original_quality_valid': row.get('test_quality_valid')}
        points.append(point)
        for threshold in thresholds:
            psnr_gate = manifest['args']['min_psnr']
            min_psnr_gate = manifest['args']['min_image_psnr']
            min_psnr = quality.get('min_psnr', min(quality.get('per_image_psnr') or [row['psnr']]))
            p, s = quality.get('per_image_psnr'), quality.get('per_image_ssim')
            if p is not None and s is not None and (len(p) != len(base) or len(s) != len(base)):
                raise ValueError('Unpaired per-image quality')
            joint = sum(a and not b and pi >= psnr_gate and si >= threshold
                        for a, b, pi, si in zip(base, detected, p, s)) / denominator if valid and p is not None and s is not None else None
            views.append({**point, 'min_ssim_view': threshold,
                'aggregate_quality_valid': row['psnr'] >= psnr_gate and min_psnr >= min_psnr_gate and row['ssim'] >= threshold,
                'per_image_quality_violation_fraction': sum(pi < psnr_gate or si < threshold for pi, si in zip(p, s)) / len(p) if p is not None and s is not None else None,
                'joint_quality_and_evasion_on_baseline_detected': joint,
                'interpretation': 'posthoc view; does not retrain or change frozen checkpoint'})
    return points, views


def plot_svg(path, points):
    groups = defaultdict(list)
    for point in points:
        if point['comparison_group'] and point['conditional_evasion'] is not None:
            groups[(point['comparison_group'], point['threat_model'])].append(point)
    sizes = [max(300, 65 + len(rows) * 18) for rows in groups.values()]
    height = sum(sizes) + 60 if sizes else 100
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}" viewBox="0 0 1000 {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<g font-family="sans-serif" font-size="12" fill="#222">',
           '<text x="30" y="22">Observed checkpoints: SSIM vs evasion among baseline-detected images. Descriptive, not a continuous budget curve.</text>']
    y0 = 55
    for panel, (group, rows) in enumerate(groups.items()):
        lo = max(0., min(r['ssim'] for r in rows) - .03)
        x = lambda v: 65 + (v - lo) / max(1. - lo, .01) * 390
        y = lambda v: y0 + 205 - v * 170
        svg.append(f'<text x="30" y="{y0}">{html.escape(str(group))}</text>')
        svg.append(f'<path d="M65 {y0+30} V{y0+205} H455" fill="none" stroke="black"/>')
        for ev in (0., .5, 1.):
            svg.append(f'<text x="28" y="{y(ev)+4}">{ev:.0%}</text>')
        for ss in (lo, (lo+1)/2, 1.):
            svg.append(f'<text x="{x(ss)-12}" y="{y0+226}">{ss:.3f}</text>')
        svg.append(f'<text x="240" y="{y0+246}">SSIM (higher is better)</text>')
        svg.append(f'<circle cx="455" cy="{y(0)}" r="4" fill="#aaa"><title>Marked reference: zero additional evasion by definition</title></circle>')
        svg.append(f'<text x="65" y="{y0+267}">Orange: non-dominated in this group; gray at (1,0): marked reference.</text>')
        front = frontier(rows)
        for i, r in enumerate(rows, 1):
            color = '#c2410c' if r in front else '#475569'
            svg.append(f'<circle cx="{x(r["ssim"]):.2f}" cy="{y(r["conditional_evasion"]):.2f}" r="5" fill="{color}"><title>{html.escape(r["method"])}</title></circle>')
            svg.append(f'<text x="{x(r["ssim"])+6:.2f}" y="{y(r["conditional_evasion"])-4:.2f}">{i}</text>')
            svg.append(f'<text x="485" y="{y0+24+i*18}" font-size="10" fill="{color}">{i}: {html.escape(r["method"])}</text>')
        y0 += sizes[panel]
    svg.extend(['</g>', '</svg>'])
    path.write_text('\n'.join(svg), encoding='utf-8')


def layer_analysis(root, output, method='natural_residual_w4_c1.0_test', top_k=8):
    path = root / 'mechanism_analysis.json'
    if not path.exists():
        return
    data = json.loads(path.read_text())
    report = json.loads((root / 'report.json').read_text())
    if data.get('selection_frozen_sha256') != report.get('selection_frozen_sha256'):
        raise ValueError('Mechanism diagnostics belong to another frozen run')
    groups = defaultdict(list)
    for row in data['rows']:
        groups[(row['method'], row['layer'])].append(row)
    ranked = []
    for (label, layer), samples in groups.items():
        gain = -sum(r['owner_error_dot'] for r in samples) / len(samples)
        cost = sum(abs(r['quality_error_dot']) for r in samples) / len(samples)
        ranked.append({'method': label, 'layer': layer, 'n': len(samples), 'local_owner_gain': gain,
            'absolute_local_quality_dot': cost,
            'negative_owner_dot_fraction': sum(r['owner_error_dot'] < 0 for r in samples) / len(samples)})
    # Normalize by a method-level floor so near-zero quality dots do not imply infinite benefit.
    for label in {r['method'] for r in ranked}:
        same = [r for r in ranked if r['method'] == label]
        floor = max(1e-12, sum(r['absolute_local_quality_dot'] for r in same) / len(same) * .1)
        for row in same:
            row['diagnostic_priority'] = max(0., row['local_owner_gain']) / (row['absolute_local_quality_dot'] + floor)
    ranked.sort(key=lambda r: (r['method'], -r['diagnostic_priority']))
    table(output / 'layer_diagnostics.csv', ranked)
    chosen = [r for r in ranked if r['method'] == method and r['local_owner_gain'] > 0 and
              r['negative_owner_dot_fraction'] >= .75 and r['layer'].endswith('.weight')][:top_k]
    if chosen:
        dump(output / 'owner_informed_steering_plan.json', {'schema_version': 1,
            'source_run': str(root.resolve()), 'source_method': method, 'source_split': 'victim_test',
            'uses_owner_information': True, 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'layers': [r['layer'] for r in chosen], 'samples': data['samples'],
            'warning': 'Exploratory endpoint Taylor diagnostic, not causal layer attribution. Owner-informed DEVELOPMENT only; requires fresh final holdout. Not blind or surrogate-only.'})


def build_report(root, thresholds=(.8, .82, .85, .86, .9, .95), output=None):
    root = Path(root)
    if any(not math.isfinite(t) or not 0 < t <= 1 for t in thresholds):
        raise ValueError('SSIM thresholds must be finite in (0,1]')
    owner = json.loads((root / 'owner_evaluation.json').read_text())
    report = json.loads((root / 'report.json').read_text())
    manifest = json.loads((root / 'manifest.json').read_text())
    output = Path(output) if output else root / 'tradeoff'
    output.mkdir(parents=True, exist_ok=False)
    points, views = quality_views(owner, report, manifest, thresholds)
    groups = defaultdict(list)
    for point in points:
        if point['comparison_group'] and point['conditional_evasion'] is not None:
            groups[(point['comparison_group'], point['threat_model'])].append(point)
    fronts = [p for rows in groups.values() for p in frontier(rows)]
    table(output / 'points.csv', points)
    table(output / 'quality_thresholds.csv', views)
    table(output / 'pareto.csv', fronts)
    plot_svg(output / 'quality_evasion.svg', points)
    layer_analysis(root, output)
    dump(output / 'analysis.json', {'source_run': str(root.resolve()), 'quality_policy': manifest['args']['quality_policy'],
        'analysis_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_sha256': {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('manifest.json', 'report.json', 'owner_evaluation.json')},
        'reference_valid': owner['reference_validation']['valid'], 'thresholds': list(thresholds),
        'interpretation': 'Observed owner-evaluated checkpoints only. Search checkpoints lacking owner metrics are NOT curve points. No global winner, no checkpoint reselection. FP32, bitwidths and threat models have separate frontiers.',
        'gate_note': 'report policy retains failed-quality candidates; changing an SSIM label alone does not improve the attack.',
        'verification': 'Derived from saved reports; does not independently rerun images/checkpoints.'})
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--output')
    p.add_argument('--ssim-thresholds', nargs='+', type=float, default=[.8, .82, .85, .86, .9, .95])
    args = p.parse_args()
    print(build_report(args.run, args.ssim_thresholds, args.output))
