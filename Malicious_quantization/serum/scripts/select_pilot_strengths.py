"""Stage 2: deterministic, predeclared preservation-only selection."""
import json
from pathlib import Path
import math

root = Path('/root/workspace/serum')
policy = json.loads((root/'manifests/stage12_queue_policy.json').read_text())['preservation_policy']
rows = []
for strength in (.25, .50, .75, 1.):
    evidence = {}
    acceptable = True
    for kind in ('clean', 'wm'):
        path = root/f'metrics/pilot/{kind}_{strength:.2f}.json'
        data = json.loads(path.read_text())
        assert data['count'] == 500
        m = data['means']
        assert all(math.isfinite(v) for v in m.values())
        retention = m['regen_clipscore'] / max(m['raw_clipscore'], 1e-12)
        ok = (m['clip_image_cosine'] >= policy['minimum_clip_image_cosine']
              and m['dino_cosine'] >= policy['minimum_dino_cosine']
              and retention >= policy['minimum_prompt_clipscore_retention']
              and m['lpips'] <= policy['maximum_lpips'])
        acceptable = acceptable and ok
        evidence[kind] = {'means': m, 'prompt_clipscore_retention': retention,
                          'acceptable': ok, 'metrics_file': str(path)}
    preservation = sum(x['means']['clip_image_cosine'] + x['means']['dino_cosine']
                       - x['means']['lpips'] for x in evidence.values()) / 2
    rows.append({'strength': strength, 'acceptable': acceptable,
                 'preservation_score': preservation, 'evidence': evidence})
eligible = [r for r in rows if r['acceptable']]
selection = {'criterion': 'image preservation only', 'policy': policy, 'candidates': rows,
             'baseline_status': 'under investigation; selection does not approve C10 baseline'}
if len(eligible) >= 2:
    # The strongest acceptable regeneration is reserved for the strong condition.
    strong = max(eligible, key=lambda r: r['strength'])
    mild = max([r for r in eligible if r['strength'] < strong['strength']],
               key=lambda r: (r['preservation_score'], -r['strength']))
    selection.update(status='selected', mild=mild['strength'], strong=strong['strength'],
                     reason='Mild maximizes CLIP + DINO - LPIPS among acceptable lower strengths; strong is the highest strength satisfying all predeclared preservation bounds on both clean and watermarked images.')
else:
    selection.update(status='no_acceptable_pair', reason='Fewer than two strengths meet the predeclared preservation criteria; do not silently relax bounds or fabricate an operating pair.')
(root/'metrics/pilot_selection.json').write_text(json.dumps(selection, indent=2))
print(json.dumps(selection, indent=2), flush=True)
