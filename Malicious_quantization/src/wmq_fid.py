"""Post-selection Clean-FID. FID-to-marked-reference is drift, not FID-to-real."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

# Audited official NVIDIA response, 95,607,719 bytes, 2026-09-25.
INCEPTION_SHA256 = 'f58cb9b6ec323ed63459aa4fb441fe750cfe39fafad6da5cb504a16f19e958f4'


def prepare_inception(cache):
    """Bounded download, atomic publication, and verified-on-reuse cache."""
    import requests
    import torch
    from filelock import FileLock
    from cleanfid.downloads_helper import inception_url
    from prepare_sleepermark import digest
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / 'inception-2015-12-05.pt'
    marker = cache / 'inception.provenance.json'
    with FileLock(str(target) + '.lock', timeout=60):
        if target.is_file():
            if digest(target) == INCEPTION_SHA256:
                marker.write_text(json.dumps({'url': inception_url, 'sha256': INCEPTION_SHA256,
                                              'hash_policy': 'pinned official response; verified before loading'}))
                return
            if marker.is_file():
                raise ValueError(f'Inception cache checksum mismatch: {target}')
        temporary = cache / (target.name + '.part')
        for attempt in range(3):
            try:
                print(f'Downloading Clean-FID Inception (attempt {attempt+1}/3)', flush=True)
                with requests.get(inception_url, stream=True, timeout=(15, 60)) as response:
                    response.raise_for_status()
                    with temporary.open('wb') as handle:
                        for chunk in response.iter_content(1024 * 1024):
                            handle.write(chunk)
                sha256 = digest(temporary)
                if sha256 != INCEPTION_SHA256:
                    raise ValueError('Downloaded Inception checksum differs from pinned response')
                # A truncated/HTML response must never become the reusable cache.
                model = torch.jit.load(str(temporary), map_location='cpu')
                del model
                temporary.replace(target)
                marker.write_text(json.dumps({'url': inception_url, 'sha256': sha256,
                    'hash_policy': 'pinned official response; verified before loading'}))
                return
            except (requests.RequestException, OSError, RuntimeError, ValueError) as error:
                temporary.unlink(missing_ok=True)
                if attempt == 2:
                    raise RuntimeError('Cannot download valid Clean-FID Inception; owner results are retained. '
                                       'Retry wmq_fid.py when the network is available.') from error
                time.sleep(attempt + 1)


def feature_fid(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if (a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]
            or min(len(a), len(b)) < 2 or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError("FID requires two finite feature matrices with at least two images each")
    ma, mb = a.mean(0), b.mean(0)
    x = (a - ma) / np.sqrt(len(a) - 1)
    y = (b - mb) / np.sqrt(len(b) - 1)
    if max(len(a), len(b)) <= a.shape[1]:
        # Exact covariance cross term; much cheaper for small diagnostic sets.
        cross = np.linalg.svd(x @ y.T, compute_uv=False).sum()
        value = ((ma - mb) ** 2).sum() + (x*x).sum() + (y*y).sum() - 2 * cross
    else:
        from cleanfid.fid import frechet_distance
        value = frechet_distance(ma, np.cov(a, rowvar=False), mb, np.cov(b, rowvar=False))
    if not np.isfinite(value) or value < -1e-5:
        raise ValueError(f"Invalid FID: {value}")
    return float(max(0, value))


def image_files(folder):
    return sorted(str(p) for p in Path(folder).rglob('*')
                  if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp'))


def evaluate(run, real_reference=None, batch_size=16, device='cuda', ordinary=False):
    import torch
    from cleanfid.inception_torchscript import InceptionV3W
    from cleanfid.fid import get_files_features
    from wmq_blind import save_csv, save_json
    run = Path(run).resolve()
    manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
    if not (run / 'selection_frozen.json').is_file():
        raise ValueError('FID evaluation requires frozen attack selection')
    frozen = json.loads((run / 'selection_frozen.json').read_text(encoding='utf-8'))
    if frozen.get('manifest_sha256') and hashlib.sha256((run / 'manifest.json').read_bytes()).hexdigest() != frozen['manifest_sha256']:
        raise ValueError('Manifest changed after selection freeze')
    root = (run / manifest['image_root']).resolve()
    branches = manifest['test_branches']
    reference = manifest.get('reference_label', 'marked_reference_test')
    if ordinary:
        if manifest.get('watermark') != 'SleeperMark':
            raise ValueError('Ordinary-prompt branches apply only to SleeperMark')
        branches = [{**b, 'label': b['label'] + '_ordinary'} for b in branches]
        reference += '_ordinary'
    labels = [b['label'] for b in branches]
    if reference not in labels or len(set(labels)) != len(labels):
        raise ValueError('Missing reference or duplicate branch labels')
    for label in labels:
        if Path(label).name != label or label in ('.', '..') or '\\' in label:
            raise ValueError('Invalid branch label')
    cache = Path(os.environ.get('WMQ_HEAVY_ROOT', Path(__file__).resolve().parents[1] / 'output_artifacts')) / 'cache' / 'clean_fid'
    cache.mkdir(parents=True, exist_ok=True)
    prepare_inception(cache)
    model = InceptionV3W(str(cache), download=False, resize_inside=False).to(device).eval()
    def features(files):
        if len(files) < 2:
            raise ValueError('Need at least two images per FID distribution')
        return get_files_features(files, model=model, num_workers=0, batch_size=batch_size,
                                  device=torch.device(device), mode='clean')
    ref_files = image_files(root / reference)
    ref = features(ref_files)
    real_files = image_files(real_reference) if real_reference else None
    real = features(real_files) if real_files else None
    if real_reference and real is None:
        raise ValueError('Empty real-reference directory')
    rows = []
    for branch in branches:
        files = image_files(root / branch['label'])
        if len(files) != len(ref_files):
            raise ValueError(f"Incomplete branch {branch['label']}: {len(files)} != {len(ref_files)}")
        values = ref if branch['label'] == reference else features(files)
        row = dict(label=branch['label'], fid_to_marked_reference=feature_fid(values, ref),
                   n_generated=len(values), n_marked_reference=len(ref),
                   small_sample_warning=min(len(values), len(ref)) < 2048,
                   feature_extractor='clean-fid InceptionV3 2048', mode='clean',
                   used_for_selection=False)
        if real is not None:
            row.update(fid_to_real=feature_fid(values, real), n_real=len(real),
                       real_reference=str(Path(real_reference).resolve()),
                       small_sample_warning=min(len(values), len(ref), len(real)) < 2048)
        rows.append(row)
        save_csv(run / ('ordinary_fid.csv' if ordinary else 'fid.csv'), rows)
        print(row, flush=True)
    save_json(run / ('ordinary_fid.json' if ordinary else 'fid.json'), {'rows': rows,
        'interpretation': 'FID-to-marked-reference measures distribution drift, not realism. '
                          'Small-sample FID is biased; compare only matched sample counts/protocols. '
                          'FID is aggregate and cannot certify individual image quality.',
        'real_reference_files': real_files})
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--real-reference', type=Path)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--device', default='cuda')
    p.add_argument('--ordinary', action='store_true')
    a = p.parse_args()
    try:
        evaluate(a.run, a.real_reference, a.batch_size, a.device, a.ordinary)
        error_path = a.run / 'fid_error.json'
        if error_path.exists():
            error_path.write_text(json.dumps({'status': 'recovered_for_requested_mode',
                                             'ordinary': a.ordinary}), encoding='utf-8')
    except Exception as error:
        if a.run.is_dir():
            (a.run / 'fid_error.json').write_text(json.dumps({'status': 'failed',
                'error': f'{type(error).__name__}: {error}', 'ordinary': a.ordinary}), encoding='utf-8')
        raise
