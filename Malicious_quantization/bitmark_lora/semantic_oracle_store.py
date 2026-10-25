"""Read-only source identities and marker-last, immutable oracle artifacts."""
import json
import math
import os
from pathlib import Path
import tempfile
import torch
from PIL import Image
from common import image_name, read_json, sha256


def reject_overlap(output, source):
    output, source = Path(output).resolve(), Path(source).resolve()
    if output == source or output in source.parents or source in output.parents:
        raise ValueError('Oracle output and source must be disjoint')


def _relative(folder, name):
    path = (folder/name).resolve()
    if path == folder or folder not in path.parents:
        raise ValueError('Artifact path escapes its root')
    return path


def verify_source_manifest(manifest):
    root = Path(manifest['source_run']).resolve()
    for name, digest in manifest['files'].items():
        path = _relative(root, name)
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f'Source artifact changed: {name}')
    for name, digest in manifest.get('external_files', {}).items():
        path = Path(name)
        if not path.is_absolute() or not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f'Source model changed: {name}')


def load_source(source):
    """Validate existing artifacts without invoking any generation/upgrade routine."""
    from semantic_assets import snapshot_identity
    from semantic_carrier import SemanticCarrier
    from semantic_trajectory import TraceReader, digest
    from semantic_report import score_rows
    source = Path(source).resolve()
    config = read_json(source/'config.json')
    locked = {'teacher_objective': 'semantic_sign_v2_trajectory', 'num_bits': 32,
              'analytic_alpha': .01, 'inference_steps': 50, 'tail_unroll_calls': 4}
    if any(config.get(k) != v for k, v in locked.items()):
        raise ValueError('Incompatible oracle source configuration')
    assets, prompts = read_json(source/'assets.json'), read_json(source/'prompts.json')
    model = Path(assets['teacher_clean']).resolve()
    identity = snapshot_identity(model)
    if identity != assets['teacher_clean_hashes']:
        raise RuntimeError('Clean model differs from frozen source snapshot')
    if assets.get('prompt_sha256') != sha256(source/'prompts.json'):
        raise RuntimeError('Source prompt identity changed')
    carrier = SemanticCarrier.load(source/'carrier', device='cpu')
    from semantic_sign import OBJECTIVE
    if len(carrier.signs) != 32 or carrier.carrier_version != OBJECTIVE:
        raise ValueError('Source requires a 32-bit RSSF sign carrier')
    reader = TraceReader(source/'teacher_trace')
    if len(reader.index) != len(prompts['train']):
        raise ValueError('Trace/prompt coverage mismatch')
    expected_protocol = {'teacher_objective': locked['teacher_objective'], 'tail_unroll_calls': 4,
                         'config_sha256': sha256(source/'config.json'),
                         'prompts_sha256': sha256(source/'prompts.json'), 'clean_model_hash': digest(identity),
                         'saved_scheduler_sha256': sha256(model/'scheduler/scheduler_config.json')}
    if any(reader.protocol.get(k) != v for k, v in expected_protocol.items()):
        raise RuntimeError('Trace protocol source identity mismatch')
    if reader.protocol.get('native', {}).get('num_inference_steps') != 50:
        raise ValueError('Trace must use 50 native scheduler steps')
    trace_files = read_json(source/'teacher_trace/complete.json')['files']
    required_trace = {'protocol.json', 'index.json'} | set(reader.index.values()) | {n.replace('.json', '.safetensors') for n in reader.index.values()}
    if not required_trace <= set(trace_files):
        raise RuntimeError('Trace completion proof omits required members')
    scores = score_rows(source, 'teacher_clean_train', len(prompts['train']), carrier=carrier)
    names = {'config.json', 'assets.json', 'prompts.json', 'carrier/complete.json',
             'carrier/key.json', 'teacher_trace/complete.json', 'teacher_trace/protocol.json',
             'teacher_trace/index.json', 'images/teacher_clean_train/image_hashes.json',
             'images/teacher_clean_train/generation.json', 'images/teacher_clean_train/metadata.jsonl',
             'scores/teacher_clean_train.json',
             'scores/teacher_clean_train.provenance.json'}
    for prefix in ('carrier', 'teacher_trace'):
        proof = read_json(source/prefix/'complete.json')
        members = proof['files'] if prefix == 'teacher_trace' else proof
        names.update(f'{prefix}/{name}' for name in members)
    image_root = source/'images/teacher_clean_train'
    generation = read_json(image_root/'generation.json')
    if (generation.get('model_hashes') != identity or generation.get('prompts_sha256') != sha256(source/'prompts.json')
            or generation.get('steps') != 50 or generation.get('split') != 'train' or not generation.get('native')):
        raise RuntimeError('Clean generation source identity mismatch')
    image_manifest = read_json(image_root/'image_hashes.json')
    if len(image_manifest) != len(scores):
        raise ValueError('Incomplete clean image manifest')
    for image_id, entry in enumerate(image_manifest):
        if entry['file_name'] != image_name(image_id) or sha256(_relative(image_root, entry['file_name'])) != entry['sha256']:
            raise RuntimeError('Clean image manifest changed')
        names.add('images/teacher_clean_train/'+entry['file_name'])
    for shard_name in set(reader.index.values()):
        shard = read_json(_relative(source/'teacher_trace', shard_name))
        if shard['protocol_sha256'] != sha256(source/'teacher_trace/protocol.json'):
            raise RuntimeError('Trace shard protocol mismatch')
        tensor_path = _relative(source/'teacher_trace', shard_name.replace('.json', '.safetensors'))
        if shard['tensor_sha256'] != sha256(tensor_path): raise RuntimeError('Trace shard tensor mismatch')
        for key, record in shard['records'].items():
            if record['metadata']['png_sha256'] != image_manifest[int(key)]['sha256']:
                raise RuntimeError('Trace/clean PNG identity mismatch')
    names.update(p.relative_to(source).as_posix() for p in image_root.glob('*.json'))
    replay = source/'teacher_trace/replay.json'
    if replay.exists():
        proof = read_json(replay)
        expected = {'trace_sha256': sha256(source/'teacher_trace/complete.json'),
                    'clean_images_sha256': sha256(image_root/'image_hashes.json')}
        if not proof.get('passed') or any(proof.get('identity', {}).get(k) != v for k, v in expected.items()):
            raise RuntimeError('Source replay proof invalid')
        names.add('teacher_trace/replay.json')
    manifest = {'source_run': str(source), 'files': {n: sha256(_relative(source, n)) for n in sorted(names)},
                'external_files': {str(model/n): file_hash for n, file_hash in identity.items()}}
    verify_source_manifest(manifest)
    return dict(config=config, assets=assets, prompts=prompts, reader=reader, carrier=carrier, scores=scores, manifest=manifest)


def _finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite oracle metadata')
    if isinstance(value, dict):
        for item in value.values(): _finite_json(item)
    elif isinstance(value, (list, tuple)):
        for item in value: _finite_json(item)


def _atomic_json(path, value):
    _finite_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name+'.', suffix='.tmp', delete=False, encoding='utf-8') as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def _freeze(path, value):
    if path.exists():
        if read_json(path) != value: raise RuntimeError(f'Frozen oracle artifact differs: {path.name}')
    else: _atomic_json(path, value)


def _tensor_file(path, tensors):
    from safetensors.torch import save_file
    for value in tensors.values():
        if not torch.isfinite(value).all(): raise ValueError('Nonfinite oracle target')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.safetensors.tmp')
    save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, str(temporary))
    os.replace(temporary, path)


class OracleStore:
    def __init__(self, output, config, source_manifest, sample_ids, protocol):
        self.output = Path(output).resolve()
        reject_overlap(self.output, source_manifest['source_run'])
        verify_source_manifest(source_manifest)
        self.config, self.manifest, self.ids, self.protocol = config, source_manifest, list(sample_ids), protocol
        if self.ids != sorted(set(self.ids)) or not self.ids or any(type(i) is not int or i < 0 for i in self.ids):
            raise ValueError('Oracle IDs must be sorted unique nonnegative integers')
        identity = {'config.json': config, 'source_manifest.json': source_manifest,
                    'oracle_sample_ids.json': self.ids, 'oracle_targets/protocol.json': protocol,
                    'oracle_targets/sample_ids.json': self.ids}
        if self.output.exists() and any(self.output.iterdir()):
            if not all((self.output/n).is_file() for n in identity):
                # Atomic individual identity writes may be interrupted before the
                # complete set exists. Recognize only exact owned identities and
                # their abandoned atomic-write temporary files at this stage.
                owned_dirs = {str(Path(n).parent).replace('\\', '/') for n in identity}
                for path in self.output.rglob('*'):
                    name = path.relative_to(self.output).as_posix()
                    if path.is_symlink(): raise ValueError('Symlink in partial oracle initialization')
                    if path.is_dir():
                        if name not in owned_dirs: raise ValueError('Unrelated oracle initialization directory')
                    elif name in identity:
                        if read_json(path) != identity[name]: raise RuntimeError('Partial oracle identity differs')
                    elif not any(path.parent == (self.output/n).parent and
                                 path.name.startswith(Path(n).name+'.') and path.name.endswith('.tmp') for n in identity):
                        raise ValueError('Existing output is not a recognized oracle run')
        for name, value in identity.items(): _freeze(self.output/name, value)
        (self.output/'oracle_images').mkdir(exist_ok=True)

    def _folder(self, index):
        if type(index) is not int or index < 0: raise ValueError('Invalid batch index')
        return self.output/'batches'/f'batch_{index:05d}'

    def _verify_proof(self, folder, proof):
        for name, digest in proof['files'].items():
            path = _relative(self.output, name)
            if not path.is_file() or sha256(path) != digest:
                raise RuntimeError(f'Committed oracle artifact changed: {name}')
            if path.suffix == '.png':
                with Image.open(path) as image: image.verify()

    def has_batch(self, index, ids):
        folder = self._folder(index)
        if not (folder/'complete.json').exists(): return False
        proof = read_json(folder/'complete.json')
        if proof.get('ids') != list(ids): raise RuntimeError('Committed batch IDs differ')
        expected_files = {p.relative_to(self.output).as_posix() for p in
                          [folder/n for n in ('targets.safetensors', 'metrics.json', 'frontier.json')] +
                          [self.output/'oracle_images'/image_name(i) for i in ids]}
        if set(proof.get('files', {})) != expected_files: raise RuntimeError('Incomplete batch artifact proof')
        self._verify_proof(folder, proof)
        rows = read_json(folder/'metrics.json')
        if [r['image_id'] for r in rows] != list(ids): raise RuntimeError('Committed rows differ')
        self._validate_tensors(ids, self.load_tensors(index))
        return True

    def load_batch(self, index):
        folder = self._folder(index)
        proof = read_json(folder/'complete.json')
        self.has_batch(index, proof['ids'])
        return {'ids': proof['ids'], 'rows': read_json(folder/'metrics.json'), 'frontier': read_json(folder/'frontier.json')}

    def load_tensors(self, index):
        from safetensors.torch import load_file
        return load_file(str(self._folder(index)/'targets.safetensors'), device='cpu')

    @staticmethod
    def _validate_tensors(ids, tensors):
        if set(tensors) != {f'{prefix}_{i}' for i in ids for prefix in ('delta', 'final_latent')}:
            raise ValueError('Incomplete oracle targets')
        for i in ids:
            delta, latent = tensors[f'delta_{i}'], tensors[f'final_latent_{i}']
            if delta.ndim != 4 or delta.shape[0] != 4 or latent.shape != (1, *delta.shape[1:]):
                raise ValueError('Invalid oracle tensor shape')
            if delta.dtype not in (torch.float16, torch.bfloat16) or latent.dtype != torch.float32:
                raise ValueError('Oracle targets require half residuals and exact FP32 final latents')
            if not torch.isfinite(delta).all() or not torch.isfinite(latent).all():
                raise ValueError('Nonfinite oracle targets')

    def commit_batch(self, index, ids, tensors, rows, frontier):
        ids = list(ids)
        if self.has_batch(index, ids): return
        if (self.output/'oracle_targets/complete.json').exists():
            raise RuntimeError('Completed oracle output is immutable')
        if not ids or len(set(ids)) != len(ids) or any(i not in self.ids for i in ids):
            raise ValueError('Invalid oracle batch IDs')
        if [r['image_id'] for r in rows] != ids or any(type(r.get('float_feasible')) is not bool for r in rows):
            raise ValueError('Oracle metrics must align with batch IDs')
        _finite_json(rows); _finite_json(frontier)
        self._validate_tensors(ids, tensors)
        pngs = [self.output/'oracle_images'/image_name(i) for i in ids]
        for path in pngs:
            with Image.open(path) as image: image.verify()
        folder = self._folder(index)
        _tensor_file(folder/'targets.safetensors', tensors)
        _atomic_json(folder/'metrics.json', rows)
        _atomic_json(folder/'frontier.json', frontier)
        paths = [folder/name for name in ('targets.safetensors', 'metrics.json', 'frontier.json')] + pngs
        _atomic_json(folder/'complete.json', {'ids': ids, 'files': {p.relative_to(self.output).as_posix(): sha256(p) for p in paths}})

    def export_targets(self):
        verify_source_manifest(self.manifest)
        target = self.output/'oracle_targets'
        complete = target/'complete.json'
        batches = []
        for folder in sorted((self.output/'batches').glob('batch_*')):
            if (folder/'complete.json').exists(): batches.append(self.load_batch(int(folder.name.split('_')[1])))
        covered = [i for b in batches for i in b['ids']]
        if sorted(covered) != self.ids or len(covered) != len(set(covered)):
            raise RuntimeError('Cannot finalize incomplete oracle sample coverage')
        if complete.exists():
            proof = read_json(complete)
            self._verify_proof(target, proof)
            if proof.get('sample_ids') != self.ids: raise RuntimeError('Completed oracle sample IDs differ')
            return proof
        selected = []
        for folder in sorted((self.output/'batches').glob('batch_*')):
            batch = self.load_batch(int(folder.name.split('_')[1])); tensors = self.load_tensors(int(folder.name.split('_')[1]))
            for row in batch['rows']:
                if row['float_feasible']: selected.append((row, tensors))
        size = self.config.get('target_shard_size', 32)
        if type(size) is not int or size <= 0: raise ValueError('Invalid target shard size')
        paths = [target/'protocol.json', target/'sample_ids.json']
        hashes = self.manifest['files']
        for start in range(0, len(selected), size):
            chunk = selected[start:start+size]; stem = target/f'shard_{start//size:05d}'
            tensors = {f'{prefix}_{row["image_id"]}': values[f'{prefix}_{row["image_id"]}'] for row, values in chunk for prefix in ('delta', 'final_latent')}
            metadata = {'sample_ids': [r['image_id'] for r, _ in chunk], 'rows': [r for r, _ in chunk],
                        'source_trace_sha256': hashes.get('teacher_trace/complete.json'), 'source_carrier_sha256': hashes.get('carrier/complete.json')}
            tensor_path, metadata_path = stem.with_suffix('.safetensors'), stem.with_suffix('.json')
            if metadata_path.exists():
                prior = read_json(metadata_path)
                if prior.get('tensor_sha256') != sha256(tensor_path):
                    raise RuntimeError('Committed oracle target shard changed')
                metadata['tensor_sha256'] = sha256(tensor_path)
                if prior != metadata: raise RuntimeError('Committed oracle target metadata differs')
                from safetensors.torch import load_file
                previous = load_file(str(tensor_path), device='cpu')
                if set(previous) != set(tensors) or any(not torch.equal(previous[k], v.cpu()) for k, v in tensors.items()):
                    raise RuntimeError('Committed oracle target tensors differ')
            else:
                _tensor_file(tensor_path, tensors)
                metadata['tensor_sha256'] = sha256(tensor_path)
                _atomic_json(metadata_path, metadata)
            paths.extend([stem.with_suffix('.safetensors'), stem.with_suffix('.json')])
        image_manifest = [{'image_id': i, 'file_name': image_name(i), 'sha256': sha256(self.output/'oracle_images'/image_name(i))} for i in self.ids]
        image_path = self.output/'oracle_images/image_hashes.json'; _freeze(image_path, image_manifest); paths.append(image_path)
        paths.extend(folder/'complete.json' for folder in sorted((self.output/'batches').glob('batch_*')))
        proof = {'sample_ids': self.ids, 'float_feasible_ids': [r['image_id'] for r, _ in selected],
                 'files': {p.relative_to(self.output).as_posix(): sha256(p) for p in paths}}
        _atomic_json(complete, proof)
        return proof
