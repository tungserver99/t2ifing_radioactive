"""Official SleeperMark assets; first-download hashes recorded, verified on reuse."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import urllib.request

REVISION = 'a78789cbac92ff5a63a2ad13885cf3893a3d9c4f'
SOURCE_SHA256 = '2b5d6f24f89e1a3cbb7a78076865632a37677e7084b4dacc25b3f59e0d70c591'
UNET_FOLDER = '1OnpVaXC6r1014oOambHETAPcF3-PILlw'
OWNER_FOLDER = '1q-CQiqhSkYESqgfRAQC43-Z-CXsXqI2F'
PREFLIGHT_ENV = 'WMQ_INTERNAL_SLEEPER_PROVENANCE_SHA256'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def prepared_paths(root):
    root = Path(root).resolve()
    configs = [p for p in (root / 'unet_download').rglob('config.json')
               if json.loads(p.read_text()).get('_class_name') == 'UNet2DConditionModel']
    if len(configs) != 1:
        raise ValueError(f'Expected one official UNet config, found {configs}')
    def unique(name):
        paths = list((root / 'owner').rglob(name))
        if len(paths) != 1:
            raise ValueError(f'Expected one {name}, found {paths}')
        return paths[0]
    return configs[0].parent, unique('decoder.pth'), unique('secret.pt'), root / 'watermarkModel.py'


def prepare(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    marker = root / 'provenance.json'
    # Set only by the parent orchestrator after a full verification in the same
    # launch. The marker token avoids both a lock race and repeated 3.4-GB hashes.
    preflight = os.environ.get(PREFLIGHT_ENV)
    if preflight:
        if not marker.exists() or digest(marker) != preflight:
            raise ValueError('SleeperMark preflight provenance token mismatch')
        return prepared_paths(root)
    import gdown
    # Exclusive lock directory: concurrent launchers cannot publish mixed assets.
    lock = root / '.preparing'
    try:
        lock.mkdir()
    except FileExistsError:
        raise RuntimeError(f'Asset preparation is already running; stale lock after a crash: {lock}')
    try:
        if marker.exists():
            data = json.loads(marker.read_text())
            for relative, expected in data['sha256'].items():
                path = (root / relative).resolve()
                if not path.is_relative_to(root) or digest(path) != expected:
                    raise ValueError(f'SleeperMark asset changed: {relative}')
        else:
            for name, folder_id in [('unet_download', UNET_FOLDER), ('owner', OWNER_FOLDER)]:
                destination = root / name
                destination.mkdir(exist_ok=True)
                result = gdown.download_folder(id=folder_id, output=str(destination), quiet=False)
                if not result:
                    raise RuntimeError('Official Google Drive download failed (permissions/quota/network); retry later')
            source = root / 'watermarkModel.py'
            url = f'https://raw.githubusercontent.com/taco-group/SleeperMark/{REVISION}/Stage2/watermarkModel.py'
            with urllib.request.urlopen(url, timeout=60) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != SOURCE_SHA256:
                raise ValueError('Official extractor source checksum mismatch')
            source.write_bytes(content)
            data = {'repository': 'https://github.com/taco-group/SleeperMark', 'revision': REVISION,
                    'unet_google_drive_folder': UNET_FOLDER, 'owner_google_drive_folder': OWNER_FOLDER,
                    'hash_policy': 'Source pinned independently. Checkpoint hashes trust-on-first-download, '
                                   'NOT publisher-authenticated; checked on every reuse.',
                    'sha256': {str(p.relative_to(root)): digest(p) for p in root.rglob('*') if p.is_file()}}
            tmp = marker.with_suffix('.tmp')
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(marker)
        return prepared_paths(root)
    finally:
        lock.rmdir()


def load_extractor(source, weights, key_path, device):
    import torch
    if digest(source) != SOURCE_SHA256:
        raise ValueError('Extractor source checksum mismatch')
    spec = importlib.util.spec_from_file_location('official_sleepermark_extractor', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    key = torch.load(key_path, map_location='cpu', weights_only=True).flatten()
    if key.numel() != 48 or not torch.all((key == 0) | (key == 1)):
        raise ValueError('Official SleeperMark checkpoint requires a binary 48-bit key')
    extractor = module.Extractor_forLatent(secret_size=48)
    extractor.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    return extractor.to(device).eval().requires_grad_(False), key.to(device)
