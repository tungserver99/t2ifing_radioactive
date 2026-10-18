"""Validate and stage the authors' public SERUM checkpoint and HF cache."""
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path
import torch
import yaml

ROOT = Path('/root/workspace/serum')
ASSETS = ROOT / 'assets/serum_author'
sys.path.insert(0, str(ROOT / 'third_party/SERUM'))

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    from src.utils.config import Config
    from src.models import WatermarkScoreModel, Watermark
    ckpt_path = ASSETS / 'latest_checkpoint.pt'
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    print('checkpoint keys:', list(ckpt), 'epoch:', ckpt.get('epoch'), flush=True)
    cfg = yaml.safe_load((ROOT/'third_party/SERUM/configs/config_sd21.yaml').read_text())
    cfg['experiment_dir'] = str(ASSETS/'experiment')
    cfg['diffusion']['device'] = 'cpu'
    cfg['diffusion']['stable_diffusion_model_id'] = str(ASSETS/'sd21_snapshot')
    config_path = ASSETS/'config_sd21_author.yaml'
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    config = Config(str(config_path))
    models = {}
    for name, cls in [('score_model', WatermarkScoreModel), ('watermark', Watermark)]:
        state = ckpt[name+'_state_dict']
        assert all(torch.isfinite(t).all() for t in state.values())
        obj = cls(config)
        obj.load_state_dict(state, strict=True)
        path = ASSETS/(name+'.pt')
        torch.save(state, path)
        models[name] = {'path': str(path), 'sha256': sha(path)}
    assert tuple(ckpt['watermark_state_dict']['grid'].shape) == (1,4,64,64)
    manifest = {'source': 'https://github.com/Hubizon/SERUM/issues/1#issuecomment-4963328577',
        'checkpoint_url': 'https://drive.google.com/file/d/12NTXeyIMFX1NWYzCJDeRX54T7BgEvXuQ/view',
        'cache_url': 'https://drive.google.com/file/d/17MFAZ2rsmaIlRNghzTbShLE73t6QCCrN/view',
        'checkpoint_sha256': sha(ckpt_path), 'checkpoint_epoch': ckpt.get('epoch'),
        'models': models, 'config': str(config_path), 'status': 'checkpoint_verified_cache_pending'}
    (ASSETS/'manifest.json').write_text(json.dumps(manifest, indent=2))
    archive = ASSETS/'sd21_original_cache.zip'
    if not archive.exists():
        print(json.dumps(manifest, indent=2), flush=True)
        return
    cache = ASSETS/'original_cache'
    cache.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            target = (cache/entry.filename).resolve()
            assert target.is_relative_to(cache.resolve()), 'Invalid archive path'
            assert (entry.external_attr >> 16) & 0o170000 != 0o120000, 'Symlink in archive requires review'
        bad = z.testzip()
        assert bad is None, bad
        z.extractall(cache)
    candidates = list(cache.rglob('model_index.json'))
    assert len(candidates) == 1, [str(p) for p in candidates]
    snapshot = candidates[0].parent
    required = ['vae/config.json','unet/config.json','text_encoder/config.json',
                'tokenizer/tokenizer_config.json','scheduler/scheduler_config.json']
    assert all((snapshot/p).is_file() for p in required)
    link = ASSETS/'sd21_snapshot'
    if link.exists():
        assert link.resolve() == snapshot.resolve()
    else:
        link.symlink_to(snapshot, target_is_directory=True)
    files = {str(p.relative_to(snapshot)): {'bytes': p.stat().st_size, 'sha256':sha(p)}
             for p in sorted(snapshot.rglob('*')) if p.is_file()}
    (ASSETS/'cache_file_hashes.json').write_text(json.dumps(files, indent=2))
    cfg['diffusion']['device'] = 'cuda'
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    manifest.update(status='verified', cache_zip_sha256=sha(archive),
        snapshot=str(snapshot), model_file_count=len(files), config_sha256=sha(config_path))
    (ASSETS/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)

if __name__ == '__main__':
    main()
