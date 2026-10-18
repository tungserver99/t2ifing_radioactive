"""Pinned author SERUM checkpoint and diffusion cache, never a substitute model."""
import argparse
try:
    import fcntl
except ImportError:  # CPU plan/tests on Windows; actual server execution remains POSIX.
    from common import windows_fcntl as fcntl
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from assets import hub_snapshot, prepare_prompts
from common import freeze, read_json, sha256, write_json

CHECKPOINT_ID = '12NTXeyIMFX1NWYzCJDeRX54T7BgEvXuQ'
CHECKPOINT_SHA256 = 'e991a694342545fa4f59cc5b3c94b6e3a93bf831dd4d507207f52d1c167c05c4'
CACHE_ID = '17MFAZ2rsmaIlRNghzTbShLE73t6QCCrN'
CACHE_SHA256 = '95877f6ab0305b92897c5f6fdba7fcd373b9031728f3e69599ffebfb6d2bc593'
TEACHER_REV = '5ede9e4bf3e3fd1cb0ef2f7a3fff13ee514fdf06'
STUDENT_REV = '133a221b8aa7292a167afc5127cb63fb5005638b'
AUTHOR_SOURCE = 'https://github.com/Hubizon/SERUM/issues/1#issuecomment-4963328577'


def verified_file(path, expected):
    path = Path(path)
    if sha256(path) != expected:
        raise RuntimeError(f'SHA256 mismatch; refusing to load {path}. Expected {expected}')
    return path


def author_file(local, cache, name, file_id, expected):
    if local:
        return verified_file(local, expected)
    path = cache / name
    if not path.exists():
        # The download endpoint returns an HTML confirmation for some Drive files.
        # Follow its form with cookies, then stream and verify before publishing.
        import http.cookiejar
        import urllib.parse
        import urllib.request
        from html.parser import HTMLParser
        class Form(HTMLParser):
            action = None
            def __init__(self):
                super().__init__(); self.fields = {}
            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if tag == 'form': self.action = attrs.get('action')
                if tag == 'input' and attrs.get('name'): self.fields[attrs['name']] = attrs.get('value', '')
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        url = f'https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t'
        temporary = path.with_suffix(path.suffix + '.partial')
        def open_with_retry(url):
            import time
            import urllib.error
            for attempt in range(3):
                try:
                    return opener.open(url, timeout=90)
                except (urllib.error.URLError, OSError) as exc:
                    if attempt == 2:
                        raise RuntimeError(f'Cannot download author {name}: {exc}. Supply the original file using --serum-checkpoint or --serum-cache-archive; checksum verification remains required.') from exc
                    print(f'Author download connection failed; retry {attempt+2}/3: {exc}', flush=True)
                    time.sleep(2)
        for _ in range(3):
            with open_with_retry(url) as response:
                if 'text/html' in response.headers.get('Content-Type', ''):
                    form = Form(); form.feed(response.read(2_000_000).decode('utf8'))
                    if not form.action or urllib.parse.urlparse(form.action).hostname not in {'drive.google.com', 'drive.usercontent.google.com'}:
                        raise RuntimeError(f'Google Drive denied download of {name}; supply the pinned local file using --serum-checkpoint/--serum-cache-archive. URL: {url}')
                    url = form.action + '?' + urllib.parse.urlencode(form.fields)
                    continue
                import time
                done, last = 0, 0
                total = int(response.headers.get('Content-Length', 0))
                with temporary.open('wb') as out:
                    while block := response.read(4 * 1024 * 1024):
                        out.write(block); done += len(block)
                        if time.monotonic() - last > 5:
                            print(f'Downloading author {name}: {done/1e6:.0f}/{total/1e6:.0f} MB', flush=True)
                            last = time.monotonic()
                if total and total != done: raise IOError(f'Incomplete author download: {done}/{total}')
                verified_file(temporary, expected)
                temporary.replace(path)
                break
        else:
            raise RuntimeError(f'Could not resolve Google Drive download for {name}')
    return verified_file(path, expected)


def safe_extract(archive, destination):
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            p = PurePosixPath(info.filename)
            if p.is_absolute() or '..' in p.parts or '\\' in info.filename or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f'Unsafe archive entry: {info.filename}')
        z.extractall(destination)


def teacher_snapshot(archive, cache):
    target = cache / 'serum_teacher'
    inventory = target / 'verified.json'
    if inventory.exists():
        prior = read_json(inventory)
        if prior['archive_sha256'] != CACHE_SHA256: raise RuntimeError('Untrusted teacher cache provenance')
        for name, digest in prior['files'].items(): verified_file(target / name, digest)
        return str(target / prior['snapshot'])
    if target.exists(): raise RuntimeError(f'Incomplete teacher cache: remove {target} and retry')
    verified_file(archive, CACHE_SHA256)
    with tempfile.TemporaryDirectory(dir=cache, prefix='serum_extract_') as temp:
        tmp = Path(temp)
        safe_extract(archive, tmp)
        snapshots = list(tmp.rglob(f'snapshots/{TEACHER_REV}/model_index.json'))
        if len(snapshots) != 1: raise RuntimeError('Author cache has no unique pinned SD2.1 snapshot')
        snapshot = snapshots[0].parent
        files = {str(p.relative_to(tmp)): sha256(p) for p in tmp.rglob('*') if p.is_file()}
        write_json(tmp / 'verified.json', {'archive_sha256': CACHE_SHA256, 'snapshot': str(snapshot.relative_to(tmp)), 'files': files})
        tmp.rename(target)
    return str(target / read_json(inventory)['snapshot'])


def prepare(config, run):
    cache = Path(config['cache']); cache.mkdir(parents=True, exist_ok=True)
    with (cache / 'assets.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = prepare_prompts(config, run, cache)
        checkpoint = author_file(config.get('serum_checkpoint'), cache, 'serum_author_epoch50.pth', CHECKPOINT_ID, CHECKPOINT_SHA256)
        # Reuse a verified extracted cache without redownloading the archive.
        archive = cache / 'serum_author_cache.zip'
        if not (cache / 'serum_teacher/verified.json').exists():
            archive = author_file(config.get('serum_cache_archive'), cache, archive.name, CACHE_ID, CACHE_SHA256)
        result['teacher'] = teacher_snapshot(archive, cache)
        result['serum_checkpoint'] = str(checkpoint.resolve())
        result['author_source'] = AUTHOR_SOURCE
        result['checkpoint_sha256'] = CHECKPOINT_SHA256
        result['teacher_archive_sha256'] = CACHE_SHA256
        result['student'] = hub_snapshot('CompVis/stable-diffusion-v1-4', STUDENT_REV, cache,
            lambda n: n == 'model_index.json' or (n.split('/')[0] in {
                'scheduler', 'tokenizer', 'text_encoder', 'unet', 'vae', 'feature_extractor'} and
                n.endswith(('.json', '.txt', '.safetensors')) and '.fp16.' not in n and '.non_ema.' not in n))
        freeze(run / 'assets.json', result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True)
    a = p.parse_args(); prepare(read_json(a.run / 'config.json'), a.run)
