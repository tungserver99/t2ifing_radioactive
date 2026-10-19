"""Small, dependency-free integrity helpers shared by the experiment stages."""
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

# All pipeline models use PyTorch. Set these before Transformers is imported,
# including when running a stage directly in a Colab subprocess.
os.environ.update(USE_TORCH="1", USE_TF="0", USE_FLAX="0", FORCE_TF_AVAILABLE="0")

HERE = Path(__file__).resolve().parent


def is_colab():
    return (bool(os.environ.get('COLAB_RELEASE_TAG') or os.environ.get('COLAB_BACKEND_VERSION'))
            or str(HERE).startswith('/content/'))


def runtime_root():
    """Local POSIX storage for environments and model caches, not experiment results."""
    return Path(os.environ.get('BITMARK_RUNTIME_ROOT',
                              '/content/bitmark_runtime' if is_colab() else str(HERE))).expanduser().resolve()


def results_root(method=None):
    """Persist runs beside the repository's launchers (including mounted Drive)."""
    default = HERE.parent.parent if method == 'semantic_fp' else HERE.parent
    return Path(os.environ.get('BITMARK_RESULTS_ROOT', str(default))).expanduser().resolve()


def require_local_runtime_path(path):
    resolved = Path(path).expanduser().resolve()
    drive = Path('/content/drive').resolve()
    if is_colab() and (resolved == drive or drive in resolved.parents):
        raise ValueError(f'Use local /content storage for runtime files, not Drive: {resolved}. '
                         'Copy completed results to Drive for backup.')
    return resolved


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".writing-")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_vendor():
    root = HERE / "vendor"
    for name, expected in read_json(root / "provenance.json")["files"].items():
        if sha256(root / name) != expected:
            raise RuntimeError(f"Vendored upstream source changed: {name}")


def image_name(i):
    return f"sample_{i:06d}.png"


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise RuntimeError(f"Resume settings/integrity mismatch: {path}")
    else:
        write_json(path, value)


def finite(values):
    if not values or not all(math.isfinite(float(x)) for x in values):
        raise ValueError("Empty or nonfinite detector scores; refusing misleading metrics")

class windows_fcntl:
    """Small advisory-lock compatibility for CPU protocol tests on Windows."""
    LOCK_EX = 1
    LOCK_NB = 2

    @staticmethod
    def flock(file, flags):
        import msvcrt
        file.seek(0)
        msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK if flags & windows_fcntl.LOCK_NB else msvcrt.LK_LOCK, 1)
