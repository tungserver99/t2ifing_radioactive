"""Create a private Python 3.11 venv, install with pip, then run the experiment.

uv is only a fallback Python downloader/venv creator when the host lacks 3.11
or ensurepip. It is installed with pip into a private directory, never globally.
"""
import fcntl
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from common import is_colab, runtime_root, require_local_runtime_path

HERE = Path(__file__).resolve().parent
UV_VERSION = "0.9.3"


def python_version(executable):
    try:
        p = subprocess.run([str(executable), "-I", "-c",
                            "import sys; print('%s.%s' % sys.version_info[:2])"],
                           check=True, capture_output=True, text=True)
        return p.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def private_uv(cache):
    target = cache / "bootstrap" / ("uv-" + UV_VERSION)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(target), PYTHONNOUSERSITE="1",
               UV_PYTHON_INSTALL_DIR=str(cache / "python"),
               UV_CACHE_DIR=str(cache / "uv"), UV_NO_CONFIG="1")
    # Calling python -m uv can select a preinstalled uv binary via UV_INTERNAL_BIN.
    # Use the wheel's executable directly and remove inherited uv launch hints.
    env.pop('UV_INTERNAL_BIN', None)
    env.pop('UV_INTERNAL_PARENT_INTERPRETER', None)
    binary = target / 'bin/uv'
    probe = (subprocess.run([str(binary), '--version'], env=env, capture_output=True, text=True)
             if binary.is_file() else None)
    if probe is None or probe.returncode or probe.stdout.split()[:2] != ['uv', UV_VERSION]:
        subprocess.run([sys.executable, "-m", "pip", "--isolated", "install", "--no-user",
                        "--upgrade", "--target", str(target), "uv==" + UV_VERSION], check=True)
    command = [str(binary)]
    verified = subprocess.run(command + ['--version'], env=env, check=True, capture_output=True, text=True)
    if verified.stdout.split()[:2] != ['uv', UV_VERSION]:
        raise RuntimeError(f'Unexpected private uv version: {verified.stdout.strip()}')
    print(verified.stdout.strip(), flush=True)
    return command, env


def ensure_environment(venv, cache):
    """Caller holds the environment lock. Existing environments are never deleted."""
    py = venv / "bin/python"
    cfg = venv / "pyvenv.cfg"
    marker = venv / ".bitmark-bootstrap.json"
    if venv.exists() and any(venv.iterdir()) and not cfg.exists() and not marker.exists():
        raise RuntimeError(f"Refusing non-venv directory: {venv}. Choose an empty BITMARK_VENV.")
    if cfg.exists():
        version = python_version(py)
        if version != "3.11":
            raise RuntimeError(f"Existing venv uses {version}, requires 3.11: {venv}. Choose another BITMARK_VENV.")
    else:
        venv.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"python": "3.11", "state": "creating"}))
        candidates = [os.environ.get("BITMARK_PYTHON"), shutil.which("python3.11"), sys.executable]
        compatible = next((p for p in candidates if p and python_version(p) == "3.11"), None)
        if compatible:
            subprocess.run([str(compatible), "-m", "venv", "--without-pip", str(venv)], check=True)
        else:
            print("Python 3.11 unavailable; downloading a private interpreter with uv.", flush=True)
            uv, env = private_uv(cache)
            subprocess.run(uv + ["python", "install", "3.11", "--no-bin"], env=env, check=True)
            subprocess.run(uv + ["venv", "--python", "3.11", "--managed-python",
                                  "--allow-existing", str(venv)], env=env, check=True)
    if subprocess.run([str(py), "-I", "-m", "pip", "--version"], capture_output=True).returncode:
        if subprocess.run([str(py), "-I", "-m", "ensurepip", "--upgrade"]).returncode:
            uv, env = private_uv(cache)
            subprocess.run(uv + ["pip", "install", "--python", str(py), "pip==25.2"], env=env, check=True)
    subprocess.run([str(py), "-I", "-c",
                    "import sys; assert sys.prefix != sys.base_prefix; assert sys.version_info[:2] == (3,11)"], check=True)
    marker.write_text(json.dumps({"python": "3.11", "state": "ready"}))
    return py


def check_colab_gpu():
    """Fail before large downloads on unsupported Colab accelerators."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
                                check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError('Enable a Colab GPU runtime (Ampere or newer) before running BitMark.') from exc
    capabilities = [float(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    if not capabilities or min(capabilities) < 8:
        raise RuntimeError('This BitMark FlashAttention build requires Ampere or newer (e.g. A100/L4). '
                           'Colab T4 is unsupported; select a compatible GPU runtime.')


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("BitMark CUDA/FlashAttention deployment requires Linux x86_64.")
    if sys.version_info < (3, 9):
        raise RuntimeError("Bootstrap requires Python >=3.9 with pip; deployment installs Python 3.11.")
    if "--setup-only" in args and args != ["--setup-only"]:
        raise ValueError("Use --setup-only by itself, then run the experiment with its flags.")
    if args != ["--setup-only"]:
        from run import parse_args
        parsed = parse_args(args)
        for path in [parsed.cache, parsed.output, parsed.resume]:
            if path is not None:
                require_local_runtime_path(path)
        if is_colab():
            check_colab_gpu()
    root = require_local_runtime_path(runtime_root())
    venv = require_local_runtime_path(os.environ.get("BITMARK_VENV", root / ".venv"))
    cache = require_local_runtime_path(os.environ.get("BITMARK_BOOTSTRAP_CACHE", root / "cache/environment"))
    if is_colab():
        # pip/torch/HF also create locks, executables and atomic temporary files.
        for key, directory in {'TMPDIR': root / 'tmp', 'PIP_CACHE_DIR': root / 'cache/pip',
                               'HF_HOME': root / 'cache/huggingface',
                               'TORCH_HOME': root / 'cache/torch',
                               'XDG_CACHE_HOME': root / 'cache/xdg'}.items():
            location = require_local_runtime_path(os.environ.get(key, directory))
            location.mkdir(parents=True, exist_ok=True)
            os.environ[key] = str(location)
        print(f'Colab local runtime: {root}; back up runs before deleting the runtime.', flush=True)
    venv.parent.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / "setup.lock").open("w") as cache_lock, Path(str(venv) + ".lock").open("w") as env_lock:
        fcntl.flock(cache_lock, fcntl.LOCK_EX)
        fcntl.flock(env_lock, fcntl.LOCK_EX)
        py = ensure_environment(venv, cache)
        print(f"Installing/checking BitMark dependencies in {venv}", flush=True)
        subprocess.run([str(py), "-I", str(HERE / "setup_env.py")], check=True)
    if args == ["--setup-only"]:
        return
    os.execv(str(py), [str(py), str(HERE / "run.py"), *args])


if __name__ == "__main__":
    main()
