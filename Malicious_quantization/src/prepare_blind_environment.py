"""Bootstrap blind-run dependencies using Conda or the explicitly selected current Python."""
import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys


def missing_requirements(path):
    problems = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "--")):
            continue
        package, expected = line.split("==", 1)
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError:
            actual = "missing"
        if (actual if "+" in expected else actual.split("+")[0]) != expected:
            problems.append(f"{package}: {actual} -> {expected}")
    return problems


def install_dependencies(requirements):
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        raise RuntimeError("The pinned CUDA wheels require Python 3.10-3.12; use Python 3.11 or Conda mode")
    requirements = Path(requirements).resolve()
    if not requirements.is_file():
        raise FileNotFoundError(f"Missing {requirements}; copy the complete repository")
    print(f"Environment Python: {sys.executable}\nEnvironment prefix: {sys.prefix}", flush=True)
    problems = missing_requirements(requirements)
    if problems:
        print("Installing pinned dependencies:\n  " + "\n  ".join(problems), flush=True)
        pip = [sys.executable, "-m", "pip"]
        if subprocess.run(pip + ["--version"], stdout=subprocess.DEVNULL).returncode:
            subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"], check=True)
        # Same interpreter throughout. No sudo, --user fallback or system-protection bypass.
        subprocess.run(pip + ["install", "--no-user", "-r", str(requirements)], check=True)
        problems = missing_requirements(requirements)
        if problems:
            raise RuntimeError("Packages still mismatch after installation: " + "; ".join(problems))
    else:
        print("Pinned dependencies already installed; skipping downloads.", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    # Fresh process is important after replacing packages; this does not download models.
    subprocess.run([sys.executable, "-c", "\n".join([
        "import torch, torchvision, diffusers, transformers, safetensors, PIL, scipy, numpy, lpips, peft, accelerate, ftfy, cleanfid, gdown",
        "from diffusers import StableDiffusionPipeline, AutoencoderKL, DDIMScheduler",
        "from transformers import CLIPTextModel, CLIPTokenizer",
        "torchvision.ops.nms(torch.tensor([[0.,0.,1.,1.]]), torch.tensor([1.]), .5)",
        "print('Dependency imports and TorchVision binary check passed.', flush=True)",
    ])], check=True)


def conda_prefix(conda, name):
    if not name or name == "base" or any(c in name for c in "/\\"):
        raise ValueError("Choose a dedicated Conda environment name (default wmq), not base or a path")
    result = subprocess.run([conda, "env", "list", "--json"], check=True, capture_output=True, text=True)
    return next((p for p in json.loads(result.stdout)["envs"] if Path(p).name == name), None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--conda")
    p.add_argument("--env-name", default="wmq")
    p.add_argument("--requirements", required=True)
    p.add_argument("--launcher")
    p.add_argument("--install-only", action="store_true")
    p.add_argument("launcher_args", nargs=argparse.REMAINDER)
    args = p.parse_args()
    forwarded = args.launcher_args[1:] if args.launcher_args[:1] == ["--"] else args.launcher_args
    if args.conda:
        if conda_prefix(args.conda, args.env_name) is None:
            print(f"Creating Conda environment {args.env_name} with Python 3.11...", flush=True)
            subprocess.run([args.conda, "create", "--yes", "--name", args.env_name, "python=3.11", "pip"], check=True)
        # Always select the requested environment, even when another one is active.
        cmd = [args.conda, "run", "--no-capture-output", "-n", args.env_name,
               "python", str(Path(__file__).resolve()), "--requirements", args.requirements]
        if args.install_only:
            cmd.append("--install-only")
        else:
            cmd.extend(["--launcher", args.launcher, "--", *forwarded])
        raise SystemExit(subprocess.run(cmd, check=False).returncode)
    install_dependencies(args.requirements)
    if args.install_only:
        return
    if not args.launcher:
        raise ValueError("--launcher is required unless using --install-only")
    environment = os.environ.copy()
    environment.update(WMQ_BOOTSTRAP_READY="1", WMQ_PYTHON=os.path.abspath(sys.executable),
                       WMQ_EXPECTED_PREFIX=os.path.abspath(sys.prefix))
    raise SystemExit(subprocess.run(["bash", args.launcher, *forwarded], env=environment, check=False).returncode)


if __name__ == "__main__":
    main()
