"""Install pinned dependencies with pip into the dedicated virtual environment."""
import hashlib
import importlib.metadata as md
import json
import platform
import subprocess
import sys
from pathlib import Path


def main():
    if (sys.version_info[:2] != (3, 11) or platform.machine() != "x86_64"
            or platform.system() != "Linux"):
        raise RuntimeError("Pinned FlashAttention wheel requires Linux x86_64, Python 3.11")
    if sys.prefix == sys.base_prefix or not (Path(sys.prefix) / "pyvenv.cfg").exists():
        raise RuntimeError("Use bash run_bitmark_lora.sh; dependencies must be installed in a venv.")
    req = Path(__file__).with_name("requirements.txt")
    stamp = Path(sys.prefix) / ".bitmark-requirements.json"
    digest = hashlib.sha256(req.read_bytes()).hexdigest()
    expected = {}
    for line in req.read_text().splitlines():
        if "==" in line:
            name, version = line.split("==")
            expected[name] = version
    expected["flash-attn"] = "2.8.3"
    def matched():
        try:
            return all((md.version(k).split('+')[0] if k == 'flash-attn' else md.version(k)) == v
                       for k, v in expected.items())
        except md.PackageNotFoundError:
            return False
    if not stamp.exists() or json.loads(stamp.read_text()).get("sha256") != digest or not matched():
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-user", "-r", str(req)], check=True)
    if not matched():
        raise RuntimeError("Installed versions differ from the pinned requirements")
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    subprocess.run([sys.executable, "-c", "import torch, torchvision, flash_attn, diffusers, transformers, peft, datasets; "
                    "from diffusers import StableDiffusionPipeline; "
                    "from transformers import T5EncoderModel; "
                    "assert torch._C._GLIBCXX_USE_CXX11_ABI; "
                    "torchvision.ops.nms(torch.tensor([[0.,0.,1.,1.]]), torch.tensor([1.]), .5); "
                    "print('BitMark dependencies and binary imports OK')"], check=True)
    pending = stamp.with_suffix(".tmp")
    pending.write_text(json.dumps({"sha256": digest, "python": sys.version}))
    pending.replace(stamp)


if __name__ == "__main__":
    main()
