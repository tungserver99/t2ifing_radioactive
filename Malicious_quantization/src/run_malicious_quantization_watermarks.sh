#!/usr/bin/env bash
set -Eeuo pipefail

# Adaptive PTQ stress test for Stable Signature and AquaLoRA.
# Default: matched W4A16 comparison (RTN, grid, gradient, reconstruction).
# RUN_PROTOCOL=legacy_grid restores the historical 50-candidate search.
# Stable Signature: public Meta decoder + SD2.1-base (public mirror fallback).
# AquaLoRA: official ppft_trained assets + SD1.5.
#
# This uses simulated weight-only PTQ: weights are genuinely rounded to low-bit
# values, then inference runs with CUDA FP16 kernels. This makes the numerical
# experiment portable; it does not claim INT4 kernel speedup.
#
# Quantizer-refinement ablations (all use the same held-out final test set):
#   REFINE_MODE=none       bash "$0"  # fixed/grid PTQ baseline
#   REFINE_MODE=scale      bash "$0"  # optimize per-group scale only
#   REFINE_MODE=scale_zero bash "$0"  # scale + integer zero-point
#   REFINE_MODE=full       bash "$0"  # scale + zero-point + rounding threshold
# REFINE_OPTIMIZER=gradient (default, autograd+STE) or zeroth (random-search ablation).
# Main compute controls: GRAD_STEPS=40 GRAD_EVAL_EVERY=4 REFINE_N=8 CALIB_N=2
# Constraint: MAX_PRED_NMSE=0.02 (timestep-weighted UNet or VAE-decode NMSE).

SCRIPT_DIR="$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export WMQ_CODE_ROOT="$SCRIPT_DIR"
WMQ_ROOT="${WMQ_ROOT:-$PROJECT_ROOT/wmq_runs}"
WMQ_CACHE="${WMQ_CACHE:-$WMQ_ROOT/hf_cache}"
WMQ_CHECK_ONLY="${WMQ_CHECK_ONLY:-0}" # 1: imports + CUDA; imports: login-node audit
case "$WMQ_CHECK_ONLY" in
  0|1|imports) ;;
  *) echo "WMQ_CHECK_ONLY must be 0, 1, or imports" >&2; exit 1 ;;
esac
WMQ_LOG_DIR="${WMQ_LOG_DIR:-$WMQ_ROOT/logs}"
mkdir -p "$WMQ_LOG_DIR"
WMQ_RUN_LOG="$(mktemp "$WMQ_LOG_DIR/wmq_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")"
exec > >(tee -a "$WMQ_RUN_LOG") 2>&1
export PYTHONUNBUFFERED=1
echo "Run log: $WMQ_RUN_LOG"
# Install dependencies on the login node beforehand. Jobs only use this prefix.
if [[ -z "${CONDA_PREFIX:-}" || ! -d "$CONDA_PREFIX/conda-meta" || ! -x "$CONDA_PREFIX/bin/python" ]]; then
  echo "Activate the prepared Conda environment first: conda activate wmq" >&2
  echo "See README.md for login-node installation and job submission instructions." >&2
  exit 1
fi
PY="$CONDA_PREFIX/bin/python"
if [[ -z "${WMQ_OUTPUT:-}" ]]; then
  mkdir -p "$WMQ_ROOT/output"
  WMQ_OUTPUT="$(mktemp -d "$WMQ_ROOT/output/run_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
fi
echo "Results: $WMQ_OUTPUT"
export PATH="$CONDA_PREFIX/bin:$PATH"
export WMQ_ROOT WMQ_OUTPUT WMQ_CACHE WMQ_CHECK_ONLY
export HF_HOME="$WMQ_CACHE"
export TORCH_HOME="${TORCH_HOME:-$WMQ_ROOT/torch_cache}"
export TOKENIZERS_PARALLELISM=false
export USE_TORCH=1 USE_TF=0 USE_FLAX=0 PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE VIRTUAL_ENV
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"

"$PY" - <<'PY'
import os
import sys
from pathlib import Path
if (Path(sys.prefix).resolve() != Path(os.environ["CONDA_PREFIX"]).resolve()
        or sys.prefix != sys.base_prefix):
    sys.exit("Python does not belong to the active Conda environment. Activate wmq again.")
if not (3, 10) <= sys.version_info[:2] <= (3, 12):
    sys.exit("Use a Conda environment with Python 3.10-3.12 (recommended: 3.11).")
print(f"Using prepared Conda environment: {sys.prefix}", flush=True)
PY

if [[ ! -r "$SCRIPT_DIR/requirements-wmq.txt" ]]; then
  echo "Missing requirements-wmq.txt next to the script; copy the complete repo." >&2
  exit 1
fi
WMQ_REQUIRED_PACKAGES="$(cat "$SCRIPT_DIR/requirements-wmq.txt")"
export WMQ_REQUIRED_PACKAGES
mkdir -p "$WMQ_ROOT" "$WMQ_OUTPUT" "$WMQ_CACHE"

"$PY" -m pip check
"$PY" - <<'PY'
import importlib
import os
import sys
import traceback
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

print(f"Python: {sys.executable}", flush=True)
sys.path.insert(0, os.environ["WMQ_CODE_ROOT"])
problems = []
for requirement in os.environ["WMQ_REQUIRED_PACKAGES"].splitlines():
    requirement = requirement.strip()
    if not requirement or requirement.startswith(("#", "--")):
        continue
    package, expected = requirement.split("==")
    try:
        actual = version(package)
    except PackageNotFoundError:
        actual = "not installed"
    print(f"{package}: {actual} (required: {expected})", flush=True)
    if (actual if "+" in expected else actual.split("+")[0]) != expected:
        problems.append(f"{package}: expected {expected}, found {actual}")
if problems:
    sys.exit("Dependency mismatch:\n  " + "\n  ".join(problems)
             + "\nInstall requirements-wmq.txt in the active Conda environment on the login node; see README.md.")

# Exercise lazy imports as well as ordinary module imports. No model downloads.
checks = {
    "numpy": [], "torch": [], "torch.nn": [], "torch.nn.functional": [],
    "torch.func": ["functional_call"], "torchvision.transforms": ["Normalize"],
    "torchvision.models": ["efficientnet_b1"],
    "diffusers": ["DDIMScheduler", "StableDiffusionPipeline"],
    "diffusers.loaders.single_file_utils": ["convert_ldm_vae_checkpoint"],
    "transformers": ["CLIPTextModel", "CLIPTokenizer", "CLIPImageProcessor"],
    "transformers.utils": ["FLAX_WEIGHTS_NAME"],
    "huggingface_hub": ["hf_hub_download"], "PIL.Image": [],
    "safetensors.torch": ["load_file"], "scipy.stats": ["binom"],
    "peft": [], "accelerate": [], "ftfy": [], "lpips": ["LPIPS"],
    "wmq_baselines": ["reconstruct", "AdaptiveRounding"],
    "wmq_diagnostics": ["quality_warning"],
}
for module, names in checks.items():
    try:
        loaded = importlib.import_module(module)
        for name in names:
            getattr(loaded, name)
        print(f"Import OK: {module}", flush=True)
    except Exception:
        problems.append(module)
        traceback.print_exc()
if problems:
    sys.exit("Import failures: " + ", ".join(problems)
             + "\nRepair the Conda environment on the login node using README.md.")

import torch
import torchvision
# Catch a mismatched torch/torchvision binary pair, even in the CPU audit.
torchvision.ops.nms(torch.tensor([[0., 0., 1., 1.]]), torch.tensor([1.]), 0.5)
if os.environ["WMQ_CHECK_ONLY"] != "imports":
    if not torch.cuda.is_available():
        sys.exit("CUDA unavailable. Run on an allocated NVIDIA GPU node; check nvidia-smi, "
                 "CUDA_VISIBLE_DEVICES and the driver. Imports passed.")
    try:
        capability = torch.cuda.get_device_capability(0)
        cuda = tuple(map(int, (torch.version.cuda or "0.0").split(".")[:2]))
        if capability >= (10, 0) and cuda < (12, 8):
            raise RuntimeError("Blackwell needs the PyTorch CUDA 12.8 wheel; reinstall on the login node using README.md")
        x = torch.randn(1, 4, 16, 16, device="cuda", dtype=torch.float16, requires_grad=True)
        conv = torch.nn.Conv2d(4, 4, 3, padding=1).to(device="cuda", dtype=torch.float16)
        conv(x).float().square().mean().backward()
        q = torch.randn(1, 2, 8, 16, device="cuda", dtype=torch.float16, requires_grad=True)
        torch.nn.functional.scaled_dot_product_attention(q, q, q).float().sum().backward()
        torch.cuda.synchronize()
        print(f"CUDA FP16 forward/backward OK: {torch.cuda.get_device_name(0)}; "
              f"torch={torch.__version__}, CUDA={torch.version.cuda}", flush=True)
    except Exception as error:
        sys.exit(f"CUDA smoke test failed: {error}\nCheck driver/GPU support for the selected PyTorch wheel.")
else:
    print("Import audit only: CUDA and model loading were not tested.", flush=True)

# Record actual transitive versions for reproducing this machine's environment.
from importlib.metadata import distributions
snapshot = sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions() if d.metadata['Name'])
Path(os.environ["WMQ_OUTPUT"], "environment.freeze.txt").write_text("\n".join(snapshot) + "\n")
PY

if [[ "$WMQ_CHECK_ONLY" != "0" ]]; then
  echo "Environment checks passed."
  exit 0
fi

"$PY" - <<'PY'
import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import time
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as TV
import diffusers
from torch.func import functional_call
from diffusers import DDIMScheduler, StableDiffusionPipeline
from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file
from scipy.stats import binom

sys.path.insert(0, os.environ.get("WMQ_CODE_ROOT", str(Path.cwd())))
from wmq_baselines import reconstruct, tensor_tree
from wmq_diagnostics import quality_warning


ROOT = Path(os.environ.get("WMQ_ROOT", Path.cwd() / "wmq_runs")).resolve()
OUT = Path(os.environ.get("WMQ_OUTPUT", ROOT / "output")).resolve()
CACHE = Path(os.environ.get("WMQ_CACHE", ROOT / "hf_cache")).resolve()
DEVICE = torch.device("cuda")
DTYPE = torch.float16
GPU_TOTAL_GB = (torch.cuda.get_device_properties(0).total_memory / 2**30
                if torch.cuda.is_available() else 0.0)
LARGE_MEMORY_GPU = GPU_TOTAL_GB >= 80.0

# The automatic large-memory profile targets RTX PRO 6000 Blackwell 96 GB.
# Every setting remains overridable for portability and reproducibility.
GEN_BATCH_SIZE = int(os.environ.get("GEN_BATCH_SIZE", "16" if LARGE_MEMORY_GPU else "1"))
METRIC_BATCH_SIZE = int(os.environ.get("METRIC_BATCH_SIZE", "32" if LARGE_MEMORY_GPU else "8"))
CALIB_BATCH_SIZE = int(os.environ.get("CALIB_BATCH_SIZE", "2" if LARGE_MEMORY_GPU else "1"))
KEEP_PRISTINE_ON_GPU = os.environ.get(
    "KEEP_PRISTINE_ON_GPU", "1" if LARGE_MEMORY_GPU else "0") == "1"
JOINT_GROUP_GRAD = os.environ.get(
    "JOINT_GROUP_GRAD", "1" if LARGE_MEMORY_GPU else "0") == "1"
USE_ATTENTION_SLICING = os.environ.get(
    "USE_ATTENTION_SLICING", "0" if LARGE_MEMORY_GPU else "1") == "1"

SS_MODEL = os.environ.get("SS_MODEL", "stabilityai/stable-diffusion-2-1-base")
SS_MODEL_FALLBACK = os.environ.get("SS_MODEL_FALLBACK", "sd2-community/stable-diffusion-2-1-base")
AQUA_MODEL = os.environ.get("AQUA_MODEL", "stable-diffusion-v1-5/stable-diffusion-v1-5")
AQUA_REPO = "georgefen/AquaLoRA-Models"
AQUA_REV = os.environ.get("AQUA_REV", "98688b2a1e762339593ee8fe96ed13762f06b732")
AQUA_FOLDER = os.environ.get("AQUA_FOLDER", "ppft_trained")
SS_DECODER_URL = "https://dl.fbaipublicfiles.com/ssl_watermarking/sd2_decoder.pth"
SS_EXTRACTOR_URL = "https://dl.fbaipublicfiles.com/ssl_watermarking/dec_48b_whit.torchscript.pt"
SS_KEY = os.environ.get("SS_KEY", "111010110101000001010111010011010100010000100111")
AQUA_KEY = os.environ.get("AQUA_KEY", "110110010011010110001111010101010101011000100110")

SEARCH_N = int(os.environ.get("SEARCH_N", "20"))
TEST_N = int(os.environ.get("TEST_N", "100"))
STEPS = int(os.environ.get("STEPS", "25"))
GUIDANCE = float(os.environ.get("GUIDANCE", "7.0"))
HEIGHT = int(os.environ.get("HEIGHT", "512"))
WIDTH = int(os.environ.get("WIDTH", "512"))
MIN_PSNR = float(os.environ.get("MIN_PSNR", "25.0"))
MAX_LPIPS = float(os.environ.get("MAX_LPIPS", "0.15"))
FPR = float(os.environ.get("FPR", "0.001"))
BASE_SEED = int(os.environ.get("BASE_SEED", "3407"))
MAX_CANDIDATES = int(os.environ.get("MAX_CANDIDATES", "0"))  # 0 = evaluate every recipe
PROMPT_FILE = os.environ.get("PROMPT_FILE", "")
# Detector-guided quantizer refinement. REFINE_MODE supports paper ablations:
# none (grid only), scale, scale_zero, or full (scale + zero-point + rounding).
REFINE_MODE = os.environ.get("REFINE_MODE", "full").lower()
REFINE_OPTIMIZER = os.environ.get("REFINE_OPTIMIZER", "gradient").lower()
REFINE_ITERS = int(os.environ.get("REFINE_ITERS", "24"))
REFINE_N = int(os.environ.get("REFINE_N", "8"))
CALIB_N = int(os.environ.get("CALIB_N", "2"))
CALIB_TIMESTEPS = int(os.environ.get("CALIB_TIMESTEPS", "5"))
MAX_PRED_NMSE = float(os.environ.get("MAX_PRED_NMSE", "0.02"))
TPR_LOSS_WEIGHT = float(os.environ.get("TPR_LOSS_WEIGHT", "0.25"))
PRED_LOSS_WEIGHT = float(os.environ.get("PRED_LOSS_WEIGHT", "0.10"))
REFINE_SEED = int(os.environ.get("REFINE_SEED", "2026"))
GRAD_STEPS = int(os.environ.get("GRAD_STEPS", "40"))
GRAD_LR = float(os.environ.get("GRAD_LR", "0.03"))
GRAD_EVAL_EVERY = int(os.environ.get("GRAD_EVAL_EVERY", "4"))
ROUND_TEMPERATURE = float(os.environ.get("ROUND_TEMPERATURE", "0.10"))
GRAD_PRED_WEIGHT = float(os.environ.get("GRAD_PRED_WEIGHT", "1.0"))
GRAD_IMAGE_WEIGHT = float(os.environ.get("GRAD_IMAGE_WEIGHT", "0.10"))
RUN_PROTOCOL = os.environ.get("RUN_PROTOCOL", "fair_w4a16")
RECON_STEPS = int(os.environ.get("RECON_STEPS", "200"))
RECON_LR = float(os.environ.get("RECON_LR", "0.001"))
RECON_CACHE_MB = int(os.environ.get("RECON_CACHE_MB", "512"))
CLEAN_MIN_BITACC = float(os.environ.get("CLEAN_MIN_BITACC", "0.80"))
CLEAN_MIN_TPR = float(os.environ.get("CLEAN_MIN_TPR", "0.90"))
CLEAN_POLICY = os.environ.get("CLEAN_POLICY", "report")
BASELINE_CHECK_ONLY = os.environ.get("BASELINE_CHECK_ONLY", "0") == "1"

PROMPTS = [
    "a professional photograph of a red fox in a snowy forest",
    "a small sailboat at sunset, realistic photography",
    "an old stone bridge over a quiet river",
    "a bowl of fresh fruit on a wooden table, studio light",
    "a futuristic city street at night, cinematic photograph",
    "a watercolor painting of mountains and a lake",
    "a close-up portrait of a tabby cat",
    "a green bicycle leaning against a brick wall",
    "a lighthouse above a stormy sea",
    "a cozy reading room with warm afternoon light",
    "a field of sunflowers beneath a blue sky",
    "an astronaut walking through a tropical jungle",
    "a ceramic teapot beside two cups",
    "a macro photograph of a butterfly on a flower",
    "a train crossing a valley in autumn",
    "a detailed illustration of a medieval market",
    "a modern glass building reflected in a pond",
    "a dog running on a beach, action photograph",
    "a quiet street in a European village",
    "an aerial photograph of winding coastal roads",
    "a robot preparing dinner in a home kitchen",
    "a misty pine forest at dawn",
    "a colorful hot air balloon above farmland",
    "an antique camera and notebooks on a desk",
]


def experiment_prompts(required):
    """Load one prompt per line/JSON item, or deterministically build a stress set.

    For paper results, set PROMPT_FILE to a preregistered COCO/DrawBench prompt
    list. The built-in expansion is intended for executable preliminary tests.
    """
    if PROMPT_FILE:
        path = Path(PROMPT_FILE)
        if not path.is_file():
            fail(f"PROMPT_FILE does not exist: {path}")
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                for key in ("prompts", "captions", "data"):
                    if key in payload:
                        payload = payload[key]
                        break
            if not isinstance(payload, list):
                fail("PROMPT_FILE JSON must contain a list or prompts/captions/data list")
            loaded = []
            for item in payload:
                if isinstance(item, str):
                    loaded.append(item.strip())
                elif isinstance(item, dict):
                    value = item.get("prompt", item.get("caption", item.get("text", "")))
                    loaded.append(str(value).strip())
        else:
            loaded = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        loaded = [prompt for prompt in loaded if prompt]
        if len(loaded) < required:
            fail(f"PROMPT_FILE has {len(loaded)} prompts; {required} are required")
        return loaded[:required]

    subjects = [
        "a red fox", "a tabby cat", "a lighthouse", "an astronaut", "a steam train",
        "a glass building", "a ceramic teapot", "a sailboat", "a green bicycle",
        "a medieval market", "a butterfly", "a reading room", "a mountain village",
        "a friendly robot", "a bowl of fruit", "a stone bridge", "a field of flowers",
        "a coastal road", "a golden retriever", "an antique camera",
    ]
    settings = [
        "at dawn", "at sunset", "in soft studio light", "during a rainstorm",
        "beneath a clear blue sky", "in winter", "in warm afternoon light",
        "at night", "surrounded by mist", "beside a quiet river",
    ]
    styles = [
        "realistic photograph", "cinematic photograph", "detailed digital illustration",
        "watercolor painting", "documentary photograph",
    ]
    expanded = list(PROMPTS)
    for style in styles:
        for setting in settings:
            for subject in subjects:
                expanded.append(f"{style} of {subject} {setting}")
                if len(expanded) >= required:
                    return expanded
    fail(f"Unable to build {required} prompts")


def fail(message):
    raise RuntimeError(message)


def download_url(url, destination):
    import urllib.request
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        print(f"Downloading {url} -> {destination}", flush=True)
        tmp = destination.with_suffix(destination.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(destination)
    return destination


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def build_pipeline(model_id, fallback_id=None):
    kwargs = dict(torch_dtype=DTYPE, safety_checker=None, requires_safety_checker=False,
                  cache_dir=str(CACHE), use_safetensors=True)
    candidates = [model_id] + ([fallback_id] if fallback_id and fallback_id != model_id else [])
    last_error = None
    pipe = None
    used_model = None
    for candidate in candidates:
        print(f"Loading {candidate}", flush=True)
        try:
            pipe = StableDiffusionPipeline.from_pretrained(candidate, **kwargs)
            used_model = candidate
            break
        except OSError as error:
            last_error = error
            if candidate != candidates[-1]:
                print(f"Could not load {candidate}; trying the declared public mirror", flush=True)
    if pipe is None:
        raise last_error
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    if USE_ATTENTION_SLICING:
        pipe.enable_attention_slicing("max")
    else:
        pipe.disable_attention_slicing()
    pipe.to(DEVICE)
    pipe._wmq_model_id = used_model
    return pipe


@torch.inference_mode()
def generate(pipe, prompts, seeds, batch_size=None):
    batch_size = min(batch_size or GEN_BATCH_SIZE, len(prompts))
    images = []
    start = 0
    while start < len(prompts):
        current = min(batch_size, len(prompts) - start)
        prompt_batch = prompts[start: start + current]
        seed_batch = seeds[start: start + current]
        generators = [torch.Generator(device=DEVICE).manual_seed(seed) for seed in seed_batch]
        try:
            batch = pipe(prompt_batch, height=HEIGHT, width=WIDTH,
                         num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                         generator=generators).images
        except torch.cuda.OutOfMemoryError:
            del generators
            cleanup()
            if current == 1:
                raise
            batch_size = max(1, current // 2)
            print(f"CUDA OOM during generation; retrying with batch_size={batch_size}", flush=True)
            continue
        images.extend(batch)
        del batch, generators
        start += current
    return images


def pil_tensor(images, normalized=False):
    tensors = []
    for image in images:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tensors.append(tensor)
    result = torch.stack(tensors).to(DEVICE)
    return result * 2 - 1 if normalized else result


def image_metrics(clean, attacked, lpips_model):
    psnr_sum = 0.0
    lpips_sum = 0.0
    count = 0
    metric_batch = min(METRIC_BATCH_SIZE, len(clean))
    start = 0
    with torch.inference_mode():
        while start < len(clean):
            current = min(metric_batch, len(clean) - start)
            x = y = None
            try:
                x = pil_tensor(clean[start: start + current])
                y = pil_tensor(attacked[start: start + current])
                mse_per = ((x - y) ** 2).flatten(1).mean(1)
                batch_psnr = -10.0 * torch.log10(mse_per.clamp_min(1e-12))
                batch_lpips = lpips_model(x * 2 - 1, y * 2 - 1).flatten()
            except torch.cuda.OutOfMemoryError:
                del x, y
                cleanup()
                if current == 1:
                    raise
                metric_batch = max(1, current // 2)
                print(f"CUDA OOM in image metrics; retrying with batch_size={metric_batch}", flush=True)
                continue
            psnr_sum += float(batch_psnr.sum().item())
            lpips_sum += float(batch_lpips.sum().item())
            count += current
            del x, y, batch_psnr, batch_lpips
            start += current
    return {"psnr": psnr_sum / count, "lpips": lpips_sum / count}


def detection_threshold(nbits, fpr):
    # Smallest number of matching bits whose random-match upper tail <= FPR.
    for matches in range(nbits + 1):
        if float(binom.sf(matches - 1, nbits, 0.5)) <= fpr:
            return matches
    return nbits + 1


def summarize_bits(predictions, key, fpr=FPR):
    truth = torch.tensor([int(x) for x in key], dtype=torch.bool)
    matches = (predictions.cpu().bool() == truth[None, :]).sum(1).numpy()
    threshold = detection_threshold(len(key), fpr)
    pvalues = [float(binom.sf(int(m) - 1, len(key), 0.5)) for m in matches]
    bitacc = matches / len(key)
    return {
        "bit_accuracy": float(bitacc.mean()),
        "bit_accuracy_std": float(bitacc.std(ddof=1)) if len(bitacc) > 1 else 0.0,
        "tpr": float((matches >= threshold).mean()),
        "threshold_matches": int(threshold),
        "mean_pvalue": float(np.mean(pvalues)),
        "per_image_bit_accuracy": bitacc.tolist(),
        "per_image_pvalue": pvalues,
    }


@torch.inference_mode()
def decode_stable_signature(extractor, images):
    transform = TV.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    predictions = []
    for start in range(0, len(images), METRIC_BATCH_SIZE):
        logits = extractor(transform(pil_tensor(images[start:start + METRIC_BATCH_SIZE])))
        if not torch.isfinite(logits).all():
            fail("Stable Signature decoder produced nonfinite logits")
        predictions.append((logits > 0).cpu())
    return torch.cat(predictions)


class AquaSecretDecoder(nn.Module):
    def __init__(self, output_size=48):
        super().__init__()
        from torchvision.models import efficientnet_b1
        self.model = efficientnet_b1(weights=None)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features,
                                             output_size * 2)
        self.output_size = output_size

    def forward(self, x):
        x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)
        return self.model(x).view(-1, self.output_size, 2)


@torch.inference_mode()
def decode_aqualora(decoder, images):
    predictions = []
    for start in range(0, len(images), METRIC_BATCH_SIZE):
        # Match the official evaluation preprocessing before tensor conversion.
        resized = [im.convert("RGB").resize((512, 512), Image.Resampling.BICUBIC)
                   for im in images[start:start + METRIC_BATCH_SIZE]]
        logits = decoder(pil_tensor(resized, normalized=True))
        if not torch.isfinite(logits).all():
            fail("AquaLoRA decoder produced nonfinite logits")
        predictions.append(logits.argmax(-1).bool().cpu())
    return torch.cat(predictions)


class MapperNet(nn.Module):
    def __init__(self, input_size=48, output_size=320):
        super().__init__()
        self.input_size = input_size
        self.bit_embeddings = nn.Embedding(input_size, output_size)

    def forward(self, x):
        positions = torch.arange(self.input_size, device=x.device)
        encoded = self.bit_embeddings(positions) * x[:, :, None]
        return encoded.sum(dim=1) / math.sqrt(self.input_size) + 1.0


def conditioned_aqualora(folder, key, scale=1.03):
    base = load_file(str(folder / "pytorch_lora_weights.safetensors"), device="cpu")
    mapper = MapperNet(len(key), 320)
    mapper.load_state_dict(torch.load(folder / "mapper.pt", map_location="cpu",
                                      weights_only=True))
    mapper.eval()
    bits = torch.tensor([[int(x) for x in key]], dtype=torch.float32)
    mapped = mapper(bits)[0]
    result = {}
    for name, tensor in base.items():
        if "unet" not in name:
            continue
        if name.endswith("up.weight"):
            result[name] = tensor
        elif name.endswith("down.weight"):
            if "proj_in" in name or "proj_out" in name:
                result[name] = tensor * mapped[:, None, None, None] * scale
            elif "attn" in name or "ff" in name:
                result[name] = mapped[:, None] * tensor * scale
    return result


def resolve_module(root, path):
    obj = root
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def aqua_target_path(down_key):
    path = down_key[len("unet."): -len(".lora.down.weight")]
    replacements = {
        ".processor.to_q_lora": ".to_q",
        ".processor.to_k_lora": ".to_k",
        ".processor.to_v_lora": ".to_v",
        ".processor.to_out_lora": ".to_out.0",
        ".processor.net.0.proj_lora": ".net.0.proj",
        ".processor.net.2_lora": ".net.2",
    }
    for old, new in replacements.items():
        path = path.replace(old, new)
    return path


def fuse_aqualora(unet, lora):
    fused = 0
    unresolved = []
    for down_key in sorted(k for k in lora if k.endswith(".lora.down.weight")):
        up_key = down_key.replace(".lora.down.weight", ".lora.up.weight")
        if up_key not in lora:
            unresolved.append((down_key, "missing up tensor"))
            continue
        path = aqua_target_path(down_key)
        try:
            module = resolve_module(unet, path)
        except (AttributeError, IndexError, KeyError, TypeError) as error:
            unresolved.append((down_key, f"{path}: {error}"))
            continue
        down = lora[down_key].float()
        up = lora[up_key].float()
        if down.ndim == 2 and up.ndim == 2:
            delta = up @ down
        elif down.ndim == 4 and up.ndim == 4 and down.shape[2:] == (1, 1) and up.shape[2:] == (1, 1):
            delta = (up[:, :, 0, 0] @ down[:, :, 0, 0])[:, :, None, None]
        else:
            unresolved.append((down_key, f"unsupported shapes {tuple(down.shape)}, {tuple(up.shape)}"))
            continue
        if tuple(delta.shape) != tuple(module.weight.shape):
            unresolved.append((down_key, f"shape {tuple(delta.shape)} != {tuple(module.weight.shape)}"))
            continue
        module.weight.data.add_(delta.to(device=module.weight.device, dtype=module.weight.dtype))
        fused += 1
    if unresolved:
        preview = "\n".join(f"  {key}: {why}" for key, why in unresolved[:10])
        fail(f"AquaLoRA fusion was incomplete ({len(unresolved)} unresolved):\n{preview}")
    if fused < 10:
        fail(f"Only {fused} AquaLoRA modules were fused; expected many more")
    print(f"Fused AquaLoRA into {fused} UNet modules", flush=True)


def clone_state(module):
    state_device = DEVICE if KEEP_PRISTINE_ON_GPU else torch.device("cpu")
    return OrderedDict((name, tensor.detach().to(state_device).clone())
                       for name, tensor in module.state_dict().items())


def restore_state(module, state):
    module.load_state_dict(state, strict=True)


def group_name(parameter_name, component):
    if component == "vae":
        if parameter_name.startswith("decoder.mid_block"):
            return "mid"
        if parameter_name.startswith("decoder.up_blocks.0") or parameter_name.startswith("decoder.up_blocks.1"):
            return "up_early"
        if parameter_name.startswith("decoder.up_blocks.2") or parameter_name.startswith("decoder.up_blocks.3"):
            return "up_late"
        if parameter_name.startswith("decoder.conv") or parameter_name.startswith("post_quant_conv"):
            return "io"
        return "other"
    if parameter_name.startswith("down_blocks"):
        return "down"
    if parameter_name.startswith("mid_block"):
        return "mid"
    if parameter_name.startswith("up_blocks"):
        return "up"
    if "conv_in" in parameter_name or "conv_out" in parameter_name:
        return "io"
    return "other"


def quantize_tensor(weight, bits, clip, scale_mul=1.0, zero_offset=0, round_threshold=0.5):
    """Per-output-channel fake quantization with attack-controlled parameters.

    The default (scale_mul=1, zero_offset=0, threshold=0.5) is the original
    symmetric round-to-nearest PTQ. Integer zero_offset shifts the representable
    grid, while round_threshold controls whether each fractional value rounds up.
    Dequantized FP weights are retained so the experiment is portable across
    CUDA GPUs; every returned value still lies on the declared low-bit grid.
    """
    if not weight.is_floating_point() or weight.ndim < 2:
        return weight
    source = weight.float()
    flat = source.reshape(source.shape[0], -1)
    max_abs = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) * clip
    qmax = float(2 ** (bits - 1) - 1)
    qmin = -qmax
    scale = (max_abs / qmax) * float(scale_mul)
    zero = float(int(zero_offset))
    transformed = flat / scale + zero
    lower = torch.floor(transformed)
    fraction = transformed - lower
    rounded = lower + (fraction >= float(round_threshold)).to(flat.dtype)
    quantized = (rounded.clamp(qmin, qmax) - zero) * scale
    return quantized.reshape_as(source).to(weight.dtype)


def apply_recipe(module, component, recipe):
    selected = set(recipe["groups"])
    changed = 0
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if not name.endswith("weight") or parameter.ndim < 2:
                continue
            # Stable Signature is carried by the VAE decoder. Its encoder is
            # unused during text-to-image generation and is outside this test.
            if component == "vae" and not (
                name.startswith("decoder.") or name.startswith("post_quant_conv.")
            ):
                continue
            if group_name(name, component) not in selected and "all" not in selected:
                continue
            group = group_name(name, component)
            controls = recipe.get("group_params", {}).get(group, {})
            parameter.copy_(quantize_tensor(
                parameter, recipe["bits"], recipe["clip"],
                scale_mul=controls.get("scale_mul", 1.0),
                zero_offset=controls.get("zero_offset", 0),
                round_threshold=controls.get("round_threshold", 0.5),
            ))
            changed += parameter.numel()
    if changed == 0:
        fail(f"Recipe selected no parameters: {recipe}")
    return changed


def candidate_recipes(component):
    groups = ["mid", "up_early", "up_late", "io"] if component == "vae" else ["down", "mid", "up", "io"]
    recipes = []
    # W4 only by default; retain other precisions for explicit research overrides.
    bitwidths = tuple(dict.fromkeys(int(b) for b in os.environ.get("WMQ_SEARCH_BITS", "4").replace(",", " ").split()))
    if not bitwidths or any(b not in (8, 6, 4, 3, 2) for b in bitwidths):
        fail("WMQ_SEARCH_BITS must list supported bitwidths: 2, 3, 4, 6, 8")
    for bits in bitwidths:
        for clip in (1.0, 0.75):
            recipes.append({"bits": bits, "clip": clip, "groups": ["all"]})
            for group in groups:
                recipes.append({"bits": bits, "clip": clip, "groups": [group]})
    if MAX_CANDIDATES > 0 and MAX_CANDIDATES < len(recipes):
        print(f"WARNING: MAX_CANDIDATES={MAX_CANDIDATES} truncates the full "
              f"{len(recipes)}-recipe registry. Set MAX_CANDIDATES=0 for all recipes.", flush=True)
        return recipes[:MAX_CANDIDATES]
    return recipes


def _tree_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, tuple):
        return tuple(_tree_cpu(item) for item in value)
    if isinstance(value, list):
        return [_tree_cpu(item) for item in value]
    if isinstance(value, dict):
        return {key: _tree_cpu(item) for key, item in value.items()}
    return value


def _tree_device(value):
    if torch.is_tensor(value):
        return value.to(DEVICE)
    if isinstance(value, tuple):
        return tuple(_tree_device(item) for item in value)
    if isinstance(value, list):
        return [_tree_device(item) for item in value]
    if isinstance(value, dict):
        return {key: _tree_device(item) for key, item in value.items()}
    return value


def _tree_clone(value):
    """Turn hook outputs created under inference_mode into ordinary constants."""
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_tree_clone(item) for item in value)
    if isinstance(value, list):
        return [_tree_clone(item) for item in value]
    if isinstance(value, dict):
        return {key: _tree_clone(item) for key, item in value.items()}
    return value


@torch.no_grad()
def capture_unet_calibration(pipe, prompts, seeds):
    """Capture real denoising states, stratified over the DDIM trajectory."""
    selected_steps = set(np.linspace(0, STEPS - 1, CALIB_TIMESTEPS, dtype=int).tolist())
    records = []
    calls = 0

    def hook(_module, args, kwargs, output):
        nonlocal calls
        step = calls % STEPS
        calls += 1
        if step not in selected_steps:
            return
        sample = output.sample if hasattr(output, "sample") else output[0]
        timestep = args[1] if len(args) > 1 else kwargs["timestep"]
        t_value = int(timestep.flatten()[0].item()) if torch.is_tensor(timestep) else int(timestep)
        alpha = float(pipe.scheduler.alphas_cumprod[t_value].item())
        records.append({
            "args": _tree_cpu(args), "kwargs": _tree_cpu(kwargs),
            "reference": sample.detach().cpu(), "timestep": t_value,
            # Later, low-noise steps receive more weight because their errors
            # propagate directly into visible fine detail.
            "weight": max(alpha, 1e-4),
        })

    handle = pipe.unet.register_forward_hook(hook, with_kwargs=True)
    try:
        generated = generate(pipe, prompts[:CALIB_N], seeds[:CALIB_N],
                             batch_size=CALIB_BATCH_SIZE)
        del generated
    finally:
        handle.remove()
    calibration_batches = math.ceil(len(prompts[:CALIB_N]) / CALIB_BATCH_SIZE)
    expected = calibration_batches * len(selected_steps)
    if len(records) < expected:
        fail(f"UNet calibration captured {len(records)} states; expected at least {expected}. "
             "Check scheduler step count/hook compatibility")
    scaling = float(pipe.vae.config.scaling_factor)
    for record in records:
        record["args"] = _tree_clone(record["args"])
        record["kwargs"] = _tree_clone(record["kwargs"])
        record["reference"] = record["reference"].clone()
        latent_input = record["args"][0].to(device=DEVICE, dtype=DTYPE)
        reference = record["reference"].to(device=DEVICE, dtype=DTYPE)
        if reference.shape[0] >= 2 and reference.shape[0] % 2 == 0:
            eps_uncond, eps_text = reference.chunk(2)
            guided = eps_uncond + GUIDANCE * (eps_text - eps_uncond)
            latent = latent_input[:guided.shape[0]]
        else:
            guided, latent = reference, latent_input
        alpha = record["weight"]
        pred_x0 = (latent - math.sqrt(max(1.0 - alpha, 0.0)) * guided) / math.sqrt(max(alpha, 1e-6))
        pred_x0 = pred_x0.clamp(-4.0, 4.0)
        reference_image = pipe.vae.decode(pred_x0 / scaling, return_dict=False)[0]
        record["reference_image"] = reference_image.detach().cpu()
    return {"kind": "timestep_noise_prediction", "records": records}


@torch.no_grad()
def capture_vae_calibration(pipe, prompts, seeds):
    """Capture final diffusion latents and clean VAE-decoder outputs."""
    records = []
    scaling = float(pipe.vae.config.scaling_factor)
    selected_prompts = prompts[:CALIB_N]
    selected_seeds = seeds[:CALIB_N]
    for start in range(0, len(selected_prompts), CALIB_BATCH_SIZE):
        prompt_batch = selected_prompts[start: start + CALIB_BATCH_SIZE]
        seed_batch = selected_seeds[start: start + CALIB_BATCH_SIZE]
        generators = [torch.Generator(device=DEVICE).manual_seed(seed) for seed in seed_batch]
        latent = pipe(prompt_batch, height=HEIGHT, width=WIDTH,
                      num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                      generator=generators, output_type="latent").images
        reference = pipe.vae.decode(latent / scaling, return_dict=False)[0]
        records.append({"latent": latent.detach().cpu(), "reference": reference.detach().cpu()})
    return {"kind": "vae_decode", "records": records}


def capture_calibration(pipe, component, prompts, seeds):
    if component == "unet":
        return capture_unet_calibration(pipe, prompts, seeds)
    return capture_vae_calibration(pipe, prompts, seeds)


@torch.inference_mode()
def prediction_nmse(pipe, component, calibration):
    numerator = 0.0
    denominator = 0.0
    if calibration["kind"] == "timestep_noise_prediction":
        for record in calibration["records"]:
            args = _tree_device(record["args"])
            kwargs = _tree_device(record["kwargs"])
            output = pipe.unet(*args, **kwargs)
            sample = output.sample if hasattr(output, "sample") else output[0]
            reference = record["reference"].to(device=DEVICE, dtype=sample.dtype)
            weight = record["weight"]
            numerator += weight * float(F.mse_loss(sample.float(), reference.float()).item())
            denominator += weight * float(reference.float().square().mean().item())
    else:
        scaling = float(pipe.vae.config.scaling_factor)
        for record in calibration["records"]:
            latent = record["latent"].to(device=DEVICE, dtype=DTYPE)
            sample = pipe.vae.decode(latent / scaling, return_dict=False)[0]
            reference = record["reference"].to(device=DEVICE, dtype=sample.dtype)
            numerator += float(F.mse_loss(sample.float(), reference.float()).item())
            denominator += float(reference.float().square().mean().item())
    return numerator / max(denominator, 1e-12)


def refinement_groups(component, recipe):
    if "all" not in recipe["groups"]:
        return list(recipe["groups"])
    # "other" weights remain on the default grid when an all-layer recipe is
    # refined; the four semantically defined carrier groups receive controls.
    groups = ["mid", "up_early", "up_late", "io"] if component == "vae" \
        else ["down", "mid", "up", "io"]
    return groups


def initial_refined_recipe(component, recipe):
    result = json.loads(json.dumps(recipe))
    result["quantizer"] = "per_channel_symmetric_controlled_rounding"
    result["group_params"] = {
        group: {"scale_mul": 1.0, "zero_offset": 0, "round_threshold": 0.5}
        for group in refinement_groups(component, recipe)
    }
    return result


def mutate_recipe(recipe, rng):
    proposal = json.loads(json.dumps(recipe))
    group = rng.choice(sorted(proposal["group_params"]))
    allowed = {
        "scale": ["scale_mul"],
        "scale_zero": ["scale_mul", "zero_offset"],
        "full": ["scale_mul", "zero_offset", "round_threshold"],
    }[REFINE_MODE]
    variable = rng.choice(allowed)
    values = proposal["group_params"][group]
    if variable == "scale_mul":
        values[variable] = float(np.clip(values[variable] * math.exp(rng.gauss(0.0, 0.12)), 0.60, 1.40))
    elif variable == "zero_offset":
        values[variable] = int(np.clip(values[variable] + rng.choice([-1, 1]), -2, 2))
    else:
        values[variable] = float(np.clip(values[variable] + rng.gauss(0.0, 0.10), 0.05, 0.95))
    return proposal, group, variable


def evaluate_attack_candidate(pipe, target, pristine, component, recipe, decoder,
                              decode_fn, key, clean_images, prompts, seeds,
                              lpips_model, calibration):
    restore_state(target, pristine)
    changed = apply_recipe(target, component, recipe)
    pred_nmse = prediction_nmse(pipe, component, calibration)
    attacked = generate(pipe, prompts, seeds)
    quality = image_metrics(clean_images, attacked, lpips_model)
    bits = summarize_bits(decode_fn(decoder, attacked), key)
    objective = (bits["bit_accuracy"] + TPR_LOSS_WEIGHT * bits["tpr"]
                 + PRED_LOSS_WEIGHT * pred_nmse)
    feasible = (quality["psnr"] >= MIN_PSNR and quality["lpips"] <= MAX_LPIPS
                and pred_nmse <= MAX_PRED_NMSE)
    return {
        "changed_parameters": changed, "bit_accuracy": bits["bit_accuracy"],
        "tpr": bits["tpr"], "psnr": quality["psnr"], "lpips": quality["lpips"],
        "prediction_nmse": pred_nmse, "objective": objective, "feasible": feasible,
    }, attacked


class GradientQuantizer(nn.Module):
    """Learnable per-group scale, zero-point and rounding controls."""
    def __init__(self, groups):
        super().__init__()
        self.groups = list(groups)
        self.log_scale = nn.ParameterDict({group: nn.Parameter(torch.zeros((), device=DEVICE))
                                           for group in groups})
        self.zero = nn.ParameterDict({group: nn.Parameter(torch.zeros((), device=DEVICE))
                                      for group in groups})
        self.round_logit = nn.ParameterDict({group: nn.Parameter(torch.zeros((), device=DEVICE))
                                             for group in groups})

    def project_(self):
        with torch.no_grad():
            for group in self.groups:
                self.log_scale[group].clamp_(math.log(0.60), math.log(1.40))
                self.zero[group].clamp_(-2.0, 2.0)
                self.round_logit[group].clamp_(math.log(0.05 / 0.95), math.log(0.95 / 0.05))

    def hard_recipe(self, component, initial_recipe):
        recipe = json.loads(json.dumps(initial_recipe))
        recipe["quantizer"] = "gradient_steered_per_channel_rounding"
        recipe["group_params"] = {}
        for group in self.groups:
            recipe["group_params"][group] = {
                "scale_mul": float(self.log_scale[group].detach().exp().cpu()),
                "zero_offset": int(self.zero[group].detach().round().cpu()),
                "round_threshold": float(self.round_logit[group].detach().sigmoid().cpu()),
            }
        return recipe


def differentiable_quantize(weight, bits, clip, log_scale, zero_value, round_logit):
    # Keep grid math in FP32: 1e-8 becomes zero in FP16 and divides by zero.
    source = weight.detach().float()
    flat = source.reshape(source.shape[0], -1)
    max_abs = flat.float().abs().amax(dim=1, keepdim=True).clamp_min(1e-8).to(flat.dtype) * clip
    qmax = float(2 ** (bits - 1) - 1)
    scale = (max_abs / qmax) * log_scale.exp().to(flat.dtype)
    # Straight-through integer zero-point: integer in the forward pass, identity
    # gradient in the backward pass.
    zero_hard = zero_value.round()
    zero = zero_value + (zero_hard - zero_value).detach()
    transformed = flat / scale + zero.to(flat.dtype)
    lower = torch.floor(transformed).detach()
    fraction = transformed - lower
    threshold = round_logit.sigmoid().to(flat.dtype)
    soft_up = torch.sigmoid((fraction - threshold) / ROUND_TEMPERATURE)
    hard_up = (fraction >= threshold).to(flat.dtype)
    up = soft_up + (hard_up - soft_up).detach()
    integer = (lower + up).clamp(-qmax, qmax)
    return ((integer - zero.to(flat.dtype)) * scale).reshape_as(source).to(weight.dtype)


def differentiable_overrides(module, component, recipe, controls, source_state, active_groups):
    selected = set(recipe["groups"])
    overrides = {}
    for name, parameter in module.named_parameters():
        if not name.endswith("weight") or parameter.ndim < 2:
            continue
        if component == "vae" and not (
            name.startswith("decoder.") or name.startswith("post_quant_conv.")
        ):
            continue
        group = group_name(name, component)
        if group not in selected and "all" not in selected:
            continue
        if group not in active_groups:
            continue
        source = source_state[name].to(device=parameter.device, dtype=parameter.dtype)
        overrides[name] = differentiable_quantize(
            source, recipe["bits"], recipe["clip"], controls.log_scale[group],
            controls.zero[group], controls.round_logit[group])
    if not overrides:
        fail(f"Gradient quantizer selected no parameters: {recipe}")
    return overrides


def differentiable_watermark_loss(name, decoder, image_tensor, key):
    truth = torch.tensor([int(x) for x in key], device=DEVICE, dtype=torch.long)
    flipped = 1 - truth
    if name == "stable_signature":
        pixels = (image_tensor.float() / 2 + 0.5).clamp(0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE)[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE)[:, None, None]
        logits = decoder((pixels - mean) / std)
        targets = flipped.to(logits.dtype).expand(logits.shape[0], -1)
        return F.binary_cross_entropy_with_logits(logits, targets)
    logits = decoder(F.interpolate(image_tensor.float(), size=(512, 512),
                                   mode="bilinear", align_corners=False))
    targets = flipped.expand(logits.shape[0], -1)
    return F.cross_entropy(logits.reshape(-1, 2), targets.reshape(-1))


def gradient_proxy_loss(name, pipe, component, decoder, key, recipe, controls, record,
                        source_state, active_groups):
    if component == "unet":
        overrides = differentiable_overrides(pipe.unet, component, recipe, controls,
                                             source_state, active_groups)
        args = _tree_device(record["args"])
        kwargs = _tree_device(record["kwargs"])
        output = functional_call(pipe.unet, overrides, args, kwargs, strict=False)
        sample = output.sample if hasattr(output, "sample") else output[0]
        reference = record["reference"].to(device=DEVICE, dtype=sample.dtype)
        pred_nmse = F.mse_loss(sample.float(), reference.float()) / reference.float().square().mean().clamp_min(1e-8)
        latent_input = args[0]
        if sample.shape[0] >= 2 and sample.shape[0] % 2 == 0:
            eps_uncond, eps_text = sample.chunk(2)
            guided = eps_uncond + GUIDANCE * (eps_text - eps_uncond)
            latent = latent_input[:guided.shape[0]]
        else:
            guided, latent = sample, latent_input
        alpha = record["weight"]
        pred_x0 = (latent - math.sqrt(max(1.0 - alpha, 0.0)) * guided) / math.sqrt(max(alpha, 1e-6))
        pred_x0 = pred_x0.clamp(-4.0, 4.0)
        image = pipe.vae.decode(pred_x0 / float(pipe.vae.config.scaling_factor), return_dict=False)[0]
        reference_image = record["reference_image"].to(device=DEVICE, dtype=image.dtype)
    else:
        overrides = differentiable_overrides(pipe.vae, component, recipe, controls,
                                             source_state, active_groups)
        post = {name[len("post_quant_conv."):]: value for name, value in overrides.items()
                if name.startswith("post_quant_conv.")}
        decoder_weights = {name[len("decoder."):]: value for name, value in overrides.items()
                           if name.startswith("decoder.")}
        latent = record["latent"].to(device=DEVICE, dtype=DTYPE) / float(pipe.vae.config.scaling_factor)
        latent = functional_call(pipe.vae.post_quant_conv, post, (latent,), strict=False)
        image = functional_call(pipe.vae.decoder, decoder_weights, (latent,), strict=False)
        reference_image = record["reference"].to(device=DEVICE, dtype=image.dtype)
        pred_nmse = F.mse_loss(image.float(), reference_image.float()) / reference_image.float().square().mean().clamp_min(1e-8)
    image_loss = F.mse_loss(image.float(), reference_image.float())
    attack_loss = differentiable_watermark_loss(name, decoder, image, key)
    total = attack_loss + GRAD_PRED_WEIGHT * pred_nmse + GRAD_IMAGE_WEIGHT * image_loss
    return total, attack_loss, pred_nmse, image_loss


def refine_quantizer_gradient(name, pipe, component, decoder, decode_fn, key, lpips_model,
                              pristine, initial_recipe, prompts, seeds, clean_images,
                              result_dir, calibration=None):
    target = pipe.vae if component == "vae" else pipe.unet
    restore_state(target, pristine)
    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    if hasattr(decoder, "parameters"):
        for parameter in decoder.parameters():
            parameter.requires_grad_(False)
    if calibration is None:
        calibration = capture_calibration(pipe, component, prompts, seeds)
    groups = refinement_groups(component, initial_recipe)
    controls = GradientQuantizer(groups)
    if REFINE_MODE == "scale":
        for value in controls.zero.values(): value.requires_grad_(False)
        for value in controls.round_logit.values(): value.requires_grad_(False)
    elif REFINE_MODE == "scale_zero":
        for value in controls.round_logit.values(): value.requires_grad_(False)
    optimizer = torch.optim.Adam([p for p in controls.parameters() if p.requires_grad], lr=GRAD_LR)
    rows = []
    best = None
    records = calibration["records"]
    joint_runtime = JOINT_GROUP_GRAD
    trainable_controls = [p for p in controls.parameters() if p.requires_grad]

    def controls_are_finite():
        return all(bool(torch.isfinite(parameter).all().item())
                   for parameter in trainable_controls)

    def restore_controls(snapshot):
        with torch.no_grad():
            for parameter, saved in zip(trainable_controls, snapshot):
                parameter.copy_(saved)

    for step in range(1, GRAD_STEPS + 1):
        record = records[(step - 1) % len(records)]
        while True:
            optimizer.zero_grad(set_to_none=True)
            active_groups = groups if joint_runtime else [groups[(step - 1) % len(groups)]]
            # Keep the complete model on the current hard low-bit grid. Large-memory
            # GPUs substitute every controlled group jointly; portable mode uses one
            # round-robin group per backward pass.
            restore_state(target, pristine)
            apply_recipe(target, component, controls.hard_recipe(component, initial_recipe))
            try:
                total, attack_loss, pred_nmse, image_loss = gradient_proxy_loss(
                    name, pipe, component, decoder, key, initial_recipe, controls, record,
                    pristine, active_groups)
                losses_finite = all(bool(torch.isfinite(value.detach()).all().item())
                                    for value in (total, attack_loss, pred_nmse, image_loss))
                if losses_finite:
                    total.backward()
                break
            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                cleanup()
                if not joint_runtime:
                    raise
                joint_runtime = False
                print(f"CUDA OOM in joint-group gradient for {name}; switching to "
                      "round-robin group gradients", flush=True)
        control_snapshot = [parameter.detach().clone() for parameter in trainable_controls]
        skip_reason = ""
        update_applied = False
        if losses_finite:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_controls, 5.0,
                                                       error_if_nonfinite=False)
            gradients_finite = (bool(torch.isfinite(grad_norm).item()) and all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
                for parameter in trainable_controls))
            if gradients_finite:
                optimizer.step()
                if controls_are_finite():
                    controls.project_()
                    update_applied = True
                else:
                    restore_controls(control_snapshot)
                    optimizer.state.clear()
                    skip_reason = "nonfinite_parameters_after_optimizer_step"
            else:
                skip_reason = "nonfinite_gradient"
        else:
            grad_norm = torch.tensor(float("nan"), device=DEVICE)
            skip_reason = "nonfinite_proxy_loss"

        if not update_applied:
            restore_controls(control_snapshot)
            optimizer.zero_grad(set_to_none=True)
            # Repeated overflows should become less likely, while a fully unstable
            # refinement can finish and safely fall back to the grid-search recipe.
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] *= 0.5
            print(f"WARNING: {name} gradient step {step:03d} skipped: {skip_reason}; "
                  f"lr={optimizer.param_groups[0]['lr']:.3g}", flush=True)
        row = {
            "step": step, "active_group": "+".join(active_groups),
            "timestep": record.get("timestep", -1),
            "proxy_total": float(total.detach().cpu()),
            "proxy_watermark": float(attack_loss.detach().cpu()),
            "proxy_prediction_nmse": float(pred_nmse.detach().cpu()),
            "proxy_image_mse": float(image_loss.detach().cpu()),
            "quantizer_grad_norm": float(grad_norm.detach().cpu()),
            "update_applied": update_applied,
            "skip_reason": skip_reason,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        del total, attack_loss, pred_nmse, image_loss
        cleanup()

        if step == 1 or step % GRAD_EVAL_EVERY == 0 or step == GRAD_STEPS:
            hard_recipe = controls.hard_recipe(component, initial_recipe)
            metrics, images = evaluate_attack_candidate(
                pipe, target, pristine, component, hard_recipe, decoder, decode_fn, key,
                clean_images, prompts, seeds, lpips_model, calibration)
            rank = (metrics["objective"], metrics["bit_accuracy"], metrics["tpr"],
                    -metrics["psnr"], metrics["lpips"])
            accepted = update_applied and metrics["feasible"] and (best is None or rank < best[0])
            if accepted:
                best = (rank, hard_recipe, metrics)
            row.update({f"hard_{key}": value for key, value in metrics.items()})
            row["accepted"] = accepted
            print(f"{name} gradient step {step:03d}: accepted={accepted} "
                  f"proxy={row['proxy_total']:.5f} hard={metrics}", flush=True)
            del images
            restore_state(target, pristine)
            cleanup()
        rows.append(row)
        save_json(result_dir / "gradient_updates.json", {"statistics": gradient_statistics(rows),
                                                        "rows": rows})

    with open(result_dir / "gradient_optimization.csv", "w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted(set().union(*(row.keys() for row in rows)))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if best is None:
        print(f"WARNING: {name} gradient refinement produced no feasible hard checkpoint; "
              "retaining grid-search recipe", flush=True)
        return initial_recipe, rows, None
    return best[1], rows, {"calibration_kind": calibration["kind"],
                           "joint_group_gradient_used": joint_runtime, **best[2]}


def refine_quantizer_zeroth(name, pipe, component, decoder, decode_fn, key, lpips_model,
                            pristine, initial_recipe, prompts, seeds, clean_images, result_dir):
    """Detector-guided projected search over quantizer parameters.

    The objective is lexicographic watermark removal. PSNR, LPIPS and
    timestep-weighted prediction NMSE are hard feasibility constraints.
    Fixed prompts/seeds make every objective evaluation paired and deterministic.
    """
    if REFINE_MODE == "none" or REFINE_ITERS == 0:
        return initial_recipe, [], None
    if REFINE_MODE not in {"scale", "scale_zero", "full"}:
        fail("REFINE_MODE must be one of: none, scale, scale_zero, full")
    target = pipe.vae if component == "vae" else pipe.unet
    restore_state(target, pristine)
    calibration = capture_calibration(pipe, component, prompts, seeds)
    current = initial_refined_recipe(component, initial_recipe)
    rng = random.Random(REFINE_SEED + (0 if name == "stable_signature" else 1))
    rows = []

    metrics, images = evaluate_attack_candidate(
        pipe, target, pristine, component, current, decoder, decode_fn, key,
        clean_images, prompts, seeds, lpips_model, calibration)
    rank = (metrics["objective"], metrics["bit_accuracy"], metrics["tpr"],
            -metrics["psnr"], metrics["lpips"])
    best = (rank, current, metrics) if metrics["feasible"] else None
    rows.append({"iteration": 0, "mutation_group": "baseline", "mutation": "baseline",
                 **metrics, "recipe": json.dumps(current, sort_keys=True)})
    del images
    cleanup()

    for iteration in range(1, REFINE_ITERS + 1):
        parent = best[1] if best is not None else current
        proposal, mutation_group, mutation = mutate_recipe(parent, rng)
        metrics, images = evaluate_attack_candidate(
            pipe, target, pristine, component, proposal, decoder, decode_fn, key,
            clean_images, prompts, seeds, lpips_model, calibration)
        proposal_rank = (metrics["objective"], metrics["bit_accuracy"], metrics["tpr"],
                         -metrics["psnr"], metrics["lpips"])
        accepted = metrics["feasible"] and (best is None or proposal_rank < best[0])
        if accepted:
            best = (proposal_rank, proposal, metrics)
        rows.append({"iteration": iteration, "mutation_group": mutation_group,
                     "mutation": mutation, "accepted": accepted, **metrics,
                     "recipe": json.dumps(proposal, sort_keys=True)})
        print(f"{name} refinement {iteration:02d}: group={mutation_group} "
              f"variable={mutation} accepted={accepted} metrics={metrics}", flush=True)
        del images
        cleanup()

    with open(result_dir / "quantizer_optimization.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if best is None:
        print(f"WARNING: {name} quantizer refinement found no candidate satisfying "
              f"prediction_NMSE<={MAX_PRED_NMSE}; retaining grid-search recipe", flush=True)
        return initial_recipe, rows, None
    return best[1], rows, {"calibration_kind": calibration["kind"], **best[2]}


def refine_quantizer(name, pipe, component, decoder, decode_fn, key, lpips_model,
                     pristine, initial_recipe, prompts, seeds, clean_images, result_dir):
    if REFINE_MODE == "none":
        return initial_recipe, [], None
    if REFINE_OPTIMIZER == "gradient":
        return refine_quantizer_gradient(
            name, pipe, component, decoder, decode_fn, key, lpips_model,
            pristine, initial_recipe, prompts, seeds, clean_images, result_dir)
    if REFINE_OPTIMIZER == "zeroth":
        return refine_quantizer_zeroth(
            name, pipe, component, decoder, decode_fn, key, lpips_model,
            pristine, initial_recipe, prompts, seeds, clean_images, result_dir)
    fail("REFINE_OPTIMIZER must be gradient or zeroth")


def save_images(images, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for idx, image in enumerate(images):
        image.save(folder / f"{idx:04d}.png")


def save_json(path, payload):
    """Atomic reports; invalid numbers remain explicit nulls in valid JSON."""
    def sanitize(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {k: sanitize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [sanitize(v) for v in value]
        return value
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(sanitize(payload), indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def quality_feasible(quality, nmse):
    return (all(math.isfinite(v) for v in (quality["psnr"], quality["lpips"], nmse))
            and quality["psnr"] >= MIN_PSNR and quality["lpips"] <= MAX_LPIPS
            and nmse <= MAX_PRED_NMSE)


def gradient_statistics(rows):
    updates = sum(bool(r.get("update_applied", False)) for r in rows)
    return {"attempted_updates": len(rows), "valid_updates": updates,
            "skipped_updates": len(rows) - updates,
            "valid_update_fraction": updates / len(rows) if rows else 0.,
            "accepted_evaluations": sum(bool(r.get("accepted", False)) for r in rows),
            "skip_reasons": {reason: sum(r.get("skip_reason") == reason for r in rows)
                             for reason in sorted({r.get("skip_reason") for r in rows
                                                   if r.get("skip_reason")})}}


def run_fair_comparison(name, pipe, component, decoder, decode_fn, key, lpips_model):
    """Shared split/target W4A16 comparison; test outcomes never select a method."""
    result_dir = OUT / name / "fair_w4a16"
    # Do not mix incomplete/old reports with a new experiment.
    result_dir.mkdir(parents=True, exist_ok=False)
    prompts = experiment_prompts(CALIB_N + SEARCH_N + TEST_N)
    calib_prompts = prompts[:CALIB_N]
    search_prompts = prompts[CALIB_N:CALIB_N + SEARCH_N]
    test_prompts = prompts[CALIB_N + SEARCH_N:]
    # Guaranteed disjoint even for large requested splits.
    all_seeds = list(range(BASE_SEED, BASE_SEED + len(prompts)))
    calib_seeds = all_seeds[:CALIB_N]
    search_seeds = all_seeds[CALIB_N:CALIB_N + SEARCH_N]
    test_seeds = all_seeds[CALIB_N + SEARCH_N:]
    target = pipe.vae if component == "vae" else pipe.unet
    target.eval().requires_grad_(False)
    selected = [n for n, p in target.named_parameters() if n.endswith("weight") and p.ndim >= 2
                and (component == "unet" or n.startswith(("decoder.", "post_quant_conv.")))]
    budget = sum(target.get_parameter(n).numel() for n in selected)
    manifest = {"protocol": "fair_w4a16", "watermark": name, "component": component,
                "weight_bits": 4, "activation_bits": 16, "simulated": True,
                "weight_grid": "symmetric_-7_to_7", "target_names": selected,
                "changed_parameters": budget, "model": getattr(pipe, "_wmq_model_id", None),
                "key": key, "assets": getattr(pipe, "_wmq_assets", {}),
                "sampling": {"steps": STEPS, "guidance": GUIDANCE, "height": HEIGHT, "width": WIDTH,
                             "scheduler": dict(pipe.scheduler.config), "batch_size": GEN_BATCH_SIZE},
                "calibration": {"prompts": calib_prompts, "seeds": calib_seeds,
                                "timesteps": CALIB_TIMESTEPS, "batch_size": CALIB_BATCH_SIZE},
                "search": {"prompts": search_prompts, "seeds": search_seeds},
                "test": {"prompts": test_prompts, "seeds": test_seeds},
                "reconstruction": {"steps": RECON_STEPS, "lr": RECON_LR, "seed": REFINE_SEED},
                "constraints": {"min_psnr": MIN_PSNR, "max_lpips": MAX_LPIPS,
                                "max_prediction_nmse": MAX_PRED_NMSE, "fpr": FPR}}
    save_json(result_dir / "manifest.json", manifest)
    clean_search = generate(pipe, search_prompts, search_seeds)
    clean_search_bits = summarize_bits(decode_fn(decoder, clean_search), key)
    baseline_valid = (clean_search_bits["bit_accuracy"] >= CLEAN_MIN_BITACC
                      and clean_search_bits["tpr"] >= CLEAN_MIN_TPR)
    validation = {"watermark": name, "split": "search", "clean": clean_search_bits,
                  "minimum_bit_accuracy": CLEAN_MIN_BITACC, "minimum_tpr": CLEAN_MIN_TPR,
                  "baseline_valid": baseline_valid, "policy": CLEAN_POLICY,
                  "interpretation": "Quality gate only; low scores do not prove checkpoint corruption."}
    save_json(result_dir / "clean_validation.json", validation)
    save_images(clean_search, result_dir / "clean_search")
    if not baseline_valid:
        quality_warning(OUT, "clean_baseline", "Clean watermark below quality thresholds", watermark=name,
                        metrics=clean_search_bits,
                        thresholds={"min_bit_accuracy": CLEAN_MIN_BITACC, "min_tpr": CLEAN_MIN_TPR},
                        action="stop_requested" if BASELINE_CHECK_ONLY or CLEAN_POLICY == "strict" else "continue")
    if BASELINE_CHECK_ONLY or (not baseline_valid and CLEAN_POLICY == "strict"):
        status = "baseline_checked" if baseline_valid else "invalid_clean_baseline"
        report = {"watermark": name, "protocol": RUN_PROTOCOL, "status": status,
                  "baseline_valid": baseline_valid, "methods": [], "validation": validation}
        save_json(result_dir / "report.json", report)
        print(f"{name}: {status}; see {result_dir / 'clean_validation.json'}", flush=True)
        return report
    clean_test = generate(pipe, test_prompts, test_seeds)
    clean_test_bits = summarize_bits(decode_fn(decoder, clean_test), key)
    save_images(clean_test, result_dir / "clean_test")
    save_json(result_dir / "clean_test.json", clean_test_bits)
    pristine = clone_state(target)
    calibration = capture_calibration(pipe, component, calib_prompts, calib_seeds)
    for record in calibration["records"]:
        if not torch.isfinite(record["reference"]).all():
            fail("Clean calibration contains nonfinite reference values")

    def replay():
        for record in calibration["records"]:
            if component == "unet":
                pipe.unet(*_tree_device(record["args"]), **_tree_device(record["kwargs"]))
            else:
                pipe.vae.decode(record["latent"].to(device=DEVICE, dtype=DTYPE)
                                / float(pipe.vae.config.scaling_factor), return_dict=False)

    def search_metrics():
        nmse = prediction_nmse(pipe, component, calibration)
        images = generate(pipe, search_prompts, search_seeds)
        quality = image_metrics(clean_search, images, lpips_model)
        bits = summarize_bits(decode_fn(decoder, images), key)
        return {"bit_accuracy": bits["bit_accuracy"], "tpr": bits["tpr"], **quality,
                "prediction_nmse": nmse, "feasible": quality_feasible(quality, nmse)}

    methods = []

    def record_method(method, recipe, search, details):
        # Nothing is selected or tuned using this final held-out evaluation.
        images = generate(pipe, test_prompts, test_seeds)
        quality = image_metrics(clean_test, images, lpips_model)
        bits = summarize_bits(decode_fn(decoder, images), key)
        report = {"method": method, "recipe": recipe, "weight_bits": 4, "activation_bits": 16,
                  "component": component, "changed_parameters": budget,
                  "baseline_valid": baseline_valid, "search": search, "clean": clean_test_bits,
                  "attacked": bits, "paired_quality": quality,
                  "calibration_prediction_nmse": search["prediction_nmse"],
                  "feasible": quality_feasible(quality, search["prediction_nmse"]),
                  "bit_accuracy_retained_percent": 100 * bits["bit_accuracy"] / max(clean_test_bits["bit_accuracy"], 1e-12),
                  "tpr_retained_percent": (100 * bits["tpr"] / clean_test_bits["tpr"]
                                           if clean_test_bits["tpr"] > 0 else None),
                  "details": details}
        folder = result_dir / method
        save_json(folder / "report.json", report)
        save_images(images, folder / "attacked_test")
        if not search["feasible"] or not report["feasible"]:
            quality_warning(OUT, "method_quality", "Method failed search or test quality gate",
                            watermark=name, method=method, metrics={"search": search, "test": quality},
                            thresholds={"min_psnr": MIN_PSNR, "max_lpips": MAX_LPIPS,
                                        "max_prediction_nmse": MAX_PRED_NMSE})
        methods.append(report)
        save_json(result_dir / "report.json", {"watermark": name, "protocol": RUN_PROTOCOL,
                  "status": "running", "baseline_valid": baseline_valid, "methods": methods})

    try:
        # Full target coverage and fixed bitwidth for every row, including grid.
        grid = []
        for clip in (1., .75):
            recipe = {"bits": 4, "clip": clip, "groups": ["all"]}
            restore_state(target, pristine)
            if apply_recipe(target, component, recipe) != budget:
                fail("Grid recipe target coverage differs from the comparison manifest")
            metrics = search_metrics()
            grid.append((recipe, metrics))
            if clip == 1.:
                record_method("rtn_w4a16", recipe, metrics, {"uses_watermark_labels": False})
        eligible = [item for item in grid if item[1]["feasible"]]
        if eligible:
            grid_recipe, grid_metrics = min(eligible, key=lambda item: (item[1]["bit_accuracy"], item[1]["tpr"]))
        else:
            grid_recipe, grid_metrics = min(grid, key=lambda item: item[1]["prediction_nmse"])
            quality_warning(OUT, "grid_fallback", "No feasible grid candidate; evaluating minimum-NMSE candidate",
                            watermark=name, metrics=grid_metrics, action="continue_with_diagnostic_fallback")
        save_json(result_dir / "grid_search.json", {"candidates": grid, "has_feasible_candidate": bool(eligible),
                                                    "selected_recipe": grid_recipe})
        restore_state(target, pristine)
        apply_recipe(target, component, grid_recipe)
        record_method("grid_w4a16", grid_recipe, grid_metrics,
                      {"uses_watermark_labels": True, "fallback_feasible": grid_metrics["feasible"]})

        if REFINE_MODE != "none":
            restore_state(target, pristine)
            folder = result_dir / "gradient_w4a16"
            folder.mkdir(exist_ok=True)
            recipe, rows, best = refine_quantizer_gradient(
                name, pipe, component, decoder, decode_fn, key, lpips_model, pristine,
                grid_recipe, search_prompts, search_seeds, clean_search, folder, calibration=calibration)
            restore_state(target, pristine)
            apply_recipe(target, component, recipe)
            metrics = search_metrics()
            # Candidate has already been checked on the same search/calibration
            # split; still recheck after materializing for the final evaluation.
            fallback = best is None or not metrics["feasible"]
            if fallback:
                recipe = grid_recipe
                restore_state(target, pristine)
                apply_recipe(target, component, recipe)
                metrics = search_metrics()
            record_method("gradient_w4a16", recipe, metrics,
                          {"uses_watermark_labels": True, **gradient_statistics(rows),
                           "selected_source": "grid_fallback" if fallback else "gradient_refinement",
                           "fallback_feasible": metrics["feasible"] if fallback else None,
                           "best_refinement_metrics": best})

        baselines = ["qdiff_unet_adapted"] if component == "unet" else ["adaround_vae", "brecq_vae_adapted"]
        for method in baselines:
            restore_state(target, pristine)
            folder = result_dir / method
            folder.mkdir(exist_ok=True)
            progress = []

            def log_progress(row):
                progress.append(row)
                save_json(folder / "reconstruction.json", progress)
                print(f"{name} {method}: {row}", flush=True)

            details = reconstruct(target, selected, replay, method, steps=RECON_STEPS,
                                  lr=RECON_LR, seed=REFINE_SEED, cache_mb=RECON_CACHE_MB,
                                  log_callback=log_progress)
            if details["changed_parameters"] != budget:
                fail(f"Unequal target coverage for {method}")
            record_method(method, {"bits": 4, "clip": 1., "groups": ["all"]}, search_metrics(), details)
    finally:
        restore_state(target, pristine)
    report = {"watermark": name, "protocol": RUN_PROTOCOL, "status": "complete",
              "baseline_valid": baseline_valid, "methods": methods,
              "quality_valid": baseline_valid and all(m["feasible"] and m["search"]["feasible"] for m in methods)}
    save_json(result_dir / "report.json", report)
    return report


def run_adaptive_search(name, pipe, component, decoder, decode_fn, key, lpips_model):
    result_dir = OUT / name
    result_dir.mkdir(parents=True, exist_ok=True)
    target = pipe.vae if component == "vae" else pipe.unet
    pristine = clone_state(target)
    all_prompts = experiment_prompts(SEARCH_N + TEST_N)
    prompts_search = all_prompts[:SEARCH_N]
    prompts_test = all_prompts[SEARCH_N: SEARCH_N + TEST_N]
    search_seeds = [BASE_SEED + i for i in range(SEARCH_N)]
    test_seeds = [BASE_SEED + 10000 + i for i in range(TEST_N)]

    clean_search = generate(pipe, prompts_search, search_seeds)
    clean_test = generate(pipe, prompts_test, test_seeds)
    clean_bits = summarize_bits(decode_fn(decoder, clean_test), key)
    clean_search_bits = summarize_bits(decode_fn(decoder, clean_search), key)
    baseline_valid = (clean_search_bits["bit_accuracy"] >= CLEAN_MIN_BITACC
                      and clean_search_bits["tpr"] >= CLEAN_MIN_TPR)
    save_json(result_dir / "clean_validation.json", {"search": clean_search_bits, "test": clean_bits,
              "minimum_bit_accuracy": CLEAN_MIN_BITACC, "minimum_tpr": CLEAN_MIN_TPR,
              "baseline_valid": baseline_valid, "policy": CLEAN_POLICY})
    save_images(clean_test, result_dir / "clean_test")
    if not baseline_valid:
        quality_warning(OUT, "clean_baseline", "Clean watermark below quality thresholds", watermark=name,
                        metrics=clean_search_bits,
                        thresholds={"min_bit_accuracy": CLEAN_MIN_BITACC, "min_tpr": CLEAN_MIN_TPR},
                        action="stop_requested" if CLEAN_POLICY == "strict" else "continue")
    if CLEAN_POLICY == "strict" and (clean_search_bits["bit_accuracy"] < CLEAN_MIN_BITACC
                                    or clean_search_bits["tpr"] < CLEAN_MIN_TPR):
        fail(f"{name} clean watermark did not pass the declared quality gate; "
             "inspect clean_validation.json and the checkpoint/key/preprocessing")
    calibration = capture_calibration(pipe, component, prompts_search, search_seeds)

    rows = []
    best = None
    for index, recipe in enumerate(candidate_recipes(component)):
        restore_state(target, pristine)
        changed = apply_recipe(target, component, recipe)
        attacked = generate(pipe, prompts_search, search_seeds)
        quality = image_metrics(clean_search, attacked, lpips_model)
        bits = summarize_bits(decode_fn(decoder, attacked), key)
        nmse = prediction_nmse(pipe, component, calibration)
        feasible = quality_feasible(quality, nmse)
        row = {
            "candidate": index, **recipe, "changed_parameters": changed,
            "search_bit_accuracy": bits["bit_accuracy"], "search_tpr": bits["tpr"],
            "search_psnr": quality["psnr"], "search_lpips": quality["lpips"],
            "calibration_prediction_nmse": nmse,
            "feasible": feasible,
        }
        rows.append(row)
        save_json(result_dir / "search_progress.json", rows)
        if not feasible:
            quality_warning(OUT, "grid_candidate", "Candidate failed quality gate", watermark=name,
                            metrics=row, thresholds={"min_psnr": MIN_PSNR, "max_lpips": MAX_LPIPS,
                                                     "max_prediction_nmse": MAX_PRED_NMSE})
        print(f"{name} candidate {index:02d}: {row}", flush=True)
        rank = (bits["bit_accuracy"], bits["tpr"], -quality["psnr"], quality["lpips"])
        if feasible and (best is None or rank < best[0]):
            best = (rank, dict(recipe), index)
        del attacked
        cleanup()

    if not rows:
        fail(f"{name}: candidate grid is empty; no quantization recipe to evaluate")
    with open(result_dir / "search.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    grid_has_feasible_candidate = best is not None
    if best is None:
        usable = [r for r in rows if all(math.isfinite(r[k]) for k in
                  ("calibration_prediction_nmse", "search_psnr", "search_lpips", "search_bit_accuracy", "search_tpr"))]
        if not usable:
            restore_state(target, pristine)
            fail(f"{name}: all candidate metrics are nonfinite; inspect search_progress.json")
        fallback = min(usable, key=lambda r: (r["calibration_prediction_nmse"], -r["search_psnr"], r["candidate"]))
        best = (None, {k: fallback[k] for k in ("bits", "clip", "groups")}, fallback["candidate"])
        quality_warning(OUT, "grid_fallback", "No feasible grid candidate; continuing refinement and test with minimum-NMSE candidate",
                        watermark=name, metrics=fallback, action="continue_with_diagnostic_fallback")

    _, grid_recipe, candidate_index = best
    refine_count = min(REFINE_N, SEARCH_N)
    recipe, refinement_rows, refinement_best = refine_quantizer(
        name, pipe, component, decoder, decode_fn, key, lpips_model, pristine,
        grid_recipe, prompts_search[:refine_count], search_seeds[:refine_count],
        clean_search[:refine_count], result_dir)
    restore_state(target, pristine)
    changed = apply_recipe(target, component, recipe)
    attacked_test = generate(pipe, prompts_test, test_seeds)
    attack_bits = summarize_bits(decode_fn(decoder, attacked_test), key)
    quality = image_metrics(clean_test, attacked_test, lpips_model)
    final_nmse = prediction_nmse(pipe, component, calibration)
    save_images(attacked_test, result_dir / "attacked_test")
    report = {
        "watermark": name,
        "baseline_valid": baseline_valid,
        "grid_has_feasible_candidate": grid_has_feasible_candidate,
        "model": getattr(pipe, "_wmq_model_id", SS_MODEL if name == "stable_signature" else AQUA_MODEL),
        "quantization": "simulated weight PTQ",
        "selected_source": ("refinement" if refinement_best is not None else
                            "grid_search" if grid_has_feasible_candidate else "grid_diagnostic_fallback"),
        "calibration_prediction_nmse": final_nmse,
        "feasible": quality_feasible(quality, final_nmse),
        "selected_grid_candidate": candidate_index,
        "grid_recipe": grid_recipe,
        "recipe": recipe,
        "refinement": {
            "mode": REFINE_MODE, "optimizer": REFINE_OPTIMIZER,
            "logged_steps": len(refinement_rows),
            "update_statistics": (gradient_statistics(refinement_rows)
                                  if REFINE_OPTIMIZER == "gradient" else None),
            "objective_images": refine_count, "max_prediction_nmse": MAX_PRED_NMSE,
            "gradient_scope_requested": (("all controlled semantic groups jointly" if JOINT_GROUP_GRAD else
                                          "one round-robin semantic group") +
                                         " and one calibration timestep-batch per update"
                                         if REFINE_OPTIMIZER == "gradient" else
                                         "one randomly mutated group-control coordinate per proposal"),
            "trajectory_proxy": ("single-step predicted-x0 differentiable proxy for UNet; "
                                 "direct final-latent VAE decode for Stable Signature"),
            "full_trajectory_backpropagation": False,
            "joint_group_gradient_requested": JOINT_GROUP_GRAD,
            "joint_group_gradient_used": ((refinement_best or {}).get(
                "joint_group_gradient_used") if REFINE_OPTIMIZER == "gradient" else None),
            "best_calibration_metrics": refinement_best,
        },
        "changed_parameters": changed,
        "search_size": SEARCH_N,
        "test_size": TEST_N,
        "quality_constraints": {"min_psnr": MIN_PSNR, "max_lpips": MAX_LPIPS},
        "clean": clean_bits,
        "attacked": attack_bits,
        "paired_quality": quality,
        "bit_accuracy_retained_percent": 100.0 * attack_bits["bit_accuracy"] / max(clean_bits["bit_accuracy"], 1e-12),
        "tpr_retained_percent": 100.0 * attack_bits["tpr"] / max(clean_bits["tpr"], 1e-12),
    }
    if not report["feasible"]:
        quality_warning(OUT, "test_quality", "Final candidate failed quality gate", watermark=name,
                        metrics={**quality, "prediction_nmse": final_nmse},
                        thresholds={"min_psnr": MIN_PSNR, "max_lpips": MAX_LPIPS, "max_prediction_nmse": MAX_PRED_NMSE})
    save_json(result_dir / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    restore_state(target, pristine)
    return report


def load_stable_signature():
    assets = ROOT / "assets"
    decoder_path = download_url(SS_DECODER_URL, assets / "sd2_decoder.pth")
    extractor_path = download_url(SS_EXTRACTOR_URL, assets / "dec_48b_whit.torchscript.pt")
    pipe = build_pipeline(SS_MODEL, SS_MODEL_FALLBACK)
    ldm_state = torch.load(decoder_path, map_location="cpu", weights_only=True)
    converted = convert_ldm_vae_checkpoint(ldm_state, dict(pipe.vae.config))
    decoder_keys = {k: v for k, v in converted.items()
                    if k.startswith("decoder.") or k.startswith("post_quant_conv.")}
    loaded = pipe.vae.load_state_dict(decoder_keys, strict=False)
    unexpected = list(loaded.unexpected_keys)
    missing_decoder = [k for k in loaded.missing_keys
                       if k.startswith("decoder.") or k.startswith("post_quant_conv.")]
    if unexpected or missing_decoder or len(decoder_keys) < 100:
        fail(f"Stable Signature decoder conversion invalid: loaded={len(decoder_keys)}, "
             f"unexpected={unexpected[:5]}, missing_decoder={missing_decoder[:5]}")
    extractor = torch.jit.load(str(extractor_path), map_location=DEVICE).eval()
    print(f"Stable Signature assets SHA256: decoder={sha256(decoder_path)}, extractor={sha256(extractor_path)}")
    return pipe, extractor


def load_aqualora():
    asset_id = hashlib.sha256(f"{AQUA_REV}/{AQUA_FOLDER}".encode()).hexdigest()[:16]
    folder = ROOT / "assets" / f"aqualora_{asset_id}"
    folder.mkdir(parents=True, exist_ok=True)
    for filename in ("mapper.pt", "msgdecoder.pt", "pytorch_lora_weights.safetensors"):
        cached = hf_hub_download(AQUA_REPO, f"{AQUA_FOLDER}/{filename}", revision=AQUA_REV,
                                 cache_dir=str(CACHE))
        destination = folder / filename
        if not destination.exists() or sha256(destination) != sha256(cached):
            import shutil
            shutil.copy2(cached, destination)
    pipe = build_pipeline(AQUA_MODEL)
    lora = conditioned_aqualora(folder, AQUA_KEY)
    fuse_aqualora(pipe.unet, lora)
    decoder = AquaSecretDecoder(len(AQUA_KEY))
    decoder.load_state_dict(torch.load(folder / "msgdecoder.pt", map_location="cpu",
                                       weights_only=True), strict=True)
    decoder.to(device=DEVICE, dtype=torch.float32).eval()
    if not all(torch.isfinite(p).all() for p in decoder.parameters()):
        fail("AquaLoRA checkpoint contains nonfinite decoder parameters")
    pipe._wmq_assets = {"repo": AQUA_REPO, "revision": AQUA_REV, "folder": AQUA_FOLDER,
                        "lora_scale": 1.03, "key": AQUA_KEY,
                        "decoder_preprocessing": "RGB bicubic 512x512, float32 [-1,1]",
                        "sha256": {p.name: sha256(p) for p in folder.iterdir() if p.is_file()}}
    return pipe, decoder


def main():
    if RUN_PROTOCOL not in {"fair_w4a16", "legacy_grid"}:
        fail("RUN_PROTOCOL must be fair_w4a16 or legacy_grid")
    if CLEAN_POLICY not in {"strict", "report"}:
        fail("CLEAN_POLICY must be strict or report")
    if not (0 <= CLEAN_MIN_BITACC <= 1 and 0 <= CLEAN_MIN_TPR <= 1):
        fail("Clean validation thresholds must be between 0 and 1")
    if RECON_STEPS < 1 or RECON_CACHE_MB < 1 or not math.isfinite(RECON_LR) or RECON_LR <= 0:
        fail("Reconstruction steps/cache/LR must be positive and finite")
    if not 0 < FPR < 1 or STEPS < 1 or not 1 <= CALIB_TIMESTEPS <= STEPS:
        fail("Require 0<FPR<1 and 1<=CALIB_TIMESTEPS<=STEPS")
    if RUN_PROTOCOL == "fair_w4a16" and REFINE_OPTIMIZER != "gradient":
        fail("fair_w4a16 uses REFINE_OPTIMIZER=gradient; zeroth is available in legacy_grid")
    if RUN_PROTOCOL == "fair_w4a16":
        for watermark in ("stable_signature", "aqualora"):
            destination = OUT / watermark / "fair_w4a16"
            if destination.exists():
                fail(f"Results already exist at {destination}. Set WMQ_OUTPUT to a new directory "
                     "to preserve earlier experiments.")
    if GUIDANCE <= 1 or HEIGHT < 8 or WIDTH < 8:
        fail("This calibration protocol requires GUIDANCE>1 and positive image dimensions")
    if not torch.cuda.is_available():
        fail("CUDA is required, but torch.cuda.is_available() is False")
    capability = torch.cuda.get_device_capability(0)
    cuda_version = tuple(int(part) for part in (torch.version.cuda or "0.0").split(".")[:2])
    torch_version = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
    if capability >= (10, 0) and (torch_version < (2, 7) or cuda_version < (12, 8)):
        fail(f"Blackwell-class GPU detected (SM {capability[0]}.{capability[1]}), but "
             f"torch={torch.__version__}, CUDA build={torch.version.cuda}. Install a "
             "PyTorch build with Blackwell support (PyTorch >=2.7, CUDA >=12.8).")
    if diffusers.__version__ != "0.35.1":
        fail(f"This script validates AquaLoRA module mapping only for diffusers==0.35.1; "
             f"found {diffusers.__version__}")
    if len(SS_KEY) != 48 or len(AQUA_KEY) != 48 or set(SS_KEY + AQUA_KEY) - {"0", "1"}:
        fail("SS_KEY and AQUA_KEY must each be 48 binary characters")
    if HEIGHT % 8 or WIDTH % 8:
        fail("HEIGHT and WIDTH must be divisible by 8")
    if SEARCH_N < 1 or TEST_N < 1:
        fail("SEARCH_N and TEST_N must be positive")
    if REFINE_N < 1 or CALIB_N < 1 or CALIB_TIMESTEPS < 1:
        fail("REFINE_N, CALIB_N and CALIB_TIMESTEPS must be positive")
    if REFINE_MODE not in {"none", "scale", "scale_zero", "full"}:
        fail("REFINE_MODE must be one of: none, scale, scale_zero, full")
    if REFINE_OPTIMIZER not in {"gradient", "zeroth"}:
        fail("REFINE_OPTIMIZER must be gradient or zeroth")
    if min(REFINE_ITERS, MAX_PRED_NMSE, TPR_LOSS_WEIGHT, PRED_LOSS_WEIGHT) < 0:
        fail("REFINE_ITERS, MAX_PRED_NMSE and loss weights must be non-negative")
    if GRAD_STEPS < 1 or GRAD_EVAL_EVERY < 1 or GRAD_LR <= 0 or ROUND_TEMPERATURE <= 0:
        fail("Gradient steps/eval interval/LR/round temperature must be positive")
    if GRAD_PRED_WEIGHT < 0 or GRAD_IMAGE_WEIGHT < 0:
        fail("Gradient preservation-loss weights must be non-negative")
    if min(GEN_BATCH_SIZE, METRIC_BATCH_SIZE, CALIB_BATCH_SIZE) < 1:
        fail("GEN_BATCH_SIZE, METRIC_BATCH_SIZE and CALIB_BATCH_SIZE must be positive")
    torch.manual_seed(BASE_SEED)
    random.seed(BASE_SEED)
    np.random.seed(BASE_SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    print(f"GPU: {torch.cuda.get_device_name(0)} ({GPU_TOTAL_GB:.1f} GiB); "
          f"torch={torch.__version__}", flush=True)
    print(f"Execution profile: large_memory={LARGE_MEMORY_GPU}, gen_batch={GEN_BATCH_SIZE}, "
          f"metric_batch={METRIC_BATCH_SIZE}, calib_batch={CALIB_BATCH_SIZE}, "
          f"pristine_on_gpu={KEEP_PRISTINE_ON_GPU}, joint_group_grad={JOINT_GROUP_GRAD}, "
          f"attention_slicing={USE_ATTENTION_SLICING}", flush=True)
    print("NOTE: low-bit weights run through FP16 CUDA kernels; latency is not an INT4 benchmark.", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    import lpips
    lpips_model = None if BASELINE_CHECK_ONLY and RUN_PROTOCOL == "fair_w4a16" else lpips.LPIPS(net="alex").to(DEVICE).eval()
    reports = []
    runner = run_fair_comparison if RUN_PROTOCOL == "fair_w4a16" else run_adaptive_search

    pipe, extractor = load_stable_signature()
    reports.append(runner("stable_signature", pipe, "vae", extractor,
                                       decode_stable_signature, SS_KEY, lpips_model))
    del pipe, extractor
    cleanup()

    pipe, decoder = load_aqualora()
    reports.append(runner("aqualora", pipe, "unet", decoder,
                                       decode_aqualora, AQUA_KEY, lpips_model))
    del pipe, decoder
    cleanup()

    if RUN_PROTOCOL == "fair_w4a16":
        save_json(OUT / "comparison_summary.json", {"protocol": RUN_PROTOCOL,
                  "status": ("complete" if all(r["status"] == "complete" for r in reports) else "incomplete"),
                  "quality_valid": all(r.get("quality_valid", False) for r in reports),
                  "reports": reports})
        table = []
        for report in reports:
            for method in report["methods"]:
                table.append({"watermark": report["watermark"], "method": method["method"],
                              "baseline_valid": method["baseline_valid"],
                              "weight_bits": 4, "activation_bits": 16,
                              "changed_parameters": method["changed_parameters"],
                              "clean_bit_accuracy": method["clean"]["bit_accuracy"],
                              "test_bit_accuracy": method["attacked"]["bit_accuracy"],
                              "clean_tpr": method["clean"]["tpr"], "test_tpr": method["attacked"]["tpr"],
                              **method["paired_quality"],
                              "calibration_nmse": method["calibration_prediction_nmse"],
                              "feasible": method["feasible"],
                              "selected_source": method["details"].get("selected_source", method["method"])})
        if table:
            with open(OUT / "comparison.csv", "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(table[0]))
                writer.writeheader()
                writer.writerows(table)
        print(f"Comparison status saved: {OUT / 'comparison_summary.json'}", flush=True)
        return

    summary = {
        "created_unix": time.time(),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "settings": {
            "search_n": SEARCH_N, "test_n": TEST_N, "steps": STEPS,
            "guidance": GUIDANCE, "height": HEIGHT, "width": WIDTH,
            "fpr": FPR, "min_psnr": MIN_PSNR, "max_lpips": MAX_LPIPS,
            "refine_mode": REFINE_MODE, "refine_iters": REFINE_ITERS,
            "refine_optimizer": REFINE_OPTIMIZER, "grad_steps": GRAD_STEPS,
            "grad_lr": GRAD_LR, "grad_eval_every": GRAD_EVAL_EVERY,
            "round_temperature": ROUND_TEMPERATURE,
            "grad_prediction_weight": GRAD_PRED_WEIGHT,
            "grad_image_weight": GRAD_IMAGE_WEIGHT,
            "gpu_total_gib": GPU_TOTAL_GB, "large_memory_profile": LARGE_MEMORY_GPU,
            "generation_batch_size": GEN_BATCH_SIZE,
            "metric_batch_size": METRIC_BATCH_SIZE,
            "calibration_batch_size": CALIB_BATCH_SIZE,
            "keep_pristine_on_gpu": KEEP_PRISTINE_ON_GPU,
            "joint_group_gradient": JOINT_GROUP_GRAD,
            "attention_slicing": USE_ATTENTION_SLICING,
            "refine_n": REFINE_N, "calib_n": CALIB_N,
            "calib_timesteps": CALIB_TIMESTEPS, "max_prediction_nmse": MAX_PRED_NMSE,
            "tpr_loss_weight": TPR_LOSS_WEIGHT, "prediction_loss_weight": PRED_LOSS_WEIGHT,
        },
        "reports": reports,
    }
    with open(OUT / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    retention_rows = []
    for report in reports:
        retention_rows.append({
            "watermark": report["watermark"],
            "model": report["model"],
            "bits": report["recipe"]["bits"],
            "clip": report["recipe"]["clip"],
            "groups": "+".join(report["recipe"]["groups"]),
            "refine_mode": report["refinement"]["mode"],
            "refine_optimizer": report["refinement"]["optimizer"],
            "calibration_kind": (report["refinement"]["best_calibration_metrics"] or {}).get("calibration_kind"),
            "calibration_prediction_nmse": (report["refinement"]["best_calibration_metrics"] or {}).get("prediction_nmse"),
            "clean_bit_accuracy_percent": 100.0 * report["clean"]["bit_accuracy"],
            "attacked_bit_accuracy_percent": 100.0 * report["attacked"]["bit_accuracy"],
            "bit_accuracy_retained_percent": report["bit_accuracy_retained_percent"],
            "clean_tpr_percent": 100.0 * report["clean"]["tpr"],
            "attacked_tpr_percent": 100.0 * report["attacked"]["tpr"],
            "tpr_retained_percent": report["tpr_retained_percent"],
            "paired_psnr": report["paired_quality"]["psnr"],
            "paired_lpips": report["paired_quality"]["lpips"],
            "test_images": report["test_size"],
        })
    with open(OUT / "watermark_retention.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(retention_rows[0].keys()))
        writer.writeheader()
        writer.writerows(retention_rows)
    print(f"Finished. Summary: {OUT / 'summary.json'}", flush=True)
    print(f"Plot-ready table: {OUT / 'watermark_retention.csv'}", flush=True)


if __name__ == "__main__":
    main()
PY
