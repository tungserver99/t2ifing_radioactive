"""Check existing model imports; never install packages or download models."""
import importlib
import importlib.metadata
import sys
import traceback

import common  # Configure PyTorch backend before any ML imports.


def check_model_imports():
    # Direct import exposes the original dependency exception instead of only
    # Transformers' top-level lazy-export ImportError.
    from transformers.models.t5.modeling_t5 import T5EncoderModel
    from transformers import AutoTokenizer, T5EncoderModel as ExportedT5, CLIPModel, CLIPProcessor
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel
    from peft import LoraConfig
    if T5EncoderModel is not ExportedT5:
        raise RuntimeError("Transformers T5 exports disagree; use a consistent existing environment")


def main():
    print(f"Python: {sys.executable}\nVersion: {sys.version}", flush=True)
    for package in ('torch', 'transformers', 'diffusers', 'peft', 'flash-attn'):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = 'NOT AVAILABLE'
        print(f'{package}: {version}', flush=True)
    try:
        check_model_imports()
    except Exception:
        traceback.print_exc()
        print('FAIL: existing environment cannot import required models. No packages installed.', flush=True)
        return 1
    print('PASS: T5, CLIP, SD and LoRA imports. GPU execution is a separate check.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
