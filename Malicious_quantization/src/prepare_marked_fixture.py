"""Experiment organizer only: export public Stable Signature model for attacker.

Model assembly happens outside the marked-model-only attacker process. No owner
extractor or clean decoder is written into the exported model directory.
"""
import argparse
import json
from pathlib import Path
import tempfile
import urllib.request
from wmq_fixture_utils import fixture_lock, publish_fixture, verify_sha256

# Pinned from 198,009,953 bytes fetched from the official Meta URL on 2026-09-15.
# This records the audited release bytes, not a publisher-provided signature.
EXPECTED_DECODER_SHA256 = "36f3a926ba080a0fe29952cd3895661736da68619a71c8166d5c44de3612965a"


def fixture_complete(root):
    """Reject partial copies that have provenance but cannot load as a pipeline."""
    root = Path(root)
    required = [root / "fixture_provenance.json", root / "model_index.json",
                root / "scheduler" / "scheduler_config.json", root / "tokenizer",
                root / "text_encoder" / "config.json", root / "unet" / "config.json",
                root / "vae" / "config.json"]
    if not all(path.exists() for path in required):
        return False
    if not any(path.is_file() for path in (root / "tokenizer").rglob("*")):
        return False
    for component in ("text_encoder", "unet", "vae"):
        folder = root / component
        if not any(p.is_file() and p.suffix in (".safetensors", ".bin") for p in folder.rglob("*")):
            return False
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--backbone", default="sd2-community/stable-diffusion-2-1-base")
    p.add_argument("--revision", default="main")
    p.add_argument("--reuse", action="store_true", help="Wait for concurrent preparation and reuse a completed fixture")
    args = p.parse_args()
    out = Path(args.output).resolve()
    with fixture_lock(out):
        if out.exists():
            if not args.reuse:
                raise ValueError("Output already exists")
            marker = out / "fixture_provenance.json"
            if not fixture_complete(out):
                raise ValueError(f"Incomplete fixture at {out}; repair it or choose a new model output directory")
            meta = json.loads(marker.read_text(encoding="utf-8"))
            if (meta.get("sha256") != EXPECTED_DECODER_SHA256 or meta.get("backbone") != args.backbone
                    or meta.get("revision_requested") != args.revision):
                raise ValueError("Existing fixture provenance does not match pinned decoder/backbone settings")
            print(f"Reusing marked fixture: {out}", flush=True)
            return
        prepare(args, out)


def prepare(args, out):
    import torch
    from diffusers import StableDiffusionPipeline
    from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
    # CPU FP32 assembly preserves the public marked checkpoint before inference.
    pipe = StableDiffusionPipeline.from_pretrained(args.backbone, revision=args.revision,
               torch_dtype=torch.float32, safety_checker=None, requires_safety_checker=False)
    url = "https://dl.fbaipublicfiles.com/ssl_watermarking/sd2_decoder.pth"
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "marked_decoder.pth"
        urllib.request.urlretrieve(url, path)
        digest = verify_sha256(path, EXPECTED_DECODER_SHA256)
        state = torch.load(path, map_location="cpu", weights_only=True)
    converted = convert_ldm_vae_checkpoint(state, dict(pipe.vae.config))
    marked = {k: v for k, v in converted.items() if k.startswith(("decoder.", "post_quant_conv."))}
    result = pipe.vae.load_state_dict(marked, strict=False)
    if (len(marked) < 100 or result.unexpected_keys or
            any(k.startswith(("decoder.", "post_quant_conv.")) for k in result.missing_keys)):
        raise ValueError("Incomplete marked VAE conversion")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a fully written fixture; interrupted exports do not look ready.
    with tempfile.TemporaryDirectory(prefix=".marked_sd21_", dir=out.parent) as staging:
        ready = Path(staging) / "model"
        pipe.save_pretrained(ready, safe_serialization=True)
        (ready / "fixture_provenance.json").write_text(json.dumps({"backbone": args.backbone,
             "revision_requested": args.revision, "marked_decoder_url": url, "sha256": digest,
             "role": "organizer; fixture construction is outside attacker access"}, indent=2), encoding="utf-8")
        publish_fixture(ready, out)
    print(f"Marked model fixture saved: {out}")


if __name__ == "__main__":
    main()
