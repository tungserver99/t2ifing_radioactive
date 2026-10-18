"""Thin adapter around pinned official BitMark generation AND detection.

No reimplementation of the watermark, visual tokenizer, or detector.
The broken upstream comprehensive_infer CLI is deliberately not used.
"""
import argparse
import contextlib
import io
import os
import sys
from pathlib import Path

from common import HERE, freeze, image_name, read_json, sha256, verify_vendor, write_json
from student import training_conditions


def official_args(assets, delta=2):
    import architecture_wrapper as official
    parser = argparse.ArgumentParser()
    official.add_common_arguments(parser)
    official.add_infinity_arguments(parser)
    return parser.parse_args([
        "--architecture", "infinity", "--pn", "1M", "--vae_type", "32",
        "--model_path", str(Path(assets["infinity"]) / "infinity_2b_reg.pth"),
        "--vae_path", str(Path(assets["infinity"]) / "infinity_vae_d32reg.pth"),
        "--text_encoder_ckpt", assets["t5"], "--model_type", "infinity_2b",
        "--add_lvl_embeding_only_first_block", "1", "--cfg", "4",
        "--watermark_scales", "2", "--watermark_context_width", "2",
        "--watermark_delta", str(delta), "--set", "01,10", "--enable_model_cache", "0",
    ])


def initialize(run, generator=False):
    import torch
    verify_vendor()
    source = HERE / "vendor" / "bitmark"
    # Upstream uses cwd-relative Infinity imports; isolate them in this subprocess.
    os.chdir(source)
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / "Infinity"))
    from architecture_wrapper import get_vae, get_architecture
    import detect_watermark
    assets = read_json(run / "assets.json")
    args = official_args(assets)
    vae = get_vae(args)
    vae.vae.eval().requires_grad_(False)
    model = None
    if generator:
        # Upstream load_tokenizer ignores its t5_path argument. Supply the identical
        # T5 classes against our pinned local snapshot, without modifying watermark code.
        import tools.run_infinity as upstream
        from transformers import AutoTokenizer, T5EncoderModel
        def pinned_t5(t5_path):
            tok = AutoTokenizer.from_pretrained(t5_path, local_files_only=True, legacy=True)
            tok.model_max_length = 512
            enc = T5EncoderModel.from_pretrained(t5_path, local_files_only=True,
                                                torch_dtype=torch.float16).cuda().eval()
            enc.requires_grad_(False)
            return tok, enc
        upstream.load_tokenizer = pinned_t5
        model = get_architecture(args, vae)
    return args, vae, model, detect_watermark


def valid_image(path, size):
    from PIL import Image
    if not path.exists():
        return False
    with Image.open(path) as im:
        im.load()
        if im.size != size or im.mode != "RGB":
            raise ValueError(f"Wrong image shape/mode: {path}")
    return True


def generate(run):
    import numpy as np
    import random
    import torch
    from PIL import Image
    args, vae, model, official = initialize(run, generator=True)
    config, prompts = read_json(run / "config.json"), read_json(run / "prompts.json")
    # Reset identical per-image seeds; delta=0 uses the same official sequential sampler.
    for split in ("train", "calibration", "test", *(["validation"] if "validation" in prompts else [])):
        conditions = (training_conditions(config) if split == "train" else
                      ("clean",) if split == "calibration" else ("clean", "wm"))
        for condition in conditions:
            out = run / "images" / f"teacher_{condition}_{split}"
            out.mkdir(parents=True, exist_ok=True)
            args.watermark_delta = 0.0 if condition == "clean" else config.get("watermark_delta", 2.0)
            watermark = official.WatermarkInference(args, vae)
            manifest = {"delta": args.watermark_delta, "all_scales": True,
                        "green_set": ["01", "10"], "prompts_sha256": sha256(run / "prompts.json"),
                        "seed": config["seed"], "resolution": [vae.tgt_h, vae.tgt_w]}
            freeze(out / "generation.json", manifest)
            entries = []
            for i, row in enumerate(prompts[split]):
                path = out / image_name(i)
                if not valid_image(path, (vae.tgt_w, vae.tgt_h)):
                    seed = config["seed"] * 1000000 + row["image_id"]
                    random.seed(seed)
                    np.random.seed(seed % 2**32)
                    torch.manual_seed(seed)
                    model.args.seed = seed
                    with torch.inference_mode():
                        _, _, image = model.gen_img([row["text"]], vae.vae, watermark)
                    arr = image.detach().cpu().numpy()
                    if arr.ndim == 4:
                        arr = arr[0]
                    if arr.dtype != np.uint8 or arr.shape != (vae.tgt_h, vae.tgt_w, 3):
                        raise RuntimeError(f"Official Infinity output contract changed: {arr.shape}, {arr.dtype}")
                    temp = path.with_suffix(".partial.png")
                    Image.fromarray(arr).save(temp)
                    temp.replace(path)
                entries.append({"file_name": path.name, "text": row["text"], "sha256": sha256(path)})
                if (i + 1) % 10 == 0 or i + 1 == len(prompts[split]):
                    print(f"Teacher {condition}/{split}: {i+1}/{len(prompts[split])}", flush=True)
            # HF imagefolder metadata: captions only; no detector output enters training.
            with (out / "metadata.jsonl").open("w") as f:
                import json
                for entry in entries:
                    f.write(json.dumps({k: entry[k] for k in ("file_name", "text")}) + "\n")
            freeze(out / "image_hashes.json", entries)


def score(run, directories):
    import torch
    args, vae, _, official = initialize(run)
    detector = official.get_detector(args)
    for directory in directories:
        path = run / "images" / directory
        files = sorted(path.glob("sample_*.png"))
        if not files:
            raise ValueError(f"No images to score: {path}")
        out = run / "scores" / f"{directory}.json"
        prior = read_json(out) if out.exists() else []
        old = {x["file"]: x for x in prior}
        result = []
        for i, file in enumerate(files):
            digest = sha256(file)
            if file.name in old and old[file.name]["sha256"] == digest:
                row = old[file.name]
            else:
                with torch.inference_mode(), contextlib.redirect_stdout(io.StringIO()):
                    metrics = official.detect(args, str(file), detector, vae,
                                              watermark_scales=vae.watermark_scales)
                row = {"file": file.name, "sha256": digest,
                       "z_score": float(metrics["z_score"]), "green_fraction": float(metrics["green_fraction"])}
            result.append(row)
            if (i + 1) % 25 == 0 or i + 1 == len(files):
                write_json(out, result)
                print(f"Official detector {directory}: {i+1}/{len(files)}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--stage", choices=["generate", "score"], required=True)
    p.add_argument("--directories", nargs="+")
    a = p.parse_args()
    run = a.run.resolve()
    if a.stage == "generate":
        generate(run)
    else:
        if not a.directories:
            p.error("--directories required for score")
        score(run, a.directories)
