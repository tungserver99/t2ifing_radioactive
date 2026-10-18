#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base-model",
        default="CompVis/stable-diffusion-v1-4",
    )
    adapter = p.add_mutually_exclusive_group(required=True)
    adapter.add_argument("--unet", type=Path)
    adapter.add_argument("--unadapted", action="store_true")
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--base-seed", type=int, default=50000)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    prompts = [
        x.rstrip("\n")
        for x in args.prompts.read_text(encoding="utf-8").splitlines()
    ]

    args.out.mkdir(parents=True, exist_ok=True)

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)

    if args.unet is not None:
        pipe.unet = UNet2DConditionModel.from_pretrained(str(args.unet), subfolder='unet', torch_dtype=dtype).to(device).eval()
    manifest = {
        "base_model": args.base_model,
        "unet": str(args.unet.resolve()) if args.unet else None,
        "num_images": len(prompts),
        "prompt_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
        "base_seed": args.base_seed,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "scheduler": type(pipe.scheduler).__name__,
        "batch_size": args.batch_size,
    }
    manifest_path = args.out / "generation_manifest.json"
    if args.resume and manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, 'Generation resume settings changed'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    pipe.set_progress_bar_config(disable=False)

    for start in range(0, len(prompts), args.batch_size):
        batch = prompts[start : start + args.batch_size]
        if args.resume and all((args.out / f'sample_{start+j:06d}.png').exists() for j in range(len(batch))):
            from PIL import Image
            valid = True
            for j in range(len(batch)):
                try:
                    with Image.open(args.out / f'sample_{start+j:06d}.png') as im:
                        im.load()
                        valid = valid and im.mode == 'RGB' and im.size == (512, 512)
                except (OSError, ValueError):
                    valid = False
            if valid:
                continue

        generators = [
            torch.Generator(device=device).manual_seed(args.base_seed + start + j)
            for j in range(len(batch))
        ]

        images = pipe(
            batch,
            negative_prompt=[""] * len(batch),
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generators,
            height=512,
            width=512,
        ).images

        for j, image in enumerate(images):
            idx = start + j
            dst = args.out / f"sample_{idx:06d}.png"
            tmp = dst.with_suffix('.png.tmp')
            image.save(tmp, format='PNG')
            os.replace(tmp, dst)


if __name__ == "__main__":
    main()
