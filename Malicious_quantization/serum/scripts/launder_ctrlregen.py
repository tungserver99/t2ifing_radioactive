#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoModel, AutoImageProcessor
from diffusers import ControlNetModel, UniPCMultistepScheduler, AutoencoderKL
from controlnet_aux import CannyDetector


IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ctrlregen-repo", type=Path, required=True)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--step", type=float, required=True)
    p.add_argument("--base-seed", type=int, default=1234)
    p.add_argument("--resume", action="store_true")

    p.add_argument(
        "--diffusion-model",
        default="SG161222/Realistic_Vision_V4.0_noVAE",
    )
    p.add_argument(
        "--image-encoder",
        default="facebook/dinov2-giant",
    )
    p.add_argument(
        "--vae",
        default="stabilityai/sd-vae-ft-mse",
    )

    p.add_argument(
        "--spatial-control",
        type=Path,
        default=None,
        help="Default: <ctrlregen-repo>/spatialnet_ckp/spatial_control_ckp_14000",
    )
    p.add_argument(
        "--semantic-control",
        type=Path,
        default=None,
        help="Default: <ctrlregen-repo>/semanticnet_ckp",
    )
    p.add_argument(
        "--semantic-weight-name",
        default="semantic_control_ckp_435000.bin",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not (0.0 < args.step <= 1.0):
        raise ValueError("--step must be in (0, 1]. Use raw images as the identity condition.")

    repo = args.ctrlregen_repo.resolve()
    sys.path.insert(0, str(repo))

    from custom_i2i_pipeline import CustomStableDiffusionControlNetImg2ImgPipeline
    from utils import color_match

    spatial_path = args.spatial_control or (
        repo / "spatialnet_ckp" / "spatial_control_ckp_14000"
    )
    semantic_path = args.semantic_control or (repo / "semanticnet_ckp")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if "cuda" in device else torch.float32

    resize_512 = transforms.Compose(
        [
            transforms.Resize(
                512,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.CenterCrop(512),
        ]
    )

    spatialnet = [
        ControlNetModel.from_pretrained(
            str(spatial_path),
            torch_dtype=dtype,
        )
    ]

    pipe = CustomStableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        args.diffusion_model,
        controlnet=spatialnet,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    pipe.costum_load_ip_adapter(
        str(semantic_path),
        subfolder="models",
        weight_name=args.semantic_weight_name,
    )

    pipe.image_encoder = AutoModel.from_pretrained(args.image_encoder).to(
        device, dtype=dtype
    )
    pipe.feature_extractor = AutoImageProcessor.from_pretrained(args.image_encoder)
    pipe.vae = AutoencoderKL.from_pretrained(args.vae).to(dtype=dtype)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_ip_adapter_scale(1.0)
    pipe.set_progress_bar_config(disable=True)
    pipe.to(device)

    processor = CannyDetector()

    files = sorted(
        p for p in args.input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXT
    )
    if not files:
        raise RuntimeError(f"No images found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "laundering_manifest.csv"

    rows = []
    previous = {}
    if args.resume and manifest_path.exists():
        with manifest_path.open(newline="", encoding="utf-8") as f:
            previous = {Path(row['output']).name: row for row in csv.DictReader(f)}

    for idx, src in enumerate(files):
        seed = args.base_seed + idx
        dst = args.output_dir / (src.stem + ".png")
        if args.resume and dst.exists():
            old = previous.get(dst.name)
            valid = False
            if old:
                assert int(old['seed']) == seed and float(old['step']) == args.step and Path(old['source']).resolve() == src.resolve(), 'Resume provenance mismatch'
                try:
                    with Image.open(dst) as im:
                        im.load()
                        valid = im.mode == 'RGB' and im.size == (512, 512)
                except (OSError, ValueError):
                    pass
            if valid:
                rows.append({'index':idx, 'source':str(src), 'output':str(dst), 'step':args.step, 'seed':seed})
                continue
            print(f'Regenerating incomplete or uncommitted image: {dst.name}', flush=True)

        input_img = Image.open(src).convert("RGB")
        input_img = resize_512(input_img)
        processed_img = processor(
            input_img,
            low_threshold=100,
            high_threshold=150,
        )

        # Match public CtrlRegen+ demo.
        torch.manual_seed(seed)

        output_img = pipe(
            "best quality, high quality",
            negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
            image=[input_img],
            control_image=[processed_img],
            ip_adapter_image=[input_img],
            strength=args.step,
            generator=torch.manual_seed(seed),
            num_inference_steps=50,
            controlnet_conditioning_scale=1.0,
            guidance_scale=2.0,
            control_guidance_start=0,
            control_guidance_end=1,
        ).images[0]

        output_img = color_match(input_img, output_img)

        # Preserve basename so prompt/image indexing remains unchanged.
        dst = args.output_dir / (src.stem + ".png")
        temporary_image = dst.with_suffix('.png.tmp')
        output_img.save(temporary_image, format='PNG')
        os.replace(temporary_image, dst)

        rows.append(
            {
                "index": idx,
                "source": str(src),
                "output": str(dst),
                "step": args.step,
                "seed": seed,
            }
        )
        # Persist provenance after each image so an interrupted job can resume.
        temporary_manifest = manifest_path.with_suffix('.tmp.csv')
        with temporary_manifest.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['index','source','output','step','seed'])
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_manifest, manifest_path)
        if (idx + 1) % 100 == 0:
            print(f'Processed {idx + 1}/{len(files)} images', flush=True)

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "source", "output", "step", "seed"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Processed {len(files)} images")
    print(f"Saved to {args.output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
