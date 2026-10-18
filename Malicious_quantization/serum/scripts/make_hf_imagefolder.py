#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from pathlib import Path

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()

    images = sorted(
        p for p in args.images.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXT
    )
    prompts = [
        x.rstrip("\n")
        for x in args.prompts.read_text(encoding="utf-8").splitlines()
    ]

    if len(images) != len(prompts):
        raise ValueError(
            f"Image/prompt mismatch: {len(images)} images vs {len(prompts)} prompts"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    expected_names = {f"{i:06d}{src.suffix.lower()}" for i, src in enumerate(images)}
    old_images = {p.name for p in args.out.iterdir() if p.suffix.lower() in IMG_EXT}
    if old_images - expected_names:
        raise ValueError("Output contains stale images; use a fresh output directory")
    metadata = args.out / "metadata.jsonl"

    with metadata.open("w", encoding="utf-8") as mf:
        for idx, (src, prompt) in enumerate(zip(images, prompts)):
            dst_name = f"{idx:06d}{src.suffix.lower()}"
            dst = args.out / dst_name

            if dst.exists():
                dst.unlink()

            # Prefer hard-link to avoid duplicating tens of GB.
            # Fall back to copy if source/output are on different filesystems.
            try:
                os.link(src.resolve(), dst)
            except OSError:
                shutil.copy2(src, dst)

            mf.write(
                json.dumps(
                    {"file_name": dst_name, "text": prompt},
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Wrote {len(images)} examples to {args.out}")


if __name__ == "__main__":
    main()
