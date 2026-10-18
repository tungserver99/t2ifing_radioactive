"""Create a deterministic held-out prompt list, excluding all supplied development prompts.

Compositional hand-authored pool, not a representative deployment distribution.
"""
import argparse
import itertools
import json
from pathlib import Path
import random


def make_prompts(manifests, count=152, seed=91027):
    excluded = {" ".join(p.casefold().split()) for filename in manifests
                for p in json.loads(Path(filename).read_text(encoding="utf-8"))["prompts"]}
    subjects = ["a ceramicist shaping a bowl", "a red fox resting beside a fallen log", "an antique mechanical clock",
                "a small glass greenhouse", "a violin maker at a wooden workbench", "a kingfisher above a stream",
                "a collection of translucent mineral specimens", "a cyclist carrying a basket of flowers",
                "a weathered astronomical observatory", "a wooden sailing boat with blue canvas"]
    scenes = ["under soft overcast light", "in warm late-afternoon sunlight", "illuminated by a single lantern",
              "with distant mountains in the background", "surrounded by green climbing plants",
              "beside a rain-covered window", "against a pale textured wall", "with reflections in still water",
              "near a stone archway", "under scattered autumn leaves"]
    styles = ["A detailed photograph of", "A watercolor illustration of", "A carefully lit cinematic image of",
              "An intricate colored-pencil drawing of"]
    pool = [f"{style} {subject}, {scene}." for subject, scene, style in itertools.product(subjects, scenes, styles)]
    pool = [p for p in pool if " ".join(p.casefold().split()) not in excluded]
    if count < 1 or count > len(pool):
        raise ValueError("Not enough unused prompts in the declared compositional pool")
    return random.Random(seed).sample(pool, count)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--development-manifests", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=152)
    p.add_argument("--seed", type=int, default=91027)
    args = p.parse_args()
    prompts = make_prompts(args.development_manifests, args.count, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(prompts) + "\n")
    print(f"Wrote {len(prompts)} prompts: {args.output}")


if __name__ == "__main__":
    main()
