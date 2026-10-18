"""Utility audit after frozen training; never an input to checkpoint selection."""
import argparse
from pathlib import Path

from common import read_json, write_json, image_name, sha256
from student import branches

CLIP_REV = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


def main(run, split="test", labels=None):
    import torch
    import torch_fidelity
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    from torch_fidelity.datasets import ImagesPathDataset
    c = read_json(run / "config.json")
    prompts = read_json(run / "prompts.json")[split]
    if split == "validation":
        prompts = prompts[:c["validation_generation_n"]]
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", revision=CLIP_REV).cuda().eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", revision=CLIP_REV)
    labels = labels or ["teacher_wm_test", "unadapted"] + [b["name"] for b in branches(c)]
    ref = run / "images" / f"teacher_clean_{split}"
    reference_files = [str(ref / image_name(i)) for i in range(len(prompts))]
    ref_dataset = ImagesPathDataset(reference_files)
    output = run / "result" / ("validation_utility.json" if split == "validation" else "utility.json")
    report = {"reference": str(ref), "clip_revision": CLIP_REV, "n": len(prompts),
              "clip_definition": "mean 2.5 * max(cos(image,text),0), same as SERUM utility script",
              "fid_warning": "Finite-sample FID is biased; compare only identical counts/reference. KID included.", "kid_note": "Subset standard deviation is not a training-seed confidence interval; tiny samples remain exploratory.",
              "branches": {}}
    if output.exists():
        old = read_json(output)
        if old["reference"] != str(ref) or old["n"] != len(prompts):
            raise ValueError("Utility reference/count changed")
        report["branches"] = old["branches"]
    for label in labels:
        folder = run / "images" / label
        values = []
        files = sorted(folder.glob("sample_*.png"))
        if len(files) != len(prompts):
            raise ValueError(f"Incomplete utility images: {folder}")
        identity = {"generated": [sha256(f) for f in files], "reference": [sha256(f) for f in reference_files]}
        if report["branches"].get(label, {}).get("image_hashes") == identity:
            print(f"Reusing verified utility: {label}", flush=True)
            continue
        for start in range(0, len(files), 16):
            images = [Image.open(p).convert("RGB") for p in files[start:start+16]]
            inputs = processor(text=[x["text"] for x in prompts[start:start+16]], images=images,
                               return_tensors="pt", padding=True, truncation=True).to("cuda")
            with torch.inference_mode():
                outputs = model(**inputs)
                value = (outputs.image_embeds * outputs.text_embeds).sum(-1).clamp_min(0) * 2.5
            values += value.cpu().tolist()
        print(f"Utility {label}: CLIP done; computing FID/KID", flush=True)
        metrics = torch_fidelity.calculate_metrics(input1=str(folder), input2=ref_dataset, cuda=True,
            isc=False, fid=True, kid=len(files) >= 2, kid_subsets=100,
            kid_subset_size=min(1000, max(2, len(files)//2)), rng_seed=c["seed"], verbose=True)
        report["branches"][label] = {"clip_score": sum(values)/len(values), "clip_per_image": values, "image_hashes": identity, **metrics}
        write_json(output, report)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--split", choices=["test", "validation"], default="test")
    p.add_argument("--labels", nargs="+")
    a = p.parse_args()
    main(a.run.resolve(), a.split, a.labels)
