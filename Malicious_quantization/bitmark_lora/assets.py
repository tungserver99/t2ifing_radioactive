"""Download the actual upstream checkpoints, pinned by revision and LFS SHA256."""
import argparse
try:
    import fcntl
except ImportError:  # CPU plan/tests on Windows; actual server execution remains POSIX.
    from common import windows_fcntl as fcntl
import json
import random
import urllib.request
import zipfile
from pathlib import Path

from common import HERE, freeze, read_json, sha256, write_json

INFINITY_REV = "e7108923249fb3c836ebfbc3f4b4d318d6131293"
T5_REV = "7d6315df2c2fb742f0f5b556879d730926ca9001"
SD_REV = "0094d483a120f3f33dafbd187ea4aa60d10de75c"
INFINITY_HASHES = {
    "infinity_2b_reg.pth": "d5958b37e7ecb914230cd2b8c28aae0ec108ce08b315720bc3dc52596cc5b3a7",
    "infinity_vae_d32reg.pth": "7a37fa3ea1b2a1ebd23de61d91a5e68202825e5a67edaef4b7c55f5fd5b9cf26",
}
COCO_URL = "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2014.zip?download=1"
# Official COCO object's ETag is also its single-part MD5; SHA256 recorded per run.
COCO_MD5 = "0a379cfc70b0e71301e0f377548639bd"


def download(url, path):
    """Stream with visible progress; a failed download never becomes a cache hit."""
    import time
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response, temp.open("wb") as out:
                total = int(response.headers.get("Content-Length", 0))
                done, last = 0, 0
                while block := response.read(4 * 1024 * 1024):
                    out.write(block)
                    done += len(block)
                    if time.monotonic() - last > 5:
                        print(f"Downloading {path.name}: {done / 1e6:.0f}/{total / 1e6:.0f} MB", flush=True)
                        last = time.monotonic()
                if total and done != total:
                    raise IOError(f"Incomplete download: {done}/{total}")
            temp.replace(path)
            return
        except Exception as exc:
            if attempt == 2:
                raise
            print(f"Download interrupted ({exc}); retry {attempt+2}/3: {path.name}", flush=True)
            time.sleep(2)


def split_prompts(annotations, seed, counts):
    # One caption per COCO image, deduplicated text across all splits.
    rows, seen_ids, seen_text = [], set(), set()
    for a in sorted(annotations, key=lambda x: (x["image_id"], x["id"])):
        text = " ".join(a["caption"].split())
        if a["image_id"] in seen_ids or text in seen_text or not text:
            continue
        seen_ids.add(a["image_id"])
        seen_text.add(text)
        rows.append({"image_id": a["image_id"], "caption_id": a["id"], "text": text})
    random.Random(seed).shuffle(rows)
    if sum(counts.values()) > len(rows):
        raise ValueError("Requested more disjoint prompts than COCO2014 validation contains")
    result, start = {}, 0
    for split, n in counts.items():
        result[split] = rows[start:start+n]
        start += n
    return result


def hub_snapshot(repo, revision, cache, keep):
    from huggingface_hub import HfApi, hf_hub_download
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise RuntimeError(f"Hub revision changed: {repo}")
    selected = [s for s in info.siblings if keep(s.rfilename)]
    if not any(s.rfilename.endswith((".safetensors", ".pth", ".bin")) for s in selected):
        raise RuntimeError(f"No checkpoint found for {repo}; no fallback model will be substituted")
    records = []
    folder = cache / repo.replace("/", "--") / revision
    for s in selected:
        print(f"Checkpoint {repo}: download/verify {s.rfilename}", flush=True)
        p = Path(hf_hub_download(repo, s.rfilename, revision=revision, local_dir=folder))
        digest = sha256(p)
        expected = s.lfs.sha256 if s.lfs else None
        if repo == "FoundationVision/Infinity":
            expected = INFINITY_HASHES[s.rfilename]
        if expected and digest != expected:
            raise RuntimeError(f"Checkpoint checksum mismatch, refuse to load: {p}")
        records.append({"file": s.rfilename, "sha256": digest, "expected_lfs_sha256": expected})
    write_json(folder / "verified.json", {"repository": repo, "revision": revision, "files": records})
    return str(folder)


def prepare_prompts(config, run, cache):
    import hashlib
    archive = cache / "annotations_trainval2014.zip"
    if not archive.exists():
        download(COCO_URL, archive)
    if hashlib.md5(archive.read_bytes()).hexdigest() != COCO_MD5:
        raise RuntimeError(f"COCO annotation checksum mismatch: {archive}")
    with zipfile.ZipFile(archive) as z:
        anns = json.loads(z.read("annotations/captions_val2014.json"))["annotations"]
    counts = {"train": config["train_n"], "calibration": config["calibration_n"], "test": config["test_n"]}
    if config.get("validation_n"):
        counts["validation"] = config["validation_n"]
    prompts = split_prompts(anns, config["seed"], counts)
    freeze(run / "prompts.json", prompts)
    print("Disjoint COCO splits: " + ", ".join(f"{k}={len(v)}" for k, v in prompts.items()), flush=True)
    return {"coco_url": COCO_URL, "coco_sha256": sha256(archive)}


def prepare(config, run):
    cache = Path(config["cache"])
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / "assets.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = prepare_prompts(config, run, cache)
        result["infinity"] = hub_snapshot("FoundationVision/Infinity", INFINITY_REV, cache,
                                          lambda n: n in INFINITY_HASHES)
        result["t5"] = hub_snapshot("google/flan-t5-xl", T5_REV, cache,
            lambda n: "/" not in n and (n.endswith(".json") or n == "spiece.model" or n.endswith(".safetensors")))
        # The original SD2.1 model was removed; this is the mirror explicitly linked by BitMark's authors.
        result["student"] = hub_snapshot("Manojb/stable-diffusion-2-1-base", SD_REV, cache,
            lambda n: n == "model_index.json" or (n.split('/')[0] in {
                "scheduler", "tokenizer", "text_encoder", "unet", "vae", "feature_extractor"} and
                n.endswith((".json", ".txt", ".safetensors")) and ".fp16." not in n))
        write_json(run / "assets.json", result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    prepare(read_json(a.run / "config.json"), a.run)
