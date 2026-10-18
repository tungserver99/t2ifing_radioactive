"""Predeclare final runs on new marked checkpoints/keys; keep secrets in owner orchestration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from run_blind_suite import configuration, write_json, collect
from wmq_science import audit_prompt_protocol


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(config):
    runs = [Path(p).resolve() for p in config["development_runs"]]
    if not runs:
        raise ValueError("Declare development runs before final replication")
    manifests = [p / "manifest.json" for p in runs]
    seen_keys = {json.loads((p / "owner_evaluation.json").read_text())["key_provenance"]["sha256"] for p in runs}
    seen_weights = {h for p in manifests for name, h in json.loads(p.read_text())["model_files_sha256"].items()
                    if name.replace("\\", "/").startswith("vae/") and name.endswith((".bin", ".safetensors"))}
    prompts_path = Path(config["prompts"]).resolve()
    prompts = [p.strip() for p in prompts_path.read_text(encoding="utf-8").splitlines() if p.strip()]
    # Default protocol: 32 calibration + 20 search + 100 test. Keep it explicit.
    if len(prompts) < 152:
        raise ValueError("Final protocol needs at least 152 prompts")
    audit_prompt_protocol(prompts[:152], 52, "final", manifests)
    assets, unique = [], set()
    for entry in config["assets"]:
        label = entry["id"]
        if not label or not all(c.isalnum() or c in "_-" for c in label) or label in unique:
            raise ValueError("Asset IDs must be unique simple directory names")
        unique.add(label)
        model = Path(entry["model"]).resolve()
        if not (model / "model_index.json").is_file():
            raise ValueError(f"Missing marked Diffusers pipeline: {model}")
        key = entry["key"]
        if len(key) != 48 or set(key) - {"0", "1"}:
            raise ValueError("Each asset needs its actual 48-bit watermark key")
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        weights = {p.relative_to(model).as_posix(): digest(p) for p in (model / "vae").rglob("*")
                   if p.suffix in (".bin", ".safetensors")}
        if not weights:
            raise ValueError("Missing marked VAE weights")
        if key_hash in seen_keys or seen_weights.intersection(weights.values()):
            raise ValueError("Final cross-key/checkpoint replication reuses a development key or VAE")
        seen_keys.add(key_hash)
        seen_weights.update(weights.values())
        extractor = Path(entry["extractor"]).resolve()
        if digest(extractor) != entry["extractor_sha256"]:
            raise ValueError("Owner extractor hash mismatch")
        assets.append(({"id": label, "model": str(model), "key_sha256": key_hash,
                        "vae_sha256": weights, "extractor_sha256": entry["extractor_sha256"]},
                       {"SS_KEY": key, "SS_EXTRACTOR": str(extractor), "SS_EXTRACTOR_SHA256": entry["extractor_sha256"],
                        "SS_KEY_SOURCE": f"user-supplied final replication asset {label}", "WMQ_METHOD_SET": "full"}))
    if not assets:
        raise ValueError("At least one real marked replication asset is required")
    return assets, manifests, prompts_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--plan-only", action="store_true")
    args = p.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    assets, manifests, prompts = prepare(config)
    seeds = config.get("seeds", [7301, 8301, 9301])
    if not seeds or len(set(seeds)) != len(seeds) or any(not isinstance(s, int) or s < 0 for s in seeds):
        raise ValueError("Invalid replication seeds")
    args.output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).resolve().parent / "run_blind_quantization.sh"
    options = configuration("science") + ["--evaluation-stage", "final", "--development-manifests", *map(str, manifests),
        "--prompts", str(prompts), "--budget-ssim", str(config.get("budget_ssim", .9)),
        "--budget-psnr", str(config.get("budget_psnr", 30.))]
    plan = {"assets": [a for a, _ in assets], "seeds": seeds, "options": options,
            "prompts_sha256": digest(prompts), "config_sha256": digest(args.config),
            "source_sha256": {p.name: digest(p) for p in script.parent.glob("*.py")},
            "launcher_sha256": digest(script),
            "development_manifest_sha256": {str(p): digest(p) for p in manifests},
            "all_results_reported": True, "owner_test_selection": False}
    write_json(args.output / "replication_plan.json", plan)
    if args.plan_only:
        return
    results = []
    for asset, owner_env in assets:
        for seed in seeds:
            if digest(prompts) != plan["prompts_sha256"] or digest(script) != plan["launcher_sha256"]:
                raise ValueError("Prompts or launcher changed after final plan freeze")
            for name, expected in plan["source_sha256"].items():
                if digest(script.parent / name) != expected:
                    raise ValueError(f"Source changed after final plan freeze: {name}")
            for name, expected in asset["vae_sha256"].items():
                if digest(Path(asset["model"]) / name) != expected:
                    raise ValueError("Marked VAE changed after final plan freeze")
            run = args.output / f"{asset['id']}_seed{seed}"
            command = ["bash", str(script), *options, "--model", asset["model"], "--seed", str(seed), "--output", str(run)]
            with (args.output / f"{run.name}.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, env={**os.environ, **owner_env}, stdout=log, stderr=subprocess.STDOUT)
            results.append({"asset": asset["id"], "seed": seed, "returncode": completed.returncode, "rows": collect(run)})
            write_json(args.output / "replication_results.json", results)
    if any(r["returncode"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
