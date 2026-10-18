"""Run upstream Diffusers trainers and audit saved adapters before inference."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from common import HERE, freeze, image_name, read_json, sha256, verify_vendor, write_json


def training_conditions(config):
    if config.get("watermark_method") == "new_watermark":
        return tuple(dict.fromkeys(b["condition"] for b in branches(config)))
    # Missing field denotes the original protocol, retained for analysis of old runs.
    return ("clean", "wm") if config.get("train_clean_controls", True) else ("wm",)


def branches(config):
    if config.get("watermark_method") == "new_watermark":
        result = config["student_branches"]
        names = [b["name"] for b in result]
        if not result or len(set(names)) != len(names):
            raise ValueError("Empty/duplicate carrier student branches")
        for b in result:
            if (not b["name"].replace("_", "").isalnum() or b["kind"] != "lora"
                    or b["condition"] not in {"clean", "wm", "fixed"}
                    or b["scope"] not in {"attention", "attention_conv"} or b["rank"] < 1):
                raise ValueError("Invalid carrier student branch")
        return result
    result = []
    for scope in config.get("adapter_scopes", ["attention"]):
        for rank in config["ranks"]:
            for condition in training_conditions(config):
                prefix = f"lora_r{rank}" + ("_conv" if scope == "attention_conv" else "")
                result.append({"name": f"{prefix}_{condition}", "kind": "lora", "rank": rank,
                               "scope": scope, "condition": condition})
    if config.get("dora", False):
        rank = max(config["ranks"])
        for condition in training_conditions(config):
            result.append({"name": f"dora_r{rank}_conv_{condition}", "kind": "lora", "rank": rank,
                           "scope": "attention_conv", "condition": condition, "use_dora": True})
    if not config["lora_only"]:
        for condition in training_conditions(config):
            result.append({"name": f"full_{condition}", "kind": "full", "condition": condition})
    if config["paper_full_ema"]:
        result.append({"name": "paper_full_wm_ema", "kind": "full", "condition": "wm", "ema": True})
    return result


def training_command(config, assets, run, branch):
    trainer = "train_text_to_image_lora.py" if branch["kind"] == "lora" else "train_text_to_image.py"
    cmd = [sys.executable, str(HERE / "vendor" / "diffusers" / trainer),
           "--pretrained_model_name_or_path", assets["student"],
           "--train_data_dir", str(run / "images" / f"teacher_{branch['condition']}_train"),
           "--output_dir", str(run / "students" / branch["name"]),
           "--resolution", "512", "--center_crop", "--train_batch_size", str(config["batch_size"]),
           "--gradient_accumulation_steps", "1", "--gradient_checkpointing",
           "--num_train_epochs", str(config.get("epochs") or 5), "--learning_rate", str(config.get("full_learning_rate", config["learning_rate"]) if branch["kind"] == "full" else config["learning_rate"]),
           "--mixed_precision", config["precision"], "--lr_scheduler", "constant", "--lr_warmup_steps", "0",
           "--checkpointing_steps", "1000000000", "--report_to", "none", "--seed", str(config["seed"]),
           "--dataloader_num_workers", "0", "--max_grad_norm", "1.0"]
    if branch["kind"] == "lora":
        cmd += ["--rank", str(branch["rank"]), "--adapter_scope", branch.get("scope", "attention")]
    if branch.get("use_dora"):
        cmd += ["--use_dora"]
    if config.get("max_train_steps"):
        cmd += ["--max_train_steps", str(config["max_train_steps"])]
    if config.get("checkpoint_steps"):
        cmd += ["--snapshot_steps", *map(str, config["checkpoint_steps"])]
    if branch.get("ema"):
        cmd += ["--use_ema"]
    return cmd


def checkpoint_hashes(directory):
    files = sorted(directory.rglob("*.safetensors"))
    configs = sorted(directory.rglob("adapter_config.json"))
    if not files:
        raise RuntimeError(f"Training returned without a checkpoint: {directory}")
    return {str(p.relative_to(directory)): sha256(p) for p in files + configs}


def train(run, name):
    import torch
    from safetensors.torch import load_file
    verify_vendor()
    config, assets = read_json(run / "config.json"), read_json(run / "assets.json")
    branch = next(b for b in branches(config) if b["name"] == name)
    out = run / "students" / name
    out.mkdir(parents=True, exist_ok=True)
    expected = read_json(run / "images" / f"teacher_{branch['condition']}_train" / "image_hashes.json")
    if len(expected) != config["train_n"]:
        raise RuntimeError("Training dataset incomplete")
    for entry in expected:
        path = run / "images" / f"teacher_{branch['condition']}_train" / entry["file_name"]
        if sha256(path) != entry["sha256"]:
            raise RuntimeError(f"Training image changed: {path}")
    command = training_command(config, assets, run, branch)
    freeze(out / "protocol.json", {"command": command, "branch": branch, "training_images": expected,
                                   "selection": ("minimum held-out denoising MSE; detector and utility excluded" if config.get("checkpoint_steps")
                                                 else "fixed final epoch; no owner scores or test samples used")})
    if (out / "complete.json").exists():
        if read_json(out / "complete.json")["checkpoints"] != checkpoint_hashes(out):
            raise RuntimeError(f"Saved checkpoint changed: {out}")
        print(f"Reusing verified trained student: {name}", flush=True)
        return
    # A partial run restarts deterministically. No silent selection of an earlier checkpoint.
    subprocess.run(command, check=True, cwd=HERE)
    checkpoint = out / "adapter_model.safetensors" if branch["kind"] == "lora" else out / "unet/diffusion_pytorch_model.safetensors"
    tensors = load_file(str(checkpoint))
    if not all(torch.isfinite(t).all().item() for t in tensors.values()):
        raise RuntimeError(f"Nonfinite trained weights: {checkpoint}")
    if branch["kind"] == "lora":
        learned = [v for k, v in tensors.items() if "lora_B" in k or "lora.up" in k]
        if not learned or not any(t.abs().max().item() > 0 for t in learned):
            raise RuntimeError("LoRA up-projections remained zero; no adaptation occurred")
    else:
        initial = Path(assets["student"]) / "unet/diffusion_pytorch_model.safetensors"
        initial_tensors = load_file(str(initial))
        if not any(not torch.equal(v, initial_tensors[k]) for k, v in tensors.items()):
            raise RuntimeError("Full UNet did not change during training")
    for step in config.get("checkpoint_steps", []):
        checkpoint_hashes(out / f"snapshot-{step}")
    audit_path = out / "optimization_audit.json"
    audit = read_json(audit_path) if audit_path.exists() else None
    write_json(out / "complete.json", {"checkpoints": checkpoint_hashes(out), "finite_weights": True,
                                       "weights_changed_verified": True,
                                       "optimizer_update_counts": audit})


def load_adapter(unet, directory, pipe=None):
    from peft import LoraConfig, set_peft_model_state_dict
    from safetensors.torch import load_file
    directory = Path(directory)
    native = directory / "adapter_model.safetensors"
    if not native.exists():
        if pipe is None:
            raise RuntimeError("Legacy adapter needs Diffusers pipeline loader")
        pipe.load_lora_weights(str(directory))
        return
    config = LoraConfig.from_pretrained(str(directory))
    unet.add_adapter(config)
    state = load_file(str(native))
    from peft import get_peft_model_state_dict
    expected = get_peft_model_state_dict(unet)
    if set(state) != set(expected) or any(state[k].shape != expected[k].shape for k in state):
        raise RuntimeError("Adapter tensor keys/shapes do not match its declared targets")
    result = set_peft_model_state_dict(unet, state)
    missing = [k for k in result.missing_keys if "lora_" in k]
    if result.unexpected_keys or missing:
        raise RuntimeError(f"Adapter did not load completely: {result.unexpected_keys}, {missing}")


def load_pipeline(run, name, checkpoint_dir=None, device="cuda", dtype=None):
    import torch
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel
    config, assets = read_json(run / "config.json"), read_json(run / "assets.json")
    dtype = dtype or (torch.float16 if device == "cuda" else torch.float32)
    pipe = StableDiffusionPipeline.from_pretrained(assets["student"], torch_dtype=dtype,
                                                  safety_checker=None, local_files_only=True)
    checkpoint = None
    if name != "unadapted":
        branch = next(b for b in branches(config) if b["name"] == name)
        root = run / "students" / name
        recorded = read_json(root / "complete.json")["checkpoints"]
        if checkpoint_dir is not None:
            directory = Path(checkpoint_dir)
        elif config.get("checkpoint_steps"):
            selected = read_json(run / "result" / "selection" / f"{name}.json")
            directory = root / f"snapshot-{selected['selected_step']}"
            if checkpoint_hashes(directory) != selected["checkpoint_hashes"]:
                raise RuntimeError(f"Selected weights changed: {directory}")
        else:
            directory = root
        checkpoint = checkpoint_hashes(directory)
        for relative, digest in checkpoint.items():
            key = str((directory / relative).relative_to(root))
            if recorded.get(key) != digest:
                raise RuntimeError(f"Changed checkpoint before generation: {directory / relative}")
        if branch["kind"] == "lora":
            load_adapter(pipe.unet, directory, pipe)
        else:
            pipe.unet = UNet2DConditionModel.from_pretrained(directory / "unet", torch_dtype=dtype,
                                                            local_files_only=True)
    pipe.to(device=device, dtype=dtype)
    pipe.unet.eval()
    pipe.vae.eval()
    pipe.text_encoder.eval()
    pipe.set_progress_bar_config(disable=True)
    return pipe, checkpoint


def generate_images(run, name, pipe, checkpoint, split="test", label=None):
    import torch
    from teacher import valid_image
    config = read_json(run / "config.json")
    prompts = read_json(run / "prompts.json")[split]
    if split == "validation":
        prompts = prompts[:config["validation_generation_n"]]
    label = label or name
    out = run / "images" / label
    out.mkdir(parents=True, exist_ok=True)
    freeze(out / "generation.json", {"checkpoint": checkpoint, "prompts_sha256": sha256(run / "prompts.json"),
           "seed": config["seed"], "split": split, "steps": config["inference_steps"], "cfg": 7.5,
           "resolution": 512, "batch_size": config["inference_batch"], "scheduler": type(pipe.scheduler).__name__,
           "dtype": "float16", "device": "cuda"})
    for start in range(0, len(prompts), config["inference_batch"]):
        batch = prompts[start:start+config["inference_batch"]]
        paths = [out / image_name(start+j) for j in range(len(batch))]
        if not all(valid_image(p, (512, 512)) for p in paths):
            generators = [torch.Generator("cuda").manual_seed(config["seed"] * 1000000 +
                          (batch[j]["image_id"] if config.get("watermark_method") == "new_watermark" else
                           {"test":500000,"validation":700000,"calibration":850000}[split] + start+j))
                          for j in range(len(batch))]
            with torch.inference_mode():
                if config.get("watermark_method") == "semantic_fp":
                    # Same ordinary sampling and original VAE conversion; fail on
                    # invalid outputs before PIL can hide a nonfinite decoder result.
                    from semantic_teacher import native_generate
                    images = native_generate(pipe, [r["text"] for r in batch], generators, config["inference_steps"])
                else:
                    images = pipe([r["text"] for r in batch], height=512, width=512, generator=generators,
                                  num_inference_steps=config["inference_steps"], guidance_scale=7.5).images
            for image, path in zip(images, paths):
                temp = path.with_suffix(".partial.png")
                image.save(temp)
                temp.replace(path)
        print(f"Student {label}: {start+len(batch)}/{len(prompts)}", flush=True)
    freeze(out / "image_hashes.json", [{"file": p.name, "sha256": sha256(p)} for p in sorted(out.glob("sample_*.png"))])


def generate(run, name, split="test"):
    if split == "calibration" and name != "unadapted":
        raise ValueError("Calibration generations must use the unadapted clean student")
    pipe, checkpoint = load_pipeline(run, name)
    generate_images(run, name, pipe, checkpoint, split=split,
                    label="unadapted_calibration" if split == "calibration" else name)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--stage", choices=["train", "generate"], required=True)
    p.add_argument("--branch", required=True)
    p.add_argument("--split", choices=["test", "calibration"], default="test")
    a = p.parse_args()
    if a.stage == "train":
        train(a.run.resolve(), a.branch)
    else:
        generate(a.run.resolve(), a.branch, a.split)
