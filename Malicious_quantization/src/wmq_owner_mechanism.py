"""Post-freeze owner diagnostics. Never imported by the attacker or used to select weights."""
import csv
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from wmq_runtime import batches, decoded01


def ownership_score(logits, key):
    """Smooth two-tail agreement; lower approaches random agreement, not bit inversion.

    This is a diagnostic surrogate, not the calibrated discrete detector statistic.
    """
    if logits.ndim != 2 or logits.shape[1] != key.numel() or not torch.isfinite(logits).all():
        raise ValueError("Invalid mechanism extractor logits")
    agreement = (torch.tanh(logits / 2) * (key.float() * 2 - 1)).mean(1)
    return agreement.square().mean()


def geometry(error, grad):
    e, g = error.detach().double(), grad.detach().double()
    dot = (e * g).sum().item()
    en, gn = e.norm().item(), g.norm().item()
    return dot, (dot / (en * gn) if en > 0 and gn > 0 else None), gn


def layer_diagnostics(vae, net, key, latent, reference, errors, subspace=None):
    """Gradient at quantized endpoint. No optimizer and no parameter writes."""
    names = list(errors)
    parameters = [vae.decoder.get_parameter(name) for name in names]
    previous = [p.requires_grad for p in parameters]
    try:
        for p in parameters:
            p.requires_grad_(True)
        x = decoded01(vae.decode(latent, return_dict=False)[0])
        mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
        std = x.new_tensor([.229, .224, .225])[None, :, None, None]
        own = ownership_score(net((x - mean) / std), key)
        quality = F.mse_loss(x, reference)
        projection = {"owner_gradient_subspace_fraction": None, "quality_gradient_subspace_fraction": None}
        if subspace is not None:
            from wmq_residual import patches
            for label, objective in (("owner", own), ("quality", quality)):
                gradient = torch.autograd.grad(objective, x, retain_graph=True)[0]
                p = patches(gradient, subspace.size)
                denominator = p.double().square().sum()
                numerator = (p @ subspace.basis.T).double().square().sum()
                projection[label + "_gradient_subspace_fraction"] = float(numerator / denominator) if denominator > 0 else None
        go = torch.autograd.grad(own, parameters, retain_graph=True)
        gq = torch.autograd.grad(quality, parameters)
        rows = []
        for name, a, b in zip(names, go, gq):
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError("Nonfinite diagnostic gradient")
            od, oc, on = geometry(errors[name], a)
            qd, qc, qn = geometry(errors[name], b)
            rows.append({"layer": name, **projection, "error_l2": errors[name].double().norm().item(),
                         "owner_error_dot": od, "owner_error_cosine": oc,
                         "owner_gradient_l2": on, "quality_error_dot": qd,
                         "quality_error_cosine": qc, "quality_gradient_l2": qn,
                         "ownership_score": own.detach().item(), "image_mse": quality.detach().item()})
        return rows
    finally:
        for p, enabled in zip(parameters, previous):
            p.requires_grad_(enabled)


def analyze(root, report, manifest, net, key, count, device, image_root, artifact_root,
            extractor_digest, reference_valid):
    from evaluate_blind_watermark import verify_frozen_run, file_sha256, resolve_artifact_root
    from diffusers import StableDiffusionPipeline, AutoencoderKL, DDIMScheduler
    from safetensors.torch import load_file

    root = Path(root)
    verify_frozen_run(root, report, image_root, artifact_root)
    artifacts = resolve_artifact_root(root, manifest, artifact_root)
    if (root / "mechanism_analysis.csv").exists() or (root / "mechanism_analysis.json").exists():
        raise FileExistsError("Mechanism results already exist")
    cfg = manifest["args"]
    model = Path(cfg["model"])
    for name, expected in manifest["model_files_sha256"].items():
        if file_sha256(model / name) != expected:
            raise ValueError(f"Marked model changed: {name}")
    pipe = StableDiffusionPipeline.from_pretrained(str(model), local_files_only=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None, requires_safety_checker=False)
    pipe.vae = AutoencoderKL.from_pretrained(str(model / "vae"), local_files_only=True, torch_dtype=torch.float32)
    pipe.scheduler = DDIMScheduler.from_config(manifest["sampling_scheduler"])
    pipe.set_progress_bar_config(disable=True)
    pipe.to(device)
    vae = pipe.vae.eval().requires_grad_(False)
    # Preserve original generation chunk size for the selected prefix of test seeds.
    start = cfg["train_n"] + cfg["search_n"]
    size = cfg["gen_batch_size"]
    generated = min(cfg["test_n"], ((count + size - 1) // size) * size)
    def generate(ids):
        generators = [torch.Generator(device=device).manual_seed(manifest["seeds"][i]) for i in ids]
        prompts = [manifest["prompts"][i] for i in ids]
        with torch.no_grad():
            z = pipe(prompts if len(ids) > 1 else prompts[0],
                     generator=generators if len(ids) > 1 else generators[0],
                     num_inference_steps=cfg["sampling_steps"], guidance_scale=cfg["guidance"],
                     height=512, width=512, output_type="latent").images
        return (z.float() / float(vae.config.scaling_factor)).cpu()
    latents = []
    for _, z in batches(list(range(start, start + generated)), generate, size, "mechanism_generate"):
        latents.extend(z.split(1))
    latents = latents[:count]
    pipe.unet.to("cpu")
    pipe.text_encoder.to("cpu")
    if device == "cuda":
        torch.cuda.empty_cache()
    pristine = {k: v.detach().cpu().clone() for k, v in vae.decoder.state_dict().items()}
    with torch.no_grad():
        references = [decoded01(vae.decode(z.to(device), return_dict=False)[0]).cpu() for z in latents]
    rows = []
    for branch in manifest["test_branches"]:
        if branch.get("role") not in ("candidate", "finetune_control"):
            continue
        vae.decoder.load_state_dict(pristine)
        is_float = branch.get("artifact_format") == "decoder_fp32"
        tensors = load_file(str(artifacts / branch["artifact"] /
                                ("decoder_fp32.safetensors" if is_float else "quantizer.safetensors")), device=device)
        subspace = None
        basis_file = artifacts / branch["artifact"] / "residual_basis.safetensors"
        if basis_file.exists():
            from wmq_residual import ResidualSubspace
            subspace = ResidualSubspace(load_file(str(basis_file), device=device)["basis"], cfg["residual_patch"])
        errors = {}
        with torch.no_grad():
            for name in ([n for n, _ in vae.decoder.named_parameters()] if is_float else manifest["target_names"]):
                weight = tensors[name] if is_float else tensors[name + ".codes"].float() * tensors[name + ".scale"]
                errors[name] = weight - pristine[name].to(device)
                vae.decoder.get_parameter(name).copy_(weight)
            if not is_float:
                for name in (branch.get('reparameterization') or {}).get('fp32_parameter_names', []):
                    bias = tensors[name + '.fp32']
                    errors[name] = bias - pristine[name].to(device)
                    vae.decoder.get_parameter(name).copy_(bias)
        del tensors
        for index, (z, ref) in enumerate(zip(latents, references)):
            for row in layer_diagnostics(vae, net, key, z.to(device), ref.to(device), errors, subspace):
                rows.append({"method": branch["label"], "parameter_space": branch.get("parameter_space", "fp32" if is_float else "quantized"),
                             "test_index": index,
                             "seed": manifest["seeds"][start + index], **row})
        print(f"Owner mechanism: {branch['label']}, {count} test samples", flush=True)
    # Verify again before publishing diagnostics; files under branches remain immutable.
    verify_frozen_run(root, report, image_root, artifact_root)
    payload = {"selection_frozen_sha256": report["selection_frozen_sha256"],
               "extractor_sha256": extractor_digest, "reference_valid": reference_valid,
               "samples": count, "gradient_location": "exported_branch_endpoint",
               "quality_loss": "RGB MSE against marked decoder on identical regenerated latent",
               "owner_score": "mean(square(mean(tanh(logit/2)*(2*key-1))))",
               "interpretation": "Negative owner_error_dot locally reduces soft two-tail agreement. Endpoint Taylor diagnostic, not an exact finite-change attribution or ownership subspace proof. Null cosine means zero vector. No PNG rounding gradient.",
               "limitations": "Test latents regenerated from frozen prompts/seeds; CUDA batch/OOM/kernel differences can change pixels. No detector metrics feed back into selection.",
               "rows": rows}
    with (root / "mechanism_analysis.json").open("x", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    with (root / "mechanism_analysis.csv").open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
