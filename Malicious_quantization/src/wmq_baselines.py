"""Diffusers W4A16 reconstruction baselines (local adaptations, not upstream replicas).

AdaRound: https://arxiv.org/abs/2004.10568
BRECQ: https://arxiv.org/abs/2102.05426
Q-Diffusion: https://arxiv.org/abs/2302.04304

Q-Diffusion adaptation uses multi-timestep replay, block AdaRound reconstruction
and separate grids for concatenated UNet shortcut inputs. BRECQ adaptation uses
block output MSE rather than a Fisher-weighted reconstruction objective.
Activations are not quantized. Reconstruction runs in FP32; inference retains
the original model dtype. No watermark labels are used by these baselines.
"""

import copy
import math

import torch
from torch import nn
from torch.func import functional_call


def tensor_tree(value, device, dtype=None):
    if torch.is_tensor(value):
        return value.detach().to(device=device, dtype=(dtype if value.is_floating_point()
                                                      and dtype is not None else value.dtype)).clone()
    if isinstance(value, tuple):
        return tuple(tensor_tree(x, device, dtype) for x in value)
    if isinstance(value, list):
        return [tensor_tree(x, device, dtype) for x in value]
    if isinstance(value, dict):
        return {k: tensor_tree(v, device, dtype) for k, v in value.items()}
    return value


def output_tensor(value):
    if torch.is_tensor(value):
        return value
    if hasattr(value, "sample"):
        return value.sample
    if isinstance(value, (tuple, list)):
        return output_tensor(value[0])
    raise TypeError(f"Unsupported reconstruction output: {type(value)}")


def tree_bytes(value):
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, (tuple, list)):
        return sum(tree_bytes(x) for x in value)
    if isinstance(value, dict):
        return sum(tree_bytes(x) for x in value.values())
    return 0


class AdaptiveRounding(nn.Module):
    """Per-weight stretched sigmoid rounding, with fixed per-output scales."""

    def __init__(self, weight, bits=4, split=0):
        super().__init__()
        if bits < 2 or not torch.isfinite(weight).all():
            raise ValueError("Rounding needs finite weights and bits >= 2")
        if split and not 0 < split < weight.shape[1]:
            raise ValueError("Shortcut split must be inside the input-channel axis")
        self.qmax = 2 ** (bits - 1) - 1
        self.split = split
        self.shape = weight.shape
        pieces = torch.split(weight.float(), [split, weight.shape[1] - split], dim=1) if split else [weight.float()]
        self.scales = []
        floors, fractions = [], []
        for index, piece in enumerate(pieces):
            scale = piece.flatten(1).abs().amax(1).clamp_min(1e-8) / self.qmax
            scale = scale.reshape([-1] + [1] * (piece.ndim - 1))
            self.register_buffer(f"scale_{index}", scale)
            self.scales.append(f"scale_{index}")
            value = piece / scale
            floors.append(value.floor())
            fractions.append(value - value.floor())
        floor = torch.cat(floors, dim=1) if split else floors[0]
        fraction = torch.cat(fractions, dim=1) if split else fractions[0]
        self.register_buffer("floor", floor)
        probability = ((fraction + 0.1) / 1.2).clamp(1e-6, 1 - 1e-6)
        self.alpha = nn.Parameter(torch.logit(probability))

    def soft_rounding(self):
        return (self.alpha.sigmoid() * 1.2 - 0.1).clamp(0, 1)

    def forward(self, hard=False):
        up = (self.alpha >= 0).float() if hard else self.soft_rounding()
        integer = (self.floor + up).clamp(-self.qmax, self.qmax)
        if self.split:
            parts = torch.split(integer, [self.split, self.shape[1] - self.split], dim=1)
            return torch.cat([p * getattr(self, scale) for p, scale in zip(parts, self.scales)], dim=1)
        return integer * self.scale_0


def shortcut_splits(unet):
    """Infer concat boundary from Diffusers' up-block channel construction."""
    splits = {}
    previous = unet.mid_block.resnets[-1].out_channels
    for block_index, block in enumerate(unet.up_blocks):
        for resnet_index, resnet in enumerate(block.resnets):
            total = resnet.in_channels
            if not 0 < previous < total:
                raise ValueError(f"Invalid up-block concat boundary {previous}/{total}")
            for leaf in ("conv1", "conv_shortcut"):
                layer = getattr(resnet, leaf, None)
                if layer is not None:
                    splits[f"up_blocks.{block_index}.resnets.{resnet_index}.{leaf}.weight"] = previous
            previous = resnet.out_channels
    return splits


def reconstruction_units(model, selected_names, blockwise):
    """Assign every selected weight exactly once; unsupported layers fail loudly."""
    selected = set(selected_names)
    units, covered = [], set()
    block_types = {"ResnetBlock2D", "BasicTransformerBlock", "Attention"}
    for name, module in model.named_modules():
        prefix = name + "." if name else ""
        owned = {prefix + n for n, p in module.named_parameters()
                 if prefix + n in selected}
        if not owned or owned & covered:
            continue
        is_layer = isinstance(module, (nn.Linear, nn.Conv2d))
        if is_layer or (blockwise and type(module).__name__ in block_types):
            units.append((name, module, owned))
            covered.update(owned)
    if covered != selected:
        raise ValueError(f"Unsupported quantization targets: {sorted(selected - covered)}")
    return units


def reconstruct(model, selected_names, replay, method, steps=200, lr=1e-3,
                seed=2026, cache_mb=512, log_callback=None):
    """Quantize in place using local FP32 reconstruction on calibration data.

    replay() runs the target on the *same* fixed calibration records for every
    method. Only one unit's inputs/outputs are cached at a time. Earlier units
    are already quantized; each current unit is still the FP teacher when hooked.
    """
    if method not in {"adaround_vae", "brecq_vae_adapted", "qdiff_unet_adapted"}:
        raise ValueError(f"Unknown baseline {method}")
    if steps < 1 or not math.isfinite(lr) or lr <= 0 or cache_mb <= 0:
        raise ValueError("Invalid reconstruction steps, learning rate or cache budget")
    model.eval().requires_grad_(False)
    splits = shortcut_splits(model) if method == "qdiff_unet_adapted" else {}
    units = reconstruction_units(model, selected_names, method != "adaround_vae")
    results = []
    for unit_index, (name, unit, owned) in enumerate(units):
        records, used_bytes = [], 0

        def hook(_module, args, kwargs, output):
            nonlocal used_bytes
            value = output_tensor(output)
            size = tree_bytes((args, kwargs, value))
            if used_bytes + size > cache_mb * 2**20:
                raise RuntimeError(f"Calibration cache for {name} exceeds {cache_mb} MiB; "
                                   "increase RECON_CACHE_MB or reduce CALIB_N/CALIB_BATCH_SIZE")
            records.append((tensor_tree(args, "cpu"), tensor_tree(kwargs, "cpu"),
                            tensor_tree(value, "cpu")))
            used_bytes += size

        handle = unit.register_forward_hook(hook, with_kwargs=True)
        try:
            with torch.no_grad():
                replay()
        finally:
            handle.remove()
        if not records:
            raise RuntimeError(f"Calibration never executed selected unit {name}")
        device = next(unit.parameters()).device
        # A separate FP32 unit avoids FP16 scale underflow and does not change
        # inference dtype or leave hooks/parametrizations attached to the model.
        teacher = copy.deepcopy(unit).to(dtype=torch.float32).eval().requires_grad_(False)
        prefix = name + "." if name else ""
        rounding = nn.ModuleDict()
        local_names = sorted(full[len(prefix):] for full in owned)
        for i, local in enumerate(local_names):
            rounding[str(i)] = AdaptiveRounding(unit.get_parameter(local),
                                                split=splits.get(prefix + local, 0))
        optimizer = torch.optim.Adam(rounding.parameters(), lr=lr)
        generator = torch.Generator().manual_seed(seed + unit_index)
        accepted, skipped = 0, 0

        def overrides(hard):
            return {local: rounding[str(i)](hard) for i, local in enumerate(local_names)}

        def hard_error():
            numerator, denominator = 0., 0.
            with torch.no_grad():
                weights = overrides(True)
                for args, kwargs, reference in records:
                    pred = output_tensor(functional_call(teacher, weights,
                        tensor_tree(args, device, torch.float32),
                        tensor_tree(kwargs, device, torch.float32)))
                    ref = reference.to(device=device, dtype=torch.float32)
                    numerator += float((pred - ref).square().sum())
                    denominator += float(ref.square().sum())
            return numerator / max(denominator, 1e-12)

        initial_error = hard_error()
        if not math.isfinite(initial_error):
            raise RuntimeError(f"Nonfinite initial reconstruction for {name}")
        best_error = initial_error
        best_state = {k: v.detach().clone() for k, v in rounding.state_dict().items()}
        for step in range(steps):
            args, kwargs, reference = records[int(torch.randint(len(records), (), generator=generator))]
            optimizer.zero_grad(set_to_none=True)
            pred = output_tensor(functional_call(teacher, overrides(False),
                tensor_tree(args, device, torch.float32), tensor_tree(kwargs, device, torch.float32)))
            ref = reference.to(device=device, dtype=torch.float32)
            loss = (pred - ref).square().mean() / ref.square().mean().clamp_min(1e-8)
            progress = step / max(steps - 1, 1)
            if progress >= .2:
                beta = 20 - 18 * (progress - .2) / .8
                regularizer = torch.stack([(1 - (2 * q.soft_rounding() - 1).abs().pow(beta)).mean()
                                           for q in rounding.values()]).mean()
                loss = loss + .01 * regularizer
            if not torch.isfinite(loss):
                skipped += 1
                continue
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(rounding.parameters(), 5.)
            if not torch.isfinite(norm):
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()
            if not all(torch.isfinite(p).all() for p in rounding.parameters()):
                rounding.load_state_dict(best_state)
                optimizer.state.clear()
                skipped += 1
                continue
            accepted += 1
            if (step + 1) % max(1, steps // 5) == 0 or step == steps - 1:
                error = hard_error()
                if math.isfinite(error) and error < best_error:
                    best_error = error
                    best_state = {k: v.detach().clone() for k, v in rounding.state_dict().items()}
        rounding.load_state_dict(best_state)
        with torch.no_grad():
            for local, quantized in overrides(True).items():
                weight = unit.get_parameter(local)
                weight.copy_(quantized.to(weight.dtype))
        row = {"unit": name, "weights": len(owned), "parameters": sum(unit.get_parameter(n).numel() for n in local_names),
               "calibration_records": len(records), "valid_updates": accepted,
               "skipped_updates": skipped, "initial_nmse": initial_error, "final_nmse": best_error}
        results.append(row)
        if log_callback:
            log_callback(row)
        del teacher, rounding, optimizer, best_state, records, pred, ref, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {"method": method, "adaptation": True, "weight_bits": 4, "activation_bits": 16,
            "reconstruction_dtype": "float32", "objective": "local_output_nmse_and_rounding_regularization",
            "implementation_notes": ["Diffusers port with sequential local teacher replay",
                                     "Per-output symmetric 15-level grid; no activation quantization",
                                     "MSE reconstruction, not Fisher-weighted BRECQ",
                                     "UNet concat inputs have separate scales on conv1 and conv_shortcut"
                                     if splits else "No shortcut split quantization",
                                     "No claim of exact reproduction of original paper results"],
            "uses_watermark_labels": False, "shortcut_splits": splits,
            "changed_parameters": sum(r["parameters"] for r in results), "units": results}
