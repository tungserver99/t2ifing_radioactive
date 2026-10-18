"""UNet quantization interventions for SleeperMark.

The disclosed public trigger is used only in labeled control branches; owner
key/extractor and TEST images are used only after the quantizer is frozen.

W4 means dequantized integer-grid weights, not a native INT4 inference kernel.
Natural images train timestep denoising with empty text conditioning, not VAE loss.
"""
import argparse
import hashlib
import json
import os
import string
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize
from PIL import Image, ImageOps

from wmq_blind import RoundingGrid, finetune_quantized_weight, pair_metrics, save_json, save_csv, release_branch_memory
from prepare_sleepermark import prepare, load_extractor, digest

EQUIV_METHODS = ('equivariance_qat', 'adversarial_equivariance_qat')
DELTA_METHODS = ('delta_cfg_equivariance', 'delta_coherent_probe')
COHERENT_METHODS = ('coherent_probe_qat', 'delta_coherent_probe')
ROLLOUT_METHODS = ('conditional_rollout_qat',)
STRUCTURED_EVOLUTION_METHODS = ('adaptive_genetic_w4', 'subspace_genetic_w4',
                                'random_subspace_genetic_w4', 'quality_genetic_w4',
                                'local_proposal_genetic_w4', 'llm_genetic_w4',
                                'prefix_invariance_genetic_w4', 'public_trigger_genetic_w4',
                                'behavior_archive_genetic_w4', 'bandit_hyperheuristic_w4',
                                'llm_hyperheuristic_w4', 'public_trigger_bandit_w4')
PUBLIC_SLEEPERMARK_TRIGGER = '*[Z]& '  # Disclosed in SleeperMark, CVPR 2025, Sec. 4.1.
EVOLUTION_METHODS = ('genetic_quantizer_w4', 'random_quantizer_w4') + STRUCTURED_EVOLUTION_METHODS
CFG_METHODS = ('cfg_reconstruction', 'prefix_consistency_qat', 'public_trigger_consistency_qat',
               'public_trigger_rollout_qat', 'public_trigger_distill_defense_w4',
               'coherent_probe_qat') + EQUIV_METHODS + DELTA_METHODS + ROLLOUT_METHODS + EVOLUTION_METHODS
METHODS = ('fixed_ptq', 'model_reconstruction', 'natural_rounding', 'natural_finetune', 'natural_joint_finetune') + CFG_METHODS
DEFAULT_METHODS = ('fixed_ptq', 'cfg_reconstruction', 'delta_cfg_equivariance', *COHERENT_METHODS)


def selected_weights(unet, scope):
    names = [name for name, p in unet.named_parameters() if p.ndim >= 2 and name.endswith('.weight')
             and (scope == 'all' or (name.startswith('up_blocks.') and '.attentions.' in name))]
    if not names:
        raise ValueError('No UNet weight matrices selected')
    return names


def state_hash(module):
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def noise_target(scheduler, latent, noise, timestep):
    kind = scheduler.config.prediction_type
    if kind == 'epsilon':
        return noise
    if kind == 'v_prediction':
        return scheduler.get_velocity(latent, noise, timestep)
    if kind == 'sample':
        return latent
    raise ValueError(f'Unsupported diffusion prediction type: {kind}')


class QuantizedWeight(nn.Module):
    def __init__(self, weight, finetune=False, group_size=0, delta_radius=None, initialization='rtn'):
        super().__init__()
        self.enabled = True
        self.quantized = True
        self.finetune = finetune
        self.group_size = group_size
        if finetune:
            self.weight = nn.Parameter(weight.detach().clone())
        else:
            from wmq_grouped_quant import GroupedW4
            from wmq_delta_quant import DeltaGrid
            self.grid = (DeltaGrid(weight, 4, delta_radius) if delta_radius is not None else
                         GroupedW4(weight, group_size, initialization) if group_size else RoundingGrid(weight, 4, learn_scale=True))

    def forward(self, original):
        if not self.enabled:
            return original
        if self.finetune:
            from wmq_grouped_quant import grouped_fake_quant
            if not self.quantized:
                return self.weight
            return grouped_fake_quant(self.weight, self.group_size) if self.group_size else finetune_quantized_weight(self.weight, 4)
        return self.grid(self.training)


def attach(unet, names, finetune=False, group_size=0, delta_radius=None, initialization='rtn'):
    modules = []
    for name in names:
        parent, leaf = name.rsplit('.', 1)
        module = unet.get_submodule(parent)
        adapter = QuantizedWeight(getattr(module, leaf), finetune, group_size, delta_radius, initialization)
        parametrize.register_parametrization(module, leaf, adapter)
        modules.append((module, leaf, adapter))
    return modules


def switch(modules, enabled=True, quantized=True):
    for _, _, adapter in modules:
        adapter.enabled, adapter.quantized = enabled, quantized


def detach(modules):
    for module, leaf, _ in modules:
        parametrize.remove_parametrizations(module, leaf, leave_parametrized=False)


def snapshot(unet, names):
    # get_parameter does not resolve parametrized weight tensors.
    return {name: getattr(unet.get_submodule(name.rsplit('.', 1)[0]), name.rsplit('.', 1)[1])
            .detach().cpu().contiguous().clone() for name in names}


@torch.no_grad()
def restore(unet, state):
    for name, value in state.items():
        unet.get_parameter(name).copy_(value)


@torch.no_grad()
def encode_text(pipe, prompts):
    ids = pipe.tokenizer(prompts, padding='max_length', truncation=True,
                         max_length=pipe.tokenizer.model_max_length, return_tensors='pt').input_ids.to(next(pipe.text_encoder.parameters()).device)
    return pipe.text_encoder(ids)[0]


def load_image(path):
    with Image.open(path) as im:
        image = ImageOps.fit(ImageOps.exif_transpose(im).convert('RGB'), (512, 512), Image.Resampling.BICUBIC)
        return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div(255).unsqueeze(0)


def train_branch(pipe, scheduler, dataset, names, method, args, output, artifact_output=None):
    if method in EVOLUTION_METHODS:
        from wmq_sleeper_evolution import train_evolution
        return train_evolution(pipe, scheduler, dataset, names, method, args, output)
    device = next(pipe.unet.parameters()).device
    rollout = method in ROLLOUT_METHODS
    quality_rollout = method == 'public_trigger_rollout_qat'
    defense_distill = method == 'public_trigger_distill_defense_w4'
    quality_selection = quality_rollout or defense_distill
    quality_prompts = []
    if quality_selection:
        from wmq_sleeper_quality_select import heldout_prompts
        if not dataset or len(dataset[0]) != 4:
            raise ValueError('Public-trigger rollout selection requires TRAIN trajectories')
        quality_prompts = heldout_prompts(dataset, args.quality_select_n, args.seed)
        heldout = set(quality_prompts)
        dataset = [row for row in dataset if row[2] not in heldout]
        if not dataset:
            raise ValueError('No FIT trajectories remain after quality prompt holdout')
    rollout_scheduler = None
    if rollout or quality_selection:
        from wmq_sleeper_rollout import make_rollout_scheduler
        rollout_scheduler = make_rollout_scheduler(pipe.scheduler, args.inference_steps)
    is_finetune = method in ('natural_finetune', 'natural_joint_finetune')
    delta_branch = method in DELTA_METHODS
    coherent = method in COHERENT_METHODS
    refined = getattr(args, 'quant_refinement', 'legacy') == 'balanced' and not is_finetune
    modules = attach(pipe.unet, names, is_finetune, getattr(args, "quant_group_size", 0),
                     args.delta_radius if delta_branch else None, getattr(args, 'weight_init', 'rtn'))
    params = [p for _, _, adapter in modules for p in adapter.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.delta_lr if delta_branch else args.ft_lr if is_finetune else args.lr,
                                 weight_decay=0, foreach=False, eps=1e-12 if delta_branch else 1e-8)
    if refined:
        code_params, scale_params = [], []
        for _, _, adapter in modules:
            grid = adapter.grid
            if hasattr(grid, 'offset'):
                code_params.append(grid.offset)
            elif getattr(grid, 'learn_code_offsets', False):
                code_params.append(grid.code_offset)
            else:
                code_params.append(grid.alpha)
            scale_params.append(grid.log_scale)
        optimizer = torch.optim.AdamW([
            {'params': code_params, 'lr': args.delta_lr if delta_branch else args.coherent_code_lr if coherent else args.code_lr},
            {'params': scale_params, 'lr': args.coherent_scale_lr if coherent else args.lr}], weight_decay=0, foreach=False, eps=1e-12)
        for group in optimizer.param_groups:
            group['initial_lr'] = group['lr']
    pipe.unet.train()
    # Parametrizations remain attached during recomputation; unlike functional_call,
    # native diffusers gradient checkpointing sees the same weights on backward.
    pipe.unet.enable_gradient_checkpointing()
    rng = torch.Generator(device=device).manual_seed(args.seed)
    order = np.random.default_rng(args.seed)
    auxiliary_order = np.random.default_rng(args.seed + 7919)
    context_order = np.random.default_rng(args.seed + 104729)
    public_trigger_contexts = {}
    rows = []
    cfg = method in CFG_METHODS
    equiv = method in (*EQUIV_METHODS, *DELTA_METHODS, *ROLLOUT_METHODS) and not coherent
    empty = encode_text(pipe, ['']).detach() if cfg else None
    trajectory = cfg and bool(dataset) and len(dataset[0]) == 4
    from wmq_sleeper_calibration import TimestepSampler, augmented_prompts, spatial_reconstruction_loss
    sampler = TimestepSampler(dataset, order) if trajectory and refined else None
    late_indices = ([i for i, item in enumerate(dataset)
                     if item[3] <= args.late_fraction * scheduler.config.num_train_timesteps] if (equiv or coherent) and trajectory else [])
    if (equiv or coherent) and (not trajectory or not late_indices):
        detach(modules)
        pipe.unet.disable_gradient_checkpointing()
        pipe.unet.eval()
        raise ValueError('Spatial QAT requires trajectory calibration with late-timestep records')
    late_sampler = TimestepSampler([dataset[i] for i in late_indices], auxiliary_order) if (equiv or coherent) and refined else None
    shared_probe = None
    count = 0 if method == 'fixed_ptq' else args.steps
    started = time.perf_counter()
    forward_calls = [0]
    def count_forward(module, inputs):
        forward_calls[0] += 1
    counter = pipe.unet.register_forward_pre_hook(count_forward)
    try:
        quality_rows, best_state, best_metrics = [], None, None
        if quality_selection:
            from wmq_sleeper_quality_select import (reference_bank, assess,
                quality_feasible, better_quality_candidate, better_defense_candidate)
            switch(modules, enabled=False)
            pipe.unet.eval()
            references = reference_bank(pipe, quality_prompts, args.seed, args.inference_steps)
            switch(modules, enabled=True)
            anchor_metrics = assess(pipe, quality_prompts, references,
                                    args.seed, args.inference_steps)
            best_metrics = anchor_metrics
            best_state = snapshot(pipe.unet, names)
            quality_rows.append({'step': 0, **anchor_metrics, 'feasible': True,
                                 'selected': True})
            pipe.unet.train()
        for step in range(count):
            if refined:
                from wmq_blind import scheduled_lr
                for group in optimizer.param_groups:
                    group['lr'] = scheduled_lr(group['initial_lr'], step+1, count, min(20, count//10))
            indices = sampler.sample(args.train_batch_size) if sampler else order.integers(len(dataset), size=args.train_batch_size)
            z = torch.cat([dataset[i][0] for i in indices]).to(device)
            condition = torch.cat([dataset[i][1] for i in indices]).to(device)
            if trajectory:
                noisy = z
                t = torch.tensor([dataset[i][3] for i in indices], device=device, dtype=torch.long)
            else:
                noise = torch.randn(z.shape, device=z.device, generator=rng)
                t = torch.randint(scheduler.config.num_train_timesteps, (len(z),), device=z.device, generator=rng)
                noisy = scheduler.add_noise(z, noise, t)
            switch(modules, enabled=False)
            with torch.no_grad():
                marked = pipe.unet(noisy, t, encoder_hidden_states=condition).sample.detach()
            if cfg:
                unconditional = empty.expand(len(z), -1, -1)
                with torch.no_grad():
                    ref_u = pipe.unet(noisy, t, encoder_hidden_states=unconditional).sample.detach()
                    ref_cfg = ref_u + 7.5 * (marked - ref_u)
                optimizer.zero_grad(set_to_none=True)
                switch(modules)
                pred_u = pipe.unet(noisy, t, encoder_hidden_states=unconditional).sample
                pred_c = pipe.unet(noisy, t, encoder_hidden_states=condition).sample
                ordinary_loss = F.mse_loss(pred_u + 7.5 * (pred_c - pred_u), ref_cfg) / 7.5**2
                ordinary_loss = ordinary_loss + F.mse_loss(pred_u, ref_u)
                ordinary_loss.backward()
                del pred_u, pred_c
                losses = [ordinary_loss.detach()]
                spatial_metrics = {}
                if equiv and args.spatial_weight > 0 and step / max(count, 1) > .1:
                    from wmq_sleeper_equivariance import probe_condition, spatial_target, predicted_x0
                    # Extra late-time batch; ordinary CFG loss above still covers the full trajectory.
                    late = ([late_indices[i] for i in late_sampler.sample(args.train_batch_size)] if late_sampler else
                            auxiliary_order.choice(late_indices, size=args.train_batch_size))
                    if rollout:
                        # DDIM.step takes one scalar timestep per batch. Retain the
                        # sampler's timestep stratum, then draw other rows in it.
                        same_t = [i for i in late_indices if dataset[i][3] == dataset[late[0]][3]]
                        late = [late[0], *auxiliary_order.choice(same_t, size=args.train_batch_size-1)]
                    late_z = torch.cat([dataset[i][0] for i in late]).to(device)
                    late_t = torch.tensor([dataset[i][3] for i in late], device=device, dtype=torch.long)
                    context = torch.cat([dataset[i][1] for i in late]).to(device)
                    use_augmented = refined and context_order.random() < args.spatial_context_probability
                    if use_augmented:
                        with torch.no_grad():
                            context = encode_text(pipe, augmented_prompts([dataset[i][2] for i in late], context_order)).detach()
                    shift = tuple(int(a * b) for a, b in zip(auxiliary_order.choice([-1, 1], size=2),
                        auxiliary_order.integers(1, args.spatial_shift + 1, size=2)))
                    switch(modules, enabled=False)
                    ramp = max(0., min(1., (step / max(count, 1) - .1) / .2))
                    if method == 'adversarial_equivariance_qat' and step % args.probe_every == 0 and ramp > 0:
                        context, spatial_metrics = probe_condition(pipe.unet, late_z, late_t, context,
                            shift, scheduler, args.probe_radius, args.probe_steps, args.probe_tokens, rng)
                    if rollout:
                        from wmq_sleeper_rollout import rollout_backward
                        late_empty = empty.expand(len(late_z), -1, -1)
                        spatial_loss, diagnostics = rollout_backward(pipe.unet,
                            lambda enabled: switch(modules, enabled=enabled), late_z, dataset[late[0]][3],
                            context, late_empty, shift, rollout_scheduler, args.rollout_horizon,
                            args.max_spatial_correction, args.spatial_weight*ramp, args.spatial_loss_mode)
                        spatial_metrics.update(diagnostics)
                    else:
                        guided_spatial = delta_branch or refined
                        if guided_spatial:
                            from wmq_sleeper_equivariance import guided_spatial_target, guided_prediction
                            late_empty = empty.expand(len(late_z), -1, -1)
                            target_x0, diagnostics = guided_spatial_target(pipe.unet, late_z, late_t,
                                context, late_empty, shift, scheduler, args.max_spatial_correction)
                        else:
                            target_x0, diagnostics = spatial_target(pipe.unet, late_z, late_t, context,
                                shift, scheduler, args.max_spatial_correction)
                        spatial_metrics.update(diagnostics)
                        switch(modules)
                        prediction = (guided_prediction(pipe.unet, late_z, late_t, context, late_empty)
                                      if guided_spatial else pipe.unet(late_z, late_t, encoder_hidden_states=context).sample)
                        student_x0 = predicted_x0(prediction, late_z, late_t, scheduler)
                        spatial_loss = args.spatial_weight * ramp * spatial_reconstruction_loss(student_x0,
                            target_x0, late_t, scheduler, guidance=7.5 if guided_spatial else 1.,
                            mode=args.spatial_loss_mode if refined else 'x0')
                        spatial_loss.backward()
                        del student_x0, prediction, target_x0
                    spatial_metrics['augmented_context'] = float(use_augmented)
                    losses.append(spatial_loss.detach())
                if coherent and args.coherent_weight > 0 and step / max(count, 1) > .1:
                    from wmq_sleeper_coherent import coherent_probe, coherent_target, sample_distinct_prompts
                    from wmq_sleeper_equivariance import guided_prediction, predicted_x0
                    anchor = (late_indices[late_sampler.sample(1)[0]] if late_sampler else int(auxiliary_order.choice(late_indices)))
                    eligible = [i for i in late_indices if dataset[i][3] == dataset[anchor][3]]
                    chosen = sample_distinct_prompts(dataset, eligible, auxiliary_order, args.coherent_batch_size)
                    probe_z = torch.cat([dataset[i][0] for i in chosen]).to(device)
                    probe_c = torch.cat([dataset[i][1] for i in chosen]).to(device)
                    probe_t = torch.full((len(chosen),), dataset[anchor][3], device=device, dtype=torch.long)
                    probe_u = empty.expand(len(chosen), -1, -1)
                    switch(modules, enabled=False)
                    inner_steps = args.probe_steps if shared_probe is None or step % args.probe_every == 0 else 0
                    probe_c, shared_probe, common, diagnostics = coherent_probe(pipe.unet, probe_z,
                        probe_t, probe_c, scheduler, shared_probe, rng, args.probe_radius,
                        args.probe_tokens, inner_steps, args.coherent_semantic_weight)
                    target, correction_rms = coherent_target(pipe.unet, probe_z, probe_t, probe_c,
                        probe_u, common, scheduler, args.max_spatial_correction)
                    switch(modules)
                    prediction = guided_prediction(pipe.unet, probe_z, probe_t, probe_c, probe_u)
                    student_x0 = predicted_x0(prediction, probe_z, probe_t, scheduler)
                    ramp = max(0., min(1., (step / max(count, 1) - .1) / .2))
                    probe_loss = args.coherent_weight*ramp*spatial_reconstruction_loss(
                        student_x0, target, probe_t, scheduler, mode=args.spatial_loss_mode)
                    probe_loss.backward()
                    losses.append(probe_loss.detach())
                    spatial_metrics.update(diagnostics, coherent_correction_rms=correction_rms)
                    del prediction, student_x0, target, common, probe_loss
                if method in ('prefix_consistency_qat', 'public_trigger_consistency_qat',
                              'public_trigger_rollout_qat', 'public_trigger_distill_defense_w4'):
                    if method in ('public_trigger_consistency_qat', 'public_trigger_rollout_qat',
                                  'public_trigger_distill_defense_w4'):
                        prompts = [dataset[i][2] for i in indices]
                        for prompt in prompts:
                            if prompt not in public_trigger_contexts:
                                public_trigger_contexts[prompt] = encode_text(
                                    pipe, [PUBLIC_SLEEPERMARK_TRIGGER+prompt]).detach()
                        prefix_cond = torch.cat([public_trigger_contexts[prompt] for prompt in prompts])
                    else:
                        # Sample from punctuation alphabet, independent of owner assets.
                        prefixes = [''.join(order.choice(list(string.punctuation), size=int(order.integers(1, 9))))
                                    + ' ' + dataset[i][2] for i in indices]
                        prefix_cond = encode_text(pipe, prefixes)
                    if defense_distill:
                        switch(modules, enabled=False)
                        with torch.no_grad():
                            trigger_teacher = pipe.unet(noisy, t, encoder_hidden_states=prefix_cond).sample.detach()
                        switch(modules, enabled=True)
                    pred = pipe.unet(noisy, t, encoder_hidden_states=prefix_cond).sample
                    ramp = max(0., min(1., (step / max(count, 1) - .2) / .3))
                    weight = (args.public_trigger_weight if method in
                              ('public_trigger_consistency_qat', 'public_trigger_rollout_qat',
                               'public_trigger_distill_defense_w4')
                              else args.prefix_weight)
                    prefix_loss = weight * (1. if defense_distill else ramp) * F.mse_loss(
                        pred, trigger_teacher if defense_distill else marked)
                    prefix_loss.backward()
                    losses.append(prefix_loss.detach())
                if quality_selection and step % args.quality_rollout_every == 0:
                    from wmq_sleeper_rollout import preserve_ordinary_rollout
                    record = dataset[int(indices[0])]
                    z_roll = record[0].to(device)
                    c_roll = record[1].to(device)
                    u_roll = empty.expand(len(z_roll), -1, -1)
                    if defense_distill:
                        prompt = record[2]
                        if prompt not in public_trigger_contexts:
                            public_trigger_contexts[prompt] = encode_text(
                                pipe, [PUBLIC_SLEEPERMARK_TRIGGER+prompt]).detach()
                        c_roll = public_trigger_contexts[prompt]
                    ramp_quality = min(1., (step + 1) / max(1, count // 10))
                    rollout_loss, diagnostics = preserve_ordinary_rollout(
                        pipe.unet, lambda enabled: switch(modules, enabled=enabled),
                        z_roll, int(record[3]), c_roll, u_roll, rollout_scheduler,
                        args.quality_rollout_horizon,
                        args.quality_rollout_weight * ramp_quality)
                    losses.append(rollout_loss.detach())
                    spatial_metrics.update(diagnostics)
                torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
                optimizer.step()
                if delta_branch or refined:
                    for _, _, adapter in modules:
                        if hasattr(adapter.grid, 'clamp_parameters'):
                            adapter.grid.clamp_parameters()
                if quality_selection and ((step + 1) % args.quality_select_every == 0 or step + 1 == count):
                    pipe.unet.eval()
                    switch(modules, enabled=True)
                    metrics = assess(pipe, quality_prompts, references,
                                     args.seed, args.inference_steps)
                    feasible = quality_feasible(metrics, anchor_metrics,
                        args.quality_select_ssim_slack, args.quality_select_psnr_slack)
                    selector = better_defense_candidate if defense_distill else better_quality_candidate
                    selected = selector(metrics, best_metrics, anchor_metrics,
                        args.quality_select_ssim_slack, args.quality_select_psnr_slack)
                    if selected:
                        best_state, best_metrics = snapshot(pipe.unet, names), metrics
                        for row in quality_rows:
                            row['selected'] = False
                    quality_rows.append({'step': step + 1, **metrics,
                                         'feasible': feasible, 'selected': selected})
                    save_csv(output / f'{method}_selection.csv', quality_rows)
                    pipe.unet.train()
                if (step + 1) % args.log_every == 0 or step + 1 == count:
                    row = {'method': method, 'step': step + 1,
                           'ordinary_cfg_loss': float(losses[0]),
                           'auxiliary_loss': float(losses[1]) if len(losses) > 1 else 0.,
                           'elapsed_seconds': time.perf_counter() - started,
                           'logical_unet_forwards': forward_calls[0],
                           'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30 if device.type == 'cuda' else 0.,
                           **{k: float(v) for k, v in spatial_metrics.items()}}
                    row['code_learning_rate'] = optimizer.param_groups[0]['lr']
                    if refined and not delta_branch:
                        with torch.no_grad():
                            grids = [adapter.grid for _, _, adapter in modules]
                            row['mean_layer_code_change_fraction'] = float(torch.stack([g.code_change_fraction() for g in grids]).mean())
                    if delta_branch:
                        with torch.no_grad():
                            grids = [adapter.grid for _, _, adapter in modules]
                            count_weights = sum(g.source.numel() for g in grids)
                            row['delta_nonzero_code_fraction'] = float(sum(g.codes().ne(0).sum() for g in grids) / count_weights)
                            row['delta_weight_rms'] = float((sum((g.codes() * g.scale).square().sum() for g in grids) / count_weights).sqrt())
                            row['optimizer_epsilon'] = 1e-12
                    rows.append(row)
                    save_csv(output / f'{method}_training.csv', rows)
                    print(row, flush=True)
                continue
            target = marked if method == 'model_reconstruction' else noise_target(scheduler, z, noise, t)
            optimizer.zero_grad(set_to_none=True)
            losses = []
            # Backward each pass before changing quantization mode, so checkpoint
            # recomputation uses exactly the mode used in its forward.
            modes = [False, True] if method == 'natural_joint_finetune' else ([False] if is_finetune else [True])
            for quantized in modes:
                switch(modules, enabled=True, quantized=quantized)
                prediction = pipe.unet(noisy, t, encoder_hidden_states=condition).sample
                loss = F.mse_loss(prediction.float(), target.float())
                if method != 'model_reconstruction':
                    loss = loss + args.preserve_weight * F.mse_loss(prediction.float(), marked.float())
                (loss / len(modes)).backward()
                losses.append(loss.detach())
            torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            optimizer.step()
            if (step + 1) % args.log_every == 0 or step + 1 == count:
                row = {'method': method, 'step': step + 1,
                       'training_loss': float(torch.stack(losses).mean()),
                       'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30 if device.type == 'cuda' else 0.}
                rows.append(row)
                save_csv(output / f'{method}_training.csv', rows)
                print(row, flush=True)
        pipe.unet.eval()
        switch(modules)
        if not is_finetune:
            initialization = []
            for name, (_, _, adapter) in zip(names, modules):
                grid = adapter.grid
                if hasattr(grid, 'initial_weight_mse'):
                    initialization.append({'name': name, 'weight_mse': float(grid.initial_weight_mse),
                                           'legacy_rtn_weight_mse': float(grid.legacy_weight_mse)})
            save_csv(output / f'{method}_initialization.csv', initialization)
        label = method + ('_fp32base_delta4' if delta_branch else
                          '' if method.endswith('_w4') else '_w4')
        result = {label: best_state if quality_selection else snapshot(pipe.unet, names)}
        if quality_selection:
            save_json(output / f'{method}_selection.json', {
                'selection_domain': 'TRAIN prompts disjoint from QAT FIT; no owner key/extractor/TEST',
                'prompt_names': quality_prompts, 'fit_prompt_names': sorted({r[2] for r in dataset}),
                'criterion': ('lowest ordinary+triggered MSE to marked teacher' if defense_distill else
                              'lowest triggered-to-clean MSE') +
                             ' subject to ordinary/triggered image-quality slack relative to initial W4',
                'ssim_slack': args.quality_select_ssim_slack,
                'psnr_slack': args.quality_select_psnr_slack,
                'selected_step': next(r['step'] for r in quality_rows if r['selected']),
                'candidates': quality_rows})
        if delta_branch and artifact_output is not None:
            from safetensors.torch import save_file
            tensors = {}
            for name, (_, _, adapter) in zip(names, modules):
                grid = adapter.grid
                tensors[name + '.codes'] = grid.codes().detach().to(torch.int8).cpu().contiguous()
                tensors[name + '.scale'] = grid.scale.detach().cpu().contiguous()
            save_file(tensors, str(Path(artifact_output) / (label + '_delta.safetensors')))
        if is_finetune:
            switch(modules, quantized=False)
            result[method + '_fp32'] = snapshot(pipe.unet, names)
        return result
    finally:
        counter.remove()
        detach(modules)
        pipe.unet.disable_gradient_checkpointing()
        pipe.unet.eval()


@torch.no_grad()
def generate(pipe, prompts, folder, args):
    folder.mkdir(parents=True, exist_ok=True)
    # Per-image seeds and batch size one keep paired generation and memory bounded.
    for i, prompt in enumerate(prompts):
        image = pipe(prompt, num_inference_steps=args.inference_steps, guidance_scale=7.5,
                     generator=torch.Generator(device='cuda').manual_seed(args.seed + 100000 + i)).images[0]
        image.save(folder / f'{i:06d}.png')
        if (i+1) % 10 == 0:
            print(f'{folder.name}: {i+1}/{len(prompts)}', flush=True)


@torch.no_grad()
def owner_scores(pipe, extractor, key, folder, fpr, seed):
    from evaluate_blind_watermark import detection_threshold, wilson_interval
    upper = detection_threshold(len(key), fpr, 'double')
    counts = []
    # Match official extraction: encode rendered RGB, sample posterior, scale latent.
    # Reuse independent posterior RNG seed across paired branches.
    rng = torch.Generator(device='cuda').manual_seed(seed + 200000)
    for path in sorted(folder.glob('*.png')):
        x = load_image(path).cuda()
        z = pipe.vae.encode(x * 2 - 1).latent_dist.sample(generator=rng) * pipe.vae.config.scaling_factor
        prediction = torch.sigmoid(extractor(z)).round()
        counts.append(int((prediction.flatten() == key).sum()))
    matches = np.asarray(counts)
    detected = (matches >= upper) | (matches <= len(key) - upper)
    low, high = wilson_interval(int(detected.sum()), len(matches))
    return {'bit_accuracy': float(matches.mean() / len(key)), 'tpr': float(detected.mean()),
            'detected': int(detected.sum()), 'n': len(matches), 'tpr_ci95_low': low, 'tpr_ci95_high': high,
            'threshold_upper_inclusive': upper, 'threshold_lower_inclusive': len(key) - upper,
            'bit_matches': counts}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--evolution-population', type=int, default=12)
    p.add_argument('--evolution-generations', type=int, default=12)
    p.add_argument('--evolution-records', type=int, default=16, help='Each of disjoint TRAIN fit/select banks')
    p.add_argument('--evolution-quality-ratio', type=float, default=1.25, help='SELECT ordinary noise MSE / RTN MSE cap')
    p.add_argument('--evolution-tail-ratio', type=float, default=2., help='Max paired SELECT record noise MSE / RTN MSE; report failed candidates and continue')
    p.add_argument('--evolution-residual-rank', type=int, default=2)
    p.add_argument('--evolution-subspace-layout', choices=['global', 'patch3'], default='global',
                   help='Experimental shared 3x3 residual basis; affects learned and random subspace controls equally')
    p.add_argument('--evolution-orthogonal-weight', type=float, default=.25)
    p.add_argument('--evolution-legacy-search', action='store_true',
                   help='Ablation: unconstrained Pareto search and shortlist used before the quality-corridor fix')
    p.add_argument('--evolution-prefix-probes', type=int, default=4,
                   help='Owner-blind random punctuation probes per late FIT/SELECT record for prefix invariance')
    p.add_argument('--llm-proposals-per-generation', type=int, default=3,
                   help='FIT-only proposals replacing part of each GA generation; total candidate budget unchanged')
    p.add_argument('--llm-max-new-tokens', type=int, default=768)
    p.add_argument('--evolution-ablate-grouping', action='store_true')
    p.add_argument('--evolution-ablate-diversity', action='store_true')
    p.add_argument('--evolution-ablate-adaptation', action='store_true')
    p.add_argument('--methods', nargs='+', choices=METHODS,
                   default=list(DEFAULT_METHODS))
    p.add_argument('--rollout-horizon', type=int, default=2,
                   help='Conditional rollout branch only: DDIM transitions, gradients truncated between states')
    p.add_argument('--delta-radius', type=float, default=.05)
    p.add_argument('--delta-lr', type=float, default=.01)
    p.add_argument('--weight-init', choices=['rtn', 'mse'], default='mse')
    p.add_argument('--quant-refinement', choices=['legacy', 'balanced'], default='balanced')
    p.add_argument('--code-lr', type=float, default=.01)
    p.add_argument('--spatial-loss-mode', choices=['x0', 'noise'], default='noise')
    p.add_argument('--spatial-context-probability', type=float, default=.5,
                   help='TRAIN-only random punctuation conditioning for spatial calibration; no owner trigger')
    p.add_argument('--quant-group-size', type=int, default=64, help='0 reproduces legacy channel quantization')
    p.add_argument('--prefix-weight', type=float, default=.25)
    p.add_argument('--public-trigger-weight', type=float, default=1.,
                   help='Known-trigger QAT control only; no owner key or extractor')
    p.add_argument('--quality-rollout-every', type=int, default=4)
    p.add_argument('--quality-rollout-horizon', type=int, default=2)
    p.add_argument('--quality-rollout-weight', type=float, default=1.)
    p.add_argument('--quality-select-n', type=int, default=4)
    p.add_argument('--quality-select-every', type=int, default=250)
    p.add_argument('--quality-select-ssim-slack', type=float, default=.03)
    p.add_argument('--quality-select-psnr-slack', type=float, default=1.)
    p.add_argument('--calibration-mode', choices=['trajectory', 'renoised'], default='trajectory')
    p.add_argument('--trajectory-points', type=int, default=8)
    p.add_argument('--late-fraction', type=float, default=.35)
    p.add_argument('--spatial-shift', type=int, default=4, help='Latent cells; used only during training')
    p.add_argument('--spatial-weight', type=float, default=1.)
    p.add_argument('--max-spatial-correction', type=float, default=.05, help='Per-sample RMS cap in x0 latent units')
    p.add_argument('--probe-radius', type=float, default=.15, help='Relative L2 radius on selected context tokens')
    p.add_argument('--probe-steps', type=int, default=2)
    p.add_argument('--probe-every', type=int, default=4)
    p.add_argument('--probe-tokens', type=int, default=4)
    p.add_argument('--coherent-batch-size', type=int, default=2,
                   help='Distinct TRAIN prompts at one timestep; only for coherent probe branches')
    p.add_argument('--coherent-weight', type=float, default=1.)
    p.add_argument('--coherent-semantic-weight', type=float, default=1.)
    p.add_argument('--coherent-code-lr', type=float, default=.003)
    p.add_argument('--coherent-scale-lr', type=float, default=.0001)
    p.add_argument('--scope', choices=['up_attentions', 'all'], default='up_attentions')
    p.add_argument('--train-n', type=int, default=256)
    p.add_argument('--test-n', type=int, default=100)
    p.add_argument('--train-batch-size', type=int, default=1)
    p.add_argument('--calibration-cache', choices=['auto', 'cpu', 'cuda'], default='auto',
                   help='Cache calibration tensors on GPU when free VRAM safely permits')
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--ft-lr', type=float, default=1e-5)
    p.add_argument('--preserve-weight', type=float, default=.5)
    p.add_argument('--log-every', type=int, default=25)
    p.add_argument('--inference-steps', type=int, default=50)
    p.add_argument('--fpr', type=float, default=.001)
    p.add_argument('--natural-images', type=Path)
    p.add_argument('--prompts', type=Path, default=Path(__file__).with_name('prompt.txt'))
    p.add_argument('--fid-real-reference', type=Path)
    p.add_argument('--skip-fid', action='store_true')
    p.add_argument('--report-min-ssim', type=float, default=.8, help='Report-only triggered-image quality threshold')
    p.add_argument('--report-min-psnr', type=float, default=25., help='Report-only threshold; never stops training')
    return p


def main():
    p = parser()
    args = p.parse_args()
    if (args.evolution_population < 4 or args.evolution_generations < 1 or args.evolution_records < 2
            or not np.isfinite([args.evolution_quality_ratio, args.evolution_tail_ratio]).all()
            or min(args.evolution_quality_ratio, args.evolution_tail_ratio) < 1):
        p.error('Invalid evolutionary search budget or quality ratio')
    if any(m in EVOLUTION_METHODS for m in args.methods) and (args.calibration_mode != 'trajectory' or args.quant_group_size < 1):
        p.error('Evolution needs trajectory calibration and positive quant-group-size')
    if (args.evolution_residual_rank < 1 or args.evolution_prefix_probes < 1
            or not np.isfinite(args.evolution_orthogonal_weight)
            or args.evolution_orthogonal_weight < 0):
        p.error('Invalid residual subspace rank/orthogonal weight')
    if (not 1 <= args.llm_proposals_per_generation < args.evolution_population
            or args.llm_max_new_tokens < 64):
        p.error('Invalid LLM proposal count/token budget')
    if min(args.steps, args.train_n, args.test_n, args.train_batch_size, args.log_every,
           args.inference_steps, args.quality_rollout_every, args.quality_rollout_horizon,
           args.quality_select_n, args.quality_select_every) < 1:
        p.error('Counts must be positive')
    if (not np.isfinite([args.quality_rollout_weight, args.quality_select_ssim_slack,
                         args.quality_select_psnr_slack]).all() or
            min(args.quality_rollout_weight, args.quality_select_ssim_slack,
                args.quality_select_psnr_slack) < 0):
        p.error('Quality rollout weight and selection slack must be nonnegative and finite')
    if not 0 < args.fpr < 1 or min(args.lr, args.ft_lr) <= 0 or args.preserve_weight < 0:
        p.error('Invalid FPR, learning rate or preservation weight')
    if (args.quant_group_size < 0 or not np.isfinite([args.prefix_weight, args.public_trigger_weight]).all()
            or min(args.prefix_weight, args.public_trigger_weight) < 0):
        p.error('Invalid quantization group size or prefix weight')
    if (min(args.trajectory_points, args.spatial_shift, args.probe_every, args.probe_tokens) < 1
            or args.probe_steps < 0 or not 0 < args.late_fraction <= 1
            or not all(np.isfinite(v) and v >= 0 for v in
                       (args.spatial_weight, args.max_spatial_correction, args.probe_radius))):
        p.error('Invalid spatial/probe calibration settings')
    if min(args.delta_radius, args.delta_lr) <= 0 or not np.isfinite([args.delta_radius, args.delta_lr]).all():
        p.error('Delta radius and learning rate must be finite and positive')
    if not np.isfinite(args.code_lr) or args.code_lr <= 0 or not 0 <= args.spatial_context_probability <= 1:
        p.error('Invalid code LR or spatial context probability')
    if args.rollout_horizon < 1:
        p.error('Rollout horizon must be positive')
    if (args.coherent_batch_size < 2 or not np.isfinite([args.coherent_weight, args.coherent_semantic_weight,
            args.coherent_code_lr, args.coherent_scale_lr]).all() or min(args.coherent_weight, args.coherent_semantic_weight) < 0
            or min(args.coherent_code_lr, args.coherent_scale_lr) <= 0):
        p.error('Invalid coherent probe settings')
    if any(m in (*EQUIV_METHODS, *DELTA_METHODS, *ROLLOUT_METHODS, *COHERENT_METHODS,
                 'public_trigger_consistency_qat', 'public_trigger_rollout_qat',
                 'public_trigger_distill_defense_w4') for m in args.methods) and args.calibration_mode != 'trajectory':
        p.error('Spatial QAT requires --calibration-mode trajectory')
    if len(set(args.methods)) != len(args.methods):
        p.error('Duplicate methods')
    if not 0 <= args.report_min_ssim <= 1 or not np.isfinite(args.report_min_psnr):
        p.error('Invalid report quality thresholds')
    if not torch.cuda.is_available():
        raise RuntimeError('SleeperMark experiment requires CUDA')
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDIMScheduler, DDPMScheduler
    from safetensors.torch import save_file, load_file
    project = Path(__file__).resolve().parents[1]
    heavy = Path(os.environ.get('WMQ_HEAVY_ROOT', project / 'output_artifacts')).resolve()
    name = 'sleepermark_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = Path(os.environ.get('WMQ_ATTACK_OUTPUT_ROOT', project / 'output_attack')) / name
    artifacts, images = heavy / 'checkpoints' / name, heavy / 'images' / name
    for folder in (output, artifacts, images):
        folder.mkdir(parents=True, exist_ok=False)
    print(f'Results: {output}\nHeavy artifacts: {artifacts}\nImages: {images}', flush=True)
    unet_path, extractor_path, key_path, extractor_source = prepare(heavy / 'models' / 'sleepermark')
    torch.manual_seed(args.seed)
    pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4',
        unet=UNet2DConditionModel.from_pretrained(unet_path), torch_dtype=torch.float32,
        safety_checker=None, requires_safety_checker=False).to('cuda')
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.requires_grad_(False).eval()
    pipe.text_encoder.requires_grad_(False).eval()
    pipe.unet.requires_grad_(False).eval()
    # Frozen components are checked at the end; interventions must touch the UNet.
    frozen_hashes = {'vae': state_hash(pipe.vae), 'text_encoder': state_hash(pipe.text_encoder)}
    names = selected_weights(pipe.unet, args.scope)
    original = snapshot(pipe.unet, names)
    original_hash = state_hash(pipe.unet)
    prompts = list(dict.fromkeys(s.strip() for s in args.prompts.read_text(encoding='utf-8').splitlines() if s.strip()))
    if len(prompts) <= args.test_n:
        raise ValueError('Need test-n held-out unique prompts plus at least one training prompt')
    shuffled = np.random.default_rng(args.seed).permutation(len(prompts))
    test_prompts = [prompts[i] for i in shuffled[:args.test_n]]
    train_prompts = [prompts[i] for i in shuffled[args.test_n:]]
    model_data, natural_data, trajectory_data = [], [], []
    evolution_plan = None
    trajectory_ids = list(range(args.train_n))
    if args.methods and all(m in EVOLUTION_METHODS for m in args.methods):
        from wmq_sleeper_evolution import sparse_calibration_plan
        pipe.scheduler.set_timesteps(args.inference_steps)
        positions = torch.linspace(0, args.inference_steps-1,
                                   min(args.trajectory_points, args.inference_steps)).round().long()
        captured_times = [int(pipe.scheduler.timesteps[i]) for i in positions]
        evolution_plan = sparse_calibration_plan(train_prompts, args.train_n, captured_times,
                                                 args.evolution_records, args.seed)
        trajectory_ids = evolution_plan['trajectory_ids']
        args._evolution_record_indices = evolution_plan['compact_record_indices']
        print(f'Evolution calibration: generating {len(trajectory_ids)}/{args.train_n} trajectories; '
              'FIT/SELECT records and original seeds unchanged', flush=True)
    if any(m in ('model_reconstruction', *CFG_METHODS) for m in args.methods):
        for generated, i in enumerate(trajectory_ids, 1):
            prompt = train_prompts[i % len(train_prompts)]
            from wmq_sleeper_equivariance import TrajectoryCapture
            capture = (TrajectoryCapture(pipe.unet, args.inference_steps, args.trajectory_points)
                       if args.calibration_mode == 'trajectory' and any(m in CFG_METHODS for m in args.methods) else None)
            try:
                with torch.no_grad():
                    latent = pipe(prompt, output_type='latent', num_inference_steps=args.inference_steps,
                                  guidance_scale=7.5, generator=torch.Generator(device='cuda').manual_seed(args.seed+i)).images
                    condition = encode_text(pipe, [prompt]).cpu()
                    model_data.append((latent.cpu(), condition, prompt))
                    if capture is not None:
                        if len(capture.records) != min(args.trajectory_points, args.inference_steps):
                            raise RuntimeError('Incomplete inference trajectory capture')
                        trajectory_data.extend((z, condition, prompt, t) for z, t in capture.records)
            finally:
                if capture is not None:
                    capture.close()
            if generated % 10 == 0 or generated == len(trajectory_ids):
                print(f'Model-only latent calibration {generated}/{len(trajectory_ids)}', flush=True)
    if evolution_plan is not None:
        if [(r[2], r[3]) for r in trajectory_data] != evolution_plan['expected_records']:
            raise RuntimeError('Captured trajectory order/timesteps differ from sparse calibration plan')
    natural_files = []
    if any(m.startswith('natural_') for m in args.methods):
        if args.natural_images is None:
            from prepare_natural_images import prepare as prepare_natural
            args.natural_images = heavy / 'datasets' / f'sleepermark_coco_{args.train_n}_{args.seed}'
            prepare_natural(args.natural_images, args.train_n, args.seed, 8)
        from wmq_fid import image_files
        natural_files = image_files(args.natural_images)
        if len(natural_files) < args.train_n:
            raise ValueError('Insufficient natural images')
        natural_files = [natural_files[i] for i in np.random.default_rng(args.seed).permutation(len(natural_files))[:args.train_n]]
        empty = encode_text(pipe, ['']).detach().cpu()
        for path in natural_files:
            with torch.no_grad():
                z = pipe.vae.encode(load_image(path).cuda() * 2 - 1).latent_dist.mode() * pipe.vae.config.scaling_factor
                natural_data.append((z.cpu(), empty))
    from wmq_sleeper_calibration import cache_datasets_on_device
    (model_data, natural_data, trajectory_data), cache_stats = cache_datasets_on_device(
        [model_data, natural_data, trajectory_data], next(pipe.unet.parameters()).device,
        args.calibration_cache)
    print(f'Calibration cache: {cache_stats}', flush=True)
    manifest = {'watermark': 'SleeperMark', 'base_model': 'CompVis/stable-diffusion-v1-4',
        'reference_label': 'marked_reference_test', 'image_root': str(images), 'artifact_root': str(artifacts),
        'test_branches': [{'label': 'marked_reference_test', 'role': 'reference'}],
        'attack_component': 'unet', 'scope': args.scope, 'selected_weight_names': names,
        'test_used_for_selection': False, 'owner_assets_used_for_training': False,
        'public_trigger_used_for_selection': any(m in args.methods for m in
            ('public_trigger_genetic_w4', 'public_trigger_consistency_qat',
             'public_trigger_rollout_qat', 'public_trigger_distill_defense_w4',
             'public_trigger_bandit_w4')),
        'selection': ('TRAIN-only held-out image quality and method-specific image MSE for the rollout/defense branches; '
                      'other methods use their predeclared selection; owner TEST never used' if
                      any(m in args.methods for m in ('public_trigger_rollout_qat',
                                                     'public_trigger_distill_defense_w4')) else
                      'Predeclared final step; no owner or test quality selection'),
        'quality_policy': ('TRAIN-only image quality gate for rollout/defense branches; '
                           'TEST SSIM/FID report only' if any(m in args.methods for m in
                               ('public_trigger_rollout_qat', 'public_trigger_distill_defense_w4')) else
                           'report_only; SSIM and FID never reject or choose checkpoints'),
        'triggered_quality_thresholds': {'ssim': args.report_min_ssim, 'psnr': args.report_min_psnr},
        'natural_conditioning': 'Empty text; unpaired public images, not paired reconstruction',
        'cfg_calibration': args.calibration_mode,
        'trajectory_records': len(trajectory_data),
        'trajectory_timesteps': sorted(set(record[3] for record in trajectory_data)),
        'calibration_cache': cache_stats,
        'evolution_calibration_plan': evolution_plan,
        'spatial_objective': 'Late-time aligned spatial teacher ensemble; balanced mode uses CFG prediction and declared loss normalization; not ownership oracle',
        'spatial_context': 'Independent TRAIN punctuation augmentation in balanced mode; no owner tokens or detector feedback',
        'conditional_rollout': 'Only conditional-minus-unconditional spatial defect is corrected; independent teacher/student DDIM paths; detached states between steps; ordinary TRAIN starting latents',
        'coherent_probe': 'Shared continuous context delta; maximize cross-image highpass agreement minus lowpass response on distinct TRAIN prompts; suppress only coherence-gated common residual; no recovered-trigger claim',
        'probe_objective': 'Continuous context spatial-defect maximization; no trigger recovery claim',
        'prefix_calibration': ('Public trigger used only in explicitly labeled public_trigger_* branches; '
                               'other branches use independently sampled punctuation or no prefix'),
        'quantization_group_size': args.quant_group_size,
        'precision': 'FP32 activations; full W4 and FP32-base + delta4 are separate intervention classes',
        'delta_base': 'Immutable marked UNet; not an unwatermarked checkpoint; no FP32 fine-tuning intermediate',
        'training_timesteps': 'Balanced CFG cycles through shuffled timestep strata; late auxiliary has separate strata; legacy samples records uniformly; natural uses DDPM',
        'source_sha256': {file: digest(Path(__file__).with_name(file)) for file in
            ('wmq_sleepermark.py', 'wmq_sleeper_calibration.py', 'wmq_sleeper_equivariance.py', 'wmq_sleeper_rollout.py', 'wmq_sleeper_coherent.py',
             'wmq_grouped_quant.py', 'wmq_delta_quant.py', 'wmq_evolution.py',
             'wmq_sleeper_evolution.py', 'wmq_residual_proxy.py', 'wmq_llm_proposals.py',
             'wmq_hyperheuristic.py', 'wmq_sleeper_quality_select.py')},
        'train_prompts': train_prompts, 'test_prompts': test_prompts,
        'natural_train_files': natural_files, 'original_unet_sha256': original_hash,
        'frozen_components': frozen_hashes,
        'arguments': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()
                      if not k.startswith('_')}}
    checkpoints = {}
    for method in args.methods:
        restore(pipe.unet, original)
        torch.manual_seed(args.seed)
        torch.cuda.reset_peak_memory_stats()
        print(f'Training UNet method: {method}', flush=True)
        dataset = (trajectory_data if method in CFG_METHODS and args.calibration_mode == 'trajectory' else
                   model_data if method in ('model_reconstruction', *CFG_METHODS) else natural_data)
        states = train_branch(pipe, scheduler, dataset,
                              names, method, args, output, artifacts)
        for label, state in states.items():
            changed = sum(not torch.equal(state[n], original[n]) for n in names)
            if not changed and method not in DELTA_METHODS:
                raise RuntimeError(f'{label} did not modify the UNet')
            path = artifacts / (label + '.safetensors')
            save_file(state, str(path))
            checkpoints[label] = {'path': str(path), 'sha256': digest(path), 'changed_matrices': changed}
            if method in DELTA_METHODS:
                delta_path = artifacts / (label + '_delta.safetensors')
                checkpoints[label]['delta_codes'] = {'path': str(delta_path), 'sha256': digest(delta_path),
                    'base_unet_sha256': original_hash, 'base_precision': 'fp32', 'delta_bits': 4,
                    'quantization_grouping': 'per_output_channel', 'whole_model_w4': False}
            manifest['test_branches'].append({'label': label,
                'role': 'defense_control' if method == 'public_trigger_distill_defense_w4' else 'attack',
                'identity_intervention': changed == 0,
                'public_trigger_used_for_selection': method in
                    ('public_trigger_genetic_w4', 'public_trigger_consistency_qat',
                     'public_trigger_rollout_qat', 'public_trigger_distill_defense_w4',
                     'public_trigger_bandit_w4'),
                'representation': 'fp32_base_plus_delta4' if method in DELTA_METHODS else 'fp32' if label.endswith('_fp32') else 'selected_weights_w4',
                'threat_model': ('owner_public_trigger_distillation_defense' if method ==
                                'public_trigger_distill_defense_w4' else
                                'public_trigger_quantizer_only' if method in
                                 ('public_trigger_genetic_w4', 'public_trigger_consistency_qat',
                                  'public_trigger_rollout_qat', 'public_trigger_bandit_w4') else
                                'frozen_marked_fp32_base_plus_quantized_delta' if method in DELTA_METHODS else
                                (('restricted_weight_finetune' if label.endswith('_fp32') else
                                  'restricted_weight_finetune_plus_quantization') if 'finetune' in label else 'quantizer_only'))})
        del states
        release_branch_memory('cuda')
    restore(pipe.unet, original)
    if state_hash(pipe.unet) != original_hash:
        raise RuntimeError('Unselected UNet weights changed')
    save_json(output / 'manifest.json', manifest)
    save_json(output / 'selection_frozen.json', {'phase': 'before_test_generation',
              'manifest_sha256': digest(output / 'manifest.json'), 'checkpoints': checkpoints,
              'test_used_for_selection': False})
    # Owner key/extractor and TEST remain post-freeze. Only the explicitly
    # labeled public-trigger control may use the paper's published trigger.
    extractor, key = load_extractor(extractor_source, extractor_path, key_path, 'cuda')
    triggered = [PUBLIC_SLEEPERMARK_TRIGGER + prompt for prompt in test_prompts]
    rows = []
    for branch in manifest['test_branches']:
        label = branch['label']
        restore(pipe.unet, original)
        if label in checkpoints:
            info = checkpoints[label]
            if digest(info['path']) != info['sha256']:
                raise ValueError('Frozen UNet checkpoint changed')
            restore(pipe.unet, load_file(info['path']))
        generate(pipe, triggered, images / label, args)
        row = {'label': label, 'representation': branch.get('representation', 'marked_reference'),
               'identity_intervention': branch.get('identity_intervention', label == 'marked_reference_test'),
               **owner_scores(pipe, extractor, key, images / label, args.fpr, args.seed)}
        # Untriggered outputs measure ordinary utility and supply empirical false positives.
        ordinary = images / (label + '_ordinary')
        generate(pipe, test_prompts, ordinary, args)
        ordinary_owner = owner_scores(pipe, extractor, key, ordinary, args.fpr, args.seed)
        row['untriggered_detection_rate'] = ordinary_owner['tpr']
        quality = []
        triggered_quality = []
        for i in range(args.test_n):
            filename = f'{i:06d}.png'
            psnr, ssim = pair_metrics(load_image(ordinary / filename).cuda(),
                                     load_image(images / 'marked_reference_test_ordinary' / filename).cuda())
            quality.append({'index': i, 'psnr': float(psnr[0]), 'ssim': float(ssim[0])})
            p_trigger, s_trigger = pair_metrics(load_image(images / label / filename).cuda(),
                load_image(images / 'marked_reference_test' / filename).cuda())
            matches = row['bit_matches'][i]
            upper, lower = row['threshold_upper_inclusive'], row['threshold_lower_inclusive']
            base_matches = (rows[0] if rows else row)['bit_matches'][i]
            baseline_detected = base_matches >= upper or base_matches <= lower
            evaded = lower < matches < upper
            quality_ok = float(p_trigger[0]) >= args.report_min_psnr and float(s_trigger[0]) >= args.report_min_ssim
            triggered_quality.append({'index': i, 'psnr': float(p_trigger[0]), 'ssim': float(s_trigger[0]),
                'bit_matches': matches, 'baseline_detected': baseline_detected, 'evaded': evaded,
                'quality_pass': quality_ok, 'joint_success': baseline_detected and evaded and quality_ok})
        row.update(psnr=float(np.mean([r['psnr'] for r in quality])),
                   ssim=float(np.mean([r['ssim'] for r in quality])))
        baseline_n = sum(r['baseline_detected'] for r in triggered_quality)
        row.update(triggered_psnr=float(np.mean([r['psnr'] for r in triggered_quality])),
                   triggered_ssim=float(np.mean([r['ssim'] for r in triggered_quality])),
                   triggered_quality_pass_fraction=float(np.mean([r['quality_pass'] for r in triggered_quality])),
                   evasion_rate=1-row['tpr'], baseline_detected_n=baseline_n,
                   joint_success_count=sum(r['joint_success'] for r in triggered_quality),
                   joint_success_rate=sum(r['joint_success'] for r in triggered_quality)/baseline_n if baseline_n else None)
        row['valid_clean_baseline'] = (row['tpr'] if not rows else rows[0]['tpr']) >= .9
        rows.append(row)
        save_json(output / f'{label}_quality.json', quality)
        save_csv(output / f'{label}_triggered_quality.csv', triggered_quality)
        save_csv(output / 'watermark_retention.csv', rows)
        save_json(output / 'owner_evaluation.json', {'rows': rows, 'detector': 'double', 'total_fpr': args.fpr,
            'extractor_sha256': digest(extractor_path), 'secret_sha256': digest(key_path),
            'posterior_protocol': 'sample, matched RNG across branches, image-to-latent scaling applied',
            'baseline_warning': None if rows[0]['valid_clean_baseline'] else
                'Invalid clean baseline: do not interpret attack TPR as successful removal'})
        print(row, flush=True)
    for component, expected in frozen_hashes.items():
        if state_hash(getattr(pipe, component)) != expected:
            raise RuntimeError(f'{component} changed during UNet experiment')
    save_json(output / 'report.json', {'status': 'complete', 'rows': rows,
              'frozen_components_verified': True, 'test_used_for_selection': False})
    del pipe, extractor
    release_branch_memory('cuda')
    if not args.skip_fid:
        from wmq_fid import evaluate
        try:
            evaluate(output, args.fid_real_reference)
            evaluate(output, args.fid_real_reference, ordinary=True)
        except Exception as error:
            save_json(output / 'fid_error.json', {'status': 'failed', 'error': f'{type(error).__name__}: {error}'})
            print(f'WARNING: FID failed; owner results retained. See {output / "fid_error.json"}', flush=True)
    print(f'Finished: {output}', flush=True)


if __name__ == '__main__':
    main()
