"""Conditional spatial distillation on short DDIM trajectories.

Only the frozen marked model supplies targets. No owner assets or clean model.
Gradients are truncated between transitions, not full-trajectory backpropagation.
"""
import torch
from torch.nn import functional as F

from wmq_sleeper_equivariance import predicted_x0, highpass, cap_rms, interior


def prediction_from_x0(x0, sample, timesteps, scheduler):
    alpha = scheduler.alphas_cumprod.to(sample.device, sample.dtype)[timesteps.long()]
    alpha = alpha.reshape(-1, 1, 1, 1)
    kind = scheduler.config.prediction_type
    if kind == 'epsilon':
        return (sample - alpha.sqrt() * x0) / (1-alpha).sqrt().clamp_min(1e-6)
    if kind == 'v_prediction':
        return (alpha.sqrt() * sample - x0) / (1-alpha).sqrt().clamp_min(1e-6)
    if kind == 'sample':
        return x0
    raise ValueError(f'Unsupported prediction type: {kind}')


@torch.no_grad()
def conditional_orbit_target(unet, sample, timesteps, condition, unconditional,
                             shift, scheduler, max_correction, guidance=7.5):
    """Correct only the spatial defect of (conditional - unconditional) x0.

    Subtraction cancels a condition-independent spatial bias exactly. It does NOT
    separate all semantic texture from ownership, nor establish trigger activation.
    """
    def responses(value):
        u = predicted_x0(unet(value, timesteps, encoder_hidden_states=unconditional).sample,
                         value, timesteps, scheduler)
        c = predicted_x0(unet(value, timesteps, encoder_hidden_states=condition).sample,
                         value, timesteps, scheduler)
        return u, c-u

    uncond, response = responses(sample)
    aligned = []
    for offset in (shift, tuple(-v for v in shift)):
        _, moved = responses(torch.roll(sample, offset, dims=(-2, -1)))
        aligned.append(torch.roll(moved, tuple(-v for v in offset), dims=(-2, -1)))
    correction = guidance * highpass((aligned[0]+aligned[1])*.5-response)
    interior(correction, shift)  # Validate border exclusion before constructing target.
    margin = max(abs(v) for v in shift)+2
    mask = torch.zeros_like(correction)
    mask[..., margin:-margin, margin:-margin] = 1
    correction = cap_rms(correction*mask, max_correction)
    target = uncond + guidance*response + correction
    return target.detach(), {
        'correction_rms': correction.square().mean().sqrt(),
        'conditional_spatial_defect': interior(highpass(aligned[0]-response), shift).square().mean(),
        'conditional_response_rms': response.square().mean().sqrt(),
    }


def make_rollout_scheduler(scheduler, inference_steps):
    from diffusers import DDIMScheduler
    # Use exactly the inference scheduler's clipping, alpha endpoint and spacing.
    result = DDIMScheduler.from_config(scheduler.config)
    result.set_timesteps(inference_steps)
    # DDIM.step in the pinned diffusers uses this fixed predecessor stride.
    if result.config.timestep_spacing != 'leading':
        raise ValueError('Short rollout currently requires DDIM leading timestep spacing')
    return result


def preserve_ordinary_rollout(unet, switch_weights, sample, timestep, condition,
                              unconditional, scheduler, horizon, weight):
    """Truncated DDIM teacher/student paths from a TRAIN trajectory latent.

    Preserve ordinary prompt behavior across multiple transitions. The teacher
    is the same marked model with the quantizer disabled. No owner signal enters.
    Backprop each transition before toggling parametrizations, then detach both
    paths to bound VRAM; this is not full-trajectory differentiation.
    """
    from wmq_sleeper_equivariance import guided_prediction

    stride = scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    if horizon < 1 or stride < 1 or timestep < 0 or weight < 0:
        raise ValueError('Invalid ordinary rollout configuration')
    timesteps = list(range(int(timestep), -1, -stride))[:horizon]
    student_z, teacher_z = sample.detach(), sample.detach()
    losses, drifts = [], []
    try:
        for t in timesteps:
            batch_t = torch.full((len(sample),), t, device=sample.device, dtype=torch.long)
            switch_weights(False)
            with torch.no_grad():
                teacher_pred = guided_prediction(unet, teacher_z, batch_t,
                                                 condition, unconditional)
                teacher_next = scheduler.step(teacher_pred, t, teacher_z, eta=0).prev_sample
            switch_weights(True)
            student_pred = guided_prediction(unet, student_z, batch_t,
                                             condition, unconditional)
            student_next = scheduler.step(student_pred, t, student_z, eta=0).prev_sample
            # The second term penalizes accumulated trajectory drift directly.
            loss = weight/len(timesteps) * (
                F.mse_loss(student_pred, teacher_pred)/7.5**2 +
                F.mse_loss(student_next, teacher_next))
            loss.backward()
            losses.append(loss.detach())
            drifts.append((student_next.detach()-teacher_next).square().mean().sqrt())
            student_z, teacher_z = student_next.detach(), teacher_next.detach()
    finally:
        switch_weights(True)
    return torch.stack(losses).sum(), {
        'quality_rollout_transitions': sample.new_tensor(len(timesteps)),
        'quality_rollout_final_latent_drift': drifts[-1],
    }


def rollout_backward(unet, switch_weights, sample, timestep, condition, unconditional,
                     shift, scheduler, horizon, max_correction, loss_weight,
                     loss_mode='noise', guidance=7.5):
    """Two independent paths: corrected frozen teacher and current quantizer.

    Each transition is supervised in predicted-x0 space; each path then takes its
    own eta=0 DDIM step. Detach BOTH states after every backward to bound memory.
    Caller applies optimizer.step only after all transitions, with weights enabled.
    A batch must share a timestep, as required by DDIMScheduler.step.
    """
    from wmq_sleeper_equivariance import guided_prediction
    from wmq_sleeper_calibration import spatial_reconstruction_loss
    stride = scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    if horizon < 1 or stride < 1 or timestep < 0:
        raise ValueError('Invalid rollout horizon/timestep/scheduler')
    # Do not repeatedly train on an already-terminal sample.
    timesteps = list(range(int(timestep), -1, -stride))[:horizon]
    student_z, teacher_z = sample.detach(), sample.detach()
    total_loss = sample.new_zeros(())
    total_correction = sample.new_zeros(())
    total_defect = sample.new_zeros(())
    drift = sample.new_zeros(())
    try:
        for t in timesteps:
            batch_t = torch.full((len(sample),), t, device=sample.device, dtype=torch.long)
            switch_weights(False)
            target_x0, stats = conditional_orbit_target(unet, teacher_z, batch_t, condition,
                unconditional, shift, scheduler, max_correction, guidance)
            with torch.no_grad():
                target_prediction = prediction_from_x0(target_x0, teacher_z, batch_t, scheduler)
                teacher_next = scheduler.step(target_prediction, t, teacher_z, eta=0).prev_sample
            switch_weights(True)
            prediction = guided_prediction(unet, student_z, batch_t, condition, unconditional, guidance)
            student_x0 = predicted_x0(prediction, student_z, batch_t, scheduler)
            loss = loss_weight / len(timesteps) * spatial_reconstruction_loss(
                student_x0, target_x0, batch_t, scheduler, guidance, mode=loss_mode)
            loss.backward()  # Checkpoint recomputation MUST see the enabled student weights.
            with torch.no_grad():
                student_next = scheduler.step(prediction.detach(), t, student_z, eta=0).prev_sample
                drift = (student_next-teacher_next).square().mean().sqrt()
            student_z, teacher_z = student_next.detach(), teacher_next.detach()
            total_loss += loss.detach()
            total_correction += stats['correction_rms']/len(timesteps)
            total_defect += stats['conditional_spatial_defect']/len(timesteps)
            del prediction, student_x0, target_x0, loss
    finally:
        switch_weights(True)
    return total_loss, {'rollout_transitions': sample.new_tensor(len(timesteps)),
                        'rollout_final_latent_drift': drift,
                        'correction_rms': total_correction,
                        'conditional_spatial_defect': total_defect}
