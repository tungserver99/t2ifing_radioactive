"""V3-only live current-weight PNDM rollout and locked gradient mathematics.

Historical trace/replay helpers and clean-selected v2 losses are untouched.
"""
import contextlib
import torch
from semantic_carrier import check_finite
from semantic_trajectory import validate_pndm, scheduler_protocol

OBJECTIVE = 'semantic_sign_v3_native_margin_refine'


def rgb8_ste(image):
    check_finite(image)
    x = (image.float().clamp(-1, 1)+1)/2
    quantized = (x*255).round()/255
    return (x+(quantized-x).detach())*2-1


def dynamic_sign_margin_loss(coords, signs):
    from semantic_sign import _validate, image_bits
    _validate(coords, signs)
    if coords.shape[1] != 32:
        raise ValueError('V3 requires the frozen 32-bit RSSF carrier')
    signed = coords.float()*signs.detach().float()
    indices = torch.argsort(signed.detach(), dim=-1, descending=True, stable=True)[:, :26]
    selected = signed.gather(1, indices)
    loss = (.02-selected).clamp_min(0).square().mean()
    check_finite(loss)
    with torch.no_grad():
        counts = (image_bits(coords) == signs).sum(-1)
        stats = {'K_train':26, 'mu':.02, 'L_M':loss.item(),
                 'selected_indices':indices.tolist(), 'per_image_K':counts.tolist(),
                 'mean_K':counts.float().mean().item(),
                 'mean_selected_margin':selected.mean().item(),
                 'selected_margin_fraction':(selected>=.02).float().mean().item(),
                 'margin_26':selected[:,-1].tolist(),
                 'selection_frequency':torch.bincount(indices.flatten(),minlength=32).tolist()}
    return loss, stats


def gradient_norm(values):
    return sum(v.double().square().sum() for v in values).sqrt()


def capped_gradient(gm, gf):
    if len(gm)!=len(gf) or not gm:
        raise ValueError('Matching nonempty gradient buffers required')
    check_finite(*gm, *gf)
    nm, nf = gradient_norm(gm).item(), gradient_norm(gf).item()
    applied = nm > 1e-12
    alpha = min(1., .5*nm/(nf+1e-12)) if applied and nf>1e-12 else 0.
    combined = [m+alpha*f if applied else torch.zeros_like(m) for m,f in zip(gm,gf)]
    norm = gradient_norm(combined).item()
    check_finite(*combined)
    return combined, {'n_M':nm,'n_F':nf,'alpha_F':alpha,'pre_clip_norm':norm,
                      'clip_coefficient':min(1.,1./(norm+1e-6)), 'update_applied':applied}


def raw_predictions(unet, scheduler, latent, timestep, text, empty, precision='bf16'):
    model_input = scheduler.scale_model_input(torch.cat([latent,latent]), timestep)
    enabled = precision != 'no'
    dtype = {'bf16':torch.bfloat16,'fp16':torch.float16,'no':torch.float32}[precision]
    with torch.autocast(latent.device.type, dtype=dtype, enabled=enabled):
        output = unet(model_input, timestep, encoder_hidden_states=torch.cat([empty,text])).sample
    output = output.float()  # Cast BEFORE CFG arithmetic, including both branches.
    check_finite(output)
    return output.chunk(2)


def live_rollout(unet, scheduler, initial, text, empty, probe_index=0,
                 precision='bf16', tail_calls=10, observer=None):
    validate_pndm(scheduler)
    if tail_calls != 10 or scheduler.num_inference_steps != 50:
        raise ValueError('V3 requires PNDM50 with exactly ten differentiable calls')
    protocol = scheduler_protocol(scheduler)
    timesteps = tuple(int(t) for t in scheduler.timesteps)
    # Audited diffusers 0.35.1 skip-PRK PNDM50 has one repeated PLMS startup call.
    if len(timesteps)!=51 or timesteps[1]!=timesteps[2] or scheduler.counter!=0:
        raise ValueError('Production PNDM50 call order/history changed')
    boundary = len(timesteps)-10
    if not 0 <= probe_index < boundary:
        raise ValueError('Preservation probe must be in the current no-grad prefix')
    latent = initial.detach().float()
    probe = None
    for i,t in enumerate(scheduler.timesteps):
        if int(t)!=timesteps[i] or scheduler.counter!=i:
            raise RuntimeError('PNDM history discontinuity')
        context = torch.no_grad() if i < boundary else contextlib.nullcontext()
        with context:
            if i==probe_index:
                probe = (latent.detach().clone(),t.detach().clone())
            u,c = raw_predictions(unet,scheduler,latent,t,text,empty,precision)
            guided = u+7.5*(c-u)
            check_finite(guided)
            latent = scheduler.step(guided,t,latent).prev_sample
            check_finite(latent)
        if observer:
            observer(i,latent,scheduler)
    if scheduler.counter != len(timesteps):
        raise RuntimeError('Incomplete current-weight native trajectory')
    return latent, probe, {'total_calls':len(timesteps),'tail_boundary':boundary,
                         'tail_timesteps':list(timesteps[boundary:]),'probe_call':probe_index,
                         'probe_timestep':int(probe[1]),'final_latent_norm':latent.detach().norm().item(),
                         'scheduler_protocol':protocol}


def initial_latents(unet, scheduler, rows, config, dtype=torch.float16):
    """Same per-row generators and randn_tensor helper as production pipeline."""
    from diffusers.utils.torch_utils import randn_tensor
    from semantic_teacher import generation_seed
    device = next(unet.parameters()).device
    generators = [torch.Generator(device).manual_seed(generation_seed(config,row)) for row in rows]
    shape = (len(rows),unet.config.in_channels,64,64)
    latent = randn_tensor(shape,generator=generators,device=device,dtype=dtype)
    return (latent*scheduler.init_noise_sigma).float()


def preservation_loss(unet, clean_unet, scheduler, probe, text, empty):
    latent,t = probe
    if latent.requires_grad:
        raise RuntimeError('Preservation state must be detached')
    u,c = raw_predictions(unet,scheduler,latent,t,text,empty)
    with torch.no_grad():
        u0,c0 = raw_predictions(clean_unet,scheduler,latent,t,text,empty)
    loss = .5*((u-u0).square().mean()+(c-c0).square().mean())
    check_finite(loss)
    return loss


def accumulate_gradients(loss, params, buffers, weight, retain_graph=False):
    values = torch.autograd.grad(loss*weight,params,retain_graph=retain_graph,allow_unused=True)
    used = []
    for index,(buffer,value) in enumerate(zip(buffers,values)):
        if value is not None:
            check_finite(value)
            buffer.add_(value.detach().float())
            used.append(index)
    return used

