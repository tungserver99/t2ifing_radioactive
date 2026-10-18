"""Differentiable short LoRA learning, not a detector loss given to the student.

The inner objective is ordinary denoising MSE. Outer gradients pass through the
teacher image, VAE encoder and AdamW updates. Evaluation uses a disjoint clean
latent/noise probe and predicted-x0 images: this is an explicit short-horizon
surrogate, not a claim of full-trajectory meta-training.
"""
import torch
from torch.func import functional_call
from torch.utils.checkpoint import checkpoint
from new_watermark.method import predict_x0, image_features, torch_scores


def adamw_step(params, grads, state, step, lr, weight_decay=.01, beta1=.9, beta2=.999, eps=1e-8):
    """Functional AdamW with smooth sqrt at zero for finite hypergradients.

    sqrt(v + 1e-24) replaces sqrt(v); the smoothing is disclosed in the protocol.
    Differentiable global norm clipping matches the student's max_grad_norm=1.
    """
    norm=torch.sqrt(sum(g.square().sum() for g in grads.values())+1e-24)
    factor=(1./(norm+1e-6)).clamp(max=1.)
    updated, new_state={},{}
    for name,p in params.items():
        g=grads[name]*factor
        m,v=state.get(name,(torch.zeros_like(p),torch.zeros_like(p)))
        m=beta1*m+(1-beta1)*g; v=beta2*v+(1-beta2)*g.square()
        mh=m/(1-beta1**step); vh=v/(1-beta2**step)
        updated[name]=p*(1-lr*weight_decay)-lr*mh/(torch.sqrt(vh+1e-24)+eps)
        new_state[name]=(m,v)
    return updated,new_state


def functional_prediction(model,params,noisy,timestep,text,use_checkpoint=True):
    names=tuple(params)
    def forward(x,t,h,*values):
        return functional_call(model,dict(zip(names,values)),(x,t,h),strict=False).sample
    args=(noisy,timestep,text,*params.values())
    # Internal checkpointing must be OFF: recomputation must re-enter functional_call.
    return checkpoint(forward,*args,use_reentrant=False) if use_checkpoint else forward(*args)


def unroll(model,initial,latent,text,scheduler,draws,lr,create_graph=True,use_checkpoint=True):
    params=initial; state={}
    for step,(noise,t) in enumerate(draws,1):
        noisy=scheduler.add_noise(latent,noise,t)
        prediction=functional_prediction(model,params,noisy,t,text,use_checkpoint)
        if scheduler.config.prediction_type=='epsilon': target=noise
        elif scheduler.config.prediction_type=='v_prediction': target=scheduler.get_velocity(latent,noise,t)
        else: raise ValueError('Unsupported inner prediction type')
        loss=(prediction.float()-target.float()).square().mean()
        values=torch.autograd.grad(loss,tuple(params.values()),create_graph=create_graph,allow_unused=True)
        grads={k:g if g is not None else torch.zeros_like(p) for (k,p),g in zip(params.items(),values)}
        params,state=adamw_step(params,grads,state,step,lr)
        if not create_graph:
            params={k:v.detach().requires_grad_(True) for k,v in params.items()}
            state={k:tuple(vv.detach() for vv in v) for k,v in state.items()}
    return params


def make_surrogate(base,rank,scope,device='cuda'):
    from diffusers import UNet2DConditionModel
    from diffusers.models.attention_processor import AttnProcessor
    from peft import LoraConfig
    model=UNet2DConditionModel.from_pretrained(base,subfolder='unet',local_files_only=True).float().to(device)
    model.requires_grad_(False)
    targets=[n for n,m in model.named_modules() if any(n.endswith('.'+s) for s in ('to_q','to_k','to_v','to_out.0'))
             or scope=='attention_conv' and isinstance(m,torch.nn.Conv2d) and m.groups==1]
    model.add_adapter(LoraConfig(r=rank,lora_alpha=rank,init_lora_weights='gaussian',target_modules=targets))
    # Explicit math attention supports double backward; CUDA flash SDPA may not.
    model.set_attn_processor(AttnProcessor())
    model.disable_gradient_checkpointing(); model.eval()
    model.requires_grad_(False)
    return model


def fresh_parameters(model,seed):
    rng=torch.Generator('cpu').manual_seed(seed)
    result={}
    for name,p in model.named_parameters():
        if '.lora_A.' in name:
            # PEFT Gaussian initialization uses std=1/rank; B starts at zero.
            value=(torch.randn(p.shape,generator=rng,dtype=torch.float32)/p.shape[0]).to(p.device)
        elif '.lora_B.' in name: value=torch.zeros_like(p)
        else: continue
        result[name]=value.detach().requires_grad_(True)
    if not result: raise RuntimeError('Surrogate has no LoRA tensors')
    return result


def transfer_loss(model,vae,clip,detector,marked_pixels,clean_pixels,train_text,
                  probe_latent,probe_text,scheduler,seed,steps,lr,margin,target):
    device=marked_pixels.device
    rng=torch.Generator('cpu').manual_seed(seed)
    # Real image -> VAE encoding bridge; student never receives teacher internal x0.
    marked=vae.encode(marked_pixels*2-1).latent_dist.mode()*vae.config.scaling_factor
    with torch.no_grad():
        clean=vae.encode(clean_pixels*2-1).latent_dist.mode()*vae.config.scaling_factor
    draws=[(torch.randn(marked.shape,generator=rng).to(device),
            torch.randint(0,scheduler.config.num_train_timesteps,(marked.shape[0],),generator=rng).to(device)) for _ in range(steps)]
    initial=fresh_parameters(model,seed+191)
    # Same initialization, data pair and denoising noise for clean/marked controls.
    clean_initial={k:p.detach().clone().requires_grad_(True) for k,p in initial.items()}
    clean_params=unroll(model,clean_initial,clean,train_text,scheduler,draws,lr,create_graph=False)
    marked_params=unroll(model,initial,marked,train_text,scheduler,draws,lr,create_graph=True)
    t=torch.full((1,),min(300,scheduler.config.num_train_timesteps-1),device=device,dtype=torch.long)
    noise=torch.randn(probe_latent.shape,generator=rng).to(device)
    noisy=scheduler.add_noise(probe_latent,noise,t)
    alpha=scheduler.alphas_cumprod.to(device)[t].reshape(-1,1,1,1)
    def score(params):
        u,v=functional_prediction(model,params,noisy.repeat(2,1,1,1),t,probe_text).chunk(2)
        x0=predict_x0(noisy,u+7.5*(v-u),alpha,scheduler.config.prediction_type)
        pixels=(vae.decode(x0/vae.config.scaling_factor).sample/2+.5).clamp(0,1)
        return torch_scores(image_features(clip,pixels),detector)
    with torch.no_grad(): clean_score=score(clean_params)
    marked_score=score(marked_params)
    # Absolute detection and clean-adjusted transfer both matter.
    gain=marked_score-clean_score
    relative=torch.nn.functional.relu(margin-gain).square().mean()/max(margin**2,1.)
    absolute=torch.nn.functional.relu(target-marked_score).square().mean()/max(target**2,1.)
    return relative+absolute, {'proxy_clean_score':float(clean_score.mean()),
        'proxy_marked_score':float(marked_score.detach().mean()),'proxy_gain':float(gain.detach().mean()),
        'inner_steps':steps,'inner_optimizer':'AdamW with sqrt(v+1e-24), fresh state, differentiable grad clipping',
        'proxy':'held-out predicted-x0, not full noise-to-image generation'}
