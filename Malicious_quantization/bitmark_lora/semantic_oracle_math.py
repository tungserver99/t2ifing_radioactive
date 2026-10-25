"""Privileged per-image controls, not a model fingerprint or teacher objective."""
import math
import torch
from semantic_carrier import check_finite
from semantic_sign import image_bits
from semantic_trajectory import restore_scheduler,replay_tail

ORACLE_CONFIG=dict(oracle_version='rssf_tail_control_oracle_v1',oracle_n=256,oracle_seed=20261026,
    oracle_batch_size=4,oracle_steps=120,oracle_lr=.03,optimizer='Adam',tail_calls=4,K_detect=24,
    oracle_bit_margin=.02,w_det_start=1.,w_det_double_every=20,w_det_max=32.,lambda_lpips=1.,
    lambda_pixel=1.,lambda_delta_relative=.01,final_png_lpips_max=.15,aggregate_clip_drop_max=.01,
    oracle_feasible_rate_min=.90,frontier_log_every=10,visual_count=16,target_shard_size=32)


def select_sample_ids(count,n=256,seed=20261026):
    if not 0<n<=count:raise ValueError('Oracle subset requires enough unique training images')
    rng=torch.Generator('cpu').manual_seed(seed)
    return sorted(torch.randperm(count,generator=rng)[:n].tolist())


def fake_quantize_rgb8_ste(x):
    check_finite(x)
    x=x.clamp(-1,1)
    # Match VaeImageProcessor -> numpy.round -> PIL RGB -> pixels()/127.5-1.
    y=((x.detach()/2+.5)*255).round()/127.5-1
    # Exact forward grid; avoid rounding cancellation in x + (y-x).detach().
    return y+(x-x.detach())


def dynamic_detector_loss(coordinates,signs,k=24,margin=.02):
    check_finite(coordinates,signs)
    if coordinates.ndim!=2 or coordinates.shape[1]!=32 or signs.shape!=(32,) or not 1<=k<=32:
        raise ValueError('Oracle detector requires N by 32 coordinates and the existing 32-bit key')
    if not ((signs==1)|(signs==-1)).all() or not math.isfinite(margin) or margin<=0:
        raise ValueError('Invalid oracle signs/margin')
    signed=coordinates*signs.detach()
    indices=torch.argsort(signed,dim=-1,descending=True,stable=True)[:,:k]
    selected=signed.gather(1,indices)
    loss=(margin-selected).clamp_min(0).square().mean(-1)
    with torch.no_grad():
        count=(image_bits(coordinates)==signs).sum(-1)
        minimum=selected[:,-1].detach()
        details={'K':count,'min_top24_margin':minimum,'selected_indices':indices.detach(),
                 'feasible':(count>=k)&(minimum>=margin)}
    check_finite(loss)
    return loss,details


def relative_energy(delta,outputs):
    check_finite(delta,outputs)
    if delta.ndim!=5 or delta.shape!=outputs.shape or delta.shape[0]!=4:
        raise ValueError('Expected four tail calls by batch by CHW controls')
    source=outputs.detach().float();d=delta.float()
    per_call=d.square().mean((2,3,4))/(source.square().mean((2,3,4))+1e-8)
    check_finite(per_call)
    return (per_call.mean(0),per_call.T,
            d.detach().square().mean((2,3,4)).sqrt().T,
            source.square().mean((2,3,4)).sqrt().T)


def replay_controls(factory,record,delta=None,device='cuda',dtype=None):
    """Only cached guided outputs + delta enter the real restored PNDM step."""
    outputs=record['tail_outputs'].detach().to(device=device,dtype=dtype)
    if outputs.shape[0]!=4:raise ValueError('Oracle requires exactly four native tail calls')
    if delta is not None:
        check_finite(delta)
        if delta.shape!=outputs.shape:raise ValueError('Oracle control shape differs from cached outputs')
    scheduler=restore_scheduler(factory(),record,device,dtype)
    counter=[0]
    def prediction(z,t):
        i=counter[0];counter[0]+=1
        return outputs[i] if delta is None else outputs[i]+delta[i]
    final=replay_tail(scheduler,record,prediction,dtype=dtype)
    if counter[0]!=4:raise ValueError('Oracle changed the native tail call count')
    check_finite(final)
    return final


def detector_weights(weights,ever_feasible,step):
    check_finite(weights)
    if step in (20,40,60,80,100):
        return torch.where(ever_feasible.to(device=weights.device),weights,(weights*2).clamp_max(32))
    return weights.clone()


class CandidateTracker:
    """Best eligible iterate by distortion; fallback never masquerades as feasible."""
    def __init__(self,count):
        self.best=[None]*count;self.rank=[None]*count;self.first=[None]*count
        self.ever_feasible=torch.zeros(count,dtype=torch.bool)

    def consider(self,index,step,delta,latent,coordinates,signs,lpips,pixel,energy,per_call,drms,erms,weight):
        check_finite(delta,latent,coordinates)
        numbers=[lpips,pixel,energy,weight,*per_call,*drms,*erms]
        if not all(math.isfinite(float(v)) for v in numbers):raise ValueError('Nonfinite oracle candidate metric')
        _,info=dynamic_detector_loss(coordinates,signs)
        k=int(info['K'].item());minimum=float(info['min_top24_margin'].item())
        feasible=bool(info['feasible'].item())
        if feasible and self.first[index] is None:self.first[index]=step
        self.ever_feasible[index]|=feasible
        # Feasible candidates always win. Diagnostic fallback favors detection.
        rank=(0,lpips,pixel,energy,step) if feasible else (1,-k,-minimum,lpips,pixel,energy,step)
        if self.rank[index] is not None and rank>=self.rank[index]:return
        self.rank[index]=rank
        self.best[index]={'float_feasible':feasible,'best_step':step,'w_det_at_best':weight,'K_float':k,
            'min_top24_margin_float':minimum,'float_lpips':lpips,'float_pixel_mse':pixel,
            'relative_delta_energy_total':energy,'relative_delta_energy_per_call':list(per_call),
            'delta_rms_per_call':list(drms),'cached_output_rms_per_call':list(erms),
            'float_coordinates':coordinates.detach().cpu().flatten().tolist(),
            'delta':delta.detach().cpu().squeeze(1).clone(),'final_latent':latent.detach().cpu().clone()}

    def results(self):
        if any(r is None for r in self.best):raise RuntimeError('No finite oracle candidate was observed')
        return [{**row,'first_feasible_step':self.first[i]} for i,row in enumerate(self.best)]
