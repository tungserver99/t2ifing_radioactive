"""Standalone read-only-source, privileged per-image cached-tail control oracle.

Only delta tensors are optimized. No teacher, LoRA, student or denoising UNet
forward is used. Actual PNG evidence, not optimization diagnostics, is final.
"""
import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import traceback
import numpy as np
import torch
from common import HERE,freeze,read_json,write_json,sha256,image_name
from semantic_carrier import check_finite
from semantic_trajectory import batch_records,clone_tree,digest,REPLAY_TOLERANCES
from semantic_oracle_math import (ORACLE_CONFIG,select_sample_ids,fake_quantize_rgb8_ste,
    replay_controls,dynamic_detector_loss,relative_energy,CandidateTracker,detector_weights,precision_drift)
from semantic_oracle_store import OracleStore,load_source,verify_source_manifest,reject_overlap

ZERO_TOLERANCES={'latent_atol':REPLAY_TOLERANCES['latent_atol'],
    'latent_rtol':REPLAY_TOLERANCES['latent_rtol'],'coordinates_atol':2e-3,
    'coordinates_rtol':2e-3,'png_pixel_max':REPLAY_TOLERANCES['png_pixel_max']}


def _single(record):
    if record['tail_outputs'].ndim==4:return batch_records([record])
    if record['tail_outputs'].ndim!=5 or record['tail_outputs'].shape[1]!=1:
        raise ValueError('Oracle optimization requires independent per-image records')
    return record


def optimize_batch(records,backend,ids,config):
    """Serial graphs inside a batch: independent Adam states live in delta only."""
    records=[_single(r) for r in records];n=len(records)
    if n!=len(ids) or not n:raise ValueError('Oracle batch/ID mismatch')
    shape=records[0]['tail_outputs'].shape
    if shape[0]!=4 or any(r['tail_outputs'].shape!=shape for r in records):raise ValueError('Incompatible control shapes')
    delta=torch.nn.Parameter(torch.zeros((4,n,*shape[-3:]),device=backend.device,dtype=torch.float32))
    optimizer=torch.optim.Adam([delta],lr=config['oracle_lr'])
    tracker=CandidateTracker(n);weights=torch.full((n,),config['w_det_start'])
    frontier=[];backend.assert_frozen()
    # Step0 is the zero-control baseline. Iterates1..120 follow 120 Adam updates.
    for step in range(config['oracle_steps']+1):
        optimizer.zero_grad(set_to_none=True);points=[]
        for i,(ident,record) in enumerate(zip(ids,records)):
            view=backend.training_view(i,record,delta[:,i:i+1])
            for key in ('latent','coordinates','detector','lpips','pixel','energy','per_call','delta_rms','cached_rms'):
                check_finite(view[key])
            tracker.consider(i,step,delta[:,i:i+1],view['latent'],view['coordinates'],view['signs'],
                float(view['lpips'].detach()),float(view['pixel'].detach()),float(view['energy'].detach()),
                view['per_call'].detach().cpu().tolist(),view['delta_rms'].detach().cpu().tolist(),
                view['cached_rms'].detach().cpu().tolist(),float(weights[i]))
            if step%config['frontier_log_every']==0:
                _,det=dynamic_detector_loss(view['coordinates'],view['signs'])
                points.append({'image_id':ident,'K_float':int(det['K'].item()),
                    'float_lpips':float(view['lpips'].detach()),
                    'relative_delta_energy_total':float(view['energy'].detach())})
            weights[i:i+1]=detector_weights(weights[i:i+1],tracker.ever_feasible[i:i+1],step)
            if step<config['oracle_steps']:
                loss=(float(weights[i])*view['detector']+config['lambda_lpips']*view['lpips']+
                      config['lambda_pixel']*view['pixel']+config['lambda_delta_relative']*view['energy'])
                check_finite(loss);loss.backward()
            del view
        if points:
            frontier.append({'step':step,'images':points})
            print(f'Oracle step {step}/{config["oracle_steps"]}: mean K={sum(p["K_float"] for p in points)/n:.2f}, '
                  f'ever float feasible={int(tracker.ever_feasible.sum())}/{n}',flush=True)
        if step<config['oracle_steps']:
            if delta.grad is None:raise RuntimeError('Oracle controls received no gradient')
            check_finite(delta.grad);optimizer.step();check_finite(delta)
    backend.assert_frozen()
    return tracker.results(),frontier


def _pil_pixels(images,device):
    return torch.stack([torch.from_numpy(np.array(im.convert('RGB'),copy=True)).permute(2,0,1).float()/127.5-1
                        for im in images]).to(device)


class FrozenOracleBackend:
    device='cuda'
    def __init__(self,source):
        from diffusers import AutoencoderKL,PNDMScheduler
        from diffusers.image_processor import VaeImageProcessor
        import lpips
        from semantic_teacher import extractor,numerical_policy
        from semantic_trajectory import native_scheduler
        numerical_policy()
        if not torch.cuda.is_available():raise RuntimeError('Oracle execution requires a CUDA GPU')
        self.source=source;self.root=Path(source['manifest']['source_run'])
        self.ext=extractor(self.root);self.carrier=source['carrier'].to(self.device)
        self.canonical_vae=AutoencoderKL.from_pretrained(source['assets']['teacher_clean'],subfolder='vae',
            torch_dtype=torch.float16,local_files_only=True).to(self.device).eval().requires_grad_(False)
        scale=2**(len(self.canonical_vae.config.block_out_channels)-1)
        self.processor=VaeImageProcessor(vae_scale_factor=scale)
        self.perceptual=lpips.LPIPS(net='alex',version='0.1').to(self.device).float().eval().requires_grad_(False)
        proto=native_scheduler(source['assets']['teacher_clean'])
        self.factory=lambda:PNDMScheduler.from_config(dict(proto.config))
        self.clip=None;self.clip_processor=None
        self.assert_frozen()

    def assert_frozen(self):
        models=[self.ext.vae,self.ext.unet,self.ext.text_encoder,self.canonical_vae,self.perceptual]
        if self.clip is not None:models.append(self.clip)
        if any(p.requires_grad or p.grad is not None for model in models for p in model.parameters()):
            raise RuntimeError('Oracle must never train a model parameter')
        if any(x.requires_grad for x in (self.carrier.mean,self.carrier.basis,self.carrier.sigmas,self.carrier.signs)):
            raise RuntimeError('Oracle carrier and key must remain detached')

    @torch.no_grad()
    def canonical_images(self,latents):
        # Same FP16 VAE + VaeImageProcessor path as native_generate().
        z=latents.to(device=self.device,dtype=torch.float16)
        raw=self.canonical_vae.decode(z/self.canonical_vae.config.scaling_factor).sample
        check_finite(raw)
        return self.processor.postprocess(raw,output_type='pil')

    @torch.no_grad()
    def begin_batch(self,ids,records):
        from semantic_teacher import pixels
        from semantic_sign_train import decoded_image
        self.ids=ids;self.text=self.ext.text([self.source['prompts']['train'][i]['text'] for i in ids])
        joined=batch_records(records)
        exact=replay_controls(self.factory,joined,device=self.device)
        expected=joined['final_latent'].to(self.device)
        torch.testing.assert_close(exact,expected,atol=0,rtol=0)
        pngs=self.canonical_images(exact)
        zero_png=_pil_pixels(pngs,self.device)
        clean=pixels([self.root/'images/teacher_clean_train'/image_name(i) for i in ids])
        if float((zero_png-clean).abs().max())>ZERO_TOLERANCES['png_pixel_max']:
            raise ValueError('Zero-control native decode no longer matches source PNG')
        zero_coordinates=self.carrier.coordinates(self.ext.encode_image(zero_png,
            [self.source['prompts']['train'][i]['text'] for i in ids]))
        stored=torch.tensor([self.source['scores'][i]['coordinates'] for i in ids],device=self.device)
        torch.testing.assert_close(zero_coordinates,stored,atol=ZERO_TOLERANCES['coordinates_atol'],rtol=ZERO_TOLERANCES['coordinates_rtol'])
        self.reference=[];self.zero=[];prepared=[]
        for j,record in enumerate(records):
            rec=clone_tree(_single(record),device=self.device)
            fp32=replay_controls(self.factory,rec,device=self.device,dtype=torch.float32)
            original=rec['final_latent'].float()
            drift=precision_drift(fp32,rec['final_latent'],
                atol=ZERO_TOLERANCES['latent_atol'],rtol=ZERO_TOLERANCES['latent_rtol'])
            print(f'Zero-control FP32 drift ID {ids[j]}: max={drift["fp32_graph_latent_max_error"]:.9g}, '
                  f'RMS={drift["fp32_graph_latent_rms_error"]:.9g}, '
                  f'outside reference tolerance={drift["fp32_graph_latent_outside_reference_tolerance"]}'
                  f'/{drift["fp32_graph_latent_elements"]} (diagnostic only)',flush=True)
            self.reference.append(decoded_image(self.ext,original).detach())
            self.zero.append({'native_latent_max_error':float((exact[j]-expected[j]).abs().max()),
                **drift,
                'canonical_png_pixel_max_error':float((zero_png[j]-clean[j]).abs().max()),
                'stored_coordinate_max_error':float((zero_coordinates[j]-stored[j]).abs().max())})
            prepared.append(rec)
        print(f'Zero-control sanity passed for IDs {ids}',flush=True)
        return prepared

    def training_view(self,index,record,delta):
        from semantic_sign_train import decoded_image,decoded_coordinates,_checkpoint
        final=replay_controls(self.factory,record,delta,device=self.device,dtype=torch.float32)
        image=decoded_image(self.ext,final)
        coords=decoded_coordinates(self.ext,self.carrier,fake_quantize_rgb8_ste(image),self.text[index:index+1])
        detector,_=dynamic_detector_loss(coords,self.carrier.signs)
        lp=_checkpoint(lambda a,b:self.perceptual(a,b).mean(),image,self.reference[index])
        pixel=(image-self.reference[index]).square().mean()
        energy,per,drms,erms=relative_energy(delta,record['tail_outputs'].to(self.device))
        return dict(latent=final,coordinates=coords,signs=self.carrier.signs,detector=detector[0],
                    lpips=lp,pixel=pixel,energy=energy[0],per_call=per[0],delta_rms=drms[0],cached_rms=erms[0])

    def _load_clip(self):
        if self.clip is not None:return
        from transformers import CLIPModel,CLIPProcessor
        from utility import CLIP_REV
        self.clip=CLIPModel.from_pretrained('openai/clip-vit-base-patch32',revision=CLIP_REV).to(self.device).eval().requires_grad_(False)
        self.clip_processor=CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32',revision=CLIP_REV)

    @torch.no_grad()
    def evaluate(self,index,ident,candidate,output):
        from PIL import Image
        from semantic_teacher import pixels
        from semantic_sign import image_evidence
        prompt=self.source['prompts']['train'][ident]['text']
        png=self.canonical_images(candidate['final_latent'])[0]
        path=output/'oracle_images'/image_name(ident);temporary=path.with_suffix('.partial.png')
        png.save(temporary);temporary.replace(path)
        clean_path=self.root/'images/teacher_clean_train'/image_name(ident)
        image=pixels([path]);clean=pixels([clean_path])
        coords=self.carrier.coordinates(self.ext.encode_image(image,[prompt]))
        evidence=image_evidence(coords,self.carrier.signs,.01)[0]
        lp=self.perceptual(image,clean).mean();check_finite(lp,coords)
        with Image.open(clean_path) as im:clean_pil=im.convert('RGB')
        with Image.open(path) as im:oracle_pil=im.convert('RGB')
        a=np.asarray(clean_pil,dtype=np.float64)/127.5-1;b=np.asarray(oracle_pil,dtype=np.float64)/127.5-1
        pixel=float(((a-b)**2).mean())
        self._load_clip()
        inputs=self.clip_processor(text=[prompt,prompt],images=[clean_pil,oracle_pil],return_tensors='pt',padding=True,truncation=True).to(self.device)
        clips=self.clip(**inputs)
        similarities=(clips.image_embeds*clips.text_embeds).sum(-1).clamp_min(0)*2.5
        check_finite(similarities)
        original=self.source['scores'][ident]
        return dict(image_id=ident,prompt=prompt,clean_K=original['K'],clean_p_img=original['p_img'],
            clean_continuous_score=original['score'],clean_coordinates=original['coordinates'],
            coordinates=coords[0].cpu().tolist(),K_png=evidence['K'],p_img_png=evidence['p_img'],
            analytic_detected_png=evidence['analytic_detected'],png_lpips=float(lp),png_pixel_mse=pixel,
            clean_clip=float(similarities[0]),oracle_clip=float(similarities[1]),
            clip_drop=float(similarities[0])-float(similarities[1]),zero_control=self.zero[index],
            execution_device=torch.cuda.get_device_name())


def protocol_for(source_manifest):
    from utility import CLIP_REV
    names=('semantic_oracle.py','semantic_oracle_math.py','semantic_oracle_store.py','semantic_oracle_report.py',
           'semantic_trajectory.py','semantic_carrier.py','semantic_sign.py','semantic_sign_train.py',
           'semantic_teacher.py','semantic_report.py','semantic_assets.py','common.py','utility.py','report.py')
    return {'oracle_version':ORACLE_CONFIG['oracle_version'],'source_manifest_sha256':digest(source_manifest),
        'code':{name:sha256(HERE/name) for name in names},
        'versions':{name:importlib.metadata.version(name) for name in
                    ('torch','diffusers','transformers','lpips','Pillow','safetensors')},
        'zero_control_tolerances':ZERO_TOLERANCES,'clip_revision':CLIP_REV,'lpips':'alex version0.1',
        'zero_control_policy':'v2: native dtype exact, canonical PNG and stored RSSF coordinates gated; finite FP32/native drift diagnostic only',
        'fp32_drift_reference_tolerance_role':'diagnostic threshold only; not an execution or feasibility gate',
        'control_dtype':'float32','stored_delta_dtype':'float16','stored_final_latent_dtype':'float32',
        'float_reference':'FP32 VAE decode of cached native final latent',
        'canonical_png_decode':'FP16 clean VAE, native VaeImageProcessor, ordinary PNG',
        'detector_view':'RGB8 STE of FP32 float image; final real PNG verifier is authoritative',
        'candidate_selection':'eligible: LPIPS, pixel MSE, relative energy, earlier step; fallback: maxK, max minimum margin, then distortion',
        'teacher_training_started':False}


def execute_batches(source,store,backend_factory,config):
    backend=None;rows=[];frontier=[];size=config['oracle_batch_size']
    for start in range(0,len(store.ids),size):
        ids=store.ids[start:start+size];index=start//size
        if store.has_batch(index,ids):
            saved=store.load_batch(index);rows.extend(saved['rows']);frontier.extend(saved['frontier'])
            print(f'Reused immutable oracle batch {index+1}: {ids}',flush=True);continue
        if backend is None:backend=backend_factory(source)
        records=[source['reader'].get(i) for i in ids]
        prepared=backend.begin_batch(ids,records)
        candidates,trace=optimize_batch(prepared,backend,ids,config)
        metrics=[];tensors={}
        for i,(ident,candidate) in enumerate(zip(ids,candidates)):
            row={k:v for k,v in candidate.items() if k not in ('delta','final_latent')}
            row.update(backend.evaluate(i,ident,candidate,store.output));metrics.append(row)
            half=candidate['delta'].half();check_finite(half)
            row['delta_storage_max_abs_error']=float((candidate['delta']-half.float()).abs().max())
            tensors[f'delta_{ident}']=half;tensors[f'final_latent_{ident}']=candidate['final_latent'].float()
        backend.assert_frozen()
        store.commit_batch(index,ids,tensors,metrics,trace)
        rows.extend(metrics);frontier.extend(trace)
        print(f'Committed oracle batch {index+1}: {min(start+size,len(store.ids))}/{len(store.ids)}',flush=True)
        del candidates,tensors,prepared;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    return rows,frontier


def run_oracle(source_path,output):
    from semantic_oracle_report import write_reports
    output=Path(output).resolve();source_path=Path(source_path).resolve();reject_overlap(output,source_path)
    print(f'Validating immutable source: {source_path}',flush=True)
    source=load_source(source_path);config=dict(ORACLE_CONFIG)
    ids=select_sample_ids(len(source['scores']),config['oracle_n'],config['oracle_seed'])
    protocol=protocol_for(source['manifest'])
    store=OracleStore(output,config,source['manifest'],ids,protocol)
    try:import fcntl
    except ImportError:
        from common import windows_fcntl as fcntl
    with (output/'.oracle.lock').open('a+b') as lock:
        if lock.seek(0,2)==0:lock.write(b'0');lock.flush()
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        complete=output/'oracle_complete.json'
        if complete.exists():
            store.export_targets()
            proof=read_json(complete)
            for name,value in proof['files'].items():
                if sha256(output/name)!=value:raise RuntimeError(f'Completed oracle report changed: {name}')
            verify_source_manifest(source['manifest'])
            print(f'Oracle already complete: {proof["status"]}; {output/"result/README.md"}',flush=True);return
        torch.manual_seed(config['oracle_seed'])
        if torch.cuda.is_available():torch.cuda.manual_seed_all(config['oracle_seed'])
        rows,frontier=execute_batches(source,store,FrozenOracleBackend,config)
        store.export_targets()
        result=write_reports(output,rows,frontier,source['manifest'],protocol,config,source_path,source['carrier'].signs.cpu())
        verify_source_manifest(source['manifest'])
        files={p.relative_to(output).as_posix():sha256(p) for directory in ('metrics','result')
               for p in (output/directory).rglob('*') if p.is_file() and p.name!='execution_error.json'}
        files['oracle_targets/complete.json']=sha256(output/'oracle_targets/complete.json')
        freeze(complete,{'status':result['status'],'source_manifest_sha256':digest(source['manifest']),
                         'protocol_sha256':digest(protocol),'files':files,'teacher_training_started':False})
        print(f'{result["status"]}: {output/"result/README.md"}',flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plan-only',action='store_true');args=parser.parse_args(argv)
    reject_overlap(args.output,args.source_run)
    if args.plan_only:
        print(json.dumps({'config':ORACLE_CONFIG,'source_run':str(args.source_run.resolve()),
            'output':str(args.output.resolve()),'resume':'automatic by committed batch',
            'source_read_only':True,'teacher_training_started':False,'gpu_work':False},indent=2));return
    try:run_oracle(args.source_run,args.output)
    except Exception as exc:
        # Only write an error inside a recognized disjoint oracle output.
        try:
            recognized=(read_json(args.output/'config.json').get('oracle_version')==ORACLE_CONFIG['oracle_version'] and
                Path(read_json(args.output/'source_manifest.json')['source_run']).resolve()==args.source_run.resolve())
        except (OSError,ValueError,KeyError,TypeError):recognized=False
        if recognized:
            write_json(args.output/'result/execution_error.json',{'status':'oracle_execution_error',
                'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),
                'teacher_training_started':False})
        raise


if __name__=='__main__':main()
