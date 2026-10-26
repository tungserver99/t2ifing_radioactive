"""CPU proofs for independent cached-tail controls; no pretrained downloads."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch


def _record(batch=1):
    from diffusers import PNDMScheduler
    from semantic_trajectory import NativeTrace,batch_records
    def factory():
        s=PNDMScheduler(skip_prk_steps=True,steps_offset=1);s.set_timesteps(50);return s
    s=factory();z=torch.randn(batch,4,3,3)*.01
    with NativeTrace(s) as trace:
        for t in s.timesteps:z=s.step(z*.05,t,z).prev_sample
    return factory,batch_records(trace.records())


def test_oracle_zero_controls_and_all_four_gradients():
    from semantic_oracle_math import replay_controls,dynamic_detector_loss
    factory,record=_record(2)
    actual=replay_controls(factory,record,device='cpu')
    torch.testing.assert_close(actual,record['final_latent'],atol=0,rtol=0)
    for key in ('boundary_latent','tail_outputs'):record[key].requires_grad_(True)
    delta=torch.zeros_like(record['tail_outputs'],requires_grad=True)
    result=replay_controls(factory,record,delta,device='cpu',dtype=torch.float32)
    loss,_=dynamic_detector_loss(result.flatten(1)[:,:32],torch.ones(32))
    loss.sum().backward()
    assert all(torch.isfinite(delta.grad[t]).all() and delta.grad[t].abs().sum()>0 for t in range(4))
    assert record['boundary_latent'].grad is None and record['tail_outputs'].grad is None


def test_dynamic_current_top24_and_safe_bits():
    from semantic_oracle_math import dynamic_detector_loss
    r=torch.cat([torch.full((1,23),.1),torch.tensor([[-.01]]),torch.full((1,8),-2.)],dim=1).requires_grad_()
    loss,info=dynamic_detector_loss(r,torch.ones(32));loss.sum().backward()
    assert set(info['selected_indices'][0].tolist())==set(range(24))
    assert r.grad[0,:23].abs().sum()==0 and r.grad[0,24:].abs().sum()==0 and r.grad[0,23]<0
    changed=r.detach().clone();changed[0,31]=.2
    loss,new=dynamic_detector_loss(changed,torch.ones(32))
    assert 31 in new['selected_indices'][0] and 23 not in new['selected_indices'][0]
    assert loss.item()==0 and new['feasible'].item()


def test_fp32_drift_diagnostics_reproduce_server_failing_element():
    from semantic_oracle_math import precision_drift
    # Actual ID27, channel1,row39,column0. FP64 agrees with FP32, not FP16.
    original=torch.tensor([-2.015625],dtype=torch.float16)
    fp32=torch.tensor([-2.021880865097046],dtype=torch.float32)
    report=precision_drift(fp32,original,atol=.002,rtol=.002)
    assert report['fp32_graph_latent_outside_reference_tolerance']==1
    assert report['fp32_graph_latent_elements']==1
    assert report['fp32_graph_latent_max_error']==pytest.approx(.006255865097046)
    assert report['fp32_graph_latent_rms_error']==pytest.approx(.006255865097046)
    assert report['source_latent_dtype']=='torch.float16'
    assert report['fp32_graph_drift_is_diagnostic'] is True
    with pytest.raises(ValueError,match='Nonfinite'):
        precision_drift(fp32*float('nan'),original,atol=.002,rtol=.002)


def test_real_id27_scheduler_fixture_reproduces_drift_and_preserves_control_gradients():
    import json
    from pathlib import Path
    from diffusers import PNDMScheduler
    from semantic_trajectory import unpack_tree
    from semantic_oracle_math import replay_controls,precision_drift
    data=json.loads((Path(__file__).parent/'fixtures/oracle_fp32_drift_id27.json').read_text())
    tensors={k:torch.tensor(v['values'],dtype=getattr(torch,v['dtype'])) for k,v in data['tensors'].items()}
    record=unpack_tree(data['state'],tensors)
    factory=lambda:PNDMScheduler.from_config(data['scheduler_config'])
    fp32=replay_controls(factory,record,device='cpu',dtype=torch.float32)
    fp64=replay_controls(factory,record,device='cpu',dtype=torch.float64)
    diagnostics=precision_drift(fp32,record['final_latent'],atol=.002,rtol=.002)
    assert diagnostics['fp32_graph_latent_outside_reference_tolerance']==1
    assert fp32[0,1,0,0].item()==pytest.approx(-2.021880865097046,abs=1e-6)
    assert float((fp32.double()-fp64).abs().max())<1e-6
    delta=torch.zeros_like(record['tail_outputs'],dtype=torch.float32,requires_grad=True)
    controlled=replay_controls(factory,record,delta,device='cpu',dtype=torch.float32)
    torch.testing.assert_close(controlled,fp32,atol=0,rtol=0)
    controlled.square().sum().backward()
    assert all(torch.isfinite(delta.grad[i]).all() and delta.grad[i].abs().sum()>0 for i in range(4))


@pytest.mark.parametrize('failure',[None,'native','png','coordinates','nonfinite'])
def test_begin_batch_reports_fp32_drift_but_retains_native_png_coordinate_gates(monkeypatch,failure):
    import semantic_oracle as oracle
    import semantic_teacher
    import semantic_sign_train
    from PIL import Image
    factory,record=_record()
    backend=oracle.FrozenOracleBackend.__new__(oracle.FrozenOracleBackend)
    backend.device='cpu';backend.root=__import__('pathlib').Path('.')
    backend.factory=factory
    backend.source={'prompts':{'train':[{'text':'test'}]},'scores':[{'coordinates':[0.]*32}]}
    backend.ext=SimpleNamespace(text=lambda prompts:torch.zeros(1,2,8),
        encode_image=lambda image,prompts:torch.ones(1,32) if failure=='coordinates' else torch.zeros(1,32))
    backend.carrier=SimpleNamespace(coordinates=lambda x:x)
    backend.canonical_images=lambda latent:[Image.new('RGB',(3,3),(255,255,255) if failure=='png' else (128,128,128))]
    monkeypatch.setattr(semantic_teacher,'pixels',lambda paths:torch.full((1,3,3,3),128/127.5-1))
    monkeypatch.setattr(semantic_sign_train,'decoded_image',lambda ext,z:z)
    def replay(factory,rec,device,dtype=None):
        value=rec['final_latent'].clone()
        if dtype is not None:return value.float()+(float('nan') if failure=='nonfinite' else .1)
        return value+(1. if failure=='native' else 0.)
    monkeypatch.setattr(oracle,'replay_controls',replay)
    if failure is not None:
        with pytest.raises((AssertionError,ValueError)):
            backend.begin_batch([0],[record])
    else:
        assert len(backend.begin_batch([0],[record]))==1
        assert backend.zero[0]['fp32_graph_latent_outside_reference_tolerance']>0
        assert backend.zero[0]['native_latent_max_error']==0
        torch.testing.assert_close(backend.reference[0],record['final_latent'])


def test_ste_matches_actual_diffusers_png_roundtrip_and_identity_gradient(tmp_path):
    from diffusers.image_processor import VaeImageProcessor
    from PIL import Image
    from semantic_oracle_math import fake_quantize_rgb8_ste
    torch.manual_seed(8)
    x=torch.rand(1,3,17,19)*2-1;x.requires_grad_()
    image=VaeImageProcessor().postprocess(x.detach(),output_type='pil')[0]
    path=tmp_path/'image.png';image.save(path)
    with Image.open(path) as im:expected=torch.from_numpy(np.array(im.convert('RGB'),copy=True)).permute(2,0,1).float()/127.5-1
    actual=fake_quantize_rgb8_ste(x)
    torch.testing.assert_close(actual[0],expected,atol=0,rtol=0)
    actual.sum().backward();assert torch.equal(x.grad,torch.ones_like(x))


def test_relative_energy_and_no_source_gradient():
    from semantic_oracle_math import relative_energy
    delta=torch.ones(4,2,4,2,2,requires_grad=True)
    outputs=torch.full_like(delta,2.,requires_grad=True)
    energy,per_call,drms,erms=relative_energy(delta,outputs)
    torch.testing.assert_close(energy,torch.full((2,),.25))
    assert per_call.shape==(2,4) and torch.equal(drms,torch.ones(2,4))
    assert torch.equal(erms,torch.full((2,4),2.))
    energy.sum().backward();assert outputs.grad is None and delta.grad.abs().sum()>0


def test_adaptive_weights_stop_after_first_feasibility():
    from semantic_oracle_math import detector_weights
    weights=torch.ones(2);ever=torch.tensor([False,True])
    for step in range(1,121):weights=detector_weights(weights,ever,step)
    assert weights.tolist()==[32.,1.]


def test_candidate_tie_order_and_best_diagnostic_fallback():
    from semantic_oracle_math import CandidateTracker
    t=CandidateTracker(1)
    delta=torch.zeros(4,1,4,2,2);latent=torch.zeros(1,4,2,2)
    coords=torch.ones(1,32)*.1
    def add(step,lp,pix,energy):
        t.consider(0,step,delta,latent,coords,torch.ones(32),lp,pix,energy,[energy]*4,[0.]*4,[1.]*4,2.)
    add(10,.1,.2,.3);add(9,.1,.2,.3);add(11,.1,.2,.2);add(12,.1,.1,.9);add(13,.09,.9,.9)
    best=t.results()[0]
    assert best['best_step']==13 and best['first_feasible_step']==10 and best['float_feasible']
    assert best['delta'].shape==(4,4,2,2) and best['final_latent'].shape==(1,4,2,2)
    f=CandidateTracker(1)
    f.consider(0,0,delta,latent,torch.full((1,32),-.1),torch.ones(32),0.,0.,0.,[0.]*4,[0.]*4,[1.]*4,1.)
    assert not f.results()[0]['float_feasible']


def test_subset_is_seeded_random_sorted_unique():
    from semantic_oracle_math import select_sample_ids
    a=select_sample_ids(5000);b=select_sample_ids(5000)
    assert a==b and a==sorted(set(a)) and len(a)==256 and a!=list(range(256))


@pytest.mark.parametrize('where',['delta','energy','coordinates','image'])
def test_oracle_rejects_nonfinite(where):
    from semantic_oracle_math import replay_controls,relative_energy,dynamic_detector_loss,fake_quantize_rgb8_ste
    bad=torch.tensor(float('nan'))
    with pytest.raises(ValueError,match='Nonfinite'):
        if where=='delta':
            factory,r=_record();replay_controls(factory,r,torch.zeros_like(r['tail_outputs'])+bad,device='cpu',dtype=torch.float32)
        elif where=='energy':relative_energy(torch.ones(4,1,4,2,2)*bad,torch.ones(4,1,4,2,2))
        elif where=='coordinates':dynamic_detector_loss(torch.ones(1,32)*bad,torch.ones(32))
        else:fake_quantize_rgb8_ste(torch.ones(1,3,2,2)*bad)


def test_optimizer_updates_only_residuals_and_keeps_best_not_last():
    from semantic_oracle import optimize_batch
    from semantic_oracle_math import ORACLE_CONFIG,replay_controls,dynamic_detector_loss,relative_energy
    factory,record=_record(1)
    class Backend:
        device='cpu'
        def __init__(self):self.frozen=torch.nn.Linear(1,1).requires_grad_(False);self.calls=0
        def training_view(self,index,rec,delta):
            self.calls+=1
            z=replay_controls(factory,rec,delta,device='cpu',dtype=torch.float32)
            coords=z.flatten(1)[:,:32]+delta.flatten(2).mean((0,2))[:,None]
            det,info=dynamic_detector_loss(coords,torch.ones(32))
            energy,per,drms,erms=relative_energy(delta,rec['tail_outputs'])
            return dict(latent=z,coordinates=coords,signs=torch.ones(32),detector=det[0],
                        lpips=z.square().mean(),pixel=z.square().mean(),energy=energy[0],
                        per_call=per[0],delta_rms=drms[0],cached_rms=erms[0])
        def assert_frozen(self):assert all(p.grad is None and not p.requires_grad for p in self.frozen.parameters())
    b=Backend();c={**ORACLE_CONFIG,'oracle_steps':3,'frontier_log_every':1}
    candidates,frontier=optimize_batch([record],b,[42],c)
    assert b.calls==4 and len(frontier)==4 and candidates[0]['best_step']<=3
    assert all(torch.isfinite(candidates[0][k]).all() for k in ('delta','final_latent'))
    assert not candidates[0]['delta'].requires_grad


def test_cli_plan_is_locked_and_creates_no_output(tmp_path,capsys):
    from semantic_oracle import main
    import json
    main(['--source-run',str(tmp_path/'historical'),'--output',str(tmp_path/'oracle'),'--plan-only'])
    plan=json.loads(capsys.readouterr().out)
    assert plan['config']['oracle_n']==256 and plan['config']['oracle_steps']==120
    assert plan['config']['oracle_seed']==20261026 and plan['config']['target_shard_size']==32
    assert plan['teacher_training_started'] is False and not (tmp_path/'oracle').exists()
    with pytest.raises(SystemExit):main(['--source-run','s','--output','o','--oracle-steps','1'])


def test_real_frozen_vae_semantic_unet_propagate_detector_gradient_to_all_controls():
    from diffusers import AutoencoderKL,UNet2DConditionModel,PNDMScheduler
    from semantic_carrier import SemanticExtractor,SemanticCarrier
    from semantic_sign_train import decoded_image,decoded_coordinates
    from semantic_trajectory import NativeTrace,batch_records
    from semantic_oracle_math import replay_controls,fake_quantize_rgb8_ste,dynamic_detector_loss
    torch.manual_seed(101)
    reference=UNet2DConditionModel(sample_size=8,in_channels=4,out_channels=4,
        down_block_types=('CrossAttnDownBlock2D',),up_block_types=('CrossAttnUpBlock2D',),
        block_out_channels=(32,),layers_per_block=1,norm_num_groups=4,cross_attention_dim=8,attention_head_dim=4)
    vae=AutoencoderKL(in_channels=3,out_channels=3,latent_channels=4,
        down_block_types=('DownEncoderBlock2D',)*4,up_block_types=('UpDecoderBlock2D',)*4,
        block_out_channels=(8,8,8,8),layers_per_block=1,norm_num_groups=4,sample_size=64)
    ext=SemanticExtractor(SimpleNamespace(unet=reference,vae=vae,tokenizer=None,text_encoder=torch.nn.Identity()),device='cpu')
    carrier=SemanticCarrier(torch.zeros(32),torch.eye(32),torch.full((32,),1000.),torch.ones(32),list(range(32)),
        carrier_version='semantic_sign_v1',medians=torch.zeros(32))
    carrier.signs.requires_grad_(True)  # The detector must still detach the secret signs.
    def factory():
        s=PNDMScheduler(skip_prk_steps=True,steps_offset=1);s.set_timesteps(50);return s
    s=factory();z=torch.randn(1,4,8,8)*.01
    with NativeTrace(s) as tr:
        for t in s.timesteps:z=s.step(z*.05,t,z).prev_sample
    record=batch_records(tr.records());record['boundary_latent'].requires_grad_(True);record['tail_outputs'].requires_grad_(True)
    delta=torch.zeros_like(record['tail_outputs'],requires_grad=True)
    final=replay_controls(factory,record,delta,device='cpu',dtype=torch.float32)
    image=decoded_image(ext,final)
    coordinates=decoded_coordinates(ext,carrier,fake_quantize_rgb8_ste(image),torch.randn(1,2,8))
    loss,_=dynamic_detector_loss(coordinates,carrier.signs);loss.sum().backward()
    assert all(torch.isfinite(delta.grad[t]).all() and delta.grad[t].abs().sum()>0 for t in range(4))
    assert all(p.grad is None and not p.requires_grad for model in (reference,vae) for p in model.parameters())
    assert carrier.signs.grad is None and record['tail_outputs'].grad is None and record['boundary_latent'].grad is None


def test_full_oracle_interrupt_resume_preserves_committed_batch_and_source(tmp_path,monkeypatch):
    from PIL import Image
    from semantic_sign import image_evidence
    from semantic_trajectory import batch_records
    from semantic_oracle_math import ORACLE_CONFIG,replay_controls,dynamic_detector_loss,relative_energy
    from common import write_json,read_json,sha256,image_name
    import semantic_oracle as oracle
    source=tmp_path/'historical';source.mkdir();write_json(source/'config.json',{'frozen':True})
    image_dir=source/'images/teacher_clean_train';image_dir.mkdir(parents=True)
    for ident in range(4):Image.new('RGB',(3,3),(128,128,128)).save(image_dir/image_name(ident))
    before={p.relative_to(source).as_posix():sha256(p) for p in source.rglob('*') if p.is_file()}
    factory,record=_record(1)
    manifest={'source_run':str(source.resolve()),'files':before,'external_files':{}}
    clean=image_evidence(torch.zeros(1,32),torch.ones(32))[0]
    ctx={'manifest':manifest,'reader':SimpleNamespace(get=lambda i:record),'scores':[{'coordinates':[0.]*32,**clean} for _ in range(4)],
         'prompts':{'train':[{'text':str(i)} for i in range(4)]},'carrier':SimpleNamespace(signs=torch.ones(32))}
    monkeypatch.setattr(oracle,'load_source',lambda path:ctx)
    monkeypatch.setattr(oracle,'ORACLE_CONFIG',{**ORACLE_CONFIG,'oracle_n':4,'oracle_batch_size':2,'oracle_steps':2,'frontier_log_every':1})
    calls=[];fail=[True]
    class Backend:
        device='cpu'
        def __init__(self,context):pass
        def assert_frozen(self):pass
        def begin_batch(self,ids,records):calls.append(list(ids));return records
        def training_view(self,i,r,delta):
            z=replay_controls(factory,r,delta,device='cpu',dtype=torch.float32)
            coords=z.flatten(1)[:,:32]
            det,_=dynamic_detector_loss(coords,torch.ones(32));energy,per,drms,erms=relative_energy(delta,r['tail_outputs'])
            return dict(latent=z,coordinates=coords,signs=torch.ones(32),detector=det[0],lpips=z.square().mean(),
                        pixel=z.square().mean(),energy=energy[0],per_call=per[0],delta_rms=drms[0],cached_rms=erms[0])
        def evaluate(self,i,ident,candidate,output):
            if ident==2 and fail[0]:raise RuntimeError('simulated scheduler interruption')
            Image.new('RGB',(3,3),(128,128,128)).save(output/'oracle_images'/image_name(ident))
            evidence=image_evidence(torch.zeros(1,32),torch.ones(32))[0]
            return dict(image_id=ident,prompt=str(ident),clean_K=clean['K'],clean_p_img=clean['p_img'],clean_continuous_score=clean['score'],
                clean_coordinates=[0.]*32,coordinates=[0.]*32,K_png=evidence['K'],p_img_png=evidence['p_img'],
                analytic_detected_png=evidence['analytic_detected'],png_lpips=0.,png_pixel_mse=0.,clean_clip=1.,oracle_clip=1.,clip_drop=0.)
    monkeypatch.setattr(oracle,'FrozenOracleBackend',Backend)
    output=tmp_path/'oracle'
    with pytest.raises(RuntimeError,match='simulated'):oracle.run_oracle(source,output)
    first=output/'batches/batch_00000/complete.json';saved=first.read_bytes();stamp=first.stat().st_mtime_ns
    fail[0]=False;oracle.run_oracle(source,output)
    assert first.read_bytes()==saved and first.stat().st_mtime_ns==stamp
    assert calls==[[0,1],[2,3],[2,3]]
    assert before=={p.relative_to(source).as_posix():sha256(p) for p in source.rglob('*') if p.is_file()}
    assert read_json(output/'oracle_complete.json')['status']=='oracle_pass'
    def forbidden(*a):raise AssertionError('Completed oracle must not reload models')
    monkeypatch.setattr(oracle,'FrozenOracleBackend',forbidden)
    oracle.run_oracle(source,output)


def test_unrelated_output_is_not_modified_when_cli_fails(tmp_path,monkeypatch):
    from common import write_json
    import semantic_oracle as oracle
    source=tmp_path/'source';source.mkdir();output=tmp_path/'other_experiment';output.mkdir()
    write_json(output/'config.json',{'other':'experiment'});write_json(output/'source_manifest.json',{})
    def reject(*a):raise ValueError('unrelated output')
    monkeypatch.setattr(oracle,'run_oracle',reject)
    with pytest.raises(ValueError,match='unrelated'):
        oracle.main(['--source-run',str(source),'--output',str(output)])
    assert not (output/'result').exists()
