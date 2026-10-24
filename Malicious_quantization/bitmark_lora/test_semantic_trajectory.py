import json
import pytest
import torch
from run import parse_args, config_from_args, main
from semantic_sign import is_sign, detection_count


def test_v2_is_same_sign_carrier():
    assert is_sign({'teacher_objective':'semantic_sign_v2_trajectory'})
    assert is_sign({'teacher_objective':'semantic_sign_v1'})
    assert detection_count(32,.01)==24


def test_v2_locked_teacher_plan(capsys):
    main(['--teacher-objective','semantic_sign_v2_trajectory','--teacher-only','--plan-only'],method='semantic_fp')
    plan=json.loads(capsys.readouterr().out)
    assert plan['students']==[]
    assert not any('student' in s.lower() for s in plan['stages'])
    c=plan['config']
    assert (c['train_n'],c['validation_n'],c['calibration_n'])==(5000,512,1000)
    assert c['teacher_train_steps']==1500 and c['train_bit_count_margin']==0
    assert c['coordinate_margin_mu']==.1 and c['tail_unroll_calls']==4
    assert c['teacher_eval_checkpoints']==[100,250,500,750,1000,1500]
    assert 'coordinate_margin_mu_candidates' not in c


def test_v2_rejects_sweep_and_non_teacher_mode():
    for extra in ([],['--teacher-only','--coordinate-margin-mu-candidates','.1','.2'],['--teacher-only','--tail-unroll-calls','3']):
        with pytest.raises(SystemExit):
            parse_args(['--teacher-objective','semantic_sign_v2_trajectory',*extra],method='semantic_fp')


def test_checkpoint_selection_keeps_diagnostic_above_lpips_gate():
    from semantic_sign_trajectory_train import choose_checkpoint
    def row(step,tpr,lp,pix=.1,clip=.005):
        return dict(step=step,metrics=dict(detection_rate=tpr),quality=dict(paired_lpips=lp,paired_pixel_mse=pix,clip_drop=clip))
    assert choose_checkpoint([row(100,.93,.2),row(250,1.,.3)])['step']==100
    assert choose_checkpoint([row(250,.94,.1),row(100,.94,.1)])['step']==100
    assert choose_checkpoint([row(100,.8,.01)]) is None


def _pndm():
    from diffusers import PNDMScheduler
    s=PNDMScheduler(skip_prk_steps=True,steps_offset=1)
    s.set_timesteps(50)
    return s


def test_cfg_equivalence():
    from types import SimpleNamespace
    from semantic_trajectory import cfg_forward
    s=_pndm();z=torch.randn(1,4,2,2);text=torch.ones(1,2,3);empty=torch.zeros_like(text)
    class Net:
        def __call__(self,z,t,encoder_hidden_states):
            return SimpleNamespace(sample=z*.1+encoder_hidden_states.mean((1,2))[:,None,None,None])
    net=Net();t=s.timesteps[0]
    a=net(z,t,encoder_hidden_states=empty).sample;b=net(z,t,encoder_hidden_states=text).sample
    torch.testing.assert_close(cfg_forward(net,s,z,t,text,empty),a+7.5*(b-a))


def test_native_trace_roundtrip_and_tail_gradients(tmp_path):
    from semantic_trajectory import NativeTrace, restore_scheduler, replay_tail, TraceWriter, TraceReader
    torch.manual_seed(1234);s=_pndm();z=torch.randn(1,4,3,3);initial=z.clone();outputs=[]
    with NativeTrace(s,4,5) as trace:
        for t in s.timesteps:
            eps=z*.05+float(t)*.00001
            outputs.append(eps.detach().clone());z=s.step(eps,t,z).prev_sample
    ordinary=_pndm();plain=initial.clone()
    for t in ordinary.timesteps:plain=ordinary.step(plain*.05+float(t)*.00001,t,plain).prev_sample
    assert torch.equal(plain,z)  # Recorder is observational.
    record=trace.records()[0]
    assert record['boundary_index']==len(s.timesteps)-4
    assert len(record['anchor_indices'])==5
    restored=restore_scheduler(_pndm(),record,'cpu')
    replay=replay_tail(restored,record,lambda latent,t:latent*.05+float(t)*.00001)
    torch.testing.assert_close(replay,z,rtol=0,atol=0)
    writer=TraceWriter(tmp_path,{'test':True},shard_size=1)
    writer.append(0,record,{'seed':0});writer.finish(1)
    loaded=TraceReader(tmp_path).get(0)
    restored=restore_scheduler(_pndm(),loaded,'cpu')
    torch.testing.assert_close(replay_tail(restored,loaded,lambda latent,t:latent*.05+float(t)*.00001),z,rtol=0,atol=0)
    # Independent losses propagate to every new scheduler prediction.
    for loss_fn in (lambda x:x.square().mean(),lambda x:(x-.2).square().mean(),lambda x:torch.relu(100.-x.mean()).square()):
        restored=restore_scheduler(_pndm(),loaded,'cpu');params=torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(.01)) for _ in range(4)]);seen=[]
        def prediction(latent,t):
            index=len(seen);out=latent*.05+params[index];out.retain_grad();seen.append(out);return out
        result=replay_tail(restored,loaded,prediction)
        loss_fn(result).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad) and p.grad.abs()>0 for p in params)
        assert all(x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum()>0 for x in seen)
        assert not loaded['boundary_latent'].requires_grad
        assert all(not v.requires_grad for v in loaded['boundary_state']['ets'])


def test_trace_corruption_is_rejected(tmp_path):
    from semantic_trajectory import NativeTrace,TraceWriter,TraceReader
    s=_pndm();z=torch.randn(1,4,2,2)
    with NativeTrace(s,4,5) as trace:
        for t in s.timesteps:z=s.step(z*.1,t,z).prev_sample
    w=TraceWriter(tmp_path,{'x':1},1);w.append(0,trace.records()[0],{});w.finish(1)
    path=tmp_path/'shard_00000.safetensors';path.write_bytes(path.read_bytes()+b'bad')
    with pytest.raises(RuntimeError):TraceReader(tmp_path)


def test_trace_protocol_change_and_detached_history(tmp_path):
    from semantic_trajectory import NativeTrace,restore_scheduler,TraceWriter
    s=_pndm();z=torch.randn(1,4,2,2,requires_grad=True)
    with NativeTrace(s) as tr:
        for t in s.timesteps:z=s.step(z*.05,t,z).prev_sample
    r=tr.records()[0]
    assert not r['boundary_latent'].requires_grad
    assert not any(t.requires_grad for t in r['boundary_state']['ets'])
    broken=__import__('copy').deepcopy(r);broken['boundary_state']['counter']-=1
    with pytest.raises(ValueError):restore_scheduler(_pndm(),broken,'cpu')
    other=_pndm();other.register_to_config(steps_offset=2)
    with pytest.raises(ValueError):restore_scheduler(other,r,'cpu')
    TraceWriter(tmp_path,{'config':'one'})
    with pytest.raises(RuntimeError):TraceWriter(tmp_path,{'config':'two'})


def test_combined_objective_detaches_targets_and_reaches_lora(monkeypatch):
    from types import SimpleNamespace
    import semantic_sign_trajectory_train as train
    from semantic_trajectory import NativeTrace
    from semantic_carrier import SemanticCarrier
    class TinyLoRA(torch.nn.Module):
        def __init__(self):
            super().__init__();self.lora_A=torch.nn.Parameter(torch.ones(1));self.lora_B=torch.nn.Parameter(torch.tensor([.01]))
        def forward(self,z,t,encoder_hidden_states):
            return SimpleNamespace(sample=.05*z+self.lora_A*self.lora_B)
    s=_pndm();z=torch.randn(1,4,3,3)*.01
    with NativeTrace(s) as trace:
        for t in s.timesteps:z=s.step(z*.05,t,z).prev_sample
    record=trace.records()[0]
    class VAE:
        config=SimpleNamespace(scaling_factor=1.)
        def decode(self,z):return SimpleNamespace(sample=z.tanh())
    class Ext:
        vae=VAE()
        def image_latent(self,x):return x
        def encode_latent(self,x,text_hidden):return x.flatten(1)[:,:32]*.01
    carrier=SemanticCarrier(torch.zeros(32),torch.eye(32),torch.ones(32),torch.ones(32),list(range(32)),
                            carrier_version='semantic_sign_v1',medians=torch.zeros(32))
    for loss_kind in ('fp','lpips','pixel'):
        if loss_kind=='pixel':carrier.signs[::2]=-1
        net=TinyLoRA();seen=[]
        original=train.cfg_forward
        def observe(*a,**kw):
            out=original(*a,**kw);out.retain_grad();seen.append(out);return out
        monkeypatch.setattr(train,'cfg_forward',observe)
        c=dict(teacher_precision='no',coordinate_margin_mu=.1,sign_k_train=24,lambda_preserve=0.,
               lambda_fp=1. if loss_kind=='fp' else 0.,lambda_quality=0. if loss_kind=='fp' else 1.,
               lambda_lpips=1. if loss_kind=='lpips' else 0.,lambda_pixel=1. if loss_kind=='pixel' else 0.)
        clean=torch.zeros(1,4,3,3);coords=torch.zeros(1,32)
        result=train.trajectory_backward(net,Ext(),carrier,_pndm,lambda a,b:(a-b).square().mean((1,2,3)),
                    [record],torch.zeros(1,2,3),torch.zeros(1,2,3),clean,coords,[0],c,lambda x:x.backward())
        assert net.lora_B.grad is not None and torch.isfinite(net.lora_B.grad).all() and net.lora_B.grad.abs().sum()>0
        assert len(seen)==5 and all(x.grad is not None and x.grad.abs().sum()>0 for x in seen[1:])
        assert clean.grad is None and coords.grad is None
        selected=(coords*carrier.signs).topk(24,dim=-1).indices
        expected_correct=int((carrier.signs.expand_as(coords).gather(-1,selected)==1).sum())
        assert result['samples'][0]['selected_clean_correct']==expected_correct
        monkeypatch.setattr(train,'cfg_forward',original)


def test_teacher_only_dispatch_stops_before_downstream(tmp_path,monkeypatch):
    import run as runner
    calls=[]
    monkeypatch.setattr(runner,'verify_vendor',lambda:None)
    monkeypatch.setattr(runner,'preflight',lambda a:{'device':'test'})
    monkeypatch.setattr(runner,'runtime_root',lambda:tmp_path/'runtime')
    monkeypatch.setattr(runner.importlib.metadata,'version',lambda p:'test')
    monkeypatch.setattr(runner,'stage',lambda run,label,command:calls.append((label,command)))
    runner.main(['--teacher-objective','semantic_sign_v2_trajectory','--teacher-only','--output',str(tmp_path/'run')],method='semantic_fp')
    assert calls[-1][0]=='05_trajectory_rescue'
    assert len(calls)==5
    assert not any('student.py' in str(command) for _,command in calls)
    assert not any('--stage generate' in ' '.join(command) for _,command in calls)


def test_native_batch_reassembly_is_exact():
    from semantic_trajectory import NativeTrace,batch_records,restore_scheduler,replay_tail,scheduler_state,assert_state_close
    torch.manual_seed(12);s=_pndm();z=torch.randn(4,4,2,2)
    with NativeTrace(s) as trace:
        for t in s.timesteps:z=s.step(z*.05,t,z).prev_sample
    record=batch_records(trace.records());restored=restore_scheduler(_pndm(),record,'cpu')
    final=replay_tail(restored,record,lambda z,t:z*.05)
    torch.testing.assert_close(final,z,atol=0,rtol=0)
    assert_state_close(scheduler_state(restored),record['final_state'],atol=0,rtol=0)


def test_incomplete_trace_shard_resumes_without_rewriting_completed(tmp_path):
    from semantic_trajectory import NativeTrace,TraceWriter,TraceReader
    s=_pndm();z=torch.randn(1,4,2,2)
    with NativeTrace(s) as tr:
        for t in s.timesteps:z=s.step(z*.05,t,z).prev_sample
    record=tr.records()[0];w=TraceWriter(tmp_path,{'config':1},shard_size=1);w.append(0,record,{'seed':123})
    original=(tmp_path/'shard_00000.safetensors').read_bytes()
    (tmp_path/'shard_00001.partial').write_bytes(b'interrupted')
    w=TraceWriter(tmp_path,{'config':1},shard_size=1)
    assert w.has(0) and not w.has(1)
    w.append(1,record,{'seed':456});w.finish(2)
    assert original==(tmp_path/'shard_00000.safetensors').read_bytes()
    assert len(TraceReader(tmp_path).index)==2


def test_v2_short_cpu_training_resumes_exactly(tmp_path):
    import random
    from types import SimpleNamespace
    from semantic_trajectory import NativeTrace
    from semantic_sign_trajectory_train import trajectory_backward
    from semantic_checkpoint import save_training_state,restore_training_state
    from semantic_carrier import SemanticCarrier
    class LoRA(torch.nn.Module):
        def __init__(self):super().__init__();self.lora=torch.nn.Parameter(torch.tensor(.01))
        def forward(self,z,t,encoder_hidden_states):return SimpleNamespace(sample=.05*z+self.lora)
    class VAE:
        config=SimpleNamespace(scaling_factor=1.)
        def decode(self,z):return SimpleNamespace(sample=z.tanh())
    class Ext:
        vae=VAE()
        def image_latent(self,x):return x
        def encode_latent(self,x,text_hidden):return x.flatten(1)[:,:32]*.01
    torch.manual_seed(42);random.seed(42);s=_pndm();z=torch.randn(1,4,3,3)*.01
    with NativeTrace(s) as tr:
        for t in s.timesteps:z=s.step(z*.05,t,z).prev_sample
    record=tr.records()[0]
    carrier=SemanticCarrier(torch.zeros(32),torch.eye(32),torch.ones(32),torch.ones(32),list(range(32)),
                            carrier_version='semantic_sign_v1',medians=torch.zeros(32))
    c=dict(teacher_precision='no',coordinate_margin_mu=.1,sign_k_train=24,lambda_preserve=1.,lambda_fp=.8,
           lambda_quality=1.,lambda_lpips=1.,lambda_pixel=1.)
    def setup():
        net=LoRA();optimizer=torch.optim.AdamW(net.parameters(),lr=1e-4);rng=torch.Generator().manual_seed(19)
        return net,optimizer,rng
    def update(net,optimizer,rng):
        anchor=int(torch.randint(5,(1,),generator=rng));optimizer.zero_grad()
        trajectory_backward(net,Ext(),carrier,_pndm,lambda a,b:(a-b).square().mean(),[record],
            torch.zeros(1,2,3),torch.zeros(1,2,3),torch.zeros(1,4,3,3),torch.zeros(1,32),[anchor],c,lambda loss:loss.backward())
        optimizer.step()
    a,opt_a,rng_a=setup();update(a,opt_a,rng_a)
    dest=tmp_path/'checkpoint-1';protocol={'teacher_objective':'semantic_sign_v2_trajectory'}
    save_training_state(dest,a,opt_a,None,rng_a,[2,0,1],1,1,protocol)
    update(a,opt_a,rng_a);update(a,opt_a,rng_a)
    b,opt_b,rng_b=setup();state=restore_training_state(dest,b,opt_b,None,rng_b,protocol)
    assert state['order']==[2,0,1] and state['offset']==1 and state['step']==1
    update(b,opt_b,rng_b);update(b,opt_b,rng_b)
    torch.testing.assert_close(a.lora,b.lora,atol=0,rtol=0)
    for key,value in opt_a.state_dict()['state'][0].items():
        torch.testing.assert_close(value,opt_b.state_dict()['state'][0][key],atol=0,rtol=0)
    assert torch.equal(rng_a.get_state(),rng_b.get_state())


def test_cached_score_resume_normalizes_carrier_device(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import semantic_teacher as teacher
    import semantic_report
    from semantic_carrier import SemanticCarrier
    from common import write_json,image_name
    write_json(tmp_path/'config.json',{'teacher_objective':'semantic_sign_v2_trajectory','score_batch':1})
    write_json(tmp_path/'prompts.json',{'train':[{'text':'x'}]})
    write_json(tmp_path/'assets.json',{})
    write_json(tmp_path/'carrier/complete.json',{})
    folder=tmp_path/'images/teacher_clean_train';folder.mkdir(parents=True)
    (folder/image_name(0)).write_bytes(b'test')
    write_json(tmp_path/'scores/teacher_clean_train.json',[])
    cpu_carrier=SimpleNamespace(signs=torch.ones(32))
    class LoadedDeviceCarrier:
        signs=torch.ones(32,device='meta')  # CPU-only stand-in for a loaded GPU carrier.
        def to(self,device):
            assert device=='cpu';return cpu_carrier
    monkeypatch.setattr(teacher,'extractor',lambda run:object())
    monkeypatch.setattr(SemanticCarrier,'load',lambda *args:LoadedDeviceCarrier())
    seen=[]
    def validate(run,label,n,carrier):
        assert carrier.signs.device.type=='cpu'
        seen.append(label)
    monkeypatch.setattr(semantic_report,'score_rows',validate)
    teacher.score(tmp_path,['teacher_clean_train'])
    assert seen==['teacher_clean_train']
