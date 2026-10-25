"""Replay rejections are complete reports, never permission to train."""
import pytest
import torch
from common import read_json, write_json


def test_image_checks_collect_all_samples_and_keep_original_gate():
    from semantic_replay_diagnostics import image_diagnostics
    clean=torch.zeros(4,3,2,2)
    replay=clean.clone();replay[0]=.01;replay[2]=.02
    reference=clean.clone();reference[0]=.01
    def distance(a,b):return (a-b).square().mean((1,2,3))
    rows=image_diagnostics(replay,clean,reference,distance,[0,1,2,3],
                           {'png_pixel_max':.1,'lpips_max':1e-5})
    assert [r['image_id'] for r in rows]==[0,1,2,3]
    assert [r['passed'] for r in rows]==[False,True,False,True]
    assert rows[0]['reference_png_lpips']>1e-5
    assert rows[0]['replay_reference_lpips']==0
    assert not rows[0]['passed']  # A PNG baseline never overrides the frozen gate.


def test_nonfinite_diagnostics_remain_errors():
    from semantic_replay_diagnostics import image_diagnostics
    x=torch.zeros(1,3,2,2);x[0,0,0,0]=float('nan')
    with pytest.raises(ValueError,match='Nonfinite'):
        image_diagnostics(x,x,x,lambda a,b:torch.zeros(1),[0],
                          {'png_pixel_max':1.,'lpips_max':1.})


def test_scheduler_integrity_errors_are_not_numerical_rejections():
    from semantic_replay_diagnostics import comparison_check,tensor_comparison
    from semantic_trajectory import assert_state_close
    def corrupt():raise ValueError('Replay scheduler scalar progression changed')
    with pytest.raises(ValueError,match='progression'):comparison_check('state',corrupt)
    result=comparison_check('numeric',lambda:torch.testing.assert_close(torch.tensor(1.),torch.tensor(2.)))
    assert result['passed'] is False
    with pytest.raises(ValueError,match='shape/dtype'):
        tensor_comparison('shape',torch.zeros(1),torch.zeros(2),0,0)
    with pytest.raises(ValueError,match='shape/dtype'):
        tensor_comparison('dtype',torch.zeros(1),torch.zeros(1,dtype=torch.float64),0,0)
    with pytest.raises(ValueError,match='shape/dtype'):
        comparison_check('state',lambda:assert_state_close(torch.zeros(1),torch.zeros(2),0,0))


def _campaign_inputs(path):
    write_json(path/'config.json',{'teacher_objective':'semantic_sign_v2_trajectory','teacher_only':True})
    write_json(path/'teacher_trace/complete.json',{'test':True})
    write_json(path/'result/fixed_key_clean_fpr.json',{'branches':{}})


def _patch_campaign_prerequisites(monkeypatch):
    import semantic_sign_report,semantic_report
    monkeypatch.setattr(semantic_sign_report,'freeze_analytic_verifier',lambda run:None)
    monkeypatch.setattr(semantic_sign_report,'validate_clean_null',lambda *a,**k:None)
    monkeypatch.setattr(semantic_report,'freeze_null',lambda *a,**k:None)


def test_failed_replay_finishes_report_without_training(tmp_path,monkeypatch,capsys):
    import semantic_sign_trajectory_train as train
    _campaign_inputs(tmp_path);_patch_campaign_prerequisites(monkeypatch)
    diagnostic={'passed':False,'samples':[{'image_id':i,'passed':False} for i in range(4)]}
    write_json(tmp_path/'teacher_trace/replay_diagnostics.json',diagnostic)
    monkeypatch.setattr(train,'verify_replay',lambda run:False)
    def forbidden(*a,**k):pytest.fail('Rejected replay must never start training or screening')
    monkeypatch.setattr(train,'train_trajectory',forbidden)
    monkeypatch.setattr(train,'screen_checkpoint',forbidden)
    train.campaign(tmp_path)
    report=read_json(tmp_path/'result/replay_failure.json')
    assert report['status']=='teacher_only_blocked'
    assert report['training_started'] is False and report['downstream_started'] is False
    assert report['replay']==diagnostic
    assert not (tmp_path/'teacher_complete.json').exists()
    assert 'replay' in (tmp_path/'result/README.md').read_text().lower()
    assert 'BLOCKED' in capsys.readouterr().out
    train.analyze_teacher_only(tmp_path)


def test_unexpected_replay_error_writes_report_and_propagates(tmp_path,monkeypatch):
    import semantic_sign_trajectory_train as train
    _campaign_inputs(tmp_path);_patch_campaign_prerequisites(monkeypatch)
    def broken(run):raise RuntimeError('CUDA out of memory')
    monkeypatch.setattr(train,'verify_replay',broken)
    with pytest.raises(RuntimeError,match='CUDA out of memory'):train.campaign(tmp_path)
    record=read_json(tmp_path/'result/teacher_execution_error.json')
    assert record['phase']=='replay_validation'
    assert record['error_type']=='RuntimeError'
    assert not (tmp_path/'teacher_complete.json').exists()


def test_blocked_analysis_rejects_changed_inputs(tmp_path):
    from semantic_replay_diagnostics import report_replay_failure,analyze_replay_failure
    _campaign_inputs(tmp_path)
    write_json(tmp_path/'teacher_trace/replay_diagnostics.json',{'passed':False,'samples':[]})
    report_replay_failure(tmp_path)
    write_json(tmp_path/'config.json',{'edited':True})
    with pytest.raises(RuntimeError,match='changed'):analyze_replay_failure(tmp_path)


def test_replay_upgrade_is_narrow_explicit_and_preserves_artifacts(tmp_path):
    from semantic_execution import upgrade_execution_manifest
    from semantic_replay_upgrade import BASELINE_HASHES,TARGET_HASHES,ADDED_FILES
    old={name:values[0] for name,values in BASELINE_HASHES.items()}
    old['unrelated.py']='unchanged'
    new={**old,**{name:values[0] for name,values in TARGET_HASHES.items()}}
    for name in ADDED_FILES:new.setdefault(name,'added-file')
    write_json(tmp_path/'config.json',{'watermark_method':'semantic_fp',
        'teacher_objective':'semantic_sign_v2_trajectory','teacher_only':True})
    write_json(tmp_path/'code_manifest.json',old)
    write_json(tmp_path/'teacher_trace/protocol.json',{'frozen_tolerance':1e-4})
    frozen=(tmp_path/'config.json').read_bytes(),(tmp_path/'teacher_trace/protocol.json').read_bytes()
    for bad in ({**new,'unrelated.py':'edited'},{**new,'semantic_trajectory.py':'unreviewed'}):
        with pytest.raises(RuntimeError,match='Unapproved'):upgrade_execution_manifest(tmp_path,bad)
        assert read_json(tmp_path/'code_manifest.json')==old
    upgrade_execution_manifest(tmp_path,new)
    assert read_json(tmp_path/'code_manifest.json')==new
    assert frozen==((tmp_path/'config.json').read_bytes(),(tmp_path/'teacher_trace/protocol.json').read_bytes())
    audit=read_json(tmp_path/'execution_upgrades/replay_diagnostics_v1.json')
    assert audit['before']==old and audit['after']==new
    upgrade_execution_manifest(tmp_path,new)


def test_replay_upgrade_rejects_started_training(tmp_path):
    from semantic_execution import upgrade_execution_manifest
    from semantic_replay_upgrade import BASELINE_HASHES,TARGET_HASHES,ADDED_FILES
    old={name:values[0] for name,values in BASELINE_HASHES.items()}
    new={**old,**{name:values[0] for name,values in TARGET_HASHES.items()}}
    for name in ADDED_FILES:new.setdefault(name,'added-file')
    write_json(tmp_path/'config.json',{'watermark_method':'semantic_fp',
        'teacher_objective':'semantic_sign_v2_trajectory','teacher_only':True})
    write_json(tmp_path/'code_manifest.json',old)
    write_json(tmp_path/'teacher_fp/trajectory/protocol.json',{'training':'started'})
    with pytest.raises(RuntimeError,match='before teacher training'):upgrade_execution_manifest(tmp_path,new)
    assert read_json(tmp_path/'code_manifest.json')==old


def test_reviewed_runtime_hashes_match_actual_checkout_and_upgrade(tmp_path):
    from common import HERE,sha256
    from semantic_execution import upgrade_execution_manifest
    from semantic_replay_upgrade import BASELINE_HASHES,ADDED_FILES
    current={p.name:sha256(p) for p in HERE.glob('*.py')}
    before={name:value for name,value in current.items() if name not in ADDED_FILES}
    before.update({name:values[0] for name,values in BASELINE_HASHES.items()})
    write_json(tmp_path/'code_manifest.json',before)
    write_json(tmp_path/'config.json',{'watermark_method':'semantic_fp',
        'teacher_objective':'semantic_sign_v2_trajectory','teacher_only':True})
    upgrade_execution_manifest(tmp_path,current)
    assert read_json(tmp_path/'code_manifest.json')==current


@pytest.mark.parametrize('png_offset,prediction_offset,passed',[(0.,0.,True),(.02,0.,False),(0.,.01,False)])
def test_real_scheduler_replay_collects_checks_and_only_proves_success(tmp_path,monkeypatch,png_offset,prediction_offset,passed):
    from types import SimpleNamespace
    from diffusers import PNDMScheduler
    import semantic_trajectory as trajectory
    from semantic_replay_diagnostics import comparison_check,image_diagnostics
    def scheduler(_=None):
        s=PNDMScheduler(skip_prk_steps=True,steps_offset=1);s.set_timesteps(50);return s
    s=scheduler();z=torch.full((4,3,2,2),.001)
    with trajectory.NativeTrace(s) as trace:
        for t in s.timesteps:z=s.step(z*.05,t,z).prev_sample
    records=trace.records();clean=z.clamp(-1,1).float()+png_offset
    class Distance:
        def cuda(self):return self
        def eval(self):return self
        def requires_grad_(self,v):return self
        def __call__(self,a,b):return (a-b).square().mean((1,2,3))
    class Net:
        def __call__(self,z,t,encoder_hidden_states):return SimpleNamespace(sample=z*.05+prediction_offset)
    pipe=SimpleNamespace(unet=Net(),
        vae=SimpleNamespace(config=SimpleNamespace(scaling_factor=1.),decode=lambda z:SimpleNamespace(sample=z)),
        _encode_prompt=lambda *a:torch.zeros(8,1,1))
    monkeypatch.setattr(trajectory,'native_scheduler',scheduler)
    write_json(tmp_path/'prompts.json',{'train':[{'text':str(i)} for i in range(4)]})
    target=tmp_path/'teacher_trace/replay.json';target.parent.mkdir()
    outcome=trajectory._measure_replay(tmp_path,{}, {'teacher_clean':'test'},
        SimpleNamespace(get=lambda i:records[i]),{'sample_ids':[0,1,2,3]},target,pipe,
        SimpleNamespace(LPIPS=lambda **k:Distance()),lambda files:clean,
        comparison_check,image_diagnostics,device='cpu')
    assert outcome is passed
    diagnostic=read_json(tmp_path/'teacher_trace/replay_diagnostics.json')
    assert len(diagnostic['samples'])==4 and len(diagnostic['checks'])==9
    assert target.exists() is passed
    if prediction_offset:assert any(not c['passed'] for c in diagnostic['checks'])
