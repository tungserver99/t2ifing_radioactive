"""Audited pre-training gate correction; immutable native artifacts stay exact."""
import hashlib
import pytest
import torch
from common import HERE,read_json,write_json,sha256,image_name


def _sources():
    registry=read_json(HERE/'replay_gate_upgrade_v2.json')
    current={p.name:sha256(p) for p in HERE.glob('*.py')}
    before={k:v for k,v in current.items() if k not in registry['added_files']}
    before.update({k:v[0] for k,v in registry['before'].items()})
    return before,current


def _run(path):
    from semantic_trajectory import REPLAY_TOLERANCES
    from semantic_sign import image_evidence
    c={'watermark_method':'semantic_fp','teacher_objective':'semantic_sign_v2_trajectory',
       'teacher_only':True,'train_n':1,'validation_n':1,'calibration_n':1,'score_batch':1}
    write_json(path/'config.json',c);write_json(path/'prompts.json',{'train':[{'text':'test'}]})
    write_json(path/'assets.json',{'source':'frozen'})
    trace=path/'teacher_trace';trace.mkdir()
    write_json(trace/'protocol.json',{'config_sha256':sha256(path/'config.json'),
        'prompts_sha256':sha256(path/'prompts.json'),'replay_tolerances':REPLAY_TOLERANCES})
    (trace/'shard_00000.safetensors').write_bytes(b'unchanged trace tensors')
    write_json(trace/'shard_00000.json',{'tensor_sha256':sha256(trace/'shard_00000.safetensors')})
    write_json(trace/'index.json',{'0':'shard_00000.json'})
    write_json(trace/'complete.json',{'count':1,'files':{
        p.name:sha256(p) for p in trace.iterdir()}})
    carrier=path/'carrier';carrier.mkdir()
    torch.save({'mean':torch.zeros(32),'basis':torch.eye(32)},carrier/'pca.pt')
    write_json(carrier/'key.json',{'basis_sha256':sha256(carrier/'pca.pt'),
        'selected_pc_indices':list(range(32)),'sigmas':[1.]*32,'signs':[1.]*32,
        'carrier_version':'semantic_sign_v1','medians':[0.]*32,
        'carrier_fit_prompt_sha256':sha256(path/'prompts.json')})
    write_json(carrier/'medians.json',{'pc_indices':list(range(32)),'medians':[0.]*32,
        'carrier_fit_prompt_sha256':sha256(path/'prompts.json')})
    write_json(carrier/'complete.json',{p.name:sha256(p) for p in carrier.iterdir()})
    for split in ('train','validation','calibration'):
        folder=path/'images'/f'teacher_clean_{split}';folder.mkdir(parents=True)
        (folder/image_name(0)).write_bytes(b'unchanged PNG '+split.encode())
        digest=sha256(folder/image_name(0))
        write_json(folder/'image_hashes.json',[{'file_name':image_name(0),'sha256':digest}])
        evidence=image_evidence(torch.zeros(1,32),torch.ones(32),.01)[0]
        write_json(path/'scores'/f'teacher_clean_{split}.json',[
            {'file':image_name(0),'sha256':digest,'coordinates':[0.]*32,**evidence}])
        write_json(path/'scores'/f'teacher_clean_{split}.provenance.json',{
            'carrier_sha256':sha256(carrier/'complete.json'),'prompt_sha256':sha256(path/'prompts.json'),
            'reference_assets_sha256':sha256(path/'assets.json'),'batch_size':1,
            'dtype':'float32','probe_timestep':0,'posterior':'mean'})
    write_json(path/'result/threshold_teacher.json',{'source':'teacher_clean_calibration',
        'score_sha256':sha256(path/'scores/teacher_clean_calibration.json')})
    write_json(trace/'replay_diagnostics.json',{'passed':False,'gate':'old PNG'})
    write_json(trace/'replay.json',{'passed':True,'gate':'old PNG proof'})
    write_json(path/'result/replay_failure.json',{'status':'teacher_only_blocked','gate':'old PNG'})
    (path/'result/README.md').write_text('old PNG failure\n')
    before,current=_sources();write_json(path/'code_manifest.json',before)
    return before,current


def _scientific_bytes(path):
    return {str(p.relative_to(path)):p.read_bytes() for p in path.rglob('*') if p.is_file()
            and (p.name in ('config.json','assets.json','prompts.json') or
                 p.parts[len(path.parts)] in ('carrier','images','scores') or
                 (p.parent==path/'teacher_trace' and not p.name.startswith('replay')))}


def test_gate_upgrade_archives_reports_and_preserves_all_scientific_bytes(tmp_path):
    from semantic_execution import upgrade_execution_manifest
    import semantic_replay_gate_upgrade
    before,current=_run(tmp_path);scientific=_scientific_bytes(tmp_path)
    old={name:(tmp_path/name).read_bytes() for name in semantic_replay_gate_upgrade.ACTIVE_OUTPUTS if (tmp_path/name).exists()}
    upgrade_execution_manifest(tmp_path,current)
    audit=read_json(tmp_path/'execution_upgrades/replay_gate_definition_v2.json')
    assert audit['policy']=='pretraining_replay_gate_definition_v2'
    assert audit['scientific_change'] is True
    for field in ('training_state_modified','config_modified','trace_modified','carrier_modified',
                  'key_modified','clean_images_modified','replay_tolerances_modified'):
        assert audit[field] is False
    assert 'replay-vs-reference float decode' in audit['gate_change']
    assert audit['before']==before and audit['after']==current
    for name,content in old.items():
        assert audit['previous_outputs'][name]['text'].encode('utf8')==content
        assert audit['previous_outputs'][name]['sha256']==hashlib.sha256(content).hexdigest()
        assert not (tmp_path/name).exists()
    assert scientific==_scientific_bytes(tmp_path)
    assert read_json(tmp_path/'code_manifest.json')==current
    upgrade_execution_manifest(tmp_path,current)  # Idempotent once fully applied.


@pytest.mark.parametrize('marker',['teacher_fp/trajectory/protocol.json',
    'teacher_fp/trajectory/checkpoint-100/training_state.pt','teacher_fp/selected.json',
    'teacher_complete.json','teacher_fp/other/adapter_model.safetensors'])
def test_gate_upgrade_rejects_teacher_training_artifacts(tmp_path,marker):
    from semantic_replay_gate_upgrade import upgrade_replay_gate
    before,current=_run(tmp_path);p=tmp_path/marker;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
    with pytest.raises(RuntimeError,match='before teacher training'):upgrade_replay_gate(tmp_path,current)
    assert read_json(tmp_path/'code_manifest.json')==before


@pytest.mark.parametrize('name',['semantic_replay_diagnostics.py','semantic_trajectory.py','semantic_replay_upgrade.py',
                                'semantic_replay_gate_upgrade.py','semantic_sign.py'])
def test_gate_upgrade_rejects_unreviewed_source(tmp_path,name):
    from semantic_replay_gate_upgrade import upgrade_replay_gate
    before,current=_run(tmp_path);current[name]='unreviewed'
    with pytest.raises(RuntimeError,match='Unapproved'):upgrade_replay_gate(tmp_path,current)
    assert read_json(tmp_path/'code_manifest.json')==before
    assert (tmp_path/'result/replay_failure.json').exists()


@pytest.mark.parametrize('name',['config.json','teacher_trace/shard_00000.safetensors','carrier/pca.pt',
    'images/teacher_clean_train/sample_000000.png','scores/teacher_clean_validation.json'])
def test_gate_upgrade_rejects_changed_immutable_artifacts(tmp_path,name):
    from semantic_replay_gate_upgrade import upgrade_replay_gate
    before,current=_run(tmp_path)
    if name=='config.json':
        c=read_json(tmp_path/name);c['seed']=99;write_json(tmp_path/name,c)
    elif name.endswith('.json'):write_json(tmp_path/name,[])
    else:(tmp_path/name).write_bytes(b'corrupt')
    with pytest.raises((RuntimeError,ValueError)):upgrade_replay_gate(tmp_path,current)
    assert read_json(tmp_path/'code_manifest.json')==before
    assert (tmp_path/'result/replay_failure.json').exists()


def test_gate_upgrade_resumes_interrupted_report_invalidation(tmp_path,monkeypatch):
    import semantic_replay_gate_upgrade as gate
    from pathlib import Path
    before,current=_run(tmp_path);original=Path.unlink;calls=[]
    def interrupted(path,*args,**kwargs):
        if path.name=='replay_failure.json' and not calls:
            calls.append(True);raise OSError('interrupted cleanup')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'unlink',interrupted)
    with pytest.raises(OSError,match='interrupted cleanup'):gate.upgrade_replay_gate(tmp_path,current)
    saved=(tmp_path/'execution_upgrades/replay_gate_definition_v2.json').read_bytes()
    assert read_json(tmp_path/'code_manifest.json')==before
    gate.upgrade_replay_gate(tmp_path,current)
    assert (tmp_path/'execution_upgrades/replay_gate_definition_v2.json').read_bytes()==saved
    assert read_json(tmp_path/'code_manifest.json')==current


@pytest.mark.parametrize('field',['score','coordinates','K','threshold_hash','null_hash'])
def test_gate_upgrade_rejects_numeric_score_corruption_before_any_mutation(tmp_path,field):
    from semantic_replay_gate_upgrade import upgrade_replay_gate
    before,current=_run(tmp_path)
    if field=='threshold_hash':
        write_json(tmp_path/'result/threshold_teacher.json',{'source':'teacher_clean_calibration','score_sha256':'0'*64})
    elif field=='null_hash':
        write_json(tmp_path/'result/random_key_null_validation.json',{
            'branches':{'teacher_clean_calibration':{'score_sha256':'0'*64}}})
    else:
        p=tmp_path/'scores/teacher_clean_train.json';rows=read_json(p)
        if field=='coordinates':rows[0][field][0]=.5
        else:rows[0][field]=99
        write_json(p,rows)
    with pytest.raises((RuntimeError,ValueError)):upgrade_replay_gate(tmp_path,current)
    assert read_json(tmp_path/'code_manifest.json')==before
    assert (tmp_path/'result/replay_failure.json').exists()
    assert not (tmp_path/'execution_upgrades/replay_gate_definition_v2.json').exists()
