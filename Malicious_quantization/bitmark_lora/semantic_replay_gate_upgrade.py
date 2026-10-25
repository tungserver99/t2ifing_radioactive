"""Audited pre-training correction of the LPIPS replay gate operand."""
from common import HERE,freeze,read_json,write_json,sha256,image_name

ACTIVE_OUTPUTS=('teacher_trace/replay_diagnostics.json','teacher_trace/replay.json',
                'result/replay_failure.json','result/README.md','result/teacher_execution_error.json')


def _immutable_snapshot(run):
    """Validate saved provenance and hash scientific inputs without rewriting them."""
    from semantic_trajectory import TraceReader,REPLAY_TOLERANCES
    from semantic_carrier import SemanticCarrier
    from semantic_report import score_rows
    c=read_json(run/'config.json');reader=TraceReader(run/'teacher_trace')
    protocol=reader.protocol
    if (protocol.get('config_sha256')!=sha256(run/'config.json') or
            protocol.get('prompts_sha256')!=sha256(run/'prompts.json') or
            protocol.get('replay_tolerances')!=REPLAY_TOLERANCES or len(reader.index)!=c['train_n']):
        raise RuntimeError('Immutable trace configuration/prompts/tolerances changed')
    files={run/name for name in ('config.json','prompts.json','assets.json','teacher_trace/complete.json','carrier/complete.json')}
    trace_proof=read_json(run/'teacher_trace/complete.json')
    for folder,proof in ((run/'teacher_trace',trace_proof['files']),
                         (run/'carrier',read_json(run/'carrier/complete.json'))):
        for name,digest in proof.items():
            path=(folder/name).resolve()
            if not path.is_relative_to(folder.resolve()) or sha256(path)!=digest:
                raise RuntimeError(f'Immutable artifact changed: {folder/name}')
            files.add(path)
    carrier_proof=read_json(run/'carrier/complete.json')
    if not {'key.json','pca.pt'}<=set(carrier_proof):raise RuntimeError('Incomplete immutable carrier proof')
    if read_json(run/'carrier/key.json')['basis_sha256']!=sha256(run/'carrier/pca.pt'):
        raise RuntimeError('Immutable carrier/key mismatch')
    carrier=SemanticCarrier.load(run/'carrier')
    for split in ('train','validation','calibration'):
        label='teacher_clean_'+split;folder=run/'images'/label;n=c[split+'_n']
        images=read_json(folder/'image_hashes.json')
        scores_path=run/'scores'/f'{label}.json';scores=read_json(scores_path)
        if len(images)!=n or len(scores)!=n:raise RuntimeError(f'Incomplete immutable data: {label}')
        for i,(image,score) in enumerate(zip(images,scores)):
            if (image['file_name']!=image_name(i) or score['file']!=image_name(i) or
                    image['sha256']!=score['sha256'] or sha256(folder/image_name(i))!=image['sha256']):
                raise RuntimeError(f'Immutable clean image/score changed: {label}/{image_name(i)}')
        provenance_path=scores_path.with_suffix('.provenance.json')
        expected={'carrier_sha256':sha256(run/'carrier/complete.json'),'prompt_sha256':sha256(run/'prompts.json'),
                  'reference_assets_sha256':sha256(run/'assets.json'),'batch_size':c['score_batch'],
                  'dtype':'float32','probe_timestep':0,'posterior':'mean'}
        if read_json(provenance_path)!=expected:raise RuntimeError(f'Immutable score provenance changed: {label}')
        # Reuse the unchanged verifier for numeric scores, coordinates and bits/K/p-values.
        score_rows(run,label,n,carrier=carrier)
        files.update(p for p in folder.rglob('*') if p.is_file())
        files.update((scores_path,provenance_path))
    for name in ('fixed_key_clean_fpr.json','random_key_null_validation.json','analytic_verifier.json','threshold_teacher.json'):
        if (run/'result'/name).exists():files.add(run/'result'/name)
    threshold=run/'result/threshold_teacher.json'
    if threshold.exists():
        proof=read_json(threshold)
        if (proof.get('source')!='teacher_clean_calibration' or
                proof.get('score_sha256')!=sha256(run/'scores/teacher_clean_calibration.json')):
            raise RuntimeError('Immutable teacher threshold score hash changed')
    null=run/'result/random_key_null_validation.json'
    if null.exists():
        for label,proof in read_json(null)['branches'].items():
            if label in ('teacher_clean_train','teacher_clean_validation','teacher_clean_calibration'):
                if proof.get('score_sha256')!=sha256(run/'scores'/f'{label}.json'):
                    raise RuntimeError(f'Immutable clean null score hash changed: {label}')
    return {p.relative_to(run).as_posix():sha256(p) for p in sorted(files)}


def upgrade_replay_gate(run,current):
    before=read_json(run/'code_manifest.json')
    if before==current:return
    c=read_json(run/'config.json')
    if (c.get('watermark_method')!='semantic_fp' or c.get('teacher_objective')!='semantic_sign_v2_trajectory'
            or c.get('teacher_only') is not True):
        raise RuntimeError('Replay gate upgrade requires the v2 teacher-only experiment')
    teacher=run/'teacher_fp'
    if ((teacher/'trajectory').exists() or (teacher/'selected.json').exists() or
            (run/'teacher_complete.json').exists() or
            (teacher.exists() and any(p.name.startswith('checkpoint') or p.name in
                ('trained.json','training_state.pt','adapter_model.safetensors') for p in teacher.rglob('*')))):
        raise RuntimeError('Replay gate upgrade is allowed only before teacher training starts')
    registry_path=HERE/'replay_gate_upgrade_v2.json';registry=read_json(registry_path)
    changed={k for k in before.keys()|current.keys() if before.get(k)!=current.get(k)}
    added=set(registry['added_files']);allowed=set(registry['before'])|added
    if (registry.get('policy')!='pretraining_replay_gate_definition_v2' or not registry['after']
            or not changed<=allowed or set(registry['after'])!=allowed
            or any(before.get(k) not in values for k,values in registry['before'].items())
            or any(current.get(k) not in values for k,values in registry['after'].items())
            or any(k in before or k not in current for k in added)
            or any(k not in current for k in before)):
        raise RuntimeError(f'Unapproved replay gate source changes: {sorted(changed)}')
    print('Verifying immutable clean data, traces, carrier/key and scores before gate upgrade...',flush=True)
    protected=_immutable_snapshot(run)
    audit_path=run/'execution_upgrades/replay_gate_definition_v2.json'
    if audit_path.exists():
        record=read_json(audit_path)
        if (record['before']!=before or record['after']!=current or record['immutable_artifacts']!=protected or
                record['policy_registry_sha256']!=sha256(registry_path)):
            raise RuntimeError('Replay gate upgrade audit or immutable artifacts changed')
    else:
        archived={}
        for name in ACTIVE_OUTPUTS:
            path=run/name
            if path.exists():archived[name]={'sha256':sha256(path),'text':path.read_bytes().decode('utf8')}
        record={'policy':registry['policy'],'before':before,'after':current,'changed_files':sorted(changed),
                'policy_registry_sha256':sha256(registry_path),'immutable_artifacts':protected,
                'previous_outputs':archived,'scientific_change':True,'training_state_modified':False,
                'config_modified':False,'trace_modified':False,'carrier_modified':False,'key_modified':False,
                'clean_images_modified':False,'replay_tolerances_modified':False,
                'gate_change':'LPIPS gate operand changed from replay-vs-saved-PNG to replay-vs-reference float decode'}
        freeze(audit_path,record)
    # Archive is committed first; interrupted invalidation can be retried safely.
    for name in ACTIVE_OUTPUTS:
        path=run/name
        if path.exists():
            if name not in record['previous_outputs'] or sha256(path)!=record['previous_outputs'][name]['sha256']:
                raise RuntimeError(f'Active replay output changed during gate upgrade: {name}')
            path.unlink()
    if _immutable_snapshot(run)!=protected:raise RuntimeError('Immutable artifacts changed during gate upgrade')
    write_json(run/'code_manifest.json',current)
    print('Applied audited pre-training replay gate definition v2; old reports archived, numeric tolerances unchanged.',flush=True)
