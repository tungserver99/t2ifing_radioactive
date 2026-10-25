"""One audited diagnostics-only upgrade for the known, pre-training v2 revision."""
from common import freeze,read_json,write_json

BASELINE_HASHES={
    'semantic_trajectory.py':['f66af5b6dadadef5309f3cf3476c8f6beff135c5e32ecbf92d7263265fc73dfa','cbfd6b09a42b94cf2b2d7661e58fcd018737875887536835d231af003c4c1871'],
    'semantic_sign_trajectory_train.py':['98aa7ababe4ad2154797c7b1830862a521d8872c9f21949ce68588c69b1b4554','aae19803f96f142917b8dd6b2b131c00d7fcb04bebb105a48390453747374915'],
    'semantic_execution.py':['55b39a9e0641052b897b18b1eb5c4b093af50a9cc9a9c27127f4b21285dd07d3','41db1aa353b411a7b7cac3471e10fa6f9ee0f0ee3f77c9970c945169ed3d4903']}
TARGET_HASHES={'semantic_trajectory.py': ['e799581840505c71a2febcc931a5e21fc318ba75504e61bf5cde3257b7a6b667', '2722c8d8b2b4b99164c67560e227bcf737c48d38aaf84831d1bfe7993e5e582a'], 'semantic_sign_trajectory_train.py': ['524922c760ef0771d55d6369a1a11f4e534d7e7b14197f3eec5ddea66238383b', '938e8e9ad1614d33dccf271cb0b815ca4fad5873d7bf0db244ddbf76e124bc47'], 'semantic_execution.py': ['a6a7ee93b4d962d759901acf61b4efb459a1832ae39c2a024dfd298605ff92e7', 'a9bf173d434db5c32bdebe76e515119a4f660d23fdc13cd00bb19c7cda1ae26a'], 'semantic_replay_diagnostics.py': ['e47bb89cd7e5b28d8e8a3d8f619ed47cb4963c332e81ef4f0959bbbba0fc7a30', 'a680ae88ccefe30bc60956ac660a743d67183bb6f8f6a0fbe12e1ba197fee74d']}
ADDED_FILES={'semantic_replay_diagnostics.py','semantic_replay_upgrade.py','test_semantic_replay_diagnostics.py'}


def upgrade_replay_diagnostics(run,current):
    before=read_json(run/'code_manifest.json');c=read_json(run/'config.json')
    if 'semantic_replay_upgrade.py' in before:
        from semantic_replay_gate_upgrade import upgrade_replay_gate
        return upgrade_replay_gate(run,current)
    if c.get('watermark_method')!='semantic_fp' or c.get('teacher_objective')!='semantic_sign_v2_trajectory' or not c.get('teacher_only'):
        raise RuntimeError('Replay diagnostics upgrade requires the v2 teacher-only experiment')
    trajectory=run/'teacher_fp/trajectory'
    if trajectory.exists() or (run/'teacher_complete.json').exists() or (run/'teacher_fp/selected.json').exists():
        raise RuntimeError('Replay diagnostics upgrade is allowed only before teacher training starts')
    changed={name for name in before.keys()|current.keys() if before.get(name)!=current.get(name)}
    if (not TARGET_HASHES or not changed<=set(BASELINE_HASHES)|ADDED_FILES
            or any(before.get(name) not in hashes for name,hashes in BASELINE_HASHES.items())
            or any(current.get(name) not in hashes for name,hashes in TARGET_HASHES.items())
            or any(name not in current for name in before)
            or any(name in before or name not in current for name in ADDED_FILES)):
        raise RuntimeError(f'Unapproved replay diagnostics source changes: {sorted(changed)}')
    freeze(run/'execution_upgrades/replay_diagnostics_v1.json',{
        'policy':'pretraining_replay_diagnostics_v1','before':before,'after':current,
        'changed_files':sorted(changed),'training_state_modified':False,
        'scientific_changes':False,'replay_tolerances_modified':False,
        'scope':'Collect replay diagnostics and report blocked execution; preserve all math and cached artifacts.'})
    write_json(run/'code_manifest.json',current)
    print('Applied audited pre-training replay diagnostics upgrade; frozen experiment and traces preserved.',flush=True)
