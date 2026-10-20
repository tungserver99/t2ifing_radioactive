"""Explicit, restricted code upgrade; never rewrites experiment or checkpoint state."""
from common import freeze, read_json, write_json

# Exact pre-optimization runtime sources at fddd62e (LF and Windows checkout).
BASELINE_HASHES = {'semantic_carrier.py': ['82193992f4c8d825abaaad2cd37d20ff630c01790da792555fd5de037352a9d3', 'fe605834e63e95f7bec2a8710c74d648cb7a12f9b7beab8b2bbb06200a940200'], 'run.py': ['1bdcbebc5097b1947a85a9d3ef2a68ee8cb23edaaa6ce2c246773d9497be41d2']}
TARGET_HASHES = {'run.py': 'd7b2cc88dc4bfe35252992239c11231557e70a9fb58cd5072ef3d07e91c1869b', 'semantic_carrier.py': 'ec446497eaa75283887d51ec1a68acae37815c5982aeb6470829bdddf1b18c68'}
ADDED_FILES = {'semantic_execution.py', 'test_semantic_execution.py'}


def upgrade_execution_manifest(run, current):
    path = run/'code_manifest.json'
    before = read_json(path)
    if before == current:
        return
    c = read_json(run/'config.json')
    if c.get('watermark_method') != 'semantic_fp' or c.get('teacher_objective') != 'threshold_cfg_v1':
        raise RuntimeError('Execution upgrade only supports the threshold_cfg_v1 experiment')
    changed = {name for name in before.keys() | current.keys() if before.get(name) != current.get(name)}
    allowed = set(BASELINE_HASHES) | ADDED_FILES
    if (not changed <= allowed or
            any(current.get(name) != digest for name, digest in TARGET_HASHES.items()) or
            any(before.get(name) not in digests for name, digests in BASELINE_HASHES.items()) or
            any(name not in current for name in before) or
            any(name in before for name in ADDED_FILES)):
        raise RuntimeError(f'Unapproved code change for execution upgrade: {sorted(changed)}')
    # Write the immutable audit before the atomic manifest replacement. Retrying
    # after interruption is safe. Loss/config/checkpoints/optimizer/RNG stay intact.
    record = {'policy':'midblock_features_v1','from_commit':'fddd62e',
              'before':before,'after':current,'changed_files':sorted(changed),
              'training_state_modified':False,
              'equivalence':'same frozen FP32 mid-block features and input gradients; skip unused decoder'}
    freeze(run/'execution_upgrades/midblock_features_v1.json', record)
    write_json(path, current)
    print('Applied audited mid-block feature execution upgrade; full optimizer/sampler/RNG checkpoints preserved.', flush=True)
