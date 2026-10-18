"""Clean pinned model assets and the existing disjoint COCO prompt protocol."""
import argparse
from pathlib import Path
from common import freeze, read_json, sha256

STUDENT_REV = '133a221b8aa7292a167afc5127cb63fb5005638b'
TEACHER_REV = '0094d483a120f3f33dafbd187ea4aa60d10de75c'
TEACHER_REPO = 'Manojb/stable-diffusion-2-1-base'


def snapshot_identity(folder):
    folder = Path(folder)
    required = ['model_index.json', 'unet/config.json', 'vae/config.json', 'scheduler/scheduler_config.json',
                'text_encoder/config.json', 'tokenizer/tokenizer_config.json']
    if any(not (folder/name).is_file() for name in required):
        raise ValueError(f'Not a complete standard SD snapshot: {folder}')
    files = {p.relative_to(folder).as_posix(): sha256(p) for p in sorted(folder.rglob('*'))
             if p.is_file() and '.cache' not in p.parts and p.name != 'verified.json'}
    for component in ('unet', 'vae', 'text_encoder'):
        if not any(n.startswith(component+'/') and n.endswith(('.bin', '.safetensors')) for n in files):
            raise ValueError(f'Missing {component} weights in {folder}')
    return files


def verified_student_snapshot(folder):
    folder = Path(folder).resolve()
    verified = read_json(folder/'verified.json')
    if verified.get('revision') != STUDENT_REV or verified.get('repository') != 'CompVis/stable-diffusion-v1-4':
        raise ValueError('Local student must be the pinned SD1.4 revision')
    expected = {r['file']: r['sha256'] for r in verified['files']}
    if snapshot_identity(folder) != expected:
        raise RuntimeError('Local student files do not match their pinned download manifest')
    return str(folder)


def prepare(config, run):
    # Import on demand: assets.py is the existing Serum split/download machinery.
    from assets import hub_snapshot, prepare_prompts
    cache = Path(config['cache']); cache.mkdir(parents=True, exist_ok=True)
    if (run/'assets.json').exists():
        prior = read_json(run/'assets.json')
        for field in ('teacher_clean', 'student'):
            if snapshot_identity(prior[field]) != prior[field+'_hashes']:
                raise RuntimeError(f'Source model changed: {field}')
        if sha256(run/'prompts.json') != prior['prompt_sha256']:
            raise RuntimeError('Frozen prompt file changed')
        return
    result = prepare_prompts(config, run, cache)
    def keep(n):
        return n == 'model_index.json' or (n.split('/')[0] in {
            'scheduler','tokenizer','text_encoder','unet','vae','feature_extractor'}
            and n.endswith(('.json','.txt','.safetensors')) and '.fp16.' not in n and '.non_ema.' not in n)
    teacher = config.get('teacher_snapshot')
    source = {'repository': TEACHER_REPO, 'revision': TEACHER_REV}
    if teacher:
        teacher = str(Path(teacher).resolve())
        source = {'local_snapshot': teacher, 'note': 'User-supplied clean SD2.1; content hashes frozen, no substitution'}
    else:
        teacher = hub_snapshot(TEACHER_REPO, TEACHER_REV, cache, keep)
    student = (verified_student_snapshot(config['student_snapshot']) if config.get('student_snapshot')
               else hub_snapshot('CompVis/stable-diffusion-v1-4', STUDENT_REV, cache, keep))
    for model,dim in ((teacher,1024),(student,768)):
        unet=read_json(Path(model)/'unet/config.json')
        text=read_json(Path(model)/'text_encoder/config.json')
        if unet.get('cross_attention_dim')!=dim or text.get('hidden_size')!=dim or unet.get('in_channels')!=4:
            raise ValueError('Unexpected SD2.1/SD1.4 model architecture; refusing a substitute model')
    result.update(method='semantic_fp', teacher_clean=str(teacher), teacher=str(teacher), student=str(student),
                  teacher_source=source, student_revision=STUDENT_REV,
                  teacher_clean_hashes=snapshot_identity(teacher), student_hashes=snapshot_identity(student),
                  prompt_sha256=sha256(run/'prompts.json'))
    freeze(run/'assets.json', result)

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True)
    a=p.parse_args(); prepare(read_json(a.run/'config.json'), a.run)
