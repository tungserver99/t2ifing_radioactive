"""Explain replay rejection without weakening the frozen scientific gate."""
import traceback
import torch
from common import read_json,write_json,sha256
from semantic_carrier import check_finite


def comparison_check(name,compare):
    try:
        compare()
    except AssertionError as exc:
        return {'name':name,'passed':False,'error':str(exc)}
    return {'name':name,'passed':True}


def tensor_comparison(name,actual,expected,atol,rtol):
    if actual.shape!=expected.shape or actual.dtype!=expected.dtype:
        raise ValueError(f'Replay tensor shape/dtype changed: {name}')
    return comparison_check(name,lambda:torch.testing.assert_close(actual,expected,atol=atol,rtol=rtol))


def image_diagnostics(replay,clean,reference,perceptual,ids,tolerances):
    check_finite(replay,clean,reference)
    if replay.shape!=clean.shape or reference.shape!=clean.shape or len(ids)!=len(clean):
        raise ValueError('Replay image batch/shape mismatch')
    distances=perceptual(replay,clean).flatten()
    baseline=perceptual(reference,clean).flatten()
    direct=perceptual(replay,reference).flatten()
    check_finite(distances,baseline,direct)
    if any(len(v)!=len(ids) for v in (distances,baseline,direct)):
        raise ValueError('Expected one LPIPS result per replay image')
    rows=[]
    for j,i in enumerate(ids):
        pixel=(replay[j]-clean[j]).abs().max().item();lp=distances[j].item()
        failed=[]
        if pixel>tolerances['png_pixel_max']:failed.append('png_pixel_max')
        if lp>tolerances['lpips_max']:failed.append('lpips_max')
        rows.append({'image_id':i,'pixel_max_abs_error':pixel,'lpips':lp,
                     'reference_png_pixel_max':(reference[j]-clean[j]).abs().max().item(),
                     'reference_png_lpips':baseline[j].item(),
                     'replay_reference_pixel_max':(replay[j]-reference[j]).abs().max().item(),
                     'replay_reference_lpips':direct[j].item(),
                     'passed':not failed,'failed_checks':failed})
    return rows


def report_replay_failure(run):
    path=run/'teacher_trace/replay_diagnostics.json';diagnostic=read_json(path)
    if diagnostic.get('passed') is not False:
        raise ValueError('A blocked replay report requires a failed diagnostic')
    report={'status':'teacher_only_blocked','success':False,'training_started':False,
            'downstream_started':False,'reason':'Clean replay failed frozen validation; teacher training was not started.',
            'replay':diagnostic,'config_sha256':sha256(run/'config.json'),
            'trace_sha256':sha256(run/'teacher_trace/complete.json'),
            'diagnostics_sha256':sha256(path)}
    write_json(run/'result/replay_failure.json',report)
    lines=['# RSSF v2 teacher-only replay report','','Status: teacher_only_blocked.',
           'Clean replay validation FAILED. No teacher training or student jobs started.',
           'The experiment is incomplete; this is not evidence that the fingerprint method failed.',
           'All replay thresholds and cached artifacts remain unchanged.','',
           '| Image | Pixel max | LPIPS vs PNG | Reference vs PNG LPIPS | Replay vs reference LPIPS | Pass |',
           '|---:|---:|---:|---:|---:|:---:|']
    for row in diagnostic.get('samples',[]):
        lines.append('| {} | {} | {} | {} | {} | {} |'.format(row['image_id'],
            row.get('pixel_max_abs_error','unavailable'),row.get('lpips','unavailable'),
            row.get('reference_png_lpips','unavailable'),row.get('replay_reference_lpips','unavailable'),row['passed']))
    for check in diagnostic.get('checks',[]):
        if not check['passed']:lines+=['',f"Failed {check['name']}: {check['error']}"]
    lines+=['','Details: teacher_trace/replay_diagnostics.json.',
            'Reference/PNG and replay/reference measurements are diagnostic only; neither overrides the frozen gate.','']
    (run/'result/README.md').write_text('\n'.join(lines),encoding='utf8')
    print(f"REPLAY VALIDATION FAILED - TEACHER BLOCKED. Report: {run/'result/README.md'}",flush=True)
    return report


def analyze_replay_failure(run):
    report=read_json(run/'result/replay_failure.json')
    paths={'config_sha256':run/'config.json','trace_sha256':run/'teacher_trace/complete.json',
           'diagnostics_sha256':run/'teacher_trace/replay_diagnostics.json'}
    if any(sha256(path)!=report[key] for key,path in paths.items()):
        raise RuntimeError('Blocked replay report inputs changed')
    print(f"{report['status']}: {run/'result/README.md'}",flush=True)
    return report


def report_execution_error(run,phase,exc):
    # Operational/integrity errors remain nonzero exits and resumable runs.
    path=run/'result/teacher_execution_error.json'
    write_json(path,{'status':'teacher_execution_error','phase':phase,'error_type':type(exc).__name__,
                     'error':str(exc),'traceback':traceback.format_exc(),'downstream_started':False})
    print(f'TEACHER EXECUTION ERROR ({phase}); details: {path}',flush=True)
