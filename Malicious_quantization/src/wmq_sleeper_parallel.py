"""Run independent SleeperMark branches concurrently and aggregate their reports."""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from wmq_sleepermark import DEFAULT_METHODS, METHODS


def split_arguments(argv):
    """Remove orchestrator flags and one --methods list from engine arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--parallel-branches', type=int)
    parser.add_argument('--parallel-vram-per-process-gib', type=float, default=18.)
    parser.add_argument('--parallel-reserve-vram-gib', type=float, default=8.)
    known, remaining = parser.parse_known_args(argv)
    methods, engine = [], []
    index = 0
    while index < len(remaining):
        arg = remaining[index]
        if arg == '--methods':
            if methods:
                raise ValueError('--methods may only be specified once')
            index += 1
            while index < len(remaining) and not remaining[index].startswith('-'):
                methods.append(remaining[index])
                index += 1
            if not methods:
                raise ValueError('--methods requires at least one method')
            continue
        engine.append(arg)
        index += 1
    methods = methods or list(DEFAULT_METHODS)
    unknown = sorted(set(methods)-set(METHODS))
    if unknown or len(methods) != len(set(methods)):
        raise ValueError(f'Invalid or duplicate methods: {unknown or methods}')
    if known.parallel_branches is not None and known.parallel_branches < 1:
        raise ValueError('--parallel-branches must be positive')
    if (not all(math.isfinite(v) for v in (known.parallel_vram_per_process_gib, known.parallel_reserve_vram_gib))
            or known.parallel_vram_per_process_gib <= 0 or known.parallel_reserve_vram_gib < 0):
        raise ValueError('Per-process VRAM must be positive and reserve nonnegative; both must be finite')
    return methods, engine, known


def automatic_parallelism(method_count, per_process_gib=18., reserve_gib=8.):
    """Size from currently free VRAM; never hardcode a GPU model name."""
    import torch
    free, total = torch.cuda.mem_get_info()
    free_gib, total_gib = free/2**30, total/2**30
    reserve = max(reserve_gib, total_gib*.08)
    capacity = max(1, int(max(0., free_gib-reserve)/max(per_process_gib, .001)))
    return min(method_count, capacity), free_gib, total_gib, reserve


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def aggregate(suite, jobs, hardware):
    rows, failures = [], []
    for job in jobs:
        method, code = job['method'], job['returncode']
        roots = sorted((suite/method).glob('sleepermark_*'))
        result = roots[-1] if roots else None
        if code or result is None:
            failures.append({'method': method, 'returncode': code, 'log': str(job['log'])})
            continue
        table = result/'watermark_retention.csv'
        if not table.exists():
            failures.append({'method': method, 'returncode': code, 'log': str(job['log']),
                             'error': 'missing watermark_retention.csv'})
            continue
        with table.open(encoding='utf-8', newline='') as handle:
            records = list(csv.DictReader(handle))
        attacks = [row for row in records if row['label'] != 'marked_reference_test']
        if len(attacks) != 1:
            failures.append({'method': method, 'returncode': code, 'log': str(job['log']),
                             'error': f'expected one attack row, found {len(attacks)}'})
            continue
        row = {'method': method, 'result_dir': str(result), 'log': str(job['log']), **attacks[0]}
        reference = next((r for r in records if r['label'] == 'marked_reference_test'), {})
        row['baseline_tpr'] = reference.get('tpr', '')
        row['baseline_bit_accuracy'] = reference.get('bit_accuracy', '')
        row['wall_seconds'] = job.get('elapsed_seconds', '')
        for filename, suffix, column in [('fid.csv', '', 'fid_triggered_to_marked_reference'),
                ('ordinary_fid.csv', '_ordinary', 'fid_ordinary_to_marked_reference')]:
            path = result/filename
            row[column] = ''
            if path.exists():
                with path.open(encoding='utf-8', newline='') as handle:
                    match = next((r for r in csv.DictReader(handle) if r['label'] == row['label']+suffix), None)
                if match is not None:
                    row[column] = match.get('fid_to_marked_reference', '')
        row['fid_status'] = ('failed' if (result/'fid_error.json').exists() else
                             'complete' if row['fid_triggered_to_marked_reference'] != '' and row['fid_ordinary_to_marked_reference'] != '' else 'not_available')
        training = result/(method+'_training.csv')
        if training.exists():
            with training.open(encoding='utf-8', newline='') as handle:
                training_rows = list(csv.DictReader(handle))
            if training_rows:
                for column in ('elapsed_seconds', 'logical_unet_forwards', 'peak_allocated_gib'):
                    row['train_'+column] = training_rows[-1].get(column, '')
        rows.append(row)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (suite/'parallel_summary.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if columns:
            writer.writeheader(); writer.writerows(rows)
    report = {'status': 'complete' if not failures else 'partial_failure', 'hardware': hardware,
              'rows': rows, 'failures': failures,
              'comparison_warning': 'Each process independently regenerates calibration/evaluation data with the same declared seed; verify manifests before comparison.'}
    atomic_json(suite/'parallel_summary.json', report)
    return failures


def main(argv=None):
    try:
        methods, engine_args, options = split_arguments(sys.argv[1:] if argv is None else argv)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    project = Path(__file__).resolve().parents[1]
    output_root = Path(os.environ.get('WMQ_ATTACK_OUTPUT_ROOT', project/'output_attack')).resolve()
    heavy_root = Path(os.environ.get('WMQ_HEAVY_ROOT', project/'output_artifacts')).resolve()
    # Full download/hash verification happens once before concurrent children.
    from prepare_sleepermark import prepare, digest, PREFLIGHT_ENV
    asset_root = heavy_root/'models'/'sleepermark'
    prepare(asset_root)
    provenance = asset_root/'provenance.json'
    preflight_token = digest(provenance)
    suite = output_root/('sleeper_parallel_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    suite.mkdir(parents=True, exist_ok=False)
    auto, free, total, reserve = automatic_parallelism(len(methods),
        options.parallel_vram_per_process_gib, options.parallel_reserve_vram_gib)
    workers = min(len(methods), options.parallel_branches or auto)
    hardware = {'free_vram_gib_at_launch': free, 'total_vram_gib': total,
                'reserve_vram_gib': reserve, 'estimated_vram_per_process_gib': options.parallel_vram_per_process_gib,
                'parallel_branches': workers, 'method_count': len(methods)}
    atomic_json(suite/'parallel_plan.json', {'methods': methods, 'engine_arguments': engine_args, **hardware})
    print(f'SleeperMark parallel suite: {suite}', flush=True)
    print(f'Launching up to {workers}/{len(methods)} branches; free VRAM {free:.1f}/{total:.1f} GiB.', flush=True)
    pending, active, finished = list(methods), [], []
    try:
        while pending or active:
            while pending and len(active) < workers:
                method = pending.pop(0)
                branch_root = suite/method
                branch_root.mkdir()
                log_path = suite/(method+'.log')
                log = log_path.open('w', encoding='utf-8')
                environment = os.environ.copy()
                environment['WMQ_ATTACK_OUTPUT_ROOT'] = str(branch_root)
                environment[PREFLIGHT_ENV] = preflight_token
                command = [sys.executable, str(Path(__file__).with_name('wmq_sleepermark.py')),
                           '--methods', method, *engine_args]
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                           env=environment, start_new_session=True)
                active.append({'method': method, 'process': process, 'log_handle': log,
                               'log': log_path, 'started': time.time()})
                print(f'[{method}] started pid={process.pid}; log={log_path}', flush=True)
            time.sleep(2)
            for job in list(active):
                code = job['process'].poll()
                if code is not None:
                    job['log_handle'].close()
                    job['returncode'] = code
                    job['elapsed_seconds'] = time.time()-job['started']
                    finished.append(job); active.remove(job)
                    print(f"[{job['method']}] finished code={code} elapsed={job['elapsed_seconds']:.1f}s", flush=True)
    except BaseException:
        for job in active:
            job['process'].terminate()
        for job in active:
            try: job['process'].wait(timeout=20)
            except subprocess.TimeoutExpired: job['process'].kill()
            job['log_handle'].close()
        raise
    failures = aggregate(suite, finished, hardware)
    print(f'Parallel summary: {suite / "parallel_summary.csv"}', flush=True)
    if failures:
        raise SystemExit(f'{len(failures)} SleeperMark branch(es) failed; see parallel_summary.json')


if __name__ == '__main__':
    main()
