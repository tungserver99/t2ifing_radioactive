"""Run predeclared blind experiments, retain failures, aggregate without test selection."""
import argparse
import codecs
import csv
import datetime
import json
import math
import os
import signal
from pathlib import Path
import subprocess
import sys
import time
import itertools


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


FOCUSED_METHODS = ["--methods", "fixed_ptq", "reconstruction", "--natural-methods",
                   "natural_rounding", "natural_residual", "natural_residual_qat", "natural_full_finetune"]
SCIENCE_METHODS = ["--methods", "fixed_ptq", "qk_rotation_ptq", "reconstruction", "--natural-methods",
    "natural_rounding", "natural_residual", "natural_residual_qat_warm", "natural_residual_cycle_qat_warm", "natural_random_subspace", "natural_frequency_subspace",
    "natural_contrastive_subspace", "natural_full_finetune", "natural_joint_finetune", "natural_teacher_rounding", "--finetune-rtn-bits", "4",
    "--quality-constraint", "off", "--quality-policy", "constrained", "--gradient-diagnostics-every", "100"]
FULL_METHODS = ["--methods", "fixed_ptq", "qk_rotation_ptq", "reconstruction", "block_reconstruction",
                "--natural-methods", "natural_rounding", "natural_rounding_scale", "natural_residual",
                "natural_qat_purification", "natural_residual_qat", "natural_qat_scale", "natural_gan_qat",
                "natural_full_finetune", "natural_gan_finetune", "natural_spectral",
                "natural_random_subspace", "natural_frequency_subspace", "natural_contrastive_subspace", "natural_teacher_rounding",
                "--finetune-rtn-bits", "4"]
TRANSFER_METHODS = ["--methods", "fixed_ptq", "reconstruction", "--natural-methods",
    "natural_rounding", "natural_residual", "natural_residual_qat_warm", "natural_delta_residual",
    # FP32 fine-tuning and its purified-teacher dependency are opt-in only.
    "--warm-qat-mode", "centered_scale",
    "--warm-checkpoint-policy", "final",
    "--quant-refinement", "balanced", "--quant-selection-start", ".5",
    "--residual-basis", "contrastive", "--reconstruction-learn-scale",
    "--quality-constraint", "off", "--quality-policy", "constrained",
    "--gradient-diagnostics-every", "100"]


def configuration(profile, preserve_weight=.5):
    # All profiles default to W4; other bitwidths require an explicit CLI override.
    # GAN has extra discriminator compute; report it, never call costs equal.
    common = ["--bits", "4", "--quality-constraint", "off", "--finetune-rtn-bits", "--quality-policy", "report",
              "--optimizer", "adamw", "--weight-decay", "0", "--lr-schedule", "warmup_cosine",
              "--natural-preservation", "lowpass", "--preserve-weight", str(preserve_weight),
              "--qat-semantic-preserve-weight", str(preserve_weight), "--warmup-steps", "20", "--cuda-math", "tf32",
              "--evaluate-final"]
    if profile == "pilot":
        return common + FOCUSED_METHODS + ["--steps", "200", "--qat-steps", "200", "--ft-steps", "200",
            "--train-batch-size", "1", "--natural-train-n", "256", "--natural-search-n", "64",
            "--eval-every", "50", "--ft-lr", ".0005", "--natural-resolution", "512"]
    methods = {'full': FULL_METHODS, 'science': SCIENCE_METHODS,
               'transfer': TRANSFER_METHODS, 'focused': FOCUSED_METHODS}[profile]
    return common + methods + ["--steps", "2000", "--qat-steps", "2000", "--ft-steps", "2000",
        "--train-batch-size", "4", "--natural-train-n", "4000", "--natural-search-n", "256",
        "--natural-resolution", "256", "--eval-every", "100", "--ft-lr", ".0005"]


def collect(run):
    owner_path = run / "owner_evaluation.json"
    if not owner_path.exists():
        return []
    owner = json.loads(owner_path.read_text())
    report = json.loads((run / "report.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    result = []
    for row in owner['rows']:
        method = row['method']
        selected = report['selections'].get(method, {})
        quality = report['branch_quality'].get(method, {})
        fields = ('method', 'comparison_group', 'role', 'bit_accuracy', 'tpr', 'n', 'psnr', 'ssim', 'lpips',
                  'test_quality_valid', 'search_quality_valid', 'reference_valid',
                  'evasion_on_originally_detected', 'tpr_drop_percentage_points',
                  'tpr_drop_ci95_low_pp', 'tpr_drop_ci95_high_pp')
        fields += ('joint_success_count', 'joint_success_denominator', 'joint_success_rate',
                   'quality_pass_rate', 'joint_min_psnr', 'joint_min_ssim', 'joint_ci95_low', 'joint_ci95_high')
        result.append({"run": str(run), "seed": manifest['args']['seed'],
            "preserve_weight": manifest['args'].get('preserve_weight'),
            "min_ssim": manifest['args'].get('min_ssim'),
            "budget_ssim": manifest['args'].get('budget_ssim'),
            "budget_psnr": manifest['args'].get('budget_psnr'),
            "shares_training_with": selected.get('shares_training_with'),
            "selected_step": selected.get('step'),
            "checkpoint_policy": selected.get('checkpoint_policy'),
            "parameter_space": selected.get('parameter_space'),
            "proxy_preferred_candidate": selected.get('proxy_preferred_candidate'),
            "quant_refinement": selected.get('quant_refinement'),
            "quant_selection_start": selected.get('quant_selection_start'),
            "quant_quality_weight": selected.get('quant_quality_weight'),
            "effective_teacher_weight": selected.get('effective_teacher_weight'),
            "teacher_signal_train_mse": selected.get('teacher_signal_train_mse'),
            "residual_basis": manifest['args'].get('residual_basis', 'pca'),
            "teacher_source": (selected.get('teacher_dependency') or {}).get('source'),
            "teacher_label": (selected.get('teacher_dependency') or {}).get('label'),
            "teacher_checkpoint_policy": (selected.get('teacher_dependency') or {}).get('checkpoint_policy'),
            "teacher_step": (selected.get('teacher_dependency') or {}).get('selected_step'),
            "teacher_search_mse": selected.get('teacher_search_mse'),
            "warm_start_source": (selected.get('warm_start_dependency') or {}).get('source_branch'),
            "warm_start_step": (selected.get('warm_start_dependency') or {}).get('source_step'),
            "cycle_weight": selected.get('cycle_weight'),
            "joint_quant_weight": selected.get('joint_quant_weight'),
            "joint_quality_weight": selected.get('joint_quality_weight'),
            "joint_quant_bits": selected.get('joint_quant_bits'),
            "joint_w4_psnr": selected.get('joint_w4_psnr'),
            "joint_w4_ssim": selected.get('joint_w4_ssim'),
            "cycle_train_encoder_images": selected.get('cycle_train_encoder_images'),
            "cycle_search_encoder_images": selected.get('cycle_search_encoder_images'),
            "reparameterization": (selected.get('reparameterization') or {}).get('kind'),
            "reparameterization_fp32_max_abs_delta": (selected.get('reparameterization') or {}).get('fp32_probe_max_abs_delta'),
            "teacher_near_identity": ((selected.get('teacher_dependency') or {}).get('target_signal') or {}).get('search', {}).get('near_identity'),
            **{key: row.get(key) for key in fields},
            "quality_violation_fraction": quality.get('quality_violation_fraction'),
            "valid_updates": selected.get('valid_updates'),
            "train_image_forwards": selected.get('train_image_forwards'),
            "discriminator_image_forwards": selected.get('discriminator_image_forwards', 0),
            "branch_seconds": report.get('branch_compute', {}).get(method, {}).get('search_and_export_seconds_including_profile')})
    return result


def run_live(command, *, stdout, stderr=subprocess.STDOUT, check=False):
    """Mirror child output to terminal while retaining the exact per-run log."""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr, start_new_session=True) as process:
        try:
            while True:
                chunk = process.stdout.read1(8192)
                rendered = decoder.decode(chunk, final=not chunk)
                if rendered:
                    stdout.write(rendered); stdout.flush()
                    sys.stdout.write(rendered); sys.stdout.flush()
                if not chunk:
                    break
            code = process.wait()
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if check and code:
        raise subprocess.CalledProcessError(code, command)
    return subprocess.CompletedProcess(command, code)


def run_suite(root, seeds, profile, extra, execute, runner=run_live, preservation_weights=None, min_ssim=None,
              quality_budgets=None):
    root.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).resolve().parent / "run_blind_quantization.sh"
    # Prevent overrides that would redirect a run and invalidate its summary.
    if any(x.split('=')[0] in ('--output', '--seed', '--image-output', '--artifact-output') for x in extra):
        raise ValueError("Suite owns output/seed/artifact paths; use --root and --seeds")
    weights = preservation_weights if preservation_weights is not None else [None]
    if not weights or any(w is not None and (not math.isfinite(w) or w < 0) for w in weights) or len(set(weights)) != len(weights):
        raise ValueError('Preservation weights must be distinct, finite and nonnegative')
    if min_ssim is not None and (not math.isfinite(min_ssim) or not 0 < min_ssim <= 1):
        raise ValueError('min-ssim must be in (0,1]')
    if preservation_weights is not None and any(x.split('=')[0] in ('--preserve-weight', '--qat-semantic-preserve-weight', '--natural-preservation') for x in extra):
        raise ValueError('CLI overrides conflict with declared preservation sweep')
    commands = []
    budgets = quality_budgets if quality_budgets is not None else [None]
    if not budgets or len(set(budgets)) != len(budgets) or any(b is not None and (not math.isfinite(b) or not 0 < b < 1) for b in budgets):
        raise ValueError("Quality budgets must be distinct SSIM values in (0,1)")
    if quality_budgets is not None and any(x.split('=')[0] in ('--budget-ssim', '--quality-constraint') for x in extra):
        raise ValueError("CLI overrides conflict with declared quality budget sweep")
    for seed in seeds:
        for weight, budget in itertools.product(weights, budgets):
            suffix = f'seed_{seed}' + (f'_preserve_{weight:g}' if weight is not None else '')
            suffix += f'_ssim{budget:g}' if budget is not None else ''
            output = root / f'{root.name}_{suffix}'
            changes = ['--min-ssim', str(min_ssim)] if min_ssim is not None else []
            if budget is not None:
                changes += ['--budget-ssim', str(budget), '--quality-constraint', 'dual']
            effective_weight = .5 if weight is None else weight
            commands.append({'seed': seed, 'preserve_weight': weight, 'budget_ssim': budget, 'run_id': suffix, 'output': str(output),
                'command': ['bash', str(script), *configuration(profile, effective_weight), *changes,
                            '--seed', str(seed), '--output', str(output), *extra],
                'status': 'pending' if execute else 'planned_not_executed'})
    protocol = {"profile": profile, "execution_mode": "execute" if execute else "plan_only",
        "suite_status": "pending" if execute else "not_executed",
        "results_available": False, "runs": commands,
        "execution": "simulated low-bit weights, FP32 VAE; no native INT4 speed claims",
        "owner_feedback_for_selection": False,
        "surrogate": "not_run_requires_independent_surrogate_assets",
        "paper_reproductions": False,
        "interpretation": "All runs reported; no best seed/method selected using owner test metrics. Repeated prompts across seeds are not independent prompt samples."}
    write_json(root / "suite_plan.json", protocol)
    rows, plot_points = [], []
    for item in commands:
        if not execute:
            continue
        protocol['suite_status'] = 'running'
        item['status'] = 'running'
        write_json(root / "suite_status.json", protocol)
        log = root / f"{item['run_id']}.log"
        print(f"Running seed {item['seed']}; log: {log}", flush=True)
        start = time.monotonic()
        try:
            with log.open('w', encoding='utf-8', newline='') as stream:
                process = runner(item['command'], stdout=stream, stderr=subprocess.STDOUT, check=False)
            item['returncode'] = process.returncode
            item['status'] = 'completed' if process.returncode == 0 else 'failed'
            rows.extend(collect(Path(item['output'])))
            protocol['results_available'] = bool(rows)
            run = Path(item['output'])
            if (run / 'owner_evaluation.json').exists():
                from wmq_tradeoff import quality_views, plot_svg, table, frontier
                points, _ = quality_views(json.loads((run/'owner_evaluation.json').read_text()),
                    json.loads((run/'report.json').read_text()), json.loads((run/'manifest.json').read_text()), [])
                for point in points:
                    point['method'] = item['run_id'] + ': ' + point['method']
                    if point['comparison_group']:
                        point['comparison_group'] += f"_seed{item['seed']}"
                plot_points.extend(points)
                plot_svg(root/'quality_evasion.svg', plot_points)
                table(root/'suite_tradeoff_points.csv', plot_points)
                groups = {}
                for point in plot_points:
                    if point['comparison_group'] and point['conditional_evasion'] is not None:
                        groups.setdefault((point['comparison_group'], point['threat_model']), []).append(point)
                table(root/'suite_pareto.csv', [p for group in groups.values() for p in frontier(group)])
            report_path = Path(item['output']) / 'report.json'
            if report_path.exists():
                failures = json.loads(report_path.read_text()).get('branch_failures', [])
                item['branch_failures'] = failures
                if failures and process.returncode == 0:
                    item['status'] = 'completed_with_branch_failures'
            if process.returncode == 0 and not (Path(item['output']) / 'owner_evaluation.json').exists():
                item['status'] = 'incomplete'
        except Exception as error:
            item.update(status='failed', error=f"{type(error).__name__}: {error}")
        item['seconds'] = time.monotonic() - start
        write_json(root / "suite_status.json", protocol)
        if rows:
            with (root / "suite_results.csv").open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
        print(f"Seed {item['seed']}: {item['status']}", flush=True)
    if execute:
        statuses = {item['status'] for item in commands}
        protocol['suite_status'] = ('completed' if statuses <= {'completed'} else
            'completed_with_issues' if statuses <= {'completed', 'completed_with_branch_failures'} else 'failed')
    write_json(root / "suite_status.json", protocol)
    return protocol


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path)
    p.add_argument('--seeds', type=int, nargs='+', default=[3407])
    p.add_argument('--profile', choices=['pilot', 'focused', 'science', 'transfer', 'full'], default='science')
    p.add_argument('--quality-budgets', type=float, nargs='+', help='Predeclared SSIM training budgets, e.g. .8 .9 .95')
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--preservation-weights', type=float, nargs='+', default=[.5],
                   help='Default .5 (one run per seed); pass several values only to request a sweep')
    p.add_argument('--min-ssim', type=float, default=.8,
                   help='Minimum mean SSIM for every branch; default 0.8; per-image dual budget is opt-in')
    p.add_argument('extra', nargs=argparse.REMAINDER)
    args = p.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        p.error('Seeds must be distinct and nonnegative')
    project_root = Path(__file__).resolve().parent.parent
    prefix = 'plan_' if args.plan_only else 'run_'
    root = args.root or project_root / 'output_attack' / (
        prefix + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    extra = args.extra[1:] if args.extra[:1] == ['--'] else args.extra
    if args.plan_only:
        print(f"Creating plan only (no attack): {root.resolve()}", flush=True)
    else:
        print(f"Starting attack suite; results: {root.resolve()}", flush=True)
    protocol = run_suite(root.resolve(), args.seeds, args.profile, extra, not args.plan_only,
                         preservation_weights=args.preservation_weights, min_ssim=args.min_ssim,
                         quality_budgets=args.quality_budgets)
    if args.plan_only:
        print("PLAN ONLY: no model, attack, or evaluator was run.")
        print(f"Suite plan: {root.resolve()}")
    else:
        print(f"Suite status/results: {root.resolve()}")
    return int(any(x['status'] in ('failed', 'incomplete', 'completed_with_branch_failures') for x in protocol['runs']))


if __name__ == '__main__':
    sys.exit(main())
