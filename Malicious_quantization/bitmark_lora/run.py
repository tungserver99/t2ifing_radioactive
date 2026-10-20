"""One-command BitMark→LoRA experiment, with matched full-finetuning controls."""
import argparse
import contextlib
import traceback
import datetime
try:
    import fcntl
except ImportError:  # CPU plan/tests on Windows; actual server execution remains POSIX.
    from common import windows_fcntl as fcntl
import importlib.metadata
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo
from pathlib import Path

from common import HERE, freeze, read_json, sha256, verify_vendor, write_json, runtime_root, require_local_runtime_path, results_root
from student import branches, training_command


def parse_args(argv=None, method="bitmark"):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dora", action=argparse.BooleanOptionalAction, default=method in {"serum", "semantic_fp"},
                   help="Add one attention+convolution DoRA branch at the largest requested rank (SERUM default on)")
    p.add_argument("--ranks", type=int, nargs="+", default=[16, 32])
    p.add_argument("--train-n", type=int, default=1000)
    p.add_argument("--calibration-n", type=int, default=5000)
    p.add_argument("--test-n", type=int, default=1000)
    p.add_argument("--epochs", type=int, help="Legacy budget override; cannot accompany --max-train-steps")
    p.add_argument("--max-train-steps", type=int, default=None, help="Shared optimizer budget; default 7500")
    p.add_argument("--checkpoint-steps", type=int, nargs="+", default=[2500, 5000, 7500])
    p.add_argument("--adapter-scopes", nargs="+", choices=["attention", "attention_conv"],
                   default=["attention", "attention_conv"])
    p.add_argument("--validation-n", type=int, default=256)
    p.add_argument("--validation-generation-n", type=int, default=128)
    p.add_argument("--validation-draws", type=int, default=2, help="Fixed noise/timestep draws per validation image")
    p.add_argument("--watermark-delta", type=float, default=2., help="Official BitMark embedding strength; separate-run ablation")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--precision", choices=["fp16", "bf16", "no"], default="fp16")
    p.add_argument("--inference-steps", type=int, default=50)
    p.add_argument("--inference-batch", type=int, default=4)
    p.add_argument("--fpr", type=float, default=.01)
    p.add_argument("--lora-only", action="store_true", help="Omit matched full controls; inference about LoRA weakness remains limited")
    p.add_argument("--paper-full-ema", action="store_true", help="Also run the paper-style full+EMA positive control")
    p.add_argument("--train-clean-controls", action="store_true", help="Also train matched clean LoRA/full controls to isolate generic adaptation effects")
    p.add_argument("--skip-utility", action="store_true", help="Explicitly omit CLIP/FID/KID (reported as unavailable)")
    p.add_argument("--cache", type=Path, default=runtime_root() / "cache")
    p.add_argument("--output", type=Path, help="Create a NEW run directory here")
    p.add_argument("--resume", type=Path, help="Resume exact recorded config from an existing run; do not supply experiment overrides")
    p.add_argument("--min-free-gb", type=float, default=70)
    p.add_argument("--plan-only", action="store_true", help="Print protocol only; default without this flag executes the experiment")
    p.add_argument("--analyze", type=Path, help="Recompute reports from an existing complete run; no GPU required")
    p.set_defaults(watermark_method=method)
    if method == "serum":
        p.description = "SERUM SD2.1 diffusion teacher -> SD1.4 LoRA/full adaptation"
        p.set_defaults(train_n=5000, validation_n=1000, batch_size=8, learning_rate=1e-5)
        p.add_argument("--serum-alpha", type=float, default=.5)
        p.add_argument("--serum-checkpoint", type=Path, help="Optional local copy of pinned author epoch-50 checkpoint")
        p.add_argument("--serum-cache-archive", type=Path, help="Optional local copy of pinned author SD2.1 cache ZIP")
        p.add_argument("--teacher-batch", type=int, default=4)
        p.add_argument("--score-batch", type=int, default=16)
        p.add_argument("--full-learning-rate", type=float, default=5e-6)
    if method == "semantic_fp":
        p.description = "Native semantic SD2.1 fingerprint -> ordinary SD1.4 adaptation"
        p.set_defaults(train_n=5000, validation_n=1000, batch_size=8, learning_rate=1e-5)
        p.set_defaults(teacher_objective="threshold_cfg_v1",cycle_every=1)
        p.add_argument("--teacher-kappas",type=float,nargs="+",default=[1.,1.5,2.])
        p.add_argument("--reuse-preparation",type=Path,help="Reuse verified clean images/carrier/calibration; train new teachers from clean weights")
        p.add_argument("--setting", choices=["pilot", "full"], default="pilot")
        p.add_argument("--teacher-snapshot", type=Path, help="Optional existing clean SD2.1 standard pipeline snapshot")
        p.add_argument("--student-snapshot", type=Path, help="Optional existing pinned SD1.4 standard pipeline snapshot")
        p.add_argument("--key-seed", type=int, default=0)
        p.add_argument("--num-bits", type=int, default=8)
        p.add_argument("--carrier-pc-min", type=int, default=8, help="Zero-based inclusive")
        p.add_argument("--carrier-pc-max", type=int, default=64, help="Zero-based inclusive")
        p.add_argument("--stable-candidate-count", type=int, default=16)
        p.add_argument("--semantic-shift", type=float, default=.5, help=argparse.SUPPRESS)
        p.add_argument("--semantic-shift-sweep", type=float, nargs="+", help=argparse.SUPPRESS)
        p.add_argument("--semantic-t-max", type=int, default=500)
        p.add_argument("--continue-from", type=Path, help="Fork a run using prepared data and saved teacher weights; fresh optimizer, original step schedule")
        p.add_argument("--teacher-precision", choices=["fp16", "bf16", "no"], default="bf16", help="Teacher only; student precision remains --precision")
        p.add_argument("--teacher-check-steps", type=int, default=32, help="Standalone teacher diagnostic optimizer updates (minimum 4)")
        p.add_argument("--teacher-check", type=Path, help="Only stress-test latest teacher adapter from an existing run; do not resume training")
        p.add_argument("--teacher-rank", type=int, default=32)
        p.add_argument("--teacher-lr", type=float, default=1e-4)
        p.add_argument("--teacher-train-steps", type=int, default=3000)
        p.add_argument("--teacher-checkpoint-every", type=int, default=100)
        p.add_argument("--teacher-micro-batch", type=int, default=1, help="Teacher forward batch; gradients accumulate to --teacher-batch")
        p.add_argument("--teacher-batch", type=int, default=4)
        p.add_argument("--score-batch", type=int, default=4)
        p.add_argument("--cycle-every", type=int, default=1)
        p.add_argument("--lambda-preserve", type=float, default=1., help=argparse.SUPPRESS)
        p.add_argument("--lambda-code", type=float, default=.10, help=argparse.SUPPRESS)
        p.add_argument("--lambda-perp", type=float, default=.01, help=argparse.SUPPRESS)
        p.add_argument("--lambda-cycle", type=float, default=.05, help=argparse.SUPPRESS)
        p.add_argument("--teacher-clip-drop-max", type=float, default=.01)
        p.add_argument("--teacher-lpips-max", type=float, default=.15)
        p.add_argument("--teacher-min-tpr", type=float, default=.90)
        p.add_argument("--full-learning-rate", type=float, default=5e-6)
    a = p.parse_args(argv)
    if method == "semantic_fp":
        supplied = list(argv if argv is not None else sys.argv[1:])
        flags = {v.split("=")[0] for v in supplied if v.startswith("--")}
        inactive={'--semantic-shift','--semantic-shift-sweep','--lambda-preserve','--lambda-code','--lambda-perp','--lambda-cycle'}
        if flags & inactive:p.error('Fixed-alpha/perpendicular/old cycle weights are inactive; use --teacher-kappas')
        if a.fpr!=.01:p.error('Threshold rescue protocol uses the frozen 1%-FPR operating point')
        if a.num_bits!=8:p.error('Threshold rescue protocol retains the 8-bit carrier')
        if a.cycle_every!=1:p.error('Threshold objective applies cycle at every semantic update (--cycle-every 1)')
        if a.teacher_clip_drop_max>.01:p.error('Threshold teacher CLIP drop hard limit cannot exceed 0.01')
        if a.teacher_min_tpr<.9:p.error('Threshold teacher validation TPR must be at least 0.90')
        if a.setting == "pilot":
            if "--ranks" not in flags: a.ranks = [32]
            if "--adapter-scopes" not in flags: a.adapter_scopes = ["attention_conv"]
            if not flags & {"--dora", "--no-dora"}: a.dora = False
        positive = [a.teacher_batch,a.teacher_micro_batch,a.score_batch,a.num_bits,a.stable_candidate_count,a.teacher_rank,
                    a.teacher_train_steps,a.teacher_checkpoint_every,a.cycle_every]
        weights = [a.lambda_preserve,a.lambda_code,a.lambda_perp,a.lambda_cycle,
                   a.teacher_clip_drop_max,a.teacher_lpips_max]
        strengths = a.semantic_shift_sweep or [a.semantic_shift]
        if any(not math.isfinite(k) or k<=0 for k in a.teacher_kappas) or len(set(a.teacher_kappas))!=len(a.teacher_kappas):
            p.error('Teacher kappas must be distinct, finite and positive')
        if (a.teacher_check_steps<4 or min(positive)<1 or a.key_seed<0 or a.carrier_pc_min<0 or a.semantic_t_max<0
                or a.carrier_pc_max-a.carrier_pc_min+1<a.stable_candidate_count
                or a.num_bits>a.stable_candidate_count or a.teacher_batch>a.train_n or a.teacher_micro_batch>a.teacher_batch
                or not 0<=a.teacher_min_tpr<=1 or a.teacher_train_steps<=750
                or any(not math.isfinite(x) or x<0 for x in weights)
                or any(not math.isfinite(x) or x<=0 for x in strengths+[a.teacher_lr,a.full_learning_rate])
                or len(set(strengths))!=len(strengths)):
            p.error("Invalid semantic carrier, teacher training, strength or quality configuration")
        if a.train_clean_controls or a.paper_full_ema:
            p.error("Semantic MVP uses the specified ordinary wm-trained students and unadapted baseline")
    if method == "serum":
        if not math.isfinite(a.serum_alpha) or not 0 < a.serum_alpha < 1:
            p.error("SERUM alpha must lie strictly between 0 and 1")
        if min(a.teacher_batch, a.score_batch) < 1 or not math.isfinite(a.full_learning_rate) or a.full_learning_rate <= 0:
            p.error("Invalid SERUM batch size/full learning rate")
    if a.epochs is not None and a.max_train_steps is not None:
        p.error("Choose --epochs or --max-train-steps, not both")
    if a.epochs is not None and a.epochs <= 0:
        p.error("--epochs must be positive")
    if a.batch_size <= 0:
        p.error("--batch-size must be positive")
    if a.max_train_steps is None:
        a.max_train_steps = (math.ceil(a.train_n / a.batch_size) * a.epochs if a.epochs else 7500)
    if any(x <= 0 for x in a.checkpoint_steps):
        p.error("Checkpoint steps must be positive")
    a.checkpoint_steps = sorted(set(x for x in a.checkpoint_steps if x <= a.max_train_steps) | {a.max_train_steps})
    if a.validation_generation_n > a.validation_n or a.validation_generation_n < 2:
        p.error("Require 2 <= validation-generation-n <= validation-n")
    if len(set(a.adapter_scopes)) != len(a.adapter_scopes):
        p.error("Adapter scopes must be unique")
    if a.resume and a.output:
        p.error("--resume and --output are mutually exclusive")
    numeric = [a.train_n, a.calibration_n, a.test_n, a.max_train_steps, a.validation_n, a.validation_draws, a.batch_size,
               a.inference_batch, a.inference_steps, *a.ranks]
    if min(numeric) <= 0 or len(set(a.ranks)) != len(a.ranks):
        p.error("Counts/ranks must be positive and ranks unique")
    if (not math.isfinite(a.watermark_delta) or a.watermark_delta <= 0 or not math.isfinite(a.learning_rate) or a.learning_rate <= 0 or not 0 < a.fpr < 1 or a.seed < 0
            or not math.isfinite(a.min_free_gb) or a.min_free_gb < 0):
        p.error("Invalid learning rate/FPR/seed/disk limit")
    if a.calibration_n * a.fpr < 1:
        p.error("Use at least 1/FPR clean calibration samples")
    if a.test_n < 2:
        p.error("Use at least two test images")
    return a


def config_from_args(a):
    ignored = {"output", "resume", "plan_only", "analyze", "min_free_gb", "teacher_check", "teacher_check_steps", "continue_from", "reuse_preparation"}
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(a).items() if k not in ignored}
    if config.get('teacher_objective')=='threshold_cfg_v1':
        for name in ('semantic_shift','semantic_shift_sweep','lambda_preserve','lambda_code','lambda_perp','lambda_cycle'):
            config.pop(name,None)
    return config


def stage(run, label, command):
    log = run / "logs" / f"{label}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    print(f"\n[{label}] log: {log}\n{' '.join(map(str, command))}", flush=True)
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false", HF_HUB_DISABLE_TELEMETRY="1")
    with log.open("a") as f:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, env=env, cwd=HERE, start_new_session=True)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                f.write(line)
                f.flush()
            code = process.wait()
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        finally:
            process.stdout.close()
    status = {"stage": label, "exit_code": code, "seconds": time.monotonic()-start,
              "finished": datetime.datetime.now().astimezone().isoformat()}
    with (run / "stages.jsonl").open("a") as f:
        f.write(json.dumps(status) + "\n")
    if code:
        raise RuntimeError(f"Stage {label} failed (exit={code}). See {log}; resume with --resume {run}")


def preflight(a):
    require_local_runtime_path(a.cache)
    if getattr(a, "watermark_method", "bitmark") == "semantic_fp":
        from semantic_teacher import check_imports as check_model_imports
    elif getattr(a, "watermark_method", "bitmark") == "serum":
        from serum_teacher import check_imports as check_model_imports
    else:
        from check_runtime import check_model_imports
    try:
        check_model_imports()
    except Exception as exc:
        raise RuntimeError(
            f"Existing Python environment cannot import pipeline models: {sys.executable}. "
            "See the chained import error above for the missing/incompatible dependency. "
            "Select a working existing interpreter with SERUM_PYTHON/ BITMARK_PYTHON; no automatic installation."
        ) from exc
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires a CUDA GPU")
    major, _ = torch.cuda.get_device_capability()
    if getattr(a, "watermark_method", "bitmark") == "bitmark":
        import flash_attn
        if major < 8:
            raise RuntimeError("Official Infinity/FlashAttention requires an Ampere-or-newer GPU")
        q = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.bfloat16)
        result = flash_attn.flash_attn_func(q, q, q)
        if not torch.isfinite(result).all():
            raise RuntimeError("FlashAttention CUDA smoke check returned nonfinite values")
    else:
        x = torch.randn(16,16,device="cuda")
        if not torch.isfinite(x @ x).all():
            raise RuntimeError("CUDA matrix smoke check returned nonfinite values")
    torch.cuda.synchronize()
    print(f"GPU: {torch.cuda.get_device_name()}; torch={torch.__version__}", flush=True)
    for path in [a.cache, (a.output or results_root(getattr(a, "watermark_method", None)))]:
        probe = path.resolve()
        while not probe.exists():
            probe = probe.parent
        free = shutil.disk_usage(probe).free / 1024**3
        if free < a.min_free_gb:
            raise RuntimeError(f"Only {free:.1f} GiB free on {probe}; requested minimum {a.min_free_gb:g}. "
                               "Choose --cache/--output on a larger disk or set an explicit --min-free-gb.")
    return {"gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
            "cuda_runtime": torch.version.cuda, "python": sys.version,
            "note": "Matching seeds does not promise bitwise equality across GPU/software versions."}


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log
    def write(self, text):
        self.log.write(text)
        return self.terminal.write(text)
    def flush(self):
        self.log.flush()
        self.terminal.flush()
    def isatty(self):
        return False


@contextlib.contextmanager
def persistent_log(run):
    with (run / "run.log").open("a", buffering=1) as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                yield
            except BaseException:
                traceback.print_exc(file=log)
                raise


def main(argv=None, method="bitmark"):
    a = parse_args(argv, method)
    if getattr(a, 'teacher_check', None):
        if a.resume or a.output or a.analyze or a.plan_only or a.continue_from or a.reuse_preparation:
            raise ValueError('--teacher-check is a standalone diagnostic, not resume/plan/analyze')
        from semantic_teacher_train import check_existing_run
        check_existing_run(a.teacher_check.resolve(), a.teacher_precision, a.teacher_check_steps)
        return
    if getattr(a,'reuse_preparation',None) and (a.resume or a.continue_from or a.analyze):
        raise ValueError('--reuse-preparation starts clean teachers in a new run, not resume/continue/analyze')
    if a.analyze:
        from report import summarize
        summarize(a.analyze.resolve())
        return
    verify_vendor()
    c = config_from_args(a)
    if getattr(a,'continue_from',None):
        if a.resume or a.analyze:raise ValueError('--continue-from creates a new run; cannot accompany resume/analyze')
        supplied=argv if argv is not None else sys.argv[1:]
        allowed={'--continue-from','--teacher-precision','--output','--min-free-gb','--plan-only'}
        if any(x.split('=')[0] not in allowed for x in supplied if x.startswith('--')):
            raise ValueError('--continue-from preserves the source experiment; only teacher precision/output/disk/plan overrides are allowed')
        from semantic_checkpoint import continuation_config
        c=continuation_config(a.continue_from,a.teacher_precision)
        a.cache=Path(c['cache'])
    if getattr(a,'reuse_preparation',None):
        supplied=argv if argv is not None else sys.argv[1:]
        forbidden={'--seed','--key-seed','--num-bits','--carrier-pc-min','--carrier-pc-max','--stable-candidate-count',
                   '--train-n','--validation-n','--calibration-n','--test-n','--fpr','--teacher-snapshot',
                   '--student-snapshot','--cache','--inference-steps','--teacher-batch','--score-batch'}
        if any(x.split('=')[0] in forbidden for x in supplied if x.startswith('--')):
            raise ValueError('--reuse-preparation preserves source data/key/null settings; do not override those fields')
        from semantic_checkpoint import reuse_preparation_config
        c=reuse_preparation_config(c,a.reuse_preparation)
        a.cache=Path(c['cache'])
    if a.resume:
        # Reject hidden overrides rather than silently ignoring user arguments.
        supplied = argv if argv is not None else sys.argv[1:]
        allowed = {"--resume", "--min-free-gb", "--plan-only"}
        if any(x.split("=")[0] not in allowed for x in supplied if x.startswith("--")):
            raise ValueError("--resume loads the frozen config; only --min-free-gb/--plan-only may accompany it")
        c = read_json(a.resume.resolve() / "config.json")
        a.watermark_method = c.get("watermark_method", "bitmark")
        a.cache = Path(c["cache"])
        a.output = a.resume.resolve()
    carrier = c.get("watermark_method", "bitmark")
    teacher_file = "semantic_teacher.py" if carrier == "semantic_fp" else "serum_teacher.py" if carrier == "serum" else "teacher.py"
    assets_file = "semantic_assets.py" if carrier == "semantic_fp" else "serum_assets.py" if carrier == "serum" else "assets.py"
    if a.plan_only and carrier == "semantic_fp":
        print(json.dumps({"config": c, "students": branches(c), "baseline": "unadapted pinned SD1.4",
            "teacher": "clean SD2.1 -> teacher-side LoRA -> merged native pipeline",
            "stages": ["assets", "clean train/validation/calibration", "PCA/stability/freeze key",
                       "teacher null", "train/quality-select/merge teacher on validation only",
                       "native teacher datasets", "unadapted null", "freeze both thresholds",
                       "ordinary student train", "MSE selection", "test", "semantic scores and utility"],
            "student_selection": "minimum held-out denoising MSE; earlier step wins ties",
            "generation": "ordinary Gaussian generators; no key, custom latents or postprocessor",
            "teacher_objective": c.get('teacher_objective','fixed_shift_v1'),
            "teacher_loss": "preserve + 0.8*(mean semantic shift error + 0.5*mean cycle shift error)" if c.get('teacher_objective')=='threshold_cfg_v1' else "historical fixed shift",
            "candidate_scheduling": "one isolated process per visible GPU; sequential on one GPU",
            "execute": "bash run_semantic_fp.sh", "gpu_work": False, "results_root": str(results_root(carrier))}, indent=2))
        return
    if a.plan_only:
        print(json.dumps({"config": c, "students": branches(c),
              "baseline": "unadapted SD1.4" if carrier == "serum" else "unadapted SD2.1",
              "teacher": (f"author SD2.1 / SERUM alpha={c['serum_alpha']}" if carrier == "serum" else
                          f"official Infinity-2B / BitMark delta={c.get('watermark_delta',2)}, all scales, G=01,10"),
              "training": "fixed step budget; minimum held-out denoising MSE, earlier step breaks ties; detector/utility excluded",
              "results_root": str(results_root()),
              "expected_updates_per_student": c.get("max_train_steps") or math.ceil(c["train_n"]/c["batch_size"])*c["epochs"],
              "detection": ("author SERUM detector; separate frozen teacher and student clean-calibration thresholds"
                            if carrier == "serum" else "official tokenizer+detector; |z| primary + signed-z secondary"),
              "execute": "bash run_serum.sh" if carrier == "serum" else "bash run_bitmark_lora.sh"}, indent=2))
        return
    if a.resume:
        run = a.resume.resolve()
    else:
        run = a.output.resolve() if a.output else results_root(carrier) / datetime.datetime.now(ZoneInfo("Australia/Sydney")).strftime(f"run_{carrier}_%Y%m%d_%H%M%S_%f")
        run.mkdir(parents=True, exist_ok=False)
    # Drive FUSE may not support flock; synchronize same-host launches on local storage.
    import hashlib
    lock_dir = runtime_root() / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / (hashlib.sha256(str(run).encode()).hexdigest() + ".lock")
    with lock_path.open("w") as lock, persistent_log(run):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(f"Persistent run: {run}", flush=True)
        hardware = preflight(a)
        freeze(run / "config.json", c)
        if ((c.get('continuation') and not (run/'continuation.json').exists()) or
                (c.get('preparation_reuse') and not (run/'preparation_reuse.json').exists())):
            from semantic_checkpoint import prepare_continuation
            prepare_continuation(run,c)
        source_files = {str(p.relative_to(HERE)): sha256(p) for p in HERE.glob("*.py")}
        source_files["vendor/provenance.json"] = sha256(HERE / "vendor/provenance.json")
        source_files["requirements.txt"] = sha256(HERE / "requirements.txt")
        if carrier == "semantic_fp":
            project_root = HERE.parent.parent
            for name in ("main.py", "run_semantic_fp.sh", "native_semantic_radioactive_fingerprint_codex_spec.md", "docs/threshold_cfg_v1.md"):
                source_files["../../" + name] = sha256(project_root / name)
            source_files["requirements-semantic.txt"] = sha256(HERE / "requirements-semantic.txt")
        freeze(run / "code_manifest.json", source_files)
        versions = {p: importlib.metadata.version(p) for p in (
            "torch", "torchvision", "diffusers", "transformers", "peft", "accelerate",
            "datasets", "pyarrow", "numpy", "scipy", "torch-fidelity", "huggingface-hub")}
        if carrier == "semantic_fp":
            versions["lpips"] = importlib.metadata.version("lpips")
        if carrier == "bitmark":
            versions["flash-attn"] = importlib.metadata.version("flash-attn")
        freeze(run / "environment.json", {"packages": versions, "hardware": hardware})
        print(f"Run: {run}\nDefault mode EXECUTES training and evaluation, not plan-only.", flush=True)
        py = sys.executable
        def call(file, label, *args):
            stage(run, label, [py, str(HERE / file), "--run", str(run), *args])
        call(assets_file, "01_assets")
        if carrier == "semantic_fp":
            if not (c.get('continuation') or c.get('preparation_reuse')):
                call(teacher_file, "01a_clean_teacher", "--stage", "clean")
                call(teacher_file, "01b_carrier", "--stage", "carrier")
                call(teacher_file, "01c_clean_calibration", "--stage", "score", "--directories", "teacher_clean_calibration")
                call("semantic_report.py", "01d_teacher_null", "--teacher-null-only")
            else:
                from semantic_report import freeze_null
                freeze_null(run,'teacher')
            call(teacher_file, "01e_train_native_teacher", "--stage", "train")
        call(teacher_file, "02_teacher", "--stage", "generate")
        call(teacher_file, "03_teacher_scores", "--stage", "score", "--directories",
             "teacher_clean_calibration", "teacher_clean_test", "teacher_wm_test")
        if carrier in {"serum", "semantic_fp"}:
            call("student.py", "03b_student_calibration_images", "--stage", "generate", "--branch", "unadapted",
                 "--split", "calibration")
            call(teacher_file, "03c_student_calibration_scores", "--stage", "score", "--directories", "unadapted_calibration")
        call("report.py", "04_freeze_threshold", "--calibrate-only")
        if c.get("checkpoint_steps"):
            call("validation.py", "04b_validation_cache", "--stage", "cache")
        call("student.py", "05_unadapted", "--stage", "generate", "--branch", "unadapted")
        for b in branches(c):
            call("student.py", f"train_{b['name']}", "--stage", "train", "--branch", b["name"])
            if c.get("checkpoint_steps"):
                from validation import validation_label
                call("validation.py", f"validate_{b['name']}", "--stage", "evaluate", "--branch", b["name"])
                labels = [validation_label(b["name"], step) for step in c["checkpoint_steps"]]
                call(teacher_file, f"validation_scores_{b['name']}", "--stage", "score", "--directories", *labels)
                if not c["skip_utility"]:
                    call("utility.py", f"validation_quality_{b['name']}", "--split", "validation", "--labels", *labels)
                call("validation.py", f"select_{b['name']}", "--stage", "select", "--branch", b["name"])
        # Final test is generated only after every branch has frozen its selection.
        for b in branches(c):
            call("student.py", f"generate_{b['name']}", "--stage", "generate", "--branch", b["name"])
        call(teacher_file, "06_student_scores", "--stage", "score", "--directories",
             "unadapted", *[b["name"] for b in branches(c)])
        # Always publish detector results before optional heavyweight utility evaluation.
        call("report.py", "07_radioactivity")
        if not c["skip_utility"]:
            if carrier == "semantic_fp":
                call("utility.py", "08_utility", "--labels", "teacher_clean_test", "teacher_wm_test", "unadapted",
                     *[b["name"] for b in branches(c)])
            else:
                call("utility.py", "08_utility")
            call("report.py", "09_final_report")
        write_json(run / "complete.json", {"status": "completed", "result": str(run / "result")})
        print(f"\nFinished. Review: {run / 'result/README.md'}", flush=True)


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    main()
