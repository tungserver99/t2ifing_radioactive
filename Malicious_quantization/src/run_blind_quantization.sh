#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")" && pwd)"
# No arguments run both watermark experiments sequentially, freeing the first
# process's CUDA allocations before starting the second. Explicit flags keep
# their existing single-watermark behavior. Stop on failure (set -e).
if [[ "$#" == "0" ]]; then
  echo '[1/2] Stable Signature: transfer suite, owner evaluation and FID'
  bash "$SCRIPT_DIR/run_blind_quantization.sh" --profile transfer
  echo '[2/2] SleeperMark: UNet experiments, owner evaluation and FID'
  exec bash "$SCRIPT_DIR/run_blind_quantization.sh" --watermark sleepermark
fi

# --direct forwards to the lower-level Stable Signature experiment.
if [[ "${1:-}" == "--direct" ]]; then
  shift
elif [[ "${1:-}" == "--profile" || "${1:-}" == "--seeds" ||
        "${1:-}" == "--preservation-weights" || "${1:-}" == "--plan-only" ||
        "${1:-}" == "--root" || "${1:-}" == "--min-ssim" ||
        "${1:-}" == "--quality-budgets" || "${1:-}" == "--" ||
        "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  SUITE_PY="${WMQ_SUITE_PYTHON:-$(type -P python3 || type -P python || true)}"
  [[ -n "$SUITE_PY" && -x "$SUITE_PY" ]] || { echo 'Python 3 is required to orchestrate the suite.' >&2; exit 1; }
  exec "$SUITE_PY" "$SCRIPT_DIR/run_blind_suite.py" "$@"
fi
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ATTACK_OUTPUT_ROOT="${WMQ_ATTACK_OUTPUT_ROOT:-$PROJECT_ROOT/output_attack}"
HEAVY_ROOT="${WMQ_HEAVY_ROOT:-$PROJECT_ROOT/output_artifacts}"
MODEL_ROOT="${WMQ_MODEL_ROOT:-$HEAVY_ROOT/models}"
CHECKPOINT_OUTPUT_ROOT="${WMQ_CHECKPOINT_OUTPUT_ROOT:-$HEAVY_ROOT/checkpoints}"
IMAGE_OUTPUT_ROOT="${WMQ_IMAGE_OUTPUT_ROOT:-$HEAVY_ROOT/images}"
DATA_ROOT="${WMQ_DATA_ROOT:-$HEAVY_ROOT/datasets}"
OWNER_ASSET_ROOT="${WMQ_OWNER_ASSET_ROOT:-$HEAVY_ROOT/owner_assets}"
CACHE_ROOT="${WMQ_CACHE_ROOT:-$HEAVY_ROOT/cache}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch}"

# Bootstrap before imports: auto prefers a dedicated Conda env, otherwise current Python.
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false USE_TORCH=1 USE_TF=0 USE_FLAX=0
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE
if [[ "${WMQ_BOOTSTRAP_READY:-0}" != "1" ]]; then
  mkdir -p "$ATTACK_OUTPUT_ROOT"
  LOG_PATH="$(mktemp "$ATTACK_OUTPUT_ROOT/run_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")"
  exec > >(tee -a "$LOG_PATH") 2>&1
  echo "Log (environment setup, fixture, attack, evaluation): $LOG_PATH"
  ENV_MODE="${WMQ_ENV_MODE:-auto}"
  case "$ENV_MODE" in auto|conda|current) ;; *) echo "WMQ_ENV_MODE must be auto, conda or current" >&2; exit 2 ;; esac
  CONDA_BIN=""
  if [[ "$ENV_MODE" != "current" ]]; then
    for candidate in "${CONDA_EXE:-}" "${HOME:-}/miniconda3/bin/conda" "${HOME:-}/anaconda3/bin/conda" /opt/conda/bin/conda; do
      if [[ -n "$candidate" && -x "$candidate" ]]; then CONDA_BIN="$candidate"; break; fi
    done
    if [[ -z "$CONDA_BIN" ]] && type -P conda >/dev/null 2>&1; then CONDA_BIN="$(type -P conda)"; fi
  fi
  if [[ "$ENV_MODE" == "conda" && -z "$CONDA_BIN" ]]; then
    echo "Conda not found; use WMQ_ENV_MODE=current to install into the current Python." >&2; exit 1
  fi
  BOOTSTRAP_PY="${WMQ_PYTHON:-}"
  if [[ -z "$BOOTSTRAP_PY" && -n "$CONDA_BIN" ]]; then
    CONDA_BASE_PATH="$("$CONDA_BIN" info --base)"
    BOOTSTRAP_PY="$CONDA_BASE_PATH/bin/python"
  fi
  if [[ -z "$BOOTSTRAP_PY" ]]; then
    BOOTSTRAP_PY="$(type -P python || type -P python3 || true)"
  fi
  [[ -n "$BOOTSTRAP_PY" && -x "$BOOTSTRAP_PY" ]] || { echo "Python not found. Set WMQ_PYTHON=/path/to/python." >&2; exit 1; }
  bootstrap=("$BOOTSTRAP_PY" "$SCRIPT_DIR/prepare_blind_environment.py"
    --requirements "$SCRIPT_DIR/requirements-wmq.txt" --launcher "$SCRIPT_DIR/run_blind_quantization.sh")
  if [[ -n "$CONDA_BIN" ]]; then
    echo "Preparing dedicated Conda environment ${WMQ_CONDA_ENV:-wmq}..."
    bootstrap+=(--conda "$CONDA_BIN" --env-name "${WMQ_CONDA_ENV:-wmq}")
  else
    echo "Preparing dependencies in current Python: $BOOTSTRAP_PY"
  fi
  exec "${bootstrap[@]}" -- "$@"
fi
PY="${WMQ_PYTHON:?Bootstrap did not select Python}"
export PATH="$(dirname -- "$PY"):$PATH"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "Using Python: $PY"

# SleeperMark uses its own UNet engine, after the shared dependency bootstrap.
if [[ "${1:-}" == "--watermark" ]]; then
  [[ "${2:-}" == "sleepermark" ]] || { echo 'Supported explicit watermark: sleepermark' >&2; exit 2; }
  shift 2
  for argument in "$@"; do
    if [[ "$argument" == "--help" || "$argument" == "-h" ]]; then
      exec "$PY" "$SCRIPT_DIR/wmq_sleepermark.py" "$@"
    fi
  done
  if [[ "${WMQ_SLEEPER_SEQUENTIAL:-0}" == "1" ]]; then
    exec "$PY" "$SCRIPT_DIR/wmq_sleepermark.py" "$@"
  fi
  exec "$PY" "$SCRIPT_DIR/wmq_sleeper_parallel.py" "$@"
fi

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  exec "$PY" "$SCRIPT_DIR/wmq_blind.py" "$@"
fi

"$PY" - <<'PY'
import os
import sys
from pathlib import Path
if Path(sys.prefix).resolve() != Path(os.environ["WMQ_EXPECTED_PREFIX"]).resolve():
    raise SystemExit("Python prefix changed after dependency preparation")
import torch, diffusers, transformers, safetensors, PIL, scipy
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable. Dependencies are prepared; run on a GPU node with a compatible NVIDIA driver.")
print(f"GPU: {torch.cuda.get_device_name(0)}; torch={torch.__version__}; diffusers={diffusers.__version__}", flush=True)
PY

DEFAULT_MODEL="$MODEL_ROOT/marked_sd21"
DEFAULT_OUTPUT="$ATTACK_OUTPUT_ROOT/run_$(date +%Y%m%d_%H%M%S)"
MODEL_PATH="$DEFAULT_MODEL"
ATTACK_OUTPUT="$DEFAULT_OUTPUT"
CUSTOM_MODEL=0
HAS_OUTPUT=0
HAS_IMAGE_OUTPUT=0
HAS_ARTIFACT_OUTPUT=0
ARTIFACT_OUTPUT=""
HAS_BITS=0
HAS_NATURAL=0
TRAIN_N=32
SEARCH_N=20
NATURAL_TRAIN_N=""
NATURAL_SEARCH_N=""
DATA_SEED=3407
NEGATIVE_N="${WMQ_NEGATIVE_N:-1000}"
NEGATIVE_IMAGES="${WMQ_NEGATIVE_IMAGES:-}"
[[ "$NEGATIVE_N" =~ ^[0-9]+$ ]] || { echo "WMQ_NEGATIVE_N must be nonnegative" >&2; exit 2; }
PROFILE="${WMQ_PROFILE:-research}"
METHOD_SET="${WMQ_METHOD_SET:-science}"
case "$PROFILE" in
  research)
    profile_defaults=(--steps 2000 --qat-steps 2000 --ft-steps 2000 --eval-every 100
      --train-batch-size 4 --natural-train-n 4000 --natural-search-n 256 --natural-resolution 256
      --optimizer adamw --weight-decay 0 --lr-schedule warmup_cosine --warmup-steps 20
      --ft-lr .0005 --natural-preservation lowpass --preserve-weight 0.5 --qat-semantic-preserve-weight 0.5
      --cuda-math tf32 --evaluate-final) ;;
  pilot) profile_defaults=(--steps 100 --cuda-math tf32) ;;
  *) echo "WMQ_PROFILE must be research or pilot" >&2; exit 2 ;;
esac
case "$METHOD_SET" in
  focused)
    # Evidence-led default: fixed baselines, the useful model-only reconstruction,
    # residual quantizers, and one unrestricted decoder upper-bound control.
    profile_defaults+=(--methods fixed_ptq reconstruction
      --natural-methods natural_rounding natural_residual natural_residual_qat natural_full_finetune) ;;
  science)
    profile_defaults+=(--bits 4 --methods fixed_ptq qk_rotation_ptq reconstruction
      --natural-methods natural_rounding natural_residual natural_residual_qat_warm natural_residual_cycle_qat_warm natural_random_subspace natural_frequency_subspace
      natural_contrastive_subspace natural_full_finetune natural_joint_finetune natural_teacher_rounding
      --quality-constraint off --quality-policy constrained --gradient-diagnostics-every 100)
    if [[ "${WMQ_MODEL_ONLY:-0}" != "1" ]]; then profile_defaults+=(--finetune-rtn-bits 4); fi ;;
  full)
    profile_defaults+=(--natural-methods natural_rounding natural_rounding_scale natural_residual
      natural_qat_purification natural_residual_qat natural_qat_scale natural_gan_qat
      natural_full_finetune natural_gan_finetune natural_spectral natural_random_subspace
      natural_frequency_subspace natural_contrastive_subspace natural_teacher_rounding)
    if [[ "${WMQ_MODEL_ONLY:-0}" != "1" ]]; then profile_defaults+=(--finetune-rtn-bits 4); fi ;;
  *) echo "WMQ_METHOD_SET must be science, focused or full" >&2; exit 2 ;;
esac
# Explicit CLI flags come last and override profile defaults, including --no-evaluate-final.
arguments=("${profile_defaults[@]}" "$@")
set -- "${arguments[@]}"
echo "Experiment profile: $PROFILE; method set: $METHOD_SET; explicit CLI flags override defaults."
for ((index=0; index<${#arguments[@]}; index++)); do
  argument="${arguments[index]}"
  case "$argument" in
    --model)
      ((index + 1 < ${#arguments[@]})) || { echo "--model requires a path" >&2; exit 2; }
      MODEL_PATH="${arguments[index+1]}"; CUSTOM_MODEL=1; ((index+=1)) ;;
    --model=*) MODEL_PATH="${argument#--model=}"; CUSTOM_MODEL=1 ;;
    --output)
      ((index + 1 < ${#arguments[@]})) || { echo "--output requires a path" >&2; exit 2; }
      ATTACK_OUTPUT="${arguments[index+1]}"; HAS_OUTPUT=1; ((index+=1)) ;;
    --output=*) ATTACK_OUTPUT="${argument#--output=}"; HAS_OUTPUT=1 ;;
    --image-output|--image-output=*) HAS_IMAGE_OUTPUT=1 ;;
    --artifact-output)
      ((index + 1 < ${#arguments[@]})) || { echo "--artifact-output requires a path" >&2; exit 2; }
      ARTIFACT_OUTPUT="${arguments[index+1]}"; HAS_ARTIFACT_OUTPUT=1; ((index+=1)) ;;
    --artifact-output=*) ARTIFACT_OUTPUT="${argument#--artifact-output=}"; HAS_ARTIFACT_OUTPUT=1 ;;
    --bits|--bits=*) HAS_BITS=1 ;;
    --natural-images|--natural-images=*) HAS_NATURAL=1 ;;
    --train-n|--search-n|--seed|--natural-train-n|--natural-search-n)
      ((index + 1 < ${#arguments[@]})) || { echo "$argument requires a value" >&2; exit 2; }
      value="${arguments[index+1]}"
      case "$argument" in
        --train-n) TRAIN_N="$value" ;; --search-n) SEARCH_N="$value" ;; --seed) DATA_SEED="$value" ;;
        --natural-train-n) NATURAL_TRAIN_N="$value" ;; --natural-search-n) NATURAL_SEARCH_N="$value" ;;
      esac
      ((index+=1)) ;;
    --train-n=*) TRAIN_N="${argument#*=}" ;;
    --search-n=*) SEARCH_N="${argument#*=}" ;;
    --seed=*) DATA_SEED="${argument#*=}" ;;
    --natural-train-n=*) NATURAL_TRAIN_N="${argument#*=}" ;;
    --natural-search-n=*) NATURAL_SEARCH_N="${argument#*=}" ;;
  esac
done
ATTACK_OUTPUT="$(realpath -m "$ATTACK_OUTPUT")"
if [[ "$HAS_ARTIFACT_OUTPUT" == "0" ]]; then
  ARTIFACT_OUTPUT="$CHECKPOINT_OUTPUT_ROOT/$(basename -- "$ATTACK_OUTPUT")"
fi
ARTIFACT_OUTPUT="$(realpath -m "$ARTIFACT_OUTPUT")"

mkdir -p "$MODEL_ROOT" "$CHECKPOINT_OUTPUT_ROOT" "$IMAGE_OUTPUT_ROOT" "$DATA_ROOT" "$OWNER_ASSET_ROOT" "$CACHE_ROOT"
if [[ "$CUSTOM_MODEL" == "0" ]]; then
  echo "[1/4] Preparing or reusing the pinned public Stable Signature fixture..."
  "$PY" "$SCRIPT_DIR/prepare_marked_fixture.py" --output "$DEFAULT_MODEL" --reuse
else
  [[ -n "${SS_KEY:-}" ]] || {
    echo "A custom --model requires SS_KEY; the public fixture key must not be assumed for another model." >&2
    exit 2
  }
  echo "[1/4] Using custom marked model: $MODEL_PATH"
fi

attack=("$PY" "$SCRIPT_DIR/wmq_blind.py")
if [[ "$CUSTOM_MODEL" == "0" ]]; then attack+=(--model "$DEFAULT_MODEL"); fi
attack+=(--prompts "$SCRIPT_DIR/prompt.txt")
if [[ "$HAS_OUTPUT" == "0" ]]; then attack+=(--output "$DEFAULT_OUTPUT"); fi
if [[ "$HAS_IMAGE_OUTPUT" == "0" ]]; then attack+=(--image-output "$IMAGE_OUTPUT_ROOT/$(basename -- "$ATTACK_OUTPUT")"); fi
if [[ "$HAS_ARTIFACT_OUTPUT" == "0" ]]; then attack+=(--artifact-output "$ARTIFACT_OUTPUT"); fi
if [[ "$HAS_BITS" == "0" ]]; then attack+=(--bits 4); fi
if [[ "${WMQ_MODEL_ONLY:-0}" == "1" && "$HAS_NATURAL" == "1" ]]; then
  echo "WMQ_MODEL_ONLY=1 conflicts with --natural-images" >&2; exit 2
fi
if [[ "${WMQ_MODEL_ONLY:-0}" != "1" && "$HAS_NATURAL" == "0" ]]; then
  TRAIN_N="${NATURAL_TRAIN_N:-$TRAIN_N}"
  SEARCH_N="${NATURAL_SEARCH_N:-$SEARCH_N}"
  [[ "$TRAIN_N" =~ ^[1-9][0-9]*$ && "$SEARCH_N" =~ ^[1-9][0-9]*$ && "$DATA_SEED" =~ ^[0-9]+$ ]] || {
    echo "Natural pool preparation requires positive train/search counts and a nonnegative seed" >&2; exit 2;
  }
  NATURAL_COUNT=$((10#$TRAIN_N + 10#$SEARCH_N + 10#$NEGATIVE_N))
  NATURAL_POOL="$DATA_ROOT/coco2017_n${NATURAL_COUNT}_seed${DATA_SEED}"
  echo "Preparing public COCO natural images for the additional branches..."
  "$PY" "$SCRIPT_DIR/prepare_natural_images.py" --output "$NATURAL_POOL" --count "$NATURAL_COUNT" --seed "$DATA_SEED" --workers "${WMQ_DOWNLOAD_WORKERS:-8}"
  attack+=(--natural-images "$NATURAL_POOL")
  if [[ -z "$NEGATIVE_IMAGES" ]]; then NEGATIVE_IMAGES="$NATURAL_POOL"; fi
fi
attack+=("$@")

echo "[2/4] Running quantization controls, natural ablations and independent FP32 decoder controls..."
echo "Result directory: $ATTACK_OUTPUT"
"${attack[@]}"
[[ -f "$ATTACK_OUTPUT/report.json" && -f "$ATTACK_OUTPUT/selection_frozen.json" ]] || {
  echo "Attack did not produce a complete frozen run at $ATTACK_OUTPUT" >&2
  exit 1
}

OFFICIAL_EXTRACTOR_SHA256="77cd0a2040b9391233bbcd79c1adf00816b196089cbb844da40035f854637a04"
if [[ -n "${SS_EXTRACTOR:-}" ]]; then
  EXTRACTOR="$(realpath -m "$SS_EXTRACTOR")"
  [[ -n "${SS_EXTRACTOR_SHA256:-}" ]] || {
    echo "A custom SS_EXTRACTOR requires SS_EXTRACTOR_SHA256." >&2
    exit 2
  }
  EXTRACTOR_SHA256="$SS_EXTRACTOR_SHA256"
else
  EXTRACTOR="$OWNER_ASSET_ROOT/dec_48b_whit.torchscript.pt"
  EXTRACTOR_SHA256="$OFFICIAL_EXTRACTOR_SHA256"
  echo "[3/4] Preparing or reusing the checksum-pinned official owner extractor..."
  "$PY" "$SCRIPT_DIR/prepare_owner_evaluator.py" --output "$EXTRACTOR"
fi

if [[ "$CUSTOM_MODEL" == "0" ]]; then
  KEY="${SS_KEY:-111010110101000001010111010011010100010000100111}"
  KEY_SOURCE="${SS_KEY_SOURCE:-facebookresearch/stable_signature public SD2 decoder example key}"
else
  KEY="$SS_KEY"
  KEY_SOURCE="${SS_KEY_SOURCE:-user-supplied key for custom marked model}"
fi
[[ ${#KEY} -eq 48 && "$KEY" != *[!01]* ]] || { echo "SS_KEY must contain exactly 48 binary digits." >&2; exit 2; }

echo "[4/4] Running owner-only evaluation after selection freeze..."
evaluation=("$PY" "$SCRIPT_DIR/evaluate_blind_watermark.py"
  --run "$ATTACK_OUTPUT" --extractor "$EXTRACTOR"
  --expected-extractor-sha256 "$EXTRACTOR_SHA256"
  --key "$KEY" --key-source "$KEY_SOURCE"
  --batch-size "${WMQ_OWNER_BATCH_SIZE:-0}"
  --mechanism-samples "${WMQ_MECHANISM_SAMPLES:-4}"
  --fpr "${FPR:-0.001}" --detector "${DETECTOR:-double}" --min-reference-tpr "${MIN_REFERENCE_TPR:-0.9}")
if [[ "${WMQ_EVAL_LPIPS:-1}" == "1" ]]; then evaluation+=(--lpips); fi
if [[ -n "$NEGATIVE_IMAGES" && "$NEGATIVE_N" != "0" ]]; then
  evaluation+=(--negative-images "$NEGATIVE_IMAGES" --negative-limit "$NEGATIVE_N")
fi
"${evaluation[@]}"

if [[ "${WMQ_EVAL_FID:-1}" == "1" ]]; then
  fid_args=(--run "$ATTACK_OUTPUT" --batch-size "${WMQ_FID_BATCH_SIZE:-16}")
  if [[ -n "${WMQ_FID_REAL_REFERENCE:-}" ]]; then fid_args+=(--real-reference "$WMQ_FID_REAL_REFERENCE"); fi
  if ! "$PY" "$SCRIPT_DIR/wmq_fid.py" "${fid_args[@]}"; then
    echo "WARNING: FID failed; see fid_error.json. Owner results are retained. Retry src/wmq_fid.py separately." >&2
  fi
fi

"$PY" - "$ATTACK_OUTPUT/owner_evaluation.json" <<'PY'
import json, sys
from pathlib import Path
owner_path = Path(sys.argv[1])
report = json.loads(owner_path.read_text(encoding="utf-8"))
validation = report["reference_validation"]
print(f"Reference validation: valid={validation['valid']}; "
      f"TPR={report['reference_tpr']:.4f}; detected={report['baseline_detected_count']}")
if report["baseline_warning"]:
    print("WARNING:", report["baseline_warning"])
    print("Attack rows were retained, but must not be interpreted as successful watermark suppression.")
status_path = owner_path.parent / "run_status.json"
if status_path.exists():
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["phase"] = "owner_evaluation_complete"
    tmp = status_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(status, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(status_path)
PY

"$PY" "$SCRIPT_DIR/wmq_bundle.py" result --run "$ATTACK_OUTPUT"

echo "Finished: $ATTACK_OUTPUT"
echo "Owner results: $ATTACK_OUTPUT/owner_evaluation.json"
echo "Plot-ready table: $ATTACK_OUTPUT/watermark_retention.csv"
echo "All search candidates: $ATTACK_OUTPUT/search.csv"
echo "Quality diagnostics: $ATTACK_OUTPUT/quality_summary.csv"
echo "Review-ready result bundle: $ATTACK_OUTPUT/result"
echo "Checkpoints: $ARTIFACT_OUTPUT"
echo "Heavy artifacts root: $HEAVY_ROOT"
