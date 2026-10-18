# Validation update, 2026-10-13

This section supersedes older output-location and fixed-final-checkpoint statements below.

- 32 CPU protocol/bootstrap/generation/selection tests pass. They check the 3×2
  rank/scope design, equal step budgets, disjoint splits, Drive results vs local
  runtime, frozen MSE-only selection (including a deliberately worse detector
  score on the selected checkpoint), tie-breaking, and resume tamper rejection.
- Real tiny-model CPU integration: attention LoRA, attention+Conv2d LoRA, and full
  UNet each execute two optimizer steps. Snapshots at steps 1/2 are exported;
  saved adapters actually change. Final/snapshot UNet predictions agree after
  reload, including the production selected-checkpoint loader. Fixed-draw validation v-prediction MSE repeats exactly on CPU.
- Mixed convolution/attention state conversion through legacy Diffusers names
  exposed an ambiguous reload during testing. Inference snapshots now store
  `adapter_model.safetensors` plus `adapter_config.json` with native PEFT names.
  Loader checks every adapter key/shape and rejects missing/unexpected LoRA tensors.
- Snapshot/source hashes are recorded; checkpoint selection excludes detector,
  utility and final test. Quality metrics are descriptive, never a stop gate.
- KID subsets use at most half the available images (capped at 1000) instead of
  repeatedly taking the entire 1000-image set. Subset standard deviation still
  does not measure training-seed uncertainty.
- GPU/Colab execution, mounted-Drive behavior, real Infinity generation and final
  CLIP/FID/KID have not been exercised on this host (no NVIDIA GPU). CPU tests
  cannot establish radioactivity strength or guarantee absence of GPU/OOM errors.

## Historical checks

# Validation, 2026-10-12

Colab compatibility follow-up:

- Six new regression tests passed for Colab local-storage defaults, Drive-path
  rejection, unchanged non-Colab defaults, direct private uv selection/version
  rejection, and T4/A100 capability guards. These were local unit tests with
  subprocess mocks; Windows used an import-only fcntl stub (no lock operations).
- Existing 18 Linux test results below predate this Colab update. The current
  change has not been executed in an actual Colab runtime or through GPU training.
- Source on Drive is permitted; venv, assets, temporary files and active run
  outputs default to `/content/bitmark_runtime` on Colab. Source/protocol parameters
  and the pinned ML package requirements are unchanged.

No-Conda launcher update:

- 18 protocol/bootstrap tests passed on Linux, including the optional clean-trained
  controls, read-only CLI routing, existing-venv reuse, and setup failure propagation.
- On a host with Python 3.12 and no `python3.11`, the bootstrap installed private
  uv 0.9.3 through pip, downloaded Python 3.11.14, created the venv, installed pip
  with ensurepip, and successfully reused the same venv on the second call.
- Restored LF bytes for 45 vendor files whose Windows CRLF checkout broke their
  recorded hashes. All vendor hashes match the unchanged provenance manifest.
  Directory-level Git attributes now enforce LF on future checkouts.
- Full dependency installation was stopped at the user's request. The new
  deployment has not completed its binary/CUDA checks or an end-to-end GPU run.
  Earlier CPU trainer checks below remain historical evidence, not GPU validation.

Follow-up: new runs now omit both clean-trained students and clean training image
generation. All 12 protocol tests pass, including reporting with no clean-trained
files and compatibility with original run manifests. The trainer implementations
and their CPU smoke validation below are unchanged.

Executed locally with Python 3.12, torch 2.7.1+cpu, torchvision 0.22.1+cpu,
diffusers 0.35.1, transformers 4.51.3, peft 0.17.1, accelerate 1.10.1,
datasets 3.6.0, pyarrow 19.0.1, NumPy 1.26.4:

- 11 protocol/integrity tests passed, including a synthetic end-to-end report,
  paired clean adjustment, altered-image rejection, ties at the FPR threshold,
  nonfinite rejection, stage failure propagation, and matched trainer commands.
- Official BitMark detector and logits processor passed CPU checks on alternating
  and constant bit sequences. This validates API wiring and score direction,
  not watermark detection on real generated images.
- Both vendored Diffusers trainers completed **two actual optimizer steps** on
  a tiny randomly initialized SD model created locally. LoRA up-projections
  changed; full UNet weights changed. Both checkpoints saved, reloaded and
  generated a 32×32 image. The input imagefolder included auxiliary JSON
  manifests, as in the real pipeline.
- Official Infinity import chain and adapter arguments checked on CPU with an
  explicit import-only FlashAttention stub. No Infinity forward pass was claimed.
- Shell syntax, Python compilation, root `--help` and `--plan-only` checked.
- Upstream source hashes and pinned model revisions/LFS hashes checked against
  official repositories. Large model download/verification runs on the GPU host.

Runtime issues found and fixed during these checks:

1. Unconstrained PyArrow 26 requires NumPy ≥2, despite resolving alongside
   NumPy 1.26. Pin PyArrow 19.0.1, verified by real imagefolder training.
2. Accelerate 1.10 rejects `log_with="none"`. Both trainer copies explicitly
   map that CLI value to Python `None`; source changes are hash-recorded.
3. LoRA parameter iterator exhaustion would disable repeated gradient clipping.
   The same `list(filter(...))` correction as SERUM is retained.

The deployment environment is Python **3.11/CUDA 12.8**, not this CPU test
environment. The local machine has no NVIDIA GPU and insufficient disk for all
checkpoints. Full Infinity image generation, FlashAttention CUDA execution,
real SD2.1 training, and final CLIP/FID/KID have **not** been run here. The launcher
exercises FlashAttention on CUDA before downloading models. No experimental
TPR or conclusion about LoRA radioactivity is fabricated or promised.

## Colab generation regression check, 2026-10-12

- Fixed upstream Infinity statistics reading `scale_bit_flips` when
  `watermark_count_bit_flip=0` (the default). Measurement state resets at every
  scale; absent measurements omit bit-flip fields. Sampling and RNG calls are
  unchanged. The patch and vendor hash are recorded in `vendor/provenance.json`.
- Executed the actual statistics block for two samples, with measurement off,
  on, and off again. Verified entropy and measured bit-flip fields.
- Exercised teacher generation orchestration with a fake model: clean/watermarked
  train/test pairing, clean calibration, image manifests, captions, and reuse of
  valid images on resume. This is a control-flow test, not an Infinity forward pass.
- Protocol tests cover matched trainer commands, clean controls, prompt splits,
  calibration, failed-stage propagation, report generation and tamper detection.
- Fixed report output to explicitly use UTF-8. Windows test imports use an
  import-only `fcntl` shim; Linux locking is not tested by that shim.
- Both shell launchers call existing Python directly and do not call installers.
  A source change makes old `code_manifest.json` incompatible with `--resume`.
  Start a new run to retain strict provenance; downloaded model cache is reused.

Full CUDA generation/training must still be checked in the user's Colab runtime.
