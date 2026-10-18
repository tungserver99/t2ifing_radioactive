# Native semantic radioactive fingerprint

Run from this directory after setting up your server environment yourself:

```bash
bash run_semantic_fp.sh
```

The shell file contains only a shebang and `python main.py "$@"`. It does not activate environments, install packages, choose an interpreter or set environment variables. All code paths are relative to this checkout. `main.py` is the sole top-level experiment entry point; the implementation is split into focused modules under `Malicious_quantization/bitmark_lora`.

## Settings

```bash
# Inspect the frozen protocol without importing/downloading models or using a GPU.
python main.py --plan-only

# Default first pilot: unadapted SD1.4, attention+Conv LoRA r32, full UNet.
bash run_semantic_fp.sh

# Full six-branch Serum comparison after examining the pilot.
bash run_semantic_fp.sh --setting full

# Teacher-only alpha sweep, selected by held-out teacher validation and quality.
bash run_semantic_fp.sh --semantic-shift-sweep 0.25 0.5 0.75

# Independent training seed; key stays the same unless explicitly changed.
bash run_semantic_fp.sh --setting full --seed 1

# Resume the exact frozen run: no experimental overrides are accepted.
bash run_semantic_fp.sh --resume Malicious_quantization/run_semantic_fp_YYYYMMDD_HHMMSS_...

# Recompute a completed report without generation/training.
python main.py --analyze Malicious_quantization/run_semantic_fp_YYYYMMDD_HHMMSS_...
```

`python main.py --help` exposes all settings, including teacher loss weights, alpha, key seed, PCA range, teacher rank/budget/batches, semantic timestep mask and student branch choices. PC 8..64 is **zero-based and inclusive**. Defaults: 8 bits chosen from the 16 clean VAE-cycle-stable PCs, alpha 0.5, teacher rank32 attention+Conv, LR1e-4, 3000 steps, batch4. Fingerprint loss ramps from step250 to750. Preservation applies to every sampled timestep; semantic losses apply only at t<=500. Raw losses, mask fractions and gradient norms are logged. Finite checks fail before optimizer updates or result serialization.

Downstream defaults are the specified Serum comparison: disjoint COCO prompt counts 5000 train /1000 validation /5000 calibration /1000 test; 7500 steps; snapshots2500/5000/7500; batch8; LoRA/DoRA LR1e-5; full LR5e-6; FP16; resolution512; inference50 steps/CFG7.5. Full includes attention r16/r32, attention+Conv r16/r32, DoRA attention+Conv r32, full UNet. Student training, `validation.py` and the vendored trainers retain their existing logic. For semantic runs, student image generation shares the ordinary native decoding helper solely to fail on nonfinite outputs before PNG conversion; prompts, seeds, sampler and adaptation are unchanged. Detection and quality diagnostics never select student checkpoints: only held-out denoising MSE, with earlier-step tie break.

## Models, data and environment

Dependencies are declared in `Malicious_quantization/bitmark_lora/requirements-semantic.txt`; nothing installs automatically. The server environment needs the pinned Diffusers/PEFT versions and LPIPS, CLIP/FID/KID dependencies. Semantic runs do not require FlashAttention, HMARK, a Serum watermark checkpoint, its grid or detector. Model/data downloads happen in the Python asset stage if missing, using the existing pinned asset/split helpers. `--plan-only` does not download anything.

The default clean teacher is the SD2.1 base mirror already pinned by this checkout (`Manojb/stable-diffusion-2-1-base`, revision `0094d483a120f3f33dafbd187ea4aa60d10de75c`). Student is `CompVis/stable-diffusion-v1-4`, revision `133a221b8aa7292a167afc5127cb63fb5005638b`. Every model file is hashed. To use the exact existing clean teacher snapshot from your Serum experiment, pass `--teacher-snapshot relative/path/to/snapshot` (a normal Diffusers pipeline directory). `--student-snapshot` accepts an existing pinned student folder with the `verified.json` emitted by the shared asset downloader. `--cache` and `--output` can point to relative server directories. No local machine paths are embedded in the launcher.

The extractor is FP32 and frozen, including VAE/text/reference UNet. It uses posterior **mean**, scaled latents, t_probe=0, spatially averaged mid-block output and an inferred channel dimension. Teacher optimization preserves input gradients through these frozen components. The cycle loss decodes, clips to the normal image range, mean-encodes and extracts features again. This differentiable cycle omits PNG quantization; candidate validation scores actual saved PNGs.

Teacher adapters are trained from the same clean teacher and seed for every alpha. Each candidate is merged, saved and reloaded as a standard pipeline, with an FP32 output-equivalence assertion (atol/rtol1e-4). Any remaining PEFT layer or nonfinite tensor fails the stage. The selected native generation path loads only model weights and ordinary prompt/generator inputs, with no key, inference hooks or custom input `latents` argument. It requests the standard pipeline final latent output, checks finiteness, and uses the original VAE and Diffusers image conversion to save ordinary PNGs; this adds no watermark transform.

## Gates and artifacts

The teacher threshold is frozen on clean calibration before candidate selection. Candidates use the whole held-out teacher validation split, with paired clean/fingerprinted prompts and Gaussian seeds. Required teacher quality gate: Serum-style CLIP drop<=0.01 and paired LPIPS<=0.15; then choose highest validation TPR, mean score as tie break, lower alpha as final tie break. No passing quality candidate stops the experiment. Validation TPR below0.90 also stops before student training (configurable `--teacher-min-tpr`). A failed frozen gate cannot be bypassed by resume. `--skip-utility` skips student CLIP/FID/KID only; mandatory teacher quality still runs.

Both teacher and unadapted-student nulls use the same key/extractor/statistic and separate conservative empirical1%-FPR thresholds. Both are frozen before scoring adapted-student test outputs. The final test never influences the carrier, key, teacher strength or checkpoint selection.

Default run directories are under `Malicious_quantization/run_semantic_fp_*`, following the existing harness. They include config/assets/prompts/environment/code hashes, carrier PCA/key/stability/provenance, per-candidate adapters/checkpoints/merged pipeline/merge test/losses/validation quality, selected teacher provenance, image hashes/seeds, student snapshots and MSE selection, fixed scores and result tables. Fingerprinted teacher data retains the harness name `teacher_wm_*`; it is generated by native merged weights, not a runtime watermark. Selected model path is in `teacher_fp/selected.json`. Reports are `result/README.md`, `radioactivity.json`, `detection_semantic.csv`, `teacher_quality.json`, `teacher_quality.csv`, `threshold_teacher.json`, `threshold_student.json`, `utility.json`, and `checkpoint_metrics.csv`.

Resume verifies frozen config, code, assets, prompt hashes, carrier/key, trained weights and selected teacher. A partial teacher candidate deterministically restarts from the same clean model/seed; saved intermediate adapters are audit snapshots, not optimizer-resume files. Completed candidates are reused only after integrity checks.

## CPU checks and remaining GPU validation

Use your current environment on the server, or the existing topmost environment locally:

```bash
python -m unittest discover -s Malicious_quantization/bitmark_lora -p 'test_semantic*.py'
```

Tests cover projection/score/bit math, deterministic key selection, PCA orthogonality, frozen extractor determinism and input gradients, relative shift, all supported prediction types, loss ramp, artifact tampering, student isolation and budgets, native pipeline arguments, candidate quality selection, resume gates, threshold families and calibration immutability. An optional scheduler comparison is skipped if Diffusers is absent; no package is installed for tests.

No model training, model download, real diffusion sampling, CLIP/LPIPS/FID/KID evaluation or real model merge was run on this CPU-only machine. Actual merge/reload equivalence is enforced automatically in the server workflow before any candidate is used. Unit-test success is not evidence of fingerprint performance or inheritance.

Second specification audit and its corrected edge cases: [docs/validation/semantic_spec_audit.md](docs/validation/semantic_spec_audit.md).
