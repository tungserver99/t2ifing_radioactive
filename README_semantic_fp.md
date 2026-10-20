# Native semantic radioactive fingerprint

The current teacher experiment is **threshold_cfg_v1**, the user-requested revision described in [docs/threshold_cfg_v1.md](docs/threshold_cfg_v1.md). It replaces the original fixed-alpha teacher loss and changes paired LPIPS from a hard gate to a diagnostic. The original specification remains historical; carrier, verifier and student protocol remain the same.

Run from the repository root after preparing your own server environment:

```bash
bash run_semantic_fp.sh
```

The launcher contains only a shebang and `python main.py "$@"`. It activates no environment, installs nothing, sets no environment variables and embeds no interpreter path. `main.py` runs every stage.

To reuse the exact clean data, PCA/key and threshold from the completed old preparation while starting all fingerprint teachers from clean SD2.1:

```bash
bash run_semantic_fp.sh --reuse-preparation run_semantic_fp_20261019_225912_275755
```

This creates a separate run and records source hashes. It does not resume the old alpha=0.5 teacher or reset a partially trained new teacher. All three new candidates start clean. Reuse uses immutable hard links with copy fallback; source artifacts must remain unchanged. A source generation batch inconsistent with the new effective teacher batch is rejected.

## Teacher objective and candidates

Default kappas are 1.0, 1.5 and 2.0. The target is the frozen teacher clean threshold plus kappa times the measured clean-score population standard deviation (ddof0). Conditional and empty-prompt unconditional preservation each have weight0.5. Semantic predicted-x0 uses CFG7.5. The bounded, detached required shift is applied along the secret signs. The semantic loss averages batch and bits; the same constraint is applied through the differentiable VAE cycle on every selected update.

```text
L_fp = L_sem + 0.5 * L_sem_cycle
L_total = L_preserve + 0.8 * L_fp
```

There is no active fixed-alpha, perpendicular, old separate cycle-weight or margin loss. Unlike the original trial, this objective applies from step zero, without the historical warmup/ramp. Semantic mask remains t<=500. The loss uses predicted-x0 proxies; final validation still uses actual native PNG generation, not proxy TPR.

Teacher defaults: clean SD2.1; rank32 attention+Conv LoRA; LR1e-4; AdamW; 3000 updates; effective batch4; microbatch1; BF16 (independent of student FP16); clip1. Frozen reference UNet/VAE/text and score math remain FP32. Activation checkpointing bounds retained differentiation graphs. Finite guards remain active before optimizer updates; gradient norm reduction uses FP64.

On three visible GPUs, three isolated candidate processes can run concurrently. On one GPU40GB, candidates run sequentially with one model-training process per GPU. This avoids placing all three teacher graphs on the same GPU. Each candidate receives the same seed/clean initialization and its own optimizer/state/artifacts. GPU memory of the revised real model must be measured on the server.

## Validation and student settings

Teacher thresholds still use 5000 clean calibration images at empirical1%FPR. The student family has its own frozen unadapted SD1.4 threshold. The carrier/key/statistic are shared. Kappa target fitting uses neither teacher validation nor final test.

Choose highest teacher validation TPR among candidates with CLIP drop<=0.01, then mean score, then lower kappa. Paired LPIPS is reported and checked for finiteness, not used as a hard gate. Require teacher validation TPR>=0.90 before students. Failed gates stop the run without changing student training. A high aggregate bit accuracy does not replace the per-image detection rate.

Student protocol is unchanged: 5000 train/1000 validation/5000 calibration/1000 test prompts; batch8; FP16; 7500 steps; snapshots2500/5000/7500; LoRA/DoRA LR1e-5 and full LR5e-6; inference50/CFG7.5/resolution512. Pilot evaluates unadapted SD1.4, attention+Conv LoRA r32 and full UNet. Full evaluates the original six branches. Only held-out denoising MSE selects student checkpoints, never fingerprint diagnostics.

```bash
# Inspect the protocol without models, downloads or GPU work.
python main.py --plan-only

# All six student branches, after the teacher candidates pass.
bash run_semantic_fp.sh --setting full

# Exact resume of an interrupted NEW run with unchanged code/config/environment.
bash run_semantic_fp.sh --resume run_semantic_fp_YYYYMMDD_HHMMSS_...

# Numerical probe of the latest saved adapters (diagnostic only).
python main.py --teacher-check run_semantic_fp_YYYYMMDD_HHMMSS_...

# Longer standalone probe or FP32 comparison.
python main.py --teacher-check run_semantic_fp_YYYYMMDD_HHMMSS_... --teacher-check-steps 64
python main.py --teacher-check run_semantic_fp_YYYYMMDD_HHMMSS_... --teacher-precision no
```

The historical --continue-from path is a weights-only warm start with reset optimizer and retains its source objective. It is not the way to start this new experiment; use --reuse-preparation to keep clean data while starting new teachers clean.

## Checkpoints, logs and models

Runs are saved beside main.py as run_semantic_fp_*. Stage console logs are in logs/. The candidate workers additionally write logs/teacher_kappa_1.0.log, teacher_kappa_1.5.log and teacher_kappa_2.0.log.

- teacher_fp/calibration_target.json: frozen tau, sigma, counts/FPR, score/carrier hashes and all target scores.
- teacher_fp/candidates/kappa_*/: protocol, losses.jsonl, checkpoint-*, final adapter, merged_pipeline, merge_test.json and validation.json.
- images/candidate_kappa_*_validation/: native candidate images; compare identical file names with images/teacher_clean_validation/.
- teacher_fp/candidates.json: campaign results; teacher_fp/selected.json: frozen selected teacher and gate outcome.
- result/teacher_quality.json and teacher_quality.csv: teacher metrics; result/README.md, radioactivity.json, detection_semantic.csv, threshold files, utility.json and checkpoint_metrics.csv: original reports.

Every teacher loss step records scalar losses and latent/cycle score diagnostics, proxy TPRs, required shifts/cap fractions, semantic mask fraction, target score, gradient norm and peak allocated GPU memory. None of these diagnostics selects a student checkpoint.

Before a candidate trains,12 isolated real-objective readiness updates exercise the mask boundaries, cycle path and optimizer. Parameters, PyTorch CPU/CUDA RNG and scratch optimizer effects are restored/discarded. A short probe does not certify every future batch. Runtime failure.json records precision, step, timesteps, microbatch losses and failing gradient boundary/parameters where available.

Teacher state checkpoints default to every100 updates and the final update. They include trainable weights, AdamW, optional FP16 scaler, completed step, permutation/offset, data RNG and CPU/CUDA/Python RNG, with integrity hashes and loss-prefix hash. Checkpoint manifests are written last. Resume restores the latest complete state, archives post-checkpoint log entries and continues at the next step. Incomplete state folders are ignored; changed complete artifacts fail. The fixed teacher LR has no stateful LR scheduler. CPU tests compare resumed/uninterrupted model and AdamW states exactly; GPU/software differences can still change numerical results.

The default clean teacher is the already pinned SD2.1 base mirror (Manojb/stable-diffusion-2-1-base revision0094d483a120f3f33dafbd187ea4aa60d10de75c); student is pinned CompVis/stable-diffusion-v1-4 revision133a221b8aa7292a167afc5127cb63fb5005638b. Existing clean snapshots are accepted through --teacher-snapshot/--student-snapshot as before. No HMARK, Serum grid, key-dependent generation input or runtime adapter is required for merged teacher generation.

## Environment and verification

Dependencies remain in Malicious_quantization/bitmark_lora/requirements-semantic.txt. Set up the server yourself; the shell file performs no setup. No model/GPU runs or downloads were performed locally.

```bash
python -m unittest discover -s Malicious_quantization/bitmark_lora -p 'test_semantic*.py'
```

Tests cover the original carrier/verifier/native generation/student isolation and the new CFG combination, threshold statistics/provenance, detached/clipped target, batch/bit normalization, cycle gradients, microbatch equivalence, candidate quality/TPR gate, selected-result resume, GPU-slot isolation and reuse without old adapters. The legacy Windows/POSIX and ZIP issues are recorded in docs/validation/semantic_threshold_cfg.md.
