# RSSF implementation and verification

Implemented against the locked semantic_sign_fingerprint_codex_spec.md. The new default is semantic_sign_v1; semantic_threshold.py and student.py source are unchanged. No model weights, real models or GPU training were loaded for local verification. Only two tiny pinned Hugging Face JSON metadata documents were read to confirm native scheduler identity.

## Scope and reuse

- Reused extractor definition and mid-block early exit, PCA/stability/sign canonicalization, LoRA target rules, full-state checkpoint helpers, merge/native generation, isolated worker scheduler and unchanged student adaptation.
- Added a versioned32-bit median-centered carrier, clean-only TopK26 squared hinge, exact key-randomization Binomial verifier and centered continuous secondary evidence.
- Full-range preservation and independent late decoded-image fingerprint/LPIPS/pixel paths contribute to one optimizer update. Weights1/.8/1, LPIPS/pixel1/1; no threshold forcing, broad mask or separate cycle objective.
- Default mu candidates.10/.20/.30 start clean with identical seeds/shared key; one exclusive process per GPU. One GPU serializes fixed candidates without adapting them to previous outcomes.
- Selection is analyticTPR>=.90, CLIPdrop<=.01, LPIPS<=.15, ranked LPIPS/pixel/mu. Failed campaign saves candidates+failure and cannot launch students.
- Preparation reuse verifies immutable clean prompts/images/assets, rebuilds key/medians/carrier/scores, and never imports old fingerprint adapters or null thresholds.
- Historical analysis dispatch remains config-driven. Explicit threshold tests request their historical objective after the intentional default change. Historical code-execution migration test uses its pinned target hashes, not the new RSSF runtime.

## Scheduler evidence

The pinned SD2.1 mirror metadata at revision0094d483a120f3f33dafbd187ea4aa60d10de75c identifies PNDMScheduler, prediction_type epsilon,1000 training timesteps, scaled_linear beta schedule, steps_offset1 and skip_prk_steps=true. Sources: [model index](https://huggingface.co/Manojb/stable-diffusion-2-1-base/resolve/0094d483a120f3f33dafbd187ea4aa60d10de75c/model_index.json), [scheduler config](https://huggingface.co/Manojb/stable-diffusion-2-1-base/resolve/0094d483a120f3f33dafbd187ea4aa60d10de75c/scheduler/scheduler_config.json).

Runtime derives the exact last5 unique integer IDs from that actual scheduler's set_timesteps(50), verifies alpha compatibility with the training DDPM schedule and freezes IDs/alphas/class/config hashes. No timestep cutoff or assumed linear spacing is hard-coded. Unsupported fractional or incompatible schedules fail clearly.

## Verification at initial RSSF commit24f4f3b

Python: D:/anaconda3/envs/topmost/python.exe. CPU-only synthetic models/fixtures, no installs or model downloads.

- Semantic suite: **89 tests;88 passed,1 optional Diffusers test skipped;0 failures/errors**. Direct Python test runner exit0.
- Full repository suite: **118 tests;113 passed,1 skipped,2 failures+2 errors**. The same four baseline Windows issues remain: ZIP-backslash extraction expectations, POSIX vs Windows paths, missing fcntl in bootstrap_env and recover_full_export. No new RSSF failures.
- compileall passes; plan-only confirms32/48 carrier, fixedmu campaign, three-term decoded objective and unchanged SD1.4 pilot commands/budget.
- Launcher verified unchanged: executable shell file, LF, shebang plus only python main.py "$@". No environment variables/setup or absolute Python path added.

Tests cover exact tails/cutoff/all32 p-value, ties, correlated-coordinate random-sign null, midpoint train-only medians/immutable key, clean-only detached TopK/hinge gradient saturation/nonselected zero gradient, native schedule dedup+alpha mismatch, frozen modules, independent semantic/LPIPS/pixel input gradients, uneven microbatch weighting, CPU AdamW+sampler/noise/timestep exact resume, successful/failed scratch-probe state restoration, quality selection/ties/gates, isolatedmu workers, clean-only prep reuse/corruption rejection, primary analytic student checkpoint diagnostics without changing MSE selection, strict score provenance, fake-key vectorization/cache, model/query evidence, continuous ROC/AUC and old reports.

Independent reviews covered carrier/math, training gradients/checkpoints/selection, CLI reuse/compatibility, reporting/cache and late-stage integration. Fixed review findings: historical continuation reachability, explicit flag normalization, summary fixture num_bits, and the primary analytic bridge into existing validation diagnostics. Decoder/feature/LPIPS graphs use activation checkpointing, but actual SD2.1/A10040GB peak memory, speed, image quality and radioactivity still require the server experiment.

## Run

```sh
bash run_semantic_fp.sh --reuse-preparation run_semantic_fp_20261020_135117_009783
```

This starts a fresh RSSF run using verified old clean data. Do not --resume an old threshold teacher to change objectives. New interrupted RSSF runs use strict --resume THEIR_RUN with unchanged code/config/environment.

A subsequent locked-spec re-audit found and fixed two edge cases (native scheduler normalization for local snapshots and stricter quality-report gates). See [the audit record](semantic_sign_spec_audit.md) for findings and fresh91-test evidence.
