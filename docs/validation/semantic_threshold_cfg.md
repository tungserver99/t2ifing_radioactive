# Threshold-CFG implementation review and CPU verification

Date: 2026-10-20. Revision protocol: ../threshold_cfg_v1.md. The new default is threshold_cfg_v1; the original fixed-shift specification is historical. This is a new teacher experiment, not evidence that the revised fingerprint succeeds.

## Reviewed requirements and changes

The active objective has two concepts: conditional/unconditional preservation and calibrated coherent fingerprint shift. CFG7.5 is applied before prediction-type-aware x0 reconstruction. The fingerprint MSE averages selected effective-batch samples and bits, with a detached [0,2.5] required shift based on tau+kappa*sigma_clean. The same formula is applied to independently scored clean/FP cycle views at every semantic update. Total loss is preserve+0.8*(sem+0.5*sem_cycle) from step0, with no legacy ramp or perpendicular/fixed-alpha term. New kappas1/1.5/2 start from clean SD2.1. BF16, microbatch1/effective4, readiness checks, finite guards, full-state checkpoints and native merged generation are preserved.

Calibration target statistics are recomputed only from the frozen clean calibration scores. Save ddof0, sample/false-positive counts, empirical/target FPR and threshold/score/carrier hashes in the candidate provenance. No teacher validation/test data fits the target.

An independent Superpowers code review checked objective math, normalization, worker isolation and provenance. Findings addressed: the legacy training protocol retains its original schema; the CLIP hard cap cannot exceed0.01; preparation reuse cannot import a different teacher batch. An additional integration test caught/prevents loss of extended candidate provenance when reconstructing a selected teacher after interruption.

The paired LPIPS metric remains reported and finite-checked, not a hard gate in this revision. Validation chooses the highest TPR among CLIP-passing candidates, with mean score then lower kappa tie breaks. Teacher TPR below0.90 prevents students and remains blocked on resume. Student models, adaptation commands, budget, FP16 and MSE-only selection remain unchanged.

The original feedback proposed simultaneous candidates. One40GB GPU cannot safely host three copies of the measured old17GB training graph. Workers therefore use one isolated process per visible GPU and serialize on one GPU; concurrency is supported across distinct devices. Clean preparation reuse never loads the old fingerprint adapter. The generic shell launcher remains one relative Python command with no environment/setup logic.

## Fresh verification

- Semantic suite:57 tests,56 passed,1 skipped (optional installed-Diffusers scheduler comparison; Diffusers absent locally).
- Full harness:86 tests,81 passed,1 skipped,2 failures,2 errors. These are the same four pre-existing issues:
  - test_bootstrap: Windows lacks fcntl.
  - test_recovery.test_recover_and_reject_incomplete_or_changed_weights: Windows lacks fcntl.
  - test_serum.test_reject_bad_artifacts_and_zip_traversal: legacy backslash ZIP traversal behavior.
  - test_validation.test_drive_results_local_environment: Windows/POSIX path expectation.
- Python compileall passed; plan-only confirms new objective/kappas, original student commands, teacher microbatch1/effective4 and checkpoint100. Worker CLI help recognizes candidate/kappa arguments without loading models.
- Launcher bytes verified LF, shebang plus python main.py "$@" only; executable Git mode100755. Scoped git diff --check passed.

New tests include hand-computed target/clip/normalization and detached clean gradients; exact CFG arithmetic and prediction conversions; microbatch loss/gradient equivalence with uneven masks and cycle every selected step; real BF16 backward/AdamW on a tiny CPU model with frozen reference parameters; clean-only calibration/provenance and tampering rejection; LPIPS diagnostic/CLIP hard cap; exclusive GPU-slot scheduling with actual parallel CPU work; candidate selection/TPR failure persistence and recovery of the full selected record; current and historical report schemas; and immutable preparation reuse without old adapters/batch changes.

No real SD2.1/SD1.4 training, merge, sampling, GPU memory measurement, package installation or downloads were performed. A short readiness probe cannot prove long-run stability. CFG one-step proxy scores need not match full generation, and capped shifts do not guarantee crossing tau. Final real-image validation/quality gates remain required; no revised TPR or inheritance claim is made here.
