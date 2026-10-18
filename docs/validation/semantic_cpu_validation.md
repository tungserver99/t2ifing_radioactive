# CPU verification — 2026-10-19

Python used: D:/anaconda3/envs/topmost/python.exe (torch2.5.1+cpu, NumPy1.26.4).

## Results

- Semantic suite: 23 tests, 22 passed, 1 skipped (Diffusers unavailable for the optional direct DDPMScheduler comparison). All three prediction-type formulas also pass independent tensor-algebra tests.
- Entire existing bitmark_lora test discovery including the new tests: 52 tests, 47 passed, 1 skipped, 2 failures, 2 errors. Full output is full_cpu_suite.log.
- Separately verified legacy protocol tests: 13 passed; legacy generation tests: 2 passed.
- Pilot/full plan-only CLIs, Python compilation and vendor integrity are checked separately without CUDA/model loading.

## Existing suite issues retained

1. test_bootstrap: bootstrap_env.py imports POSIX fcntl, which is absent on this Windows machine. This environment-setup module is not called by the new launcher or semantic workflow.
2. test_recovery.RecoveryTests.test_recover_and_reject_incomplete_or_changed_weights: recover_full_export.py imports POSIX fcntl. The recovery command is not called by the semantic workflow.
3. test_serum.SerumTests.test_reject_bad_artifacts_and_zip_traversal: the existing safe_extract check accepts a single backslash path (`a\b`). This was reproduced directly. Semantic assets do not use this ZIP extractor or download the Serum author watermark/archive.
4. test_validation.ValidationTests.test_drive_results_local_environment: compares POSIX /content paths against Windows drive-resolved absolute paths. Server POSIX behavior was not executed locally.

These failures are reported rather than hidden or described as a green full suite. The source snapshot has no .git history, so the evidence for their origin is the inspected unchanged implementation/assumptions, not a baseline commit comparison.

## Review and limits

The initial independent reviewer could not read files because its tool invocation hung; no assurance came from that attempt. A subsequent reviewer examined pasted source for the extractor, losses, teacher training, merge, native generation and orchestration sequence without tools. It found a conditional autograd concern for direct native_generate calls; existing production callers already used inference_mode, and the helper now also enforces inference_mode itself. A regression test failed before that fix and passed after it. This is a limited static review, not a complete independent filesystem/diff audit.

No installation, real diffusion model download, GPU sampling, model training, real-model feature extraction, CLIP/LPIPS/FID/KID evaluation or real-model adapter merge was run. The model merge/reload equivalence assertion is implemented as a mandatory server-stage check; its real-model numerical result is still pending server execution. Tests establish software logic, not scientific effectiveness.

## Decisions recorded

- The supplied detailed specification is the implementation brief; work was performed in place because the workspace has no Git repository. No branch/commit was created.
- Default clean teacher uses the SD2.1 mirror/revision already pinned in the existing asset helper. A user-supplied clean --teacher-snapshot supports exact reuse of the previous Serum teacher. Actual source hashes are recorded; equality with the Serum author snapshot is not claimed for the default mirror.
- Default setting is the pilot (r32 attention+Conv LoRA and full UNet); --setting full exposes all six specified branches. Student adaptation budgets are unchanged.
- To enforce finite checks without inference hooks or a watermark transform, native generation requests standard pipeline final latents, uses the original VAE decode and ordinary Diffusers PIL conversion, and checks both tensors before conversion. The same guard is used only for semantic student generation; student training/selection logic remains unchanged.
- Partial teacher candidates restart deterministically from the clean teacher/seed. Intermediate adapter snapshots are audit artifacts, not optimizer-resume checkpoints.
