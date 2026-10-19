# Teacher numerical stability and early readiness checks

The supplied failure is a nonfinite total gradient norm at clipping, with finite preceding losses. It does not establish whether individual gradients became NaN/Inf or an FP32 norm reduction overflowed. The last displayed cycle iteration used 17.38 GiB; the traceback is not an OOM.

## Changes

- Separate teacher_precision (new-run default BF16) from unchanged student precision FP16. Historical config interpretation remains FP16 when the teacher key is absent. Spec section 9.1 explicitly allows FP16/BF16 for teacher training; no student protocol, LR or objective is altered.
- No GradScaler in BF16. Actual nonfinite gradients still fail before optimizer updates. Finite gradients use FP64 norm reduction before clipping at 1.0, avoiding false FP32 norm overflow. No silent skipped updates or zero replacement.
- Fresh teacher training first performs 12 isolated readiness optimizer updates, with full ramp/cycle and timestep boundaries. Trainable parameters, gradients and PyTorch CPU/CUDA RNG states are restored, and the scratch optimizer is discarded. The actual optimizer starts empty as before.
- Standalone main.py --teacher-check RUN uses the latest available saved candidate checkpoint, original verified data and carrier, with 32 scratch updates. It records source hashes and writes diagnostic reports separately. Original optimizer state is unavailable, so this is not exact failing-step replay or training resume. More probes are selectable with --teacher-check-steps.
- Backward boundary guards and failure.json include stage/step, precision, timesteps, microbatch loss components and invalid parameter names where available. Hooks indicate the observed failing gradient boundary, not necessarily the internal operation that caused it.

## CPU verification (2026-10-19)

Seven new tests cover separation of teacher/student precision; rejecting actual invalid gradients without changing weights; large finite gradients whose FP32 norm overflows; finite forward with FP16 backward overflow contrasted with BF16; numerical checkpoint selection; named boundary diagnostics; and stress-test weights/RNG restoration on success and failure. The stress test uses the real objective and checkpointed tiny frozen extractor, not model downloads.

Full harness suite: 70 tests, 65 passed, 1 skipped (Diffusers unavailable), 2 failures and 2 errors. Existing errors: test_bootstrap and test_recovery.test_recover_and_reject_incomplete_or_changed_weights require POSIX fcntl. Existing failures: test_serum.test_reject_bad_artifacts_and_zip_traversal, test_validation.test_drive_results_local_environment (Windows/POSIX path expectation). These four are unchanged from the previous baseline.

Actual SD2.1 BF16 numerical/GPU-memory behavior must be checked on the user's server. No GPU runs, model downloads, installs or environment changes were performed locally. Passing a short check does not certify long-run stability or reproduce an unsaved failing batch/optimizer state.
