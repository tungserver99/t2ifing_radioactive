# RSSF v2 trajectory rescue verification

Date: 2026-10-24 (Australia/Sydney).

Implementation follows the locked rssf_teacher_trajectory_rescue_codex_spec.md.
Changes are scoped to new native-trajectory/training modules plus the existing
runner, teacher generation/scoring, objective classification, asset preparation
and paired-quality interfaces. Historical v1/threshold paths remain separate.

## Local verification

Interpreter: D:/anaconda3/envs/topmost/python.exe, CPU PyTorch 2.5.1.
Diffusers 0.35.1 and huggingface-hub 0.34.4 were loaded from a temporary local
test dependency directory; the existing environment was not changed.

- All 105 semantic CPU tests pass, including unchanged Binomial/verifier regressions.
- New tests cover objective classification/locked config, native recorder
  transparency, complete declarative PNDM state, CFG equivalence, exact replay,
  original-batch reconstruction, protocol and shard corruption rejection,
  incomplete-shard resume, four-call gradients, detached targets, actual
  teacher-only runner dispatch and exact interrupted optimizer/sampler resume.
- Shell syntax passes. Both `bash run_semantic_fp.sh --plan-only` and
  `bash run_semantic_fp.sh --resume RUN_DIR --plan-only` were executed and their
  JSON verified: same frozen v2 configuration, zero student stages.
- Independent review found/fixed a cached-score CPU/GPU device mismatch, with
  a failing-then-passing CPU regression test.

Broader harness verification: 130 passed, 3 failed (plus 2 subtests). The failures
are in unchanged platform-specific code: recover_full_export imports POSIX
fcntl; Serum ZIP traversal expectation differs on Windows; a Colab path test
expects POSIX paths. test_bootstrap was excluded from this broader run because
its POSIX fcntl import fails during collection on Windows. These are outside the
teacher rescue path; they were not changed in this revision.

## Server-only verification

No real model generation, training, GPU test, or scientific utility recheck was
performed locally. The server campaign must pass its mandatory instrumentation
and exact FP16 clean tail replay/anchor tests before any teacher optimization.
Replay tolerances are declared in the trace protocol and measured errors are
saved in teacher_trace/replay.json. The final teacher quality is determined only
by native FP16 validation with the frozen TPR/CLIP/LPIPS gates, never by CPU tests
or BF16 training diagnostics. Failure stops cleanly without student work.
