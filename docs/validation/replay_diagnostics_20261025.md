# Replay failure audit and operational fix

The reported failure was at the mandatory clean replay gate, before teacher
optimization. Pixel max 0.0044021606 passed the frozen limit 2/255 + 0.002;
LPIPS 0.0001288053 exceeded 0.0001. These values do not establish whether the
fingerprint teacher succeeds: no teacher had been trained yet.

## Specification audit

- Section 14 explicitly requires stopping before training on failed replay.
  The previous ValueError enforced that prerequisite, but discarded per-sample
  diagnostic evidence after the first failing sample.
- Sections 28, 29, 30, 44 and 45 already have an appropriate campaign path:
  train once to 1500 steps, screen all six fixed milestones, select by frozen
  eligibility/utility rules, evaluate the selected teacher on 512 native images,
  run trajectory drift, and record a completed/failed teacher-only result.
  A missed final utility gate is an experimental outcome, not an exception.
- Sections 31 and 49 require exact resumability and immutable traces. Updating
  source and simply using --resume would fail code_manifest.json verification.
- Section 52 requires real numerical/integrity errors to remain errors. OOM,
  corrupt artifacts, nonfinite tensors or broken scheduler protocols must not
  be swallowed to create a superficially successful job.

## Behavior after this change

1. Reuse verified clean data, traces, carrier and scores.
2. Replay the original native four-sample batch. Collect all numerical anchor,
   latent and final-state comparison failures and all four image comparisons.
3. Additionally decode the original cached final latent with the clean FP16
   VAE. Measure original-decoded vs saved PNG and replay vs original-decoded
   pixel/LPIPS errors. These separate possible PNG/decoder and replay effects;
   they never replace or relax the frozen gate.
4. Write teacher_trace/replay_diagnostics.json on either numerical outcome.
5. On replay rejection, write result/replay_failure.json and result/README.md,
   print TEACHER BLOCKED, and return normally. Do not write replay.json or
   teacher_complete.json, and do not call training, screening or students.
6. On success, write the existing replay proof and follow the unchanged full
   training/screening/validation/report path.
7. Unexpected campaign errors write result/teacher_execution_error.json with
   phase/type/traceback, then propagate as a nonzero process exit.

A report of teacher_only_blocked means the diagnostic job ended normally but
the teacher experiment has not completed. It must not be counted as a successful
or a negative fingerprint-teacher result. Hard external PBS kills cannot be
converted into Python reports; resume remains necessary for those.

## Resume the existing HPC run

Copy these updated runtime files to the same relative project paths on HPC:

- Malicious_quantization/bitmark_lora/semantic_trajectory.py
- Malicious_quantization/bitmark_lora/semantic_sign_trajectory_train.py
- Malicious_quantization/bitmark_lora/semantic_execution.py
- Malicious_quantization/bitmark_lora/semantic_replay_diagnostics.py
- Malicious_quantization/bitmark_lora/semantic_replay_upgrade.py
- Malicious_quantization/bitmark_lora/test_semantic_replay_diagnostics.py

Update your server's PBS script to use the following launch command, then submit
that script from the project directory. No PBS file is included in the repository.

```bash
bash run_semantic_fp.sh --resume \
  /shared/homes/u26466553/projects/t2ifing_radioactive/run_semantic_fp_20261024_213309_944621 \
  --upgrade-execution
```

For this v2 revision the explicit upgrade is accepted only before teacher
training starts, with exact known old hashes and reviewed new runtime hashes
(both LF and CRLF checkouts supported). Unrelated source changes are rejected.
The old/new manifests are audited at execution_upgrades/replay_diagnostics_v1.json.
Configuration, frozen replay tolerances and trace protocol are never rewritten.
Do not delete or manually edit code_manifest.json to bypass this guard.

Blocked reports can be inspected without loading GPU models:

```bash
bash run_semantic_fp.sh --analyze RUN_DIR
cat RUN_DIR/result/README.md
cat RUN_DIR/teacher_trace/replay_diagnostics.json
```

## Verification

Final semantic CPU suite: **124 passed, 2 subtests passed**, with 10 upstream
Diffusers/torchvision/LPIPS warnings. This run includes the review fix and the
actual-checkout source-hash upgrade test.

The focused replay/upgrade regression suite passed 16 tests (including existing
execution-upgrade tests). New cases cover all four samples, strict unchanged
gates, no false success proof, blocked reports, analysis provenance, errors
propagating, scheduler structure/shape/dtype rejection, exact runtime source
hash acceptance, and actual PNDM replay with a small deterministic CPU model.
The image metric in the small replay orchestration test is a deterministic
stand-in; it does not establish a pretrained LPIPS baseline.

The full harness command was attempted and stopped during collection because
test_bootstrap.py imports the POSIX-only fcntl module on Windows. Running the
remaining harness with --ignore=test_bootstrap.py produced 148 passed and three
existing Windows/platform failures (also documented in the prior v2 audit):

- test_recovery.py::RecoveryTests::test_recover_and_reject_incomplete_or_changed_weights
  (POSIX fcntl import).
- test_serum.py::SerumTests::test_reject_bad_artifacts_and_zip_traversal
  (Windows ZIP-path semantics).
- test_validation.py::ValidationTests::test_drive_results_local_environment
  (POSIX absolute path expectation on Windows).

Bash syntax validation of the PBS content supplied in chat, Python compilation of all
changed runtime modules and git diff --check passed. Independent review found
an overly broad exception catch; a failing regression was added, the catch and
shape/dtype classification fixed, and follow-up review found no further
actionable issue.

No actual pretrained SD2.1 CUDA replay or HPC job was run locally. The existing
GPU discrepancy is not claimed fixed: this change makes the next run explain
it completely while preserving the locked replay and training rules.
