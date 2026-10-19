# Teacher cycle memory and output location fix

The supplied server traceback reaches the first differentiable VAE-cycle loss after preservation-only warmup, inside VAE encode. The 39.49 GiB GPU reported 37.43 GiB allocated, 888 MiB reserved but unallocated, and only 156 MiB free when requesting another 512 MiB. This is activation pressure from the concurrent FP32 reference/cycle graphs, rather than evidence that allocator fragmentation alone caused the failure.

Teacher forwards now default to microbatch 1 with gradient accumulation over the original effective batch 4. Preservation is normalized over all effective-batch samples; semantic terms are normalized over all timestep-mask-selected samples, including uneven microbatches. Gradient unscale, clipping, optimizer update and scheduler/budget semantics remain once per original step. Frozen reference feature extraction and differentiable VAE cycles use non-reentrant activation checkpointing; input gradients are preserved. No loss is disabled and no precision, training budget or student setting is reduced. Logs record effective batch, microbatch and peak allocated GPU memory.

Semantic runs now default to run_semantic_fp_* beside main.py, including logs/. Explicit --output and BITMARK_RESULTS_ROOT overrides are retained. Legacy harness methods retain their previous default location. Existing run directories are not moved or rewritten. Resume still checks frozen code/config hashes: an old run from before this change is not silently migrated or accepted under changed code. Start a new run with this version.

## Verification on CPU (2026-10-19)

- Semantic suite: 34 tests, 33 passed, 1 skipped (Diffusers unavailable).
- New tests compare microbatch sizes 1/2/full, effective batch 5 with uneven timestep masks, warmup/noncycle/cycle steps, and no selected semantic samples. Losses and parameter gradients agree within float32 tolerance.
- Checkpointed extractor features/cycle/input gradients match the ordinary path. Frozen parameters remain frozen; scoped hooks are removed. Saved tensor element counts decrease on the small CPU model.
- Default and explicit output roots verified; plan-only confirms teacher effective batch 4, microbatch 1 and the project-root results path.
- Python compileall passed.
- Full harness suite: 63 tests, 58 passed, 1 skipped, 2 failures and 2 errors. These are the same four previously documented legacy issues: missing Windows fcntl in bootstrap/recovery, ZIP backslash traversal rejection, and POSIX path expectation on Windows. No new failure was introduced.

No real model/GPU execution, downloads or environment setup was performed. These checks establish loss/gradient and path logic, not an actual SD2.1 peak-memory measurement or a guarantee that every GPU configuration fits.
