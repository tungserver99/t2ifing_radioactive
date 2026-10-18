# Native semantic fingerprint implementation plan

Goal: execute native_semantic_radioactive_fingerprint_codex_spec.md using the existing Serum downstream harness.
Architecture: new semantic modules alongside bitmark_lora; root main.py; command-only relative launcher. Execute inline using Superpowers, with one independent final review.

- [x] CPU contracts for key, PCA, relative shift, scheduler types and native generation.
- [x] Clean assets; deterministic extractor, PCA/stability and frozen key artifacts.
- [x] Teacher-only LoRA objective, timestep mask, warmup/ramp, cycle gradients and merge/reload verification.
- [x] Validation-only strength/quality selection; independent clean teacher/student nulls.
- [x] Shared student/validation/utility integration; pilot/full settings.
- [x] Limited independent static core review, CPU regression verification and documentation; actual GPU/model checks remain unexecuted as requested.

Ruling: supplied detailed specification is the implementation brief; no redesign is required. Workspace is a source snapshot without .git, so worktrees/commit steps do not apply.
Ruling: use the clean SD2.1 mirror and revision already pinned in assets.py by default; optionally accept an existing clean snapshot. Source hashes and architecture are recorded and checked. No Serum author watermark download is needed.
Ruling: default to the pilot (attention+Conv LoRA r32 and full UNet), with --setting full exposing the six-branch table. All budgets remain unchanged.

Review focus: resume integrity, tied calibration, prediction-type algebra, gradient flow through frozen modules, standard merged inference, matching student hyperparameters.

Verification: 23 semantic tests: 22 pass, 1 skip. Full suite: 52 discovered, 47 pass, 1 skip, 4 retained existing/platform issues. See docs/validation/semantic_cpu_validation.md.
