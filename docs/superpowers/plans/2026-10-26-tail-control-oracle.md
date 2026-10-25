# RSSF Tail Control Oracle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the user-authorized locked standalone oracle experiment, then test, review, commit and push; provide PBS content only in chat.

**Architecture:** New modules only; historical teacher execution is unchanged. A differentiable controls module owns residual math/selection, a source/store module validates read-only inputs and atomic batch commits, a report module evaluates PNG-authoritative evidence, and semantic_oracle.py orchestrates the fixed protocol and frozen-model backend.

**Tech Stack:** Existing PyTorch, Diffusers PNDM, safetensors, PIL, LPIPS and CLIP dependencies.

**Spec:** rssf_tail_control_oracle_codex_spec.md (user-supplied locked design, approved for implementation in the request).

## Global Constraints

- Fixed 256 samples, CPU RNG seed 20261026, sorted unique IDs, batch 4, 120 Adam steps, lr .03.
- Four cached guided outputs plus independent FP32 residuals; real restored PNDM; no denoising UNet call.
- Same carrier/verifier, K_detect=24, dynamic current top24 hinge with .02 optimization margin.
- Objective per image: w_det*L_det + LPIPS_float + pixel_MSE_float + .01*relative_delta_energy.
- Double detector weights only for never-feasible images after steps20,40,60,80,100, capped32.
- STE detector path matches actual project PNG rounding; quality reference is float decode.
- Final PNG score and paired PNG quality are authoritative; pass requires feasible rate>=.90 and aggregate CLIP drop<=.01.
- Source run is read-only, independently hashed at start/resume/end. Output is disjoint and automatically resumes committed batches.
- No model training/imprinting/student work, no sweep, no PBS repository file, no source-run upgrade.

## Task 1: Pure controls and candidate tracking

- [ ] Add failing tests for cached zero replay, gradients to all four residuals, dynamic top24, STE PNG equivalence, finite guards and candidate/adaptive-weight rules.
- [ ] Implement semantic_oracle_math.py using existing scheduler and decoded-image helpers.
- [ ] Verify no source/model gradients and finite nonzero gradients through a real tiny Diffusers path.

## Task 2: Read-only provenance and resumable output

- [ ] Add failing tests for incompatible source, output overlap, immutable source hashes, committed-batch reuse and interrupted commit recovery.
- [ ] Implement semantic_oracle_store.py: source manifest, code/protocol identity, atomic batch safetensors/JSON commit, verification and final 32-target shard export.
- [ ] Keep derived PNGs/metrics within the oracle output; never call historical run orchestration.

## Task 3: PNG-authoritative reporting

- [ ] Add failing tests for float/PNG detection disagreement, exact feasible count, aggregate CLIP decision, clean-K bins, Wilson CI, frontier and deterministic representative visuals.
- [ ] Implement semantic_oracle_report.py with existing image_evidence/model_evidence and quality conventions.

## Task 4: Standalone frozen-model execution

- [ ] Implement semantic_oracle.py CLI (--source-run, --output, --plan-only), auto-resume and fixed config.
- [ ] Verify zero controls in original dtype against cached final latent; separately validate the declared FP32 graph baseline.
- [ ] Validate decoded clean coordinates through the canonical PNG/scorer path using stored clean scores; freeze named strict tolerances in the oracle protocol.
- [ ] Optimize per sample with serial gradient accumulation inside batch4 to limit graph memory; save best eligible candidate or best diagnostic fallback.
- [ ] Canonical native FP16 VAE evaluation and actual PNG readback use the project image processor/scorer. Keep float/STE, float quality and PNG evidence explicitly separated.
- [ ] Commit batch only after its residuals, latents, PNGs, metrics and frontier are complete. Finalize targets/report and reverify source hashes.

## Task 5: Verification and delivery

- [ ] Run oracle tests and the full semantic suite; attempt broader harness and disclose existing platform failures.
- [ ] Independent code review; fix concrete findings with regression tests.
- [ ] Document CLI, automatic batch resume, output paths, fixed scientific interpretation and GPU validation limits.
- [ ] Add the supplied spec and implementation files; exclude unrelated validation.json; commit/push main.
- [ ] Return an explicit PBS script in chat with CPU thread limits and a direct live log; no PBS file in repo.

## Review Focus

- FP16 cached exactness must not be confused with FP32 optimization/canonical PNG paths; test both explicitly.
- A float-feasible candidate losing a bit at actual PNG verification must not be counted as PNG feasible.
- Quantized residual storage must not silently substitute different values for the selected float solution; selected final latents remain exact FP32.
- A crash after partial output writes must not make that batch appear committed or rewrite previous committed batches.
- Output overlap, changed source bytes, stale output protocol/code, nonfinite values or missing provenance must reject before optimization.

## Execution ledger

Tasks1-4 implemented and verified. Final semantic suite:181 passed plus2 subtests;
32 new oracle tests cover the frozen scientific and resume rules. Independent
review and final follow-up found no remaining actionable issues. Full harness
collection is blocked by the pre-existing Windows fcntl import; see
docs/validation/semantic_oracle_20261026.md. Git delivery is recorded in the
commit history and final handoff message.

Ruling: zero-control coordinate validation uses the canonical native FP16 PNG
path, separately from the differentiable FP32 reference — this avoids confusing
PNG/decode precision with consistency of the saved clean score.

Ruling: use serial per-image graphs within the fixed batch4, with one Adam
parameter containing independent slices — no model state is optimized and no
cross-image objective coupling or graph retention is introduced.

Ruling: include baseline step0 and all120 post-update iterates; apply detector
weight increases after evaluating steps20/40/60/80/100, before the next update.

Ruling: diagnostic fallback maximizes K then minimum Top24 margin, followed by
distortion/energy and earlier step. It never sets float_feasible=true unless the
specified detector and safety-margin conditions actually hold.

Ruling: store residuals in FP16 only after selection, retain final latents in
FP32, and record storage error — the final PNG must represent the selected
latent, not an unvalidated replay of rounded residuals.
