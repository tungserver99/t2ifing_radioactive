# RSSF teacher trajectory rescue implementation plan

> **For agentic workers:** Use superpowers:executing-plans for inline implementation.

**Goal:** Implement the locked teacher-only rescue without changing the successful carrier/verifier.
**Architecture:** Observe native generation, store immutable PNDM traces, reuse existing LoRA/checkpoint/merge/scoring machinery with a versioned four-call objective.
**Tech Stack:** Python/PyTorch, pinned Diffusers 0.35.1, safetensors, existing shell entry point.
**Spec:** rssf_teacher_trajectory_rescue_codex_spec.md (locked design supplied and implementation authorized by user).

## Global constraints
- Preserve historical semantic_sign_v1 and threshold_cfg_v1 artifacts/analysis.
- Teacher-only: 5000 train / 1000 calibration / 512 validation, one mu=.10, K_train=24, 1500 updates, effective batch 4, microbatch 1, BF16 training, FP16 native evaluation.
- Four actual PNDM calls, five pre-tail anchors, real final PNG quality/clean coordinates; strict replay gate before training.
- No GPU/model execution on this machine; use D:/anaconda3/envs/topmost/python.exe for CPU verification.
- Single server entry point run_semantic_fp.sh; resume exact frozen protocol.

## Review focus
- Interrupted shard writes: commit shard metadata last, validate committed hashes; regenerate only incomplete shards.
- Scheduler extra state or config changes: declarative state roundtrip, protocol rejection and exact replay.
- Boundary history and target tensors: detached constants, differentiable new outputs for all four calls.
- Milestones 250/750 outside the 100-update cadence: save full checkpoints at their union.
- No eligible checkpoint / failed full utility: diagnostic report, no downstream execution.

### Task 1: Protocol and orchestration
- [x] Add CPU failing tests for objective classification, locked v2 config, teacher-only plan, selection rule.
- [x] Separate sign carrier version from sign objectives; add v2 arguments/config and teacher-only early return.
- [x] Run protocol/sign regression tests.

### Task 2: Native traces and replay
- [x] Add CPU tests for observational recorder, explicit scheduler state roundtrip, cached replay, CFG equivalence and four-call gradients.
- [x] Implement semantic_trajectory.py: PNDM validation, detached declarative state, sharded safetensors, strict protocol and replay checks.
- [x] Integrate clean generation trace capture in the same pass; retain historical generation behavior.
- [x] Run CPU trajectory tests (no model downloads/inference).

### Task 3: Optimization, selection and reports
- [x] Implement semantic_sign_trajectory_train.py using existing resume/LoRA helpers; separate branch backward and full milestone checkpoints.
- [x] Screen deterministic first 128 validation rows, select eligible minimum LPIPS/pixel/step, merge once selected, full 512 validation and drift/pairs.
- [x] Write teacher-only completion/failure reports; update README with one-command execution/resume.
- [x] Run all semantic CPU tests, syntax and plan-only verification; review diff against locked spec.
- [x] Stage explicit task files, commit and push; leave unrelated validation.json untouched.

## Execution record

- Task 1 complete: objective/carrier separation and teacher-only configuration/dispatch tests pass.
- Task 2 complete: real pinned PNDM CPU tests cover observational tracing, safe serialization, exact single/batch replay, protocol/corruption rejection, incomplete shards and four-call gradients.
- Task 3 complete: full AdamW/sampler/RNG resume matches uninterrupted tiny v2 optimization; native screening/full validation/reporting are wired for server execution.
- Independent read-only reviewer found one important issue: cached-score resume passed CUDA signs to CPU validation. Reproduced with a failing CPU device-boundary test, fixed by normalizing the carrier to CPU; regression passes.
- Review scope remained the locked supplied specification. No carrier/verifier changes, GPU/model execution, new sweeps or students.
- User authorized commit/push at the outset; integrate into main with a fast-forward and push without force after final verification.

- Final diagnostic check: r=0 is counted using the unchanged image_bits tie rule; a failing-then-passing assertion covers negative secret signs at zero.
