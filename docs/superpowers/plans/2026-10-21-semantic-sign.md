# RSSF implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development for independent carrier/verifier and reporting tasks; parent integrates teacher execution and CLI. Check each deliverable against the locked spec.

**Goal:** Implement the approved RSSF revision while retaining the extractor, native generation, student pipeline and historical analysis.
**Architecture:** Add semantic_sign_v1 modules and versioned carrier artifacts. Dispatch only new runs to decoded-image easiest-bit hinge + LPIPS/MSE teacher loss. Reuse existing LoRA training/checkpoint/merge, isolated workers and all student commands.
**Tech Stack:** Existing PyTorch/Diffusers/PEFT/LPIPS; exact binomial math from Python math.comb.
**Spec:** semantic_sign_fingerprint_codex_spec.md (locked user-approved design).

## Global constraints
- CPU logic only locally, Python D:/anaconda3/envs/topmost/python.exe; no installs/downloads/model training.
- New defaults: 32/48 PCs in 8..64, mu=[.1,.2,.3], alpha=.01, K_detect=24 derived, K_train=26, 5 native late timesteps, CFG7.5, LoRA32 attention_conv, 3000 updates, effective4/micro1 BF16.
- LPIPS+pixel paired loss; preserve/fp/quality weights1/.8/1. LPIPS/pixel weights1/1. No old threshold loss or separate latent/cycle losses in new branch.
- Analytic TPR>=.90, CLIP drop<=.01, LPIPS<=.15. Select min LPIPS, then pixel MSE, then mu. Keep students unchanged.
- Native generation and launcher unchanged; historical threshold module unchanged. Full-state resume remains strict.

## Review focus
- Exact-zero signs and impossible alpha cutoffs must be deterministic.
- Median fit only carrier train, stable PC pool fixed before key generation.
- Scheduler integer IDs must match alpha schedules; noninteger schedules rejected.
- Uneven microbatch weighting and decoded gradients must match effective batch.
- Old 8-bit reuse cannot import key/weights/threshold into new RSSF run; historical reporting remains usable.

## Tasks
- [x] Carrier/math: write failing CPU tests, implement semantic_sign.py exact tails, explicit bits, detached clean TopK hinge and versioned medians in semantic_carrier.py; verify old loader unchanged.
- [x] Reporting: tests then semantic_sign_report.py primary analytic image/model/query/null metrics, secondary continuous ROC/AUC, paired pixel and visual samples; dispatch new objective without changing historical reports.
- [x] Teacher objective: tests then semantic_sign_train.py full-range preservation + independent native late timestep decoded image hinge/LPIPS/MSE; frozen graph checkpointing and finite checks.
- [x] Execution integration: reuse train_one optimizer/checkpoint and merge; attach sign runtime under new branch; derive actual native scheduler via model_index metadata; validate alphas, freeze exact IDs; scratch readiness preserves weights/RNG.
- [x] Campaign/CLI: new default, locked first settings, independent mu workers from clean; strict gates and low-distortion selection; coherent resume/diagnostic/report provenance.
- [x] Prep reuse: reuse only verified clean prompts/images/assets, rebuild carrier/key and scores. Never continue old adapters. Fresh command needs no prior run.
- [x] Verification/review: all semantic unit tests + whole suite, compile, plan-only, shell bytes; independent final spec/code review; fix blockers.
- Delivery (completion evidenced by Git commit/remote HEAD): update README/spec reference, stage owned files, commit and push, verify remote; give one bash command.

## Ledger
Ruling: execute approved spec directly in the shared workspace ? user explicitly requested edits, verification, commit and push; no additional design approval is required.
Ruling: use existing isolated one-worker-per-visible-GPU scheduler ? one40GB GPU serializes fixed independent candidates, multiple GPUs parallelize; no adaptive candidate settings or warm starts.
Ruling: keep historical threshold implementation untouched; historical config-driven analysis remains separate from new defaults.

Ruling: analytic image evidence becomes the checkpoint diagnostic statistic K only after the denoising-MSE winner is frozen ? prevents a late report crash while preserving student selection discipline.
Verification:89 semantic tests run,88 passed and1 optional skip; full118 has only the same4 Windows baseline nonpasses; no GPU/model run. See docs/validation/semantic_sign_v1.md.
