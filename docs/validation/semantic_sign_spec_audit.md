# RSSF locked-spec audit, 2026-10-21

Audited initial RSSF commit24f4f3b against all sections of semantic_sign_fingerprint_codex_spec.md. Independent read-only review corroborated the findings; scoped re-review accepted both fixes. No model weights, GPU execution, installations or real training were used.

## Confirmed implementation coverage

| Spec sections | Checked behavior | Implementation |
|---|---|---|
|1-3,29 | versioned objective/carrier; historical threshold module and student source preserved | semantic_sign.py, semantic_carrier.py, CLI/report dispatch |
|4-6 | PC8..64, top48 stability pool,32 unique indices, secret signs sampled after clean-only ranking; train-only midpoint medians; explicit zero->+1 | save_carrier, coordinates, image_bits |
|7-10 | exact math.comb tails, derived24 cutoff, exact all32 p-value, primary image/model evidence, fake-key/fixed-key null checks, secondary continuous scores | semantic_sign.py, semantic_sign_report.py |
|11-14 | detached TopK26 from clean signed margins; squared hinge mean over selected bits/batch; no pressure beyond mu; no empirical threshold | sign_hinge_loss |
|15 | final5 unique exact integer IDs from actual native scheduler; matched alpha schedules; incompatible/fractional schedules rejected | sign_schedule, derive_late_timesteps |
|16-20 | same latent/prompt batch; uniform full-range preservation and independent late path; CFG7.5; prediction-aware x0; decode/clamp then posterior mean image extractor; frozen LPIPS/pixel; weights1/.8/1 | sign_teacher_backward |
|21-24 | clean LoRA32 attention+Conv, LR1e-4,3000 updates,4/1 batch,BF16; independent fixedmu campaign; native validation and ordinary merged generation | train_sign_one, train_sign_candidates, existing merge/generation |
|23,36 | analyticTPR>=.90, CLIPdrop<=.01, LPIPS<=.15; passing candidates ranked LPIPS/pixel/mu; no students on failure | choose_sign_candidate |
|25,27-28 | student code/budget/MSE selection unchanged; K diagnostics only after selection; clean data reuse excludes old carrier/key/threshold/weights | existing student/validation pipeline, report.detector_value, prepare_sign_reuse |
|26,30-32 | per-image bits/K/p, model/query evidence, secondary ROC/AUC/calibration, quality/visuals, provenance and finite/state tests | score/report modules and semantic CPU suite |
|33-37 | no learned detector/new encoder, two-phase training, runtime key or new student; locked first-run defaults | CLI defaults/validation + native generation signature |

## Findings and fixes

1. **Supported local snapshot scheduler normalization.** Previously sign_schedule loaded the saved scheduler directly. StableDiffusionPipeline initialization can normalize steps_offset to1 and clip_sample toFalse, so raw scheduler IDs may differ from actual native generation even when alpha values match. The pinned default already has steps_offset1/clip_sampleFalse and is unaffected. Fix: construct a scheduler-only empty StableDiffusionPipeline shell (all model components None), then derive late IDs from its scheduler. This invokes the installed library's actual normalization without loading weights or duplicating private normalization rules. See [Diffusers0.35.1 constructor](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L184).

   A CPU regression reproduced raw late IDs[4,2,0] versus pipeline IDs[5,3,1] for a scheduler double with offset0; now passes. It also checks offset1 remains unchanged. An optional actual PNDM + empty pipeline CPU integration test is included and skips explicitly when Diffusers is absent.

2. **Stricter user-configured quality gates in reports.** Candidate selection already honored stricter configured caps, but quality report identity/passes used fixed.01/.15. Fix: both report and selector now use min(configured_cap, locked_cap). A regression with LPIPS.12 and configured limit.10 reproduced the wrong report pass before the fix and rejects correctly afterward. Historical threshold quality cache shape remains unchanged.

Neither fix changes the default carrier, key, hinge, objective weights, mu campaign or student behavior.

## Fresh verification

- Semantic CPU discovery:91 tests,89 passed,2 optional Diffusers skips,0 failures/errors; Python exit0.
- Both new defects were observed as failing regressions before implementation and pass after the fixes.
- compileall and git diff whitespace checks pass.
- Previous full-repository verification at24f4f3b had only four pre-existing Windows/POSIX failures, recorded in semantic_sign_v1.md; this audit does not claim those unrelated tests were fixed.

Remaining empirical questions are actual A10040GB peak memory/speed, long-run numerical behavior, image utility and inherited radioactivity. CPU logic checks and source review cannot establish these results. The default pinned scheduler behavior is unchanged by this audit; an already-running default24f4f3b experiment need not be restarted solely for these two edge-case fixes. Strict code/config resume provenance still applies, so do not mix source revisions inside a running experiment.
