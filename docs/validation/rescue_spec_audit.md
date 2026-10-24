# RSSF teacher trajectory rescue specification audit

Date: 2026-10-24. Authority: ../../rssf_teacher_trajectory_rescue_codex_spec.md.
The audit covers the implementation and follow-up fixes, with an independent
read-only reviewer. No pretrained SD model generation/training or GPU test ran.

| Spec sections | Checked implementation | Outcome |
|---|---|---|
| 1-6 | Objective/carrier versions separated; original PCA/medians/key/Binomial math retained; clean-only detached easiest24 hinge, mu=.10 | Matches |
| 7-15 | Saved PNDM config checked; actual step instrumentation; complete mutable state; five native anchors; four-call boundary; sharded lossless traces; mandatory clean replay/anchor gate; final PNG clean scores | Matches; real FP16 clean replay runs on server before optimization |
| 16-24 | Single optimizer update, separate preservation/image backward, one native CFG helper, actual scheduler tail autograd, final decoded image clamp and LPIPS/pixel/RSSF losses | Matches; real-component CPU gradient coverage added |
| 25-29 | One teacher/one margin;1500 updates; full checkpoints at100 cadence plus milestones100/250/500/750/1000/1500; fixed128 screening; TPR/CLIP eligibility then LPIPS/pixel/step ranking; full512 validation | Matches |
| 30-33 | Teacher-only early stop;5000 train/1000 calibration/512 validation; no student/test-image/FID/KID stages; PNG and traces together; full AdamW/sampler/RNG resume | Matches |
| 34-39 | Observational wrapping calls original step once; declarative state and PLMS validation; gradients through all4 calls; micro1/effective4; checkpointed paths; BF16 teacher/FP16 inference | Matches |
| 40-45 |16-example mean/median trajectory drift; per-step diagnostics;8 checkpoint pairs/16 selected pairs; fixed final gates; failed campaign records no downstream jobs | Matches after CSV resume fix |
| 46-50 | Two focused new modules, existing runner/teacher/scorer/checkpoint/merge reuse, pinned configuration, strict manifests and historical analysis paths | Matches |
| 51 | Existing verifier and new protocol/replay/CFG/gradient/detachment/dispatch/resume tests |112 semantic CPU tests plus2 subtests pass; true LPIPS/PEFT/VAE gradient gap closed |
| 52-57 | Finite guards, locked first-run values, no carrier redesign/key reroll/sweeps/students/two-phase/full50-step graph | Matches |

## Findings resolved

- Drift JSON used to be treated as completion before CSV existed. The CSV is now
  an atomic deterministic export recoverable from the committed JSON; missing,
  interrupted and inconsistent files are tested without model execution.
- Earlier gradient tests substituted MSE for LPIPS. Separate real-component CPU
  cases now exercise lpips.LPIPS, tiny random Diffusers modules and actual PEFT
  adapters through the production loss path and every scheduler tail call.

## Validation limits

The unit tests establish code behavior, gradient connectivity and deterministic
CPU replay/resume. They do not establish pretrained SD2.1 FP16 replay accuracy,
GPU memory fit,24-hour wall-clock completion or improved image utility. The
existing server replay gate aborts before training on cache mismatch; actual
merged-teacher native validation alone determines TPR/CLIP/LPIPS success.
No outstanding confirmed method or resume defect remains from this audit.
