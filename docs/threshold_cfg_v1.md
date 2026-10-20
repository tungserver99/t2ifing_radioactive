# Threshold-calibrated coherent shift (threshold_cfg_v1)

User-requested protocol revision, 2026-10-20. This supersedes the teacher objective, candidate strength and teacher quality policy in the original native_semantic_radioactive_fingerprint_codex_spec.md. That original file describes the historical fixed-shift experiment. Carrier/key/extractor/verifier, teacher/student model families and student adaptation/selection remain unchanged.

## Review assessment

The proposed change is a defensible teacher rescue experiment, not a proven correction guaranteeing higher TPR. A fixed relative shift need not clear an absolute clean-null decision boundary. Aligning conditional/unconditional predictions with inference CFG removes one mismatch, but a predicted-x0 one-step proxy is still different from a full denoising trajectory. The clean score spread is measured, not assumed to be 0.35. No validation/test score is used to fit its target.

Three simultaneous teachers on one 40 GB GPU would risk OOM (the old single-teacher cycle already allocated roughly 17 GB). The implementation uses isolated processes with one candidate per visible GPU: three GPUs can run all three concurrently; one GPU runs them in sequence. This resource adaptation does not change candidate initialization, seeds, budgets or selection.

## Frozen target and loss

Use the original teacher clean-calibration scores and frozen empirical 1%-FPR threshold tau. Recompute and verify their provenance, including score-coordinate consistency, image hashes and conservative calibration. Let sigma_clean be the population standard deviation (ddof=0) of these scores; default calibration size remains 5000. Save the sample count, false positives, std convention, threshold/score/carrier hashes and candidate targets in teacher_fp/calibration_target.json.

Each candidate kappa in [1.0, 1.5, 2.0] has T=tau+kappa*sigma_clean. Do not hard-code the observed previous-run threshold 0.8586266.

For identical noisy latents/timesteps, compute clean and trainable conditional and empty-prompt unconditional predictions. Preservation is half the conditional MSE plus half the unconditional MSE. Combine both branches in FP32 with CFG=7.5 and reconstruct predicted-x0 using the scheduler's prediction_type (epsilon/v_prediction/sample).

Extract clean and fingerprinted semantic coordinates q0 and qfp. Compute r0=mean(signs*q0), detached a=clamp(T-r0,0,2.5), and L_sem=mean((qfp-q0-a*signs)^2), averaging over both selected samples and bits. Apply exactly the same formula independently to the clean/fingerprinted differentiable VAE-cycle coordinates, deriving its shift from the clean cycle score.

L_fp = L_sem + 0.5*L_sem_cycle
L_total = L_preserve + 0.8*L_fp

There is no fixed-alpha target, perpendicular penalty, separate old cycle weight or extra margin loss in the active objective. The new objective applies from step zero with coefficient 0.8, without the historical fixed-shift warmup/ramp. Cycle runs on every semantic update; the timestep mask remains t<=500. Preservation averages all effective-batch samples; semantic losses average all selected samples in that effective batch, including uneven microbatches.

If the target is clipped, zero loss only implies rfp=r0+clamp(T-r0,0,2.5), not necessarily rfp=T. Likewise a sample already above T is not pushed downward. The loss is therefore a bounded coherent-shift objective, not a guarantee that every sample exceeds tau. Log cap fractions to expose this limitation.

## Execution and unchanged components

Every new candidate starts from the original clean SD2.1 snapshot with the same seed, rank32 attention+Conv LoRA, LR1e-4, 3000 optimizer updates, effective batch4/microbatch1 and BF16. Preserve the frozen FP32 reference extractor and VAE, checkpointed differentiation, finite-value guards, FP64 norm reduction/clip1, readiness probe, full-state checkpoints every100 and strict resume provenance. There is no GPU/model download or training in local CPU tests.

--reuse-preparation OLD_RUN reuses verified clean images, prompts, PCA/key and clean null in a separately recorded run. It never imports old fingerprint weights or optimizer history. Reused source data/key settings are explicit in config and lineage. A different source generation batch is rejected instead of silently changing the new effective batch. Use --resume only for an interrupted run of this same frozen code/config/objective. The old fixed-shift implementation remains available only for interpreting explicitly historical configurations, not as an alternative default objective.

Native generation still uses merged ordinary weights, normal Gaussian generators and CFG7.5, without runtime keys, custom input latents or image postprocessing watermark injection. Student remains SD1.4 with the original pilot/full branches, batch8, FP16, 7500 steps and MSE-only checkpoint selection.

## Selection and diagnostics

Train/evaluate all three candidates against the same clean threshold and held-out teacher validation split. CLIP drop must be <=0.01 (a stricter configured cap is allowed). Paired LPIPS remains finite-checked and reported, but is not a hard gate for this revision. Choose highest validation TPR among CLIP-passing candidates; break ties by mean score then lower kappa. Require validation TPR>=0.90 before any student training. No passing quality candidate, insufficient TPR, nonfinite values or failed integrity checks stops the experiment. FID/KID/CLIP remain reported through the original utility workflow; no student changes are made to rescue weak inheritance.

Loss logs include mean_r0, mean_rfp, mean_required_shift, proxy_tpr, cycle_mean_r0, cycle_mean_rfp, cycle_mean_required_shift, cycle_proxy_tpr, both shift_cap_fraction values, target_score, semantic_fraction and VRAM. Proxy diagnostics refer only to masked samples; when none are selected they are null. Proxy TPR does not substitute for validation on actual saved PNGs.

Each candidate has its own checkpoint/merged model/validation files and log. Keep full calibration and model/config provenance in validation.json and selected.json, so the operating point can be audited without guessing where a threshold originated.
