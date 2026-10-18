# Second audit of the native semantic fingerprint implementation

Date: 2026-10-19. Scope: compare the supplied specification with the actual semantic method, its shared downstream harness and pinned-library API source. This audit used CPU logic tests only; it did not install packages or run diffusion models.

## Conclusion

The core implementation follows the specified MVP: teacher-side relative coherent semantic displacement, fixed clean reference features and keyed PCA coordinates, native merged teacher generation, passive student adaptation and a fixed semantic verifier. No mathematical change to the fingerprint objective was needed in this audit. Three robustness issues were reproduced and corrected below. Real SD2.1 merge equivalence and scientific effectiveness remain unverified locally.

## Findings corrected

| Finding | Evidence before fix | Correction |
|---|---|---|
| Calibration trusted scalar scores without checking their consistency with the saved coordinates and frozen key. | A fixture with score100 but all-zero coordinates was allowed to freeze a teacher threshold. | score_rows now verifies coordinate shape, finiteness and score = mean(signs * coordinates) before calibration, validation diagnostics or reporting. Test now rejects the fixture before writing a threshold. |
| Candidate paths rounded alpha to six significant digits. | The production `:g` spelling maps both0.5000001 and0.5000002 to shift_0.5, making a valid sweep hit a protocol/integrity conflict. | candidate_tag uses a round-trip float representation; distinct finite positive strengths have distinct candidate folders and score labels. Standard0.25/0.5/0.75 labels retain their spelling. |
| choose_candidate did not itself reject nonfinite cached metrics. | Its direct API returned a candidate whose mean score was NaN/Inf. JSON serialization elsewhere already rejected nonfinite numbers, but the selection function lacked a local guard. | Explicit finite/range validation for detection, mean score, optional bit accuracy and measured quality values before selection. |

Tests for all three cases failed before production changes and passed afterward. These are validation/artifact fixes; they do not change the intended displacement, student loss or evaluation budget.

## Specification-to-code audit

| Requirement | Evidence and status |
|---|---|
| No HMARK code/checkpoints/targets; no Serum grid or detector | Semantic assets request only clean teacher/student models and shared COCO splits. Semantic scoring uses reference features and the PCA key. Source inspection found no invocation of the forbidden components in this method path. |
| Fixed clean SD2.1 extractor | SemanticExtractor freezes and evaluates VAE, text encoder and reference UNet. Feature dimension is inferred from captured mid-block output. |
| Deterministic VAE encoding | The image and cycle paths use latent_dist.mean * scaling_factor. No posterior sampling is used by the semantic extractor. Student denoising training/validation retains its original ordinary posterior-sampling policy, as required. |
| Probe timestep0 and GAP | encode_latent passes zero timesteps and averages captured BCHW mid-block activations over both spatial axes. Hooks are scoped to reference feature extraction, not teacher production inference. |
| Clean-only PCA and stable carrier | carrier_stage reads only teacher_clean_train; fit_pca centers FP32 features and saves mean, full PCA basis and sample variances. Cycle stability ranks correlation, sign retention and PC index. Default indices8..64 are zero-based inclusive; top16 candidates precede seeded selection of8 directions/signs. |
| Frozen key and provenance | PCA/key/stability/provenance digests are saved and checked before reuse. Config/code/assets/prompts are frozen. No key regeneration is performed for a completed carrier. |
| Relative coherent shift | semantic_losses computes ((f_fp - f_clean) @ U) / (sigma + epsilon) and targets alpha * signs. Carrier mean cancels; there is no absolute-coordinate-collapse objective. |
| Loss reductions | Nonzero hand calculation verifies code2.5, perpendicular penalty12, and gradients[0.5,0.75,4] for a specified three-dimensional example. Translation of both clean/fingerprinted features leaves these losses unchanged. |
| Prediction-type-aware x0 | Helper supports epsilon, v_prediction and sample; unsupported types fail. Tested against the actual prediction branch extracted from the bundled DDPM source, plus independent algebra tests; the corresponding pinned v0.35.1 source was also inspected. No training-time predicted-x0 clipping is added. |
| Teacher-only gradients | Clean teacher forward is no_grad; fingerprint forward has gradients. A CPU toy computation through predicted-x0 -> frozen reference features and decode/mean-encode cycle yields nonzero gradients only for a trainable teacher prediction and decreases the objective after an optimizer step. This is not a real SD/PEFT training run. |
| Preservation, perpendicular drift, cycle | Preservation spans all sampled timesteps. Semantic/cycle losses are restricted to t<=semantic_t_max; cycle_every and warmup/ramp match the specified defaults. Raw components, valid-sample fraction and gradient norm are recorded. |
| Teacher LoRA scope | Target enumeration follows the existing attention+Conv2d rule: attention projections plus Conv2d with groups1. No student rank/scope expansion is used as a new objective. |
| Native merge | Code loads the adapter into clean SD2.1, fuses it, removes PEFT structures, saves/reloads a normal pipeline, and compares fixed-input FP32 UNet outputs with atol/rtol1e-4. The API sequence was checked against pinned Diffusers/PEFT source. Actual model-level equivalence is pending server execution. |
| Ordinary teacher inference | native_generate passes standard prompts/generators and sampling settings. It does not supply input latents or load a key. Standard output latents are checked, decoded by the original VAE and converted by the ordinary Diffusers image processor; no added semantic/watermark transform or inference hook exists. |
| Teacher-only strength selection | Candidates start from the same clean teacher/seed, use held-out teacher validation and paired seeds, pass CLIP/LPIPS quality checks, then rank by validation detection/mean score. No final test is read for this decision. |
| Separate nulls | Teacher null uses clean SD2.1 calibration; student null uses unadapted SD1.4 calibration. The score/key/extractor are identical; thresholds differ. Both are frozen before adapted-student test scoring. Conservative tie handling is inherited from the shared calibrator. |
| Student isolation | Actual trainer commands receive the student model, ordinary image/caption dataset and standard denoising settings. They contain no key, verifier, teacher weights or fingerprint loss. |
| Frozen comparison budgets | Full setting retains six trained branches,7500 steps, snapshots2500/5000/7500, batch8, LoRA/DoRA LR1e-5, full LR5e-6, resolution512 and the specified data counts. Default pilot uses attention+Conv r32 and full UNet. |
| Student selection | Shared validation chooses minimum held-out denoising MSE with earlier-step ties. Its selection is frozen before reading detector/utility diagnostics. |
| Detection/utility reporting | Reports include detection, Wilson intervals, mean score, aggregate bit recovery, student CLIP/FID/KID, and paired teacher CLIP/LPIPS tables. No effectiveness or quality-preservation claim follows merely from generating the table. |
| Resume | Frozen config/code/key and image/model digests are checked. A failed teacher gate cannot be bypassed by resuming; interruption after selected.json can recover the quality manifest. Partial teacher optimization restarts deterministically rather than resuming optimizer state. |
| Transfer-aware meta-objective | Not implemented, matching its explicit exclusion from the first MVP. |

## Verification results

Using D:/anaconda3/envs/topmost/python.exe:

```text
python -m unittest discover -s Malicious_quantization/bitmark_lora -p 'test_semantic*.py'
29 tests: 28 passed, 1 skipped.
```

The skipped test compares directly with an installed Diffusers DDPMScheduler; Diffusers is absent locally. The new source-branch comparison runs without importing that package. Syntax parsing, vendor hashes, pilot/full plan-only and the7500-step/batch8 defaults passed.

The full harness discovery found58 tests:53 passed,1 skipped,2 failures and2 errors. The same four pre-existing cases reported before this audit remain: test_bootstrap (POSIX fcntl), test_recovery.test_recover_and_reject_incomplete_or_changed_weights (fcntl), test_serum.test_reject_bad_artifacts_and_zip_traversal (single-backslash archive path), and test_validation.test_drive_results_local_environment (Windows versus POSIX path expectation). See semantic_audit_full_cpu_suite.log. The full suite is not green.

## Limits and source comparability

- No real diffusion model, real teacher/student training, real model merge, GPU feature determinism or CLIP/LPIPS/FID/KID measurement was executed. Mandatory runtime assertions are implemented; their presence does not mean they already passed on SD2.1.
- Default clean teacher is the pinned SD2.1 mirror used by the checkout's asset helper. Its byte-for-byte equality with the Serum author snapshot has not been established. For a comparison requiring exactly the old Serum teacher weights, use --teacher-snapshot with that clean snapshot; the reference and trainable teacher will both use it.
- Production uses the imported pinned library versions. The bundled third_party Diffusers tree identifies itself as0.41.0.dev0, so its source alone was not treated as proof of0.35.1 API compatibility. The pinned official sources below were inspected separately.
- Existing frozen runs reject changed code_manifest hashes by design. Do not edit recorded hashes to force an old experiment to resume under changed code.

## Primary library sources checked

- [Diffusers v0.35.1 StableDiffusion LoRA loader](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/loaders/lora_pipeline.py): fusion API and its StableDiffusion component defaults.
- [Diffusers v0.35.1 base LoRA loader](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/loaders/lora_base.py): unload behavior.
- [Diffusers v0.35.1 DDPM scheduler](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/schedulers/scheduling_ddpm.py): all three prediction conversions before optional clipping.
- [PEFT v0.17.1 LoRA layers](https://github.com/huggingface/peft/blob/v0.17.1/src/peft/tuners/lora/layer.py): Conv2d adapter/merge implementation and grouped-convolution restrictions.
