# Redundant semantic sign fingerprint (RSSF)

The current default is `semantic_sign_v1`, defined by [the locked specification](semantic_sign_fingerprint_codex_spec.md). The historical `threshold_cfg_v1` implementation remains separate; old completed runs can still be analyzed with `--analyze OLD_RUN`.

Run from the repository root in your own server environment:

```bash
bash run_semantic_fp.sh
```

Use existing verified clean preparation to save generation time:

```bash
bash run_semantic_fp.sh --reuse-preparation run_semantic_fp_20261020_135117_009783
```

This reuses only clean images/prompts/assets with matching hashes, Gaussian seeds, snapshots and preprocessing. It rebuilds PCA/stability, the top48 PC pool in PC8..64, the32-bit secret, train-only midpoint medians, scores and calibration. It never imports old keys, thresholds or fingerprint weights. Key seed remains the source experiment key seed; all new candidates share the newly frozen carrier/key. No existing result is overwritten.

## Teacher and verifier

32 selected median-centered semantic coordinates define bits (`r>=0` maps to +1). Exact Binomial(32,.5) tails derive `K_detect=24` at alpha=.01; the training buffer+2 gives26 easiest clean-selected bits. The squared hinge uses only those clean-selected bits and stops pushing each bit after positive margin mu.

Three independent margins mu=.10/.20/.30 start from the same clean SD2.1 and seed. Teacher remains rank32 attention+Conv LoRA, LR1e-4, AdamW,3000 updates, effective batch4/micro1, BF16. Native inference stays ordinary merged SD2.1 with Gaussian generators and CFG7.5.

Each optimizer update has full-range conditional/unconditional preservation plus an independent late decoded-image fingerprint/quality path. Five unique final native inference timestep IDs come from the actual saved scheduler at50 steps; selected alphas must match the training scheduler. The real differentiable image channel is VAE decode/clamp[-1,1] -> posterior mean encode -> frozen t0 mid-block GAP -> centered PCA coordinates. Quality is frozen AlexNet LPIPS plus pixel MSE on those same paired images.

```text
L_total = L_preserve + 0.8*L_fp + (LPIPS + pixel_MSE)
```

No empirical threshold, kappa, relative shift, broad semantic timestep mask, perpendicular penalty or separate cycle term is active in RSSF. This is one teacher-training phase; the two paths contribute to one optimizer update.

Before student training, actual native validation PNGs must pass analytic TPR>=.90, CLIP drop<=.01 and paired LPIPS<=.15. Passing candidates are ranked by lowest LPIPS, then pixel MSE, then mu. A failed campaign saves all diagnostics and stops without modifying the key/gates or launching students.

The analytic null randomizes the precommitted secret signs conditional on a key-independent image/model; it does not assume independent PCA coordinates. It is not a guarantee of nominal image FPR for the one deployed key. Fixed-key clean FPR and1000 deterministic fake-sign null checks are reported without rerolling the key. Continuous scores with disjoint teacher/student clean calibration remain secondary metrics and never enter teacher loss.

## Students and checkpoints

SD1.4 students are unchanged: pilot unadapted baseline + attention/Conv LoRA r32 + full UNet;7500 updates, batch8, FP16, checkpoint2500/5000/7500, selection solely by held-out denoising MSE. Fingerprint diagnostics are reporting only. `--setting full` retains the existing full set of branches.

```bash
python main.py --plan-only
bash run_semantic_fp.sh --resume YOUR_NEW_RSSF_RUN
python main.py --analyze COMPLETED_RUN
python main.py --teacher-check YOUR_NEW_RSSF_RUN
```

Strict resume uses the same frozen config/code/environment and restores full weights, AdamW/scaler, sampler order/offset and CPU/CUDA/Python RNG. It continues from the latest complete checkpoint every100 updates, replaying any later unsaved updates. Twelve scratch readiness updates check gradients/VRAM and restore weights/RNG before training. `--teacher-check` is only a fresh-optimizer diagnostic. `--upgrade-execution` belongs to the historical threshold optimization and cannot migrate a threshold teacher into RSSF. Historical `--continue-from` retains its historical source objective; RSSF does not warm-start candidates.

## Results and images

Runs are beside main.py, named `run_semantic_fp_*`.

- `carrier/`: versioned PCA, clean stability table, immutable32-bit key, all candidate-PC medians, provenance/hashes.
- `teacher_fp/sign_protocol.json`: actual native scheduler and exact late IDs/alphas.
- `teacher_fp/candidates/mu_*/`: protocol, losses/diagnostics, full checkpoints, merged pipeline, merge test and native validation results.
- `teacher_fp/selected.json`: selected teacher and frozen policy; `teacher_fp/failure.json` explains an unsuccessful campaign.
- `images/candidate_mu_*_validation/`: candidate images; compare same file names against `teacher_clean_validation/`.
- `images/teacher_wm_test/`, `images/lora_r32_conv_wm/`, `images/full_wm/`, `images/unadapted/`: final test images.
- `result/README.md`: primary analytic image/model evidence, secondary continuous detection/AUC and utility.
- `result/analytic_verifier.json`: exact tail table, cutoff, null definition and tie rule.
- `result/random_key_null_validation.json`, `fixed_key_clean_fpr.json`: clean checks, not key-selection tools.
- `result/model_level_verification.json`, `query_budget.json`/CSV: K_M, exact p_M and deterministic query budgets10..1000.
- `result/teacher_candidates/`: paired LPIPS/pixel/CLIP and saved clean/fingerprint/difference visual samples.
- `scores/*.json`: centered coordinates, bits/matches, K, exact p_img, analytic detection, continuous score and image hashes.
- `logs/`: persistent stage logs and isolated `teacher_coordinate_margin_mu_*.log` files.

Teacher remains the pinned SD2.1 base mirror and student pinned SD1.4. Environment setup is your responsibility; the launcher installs nothing. No real models, downloads or GPU runs are performed during local verification.
