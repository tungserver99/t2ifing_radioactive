# RSSF per-image tail-control oracle

This is a privileged per-image control oracle and not a deployable
fingerprinting method. It asks whether the existing fixed RSSF carrier admits
low-distortion, per-image detectable solutions with separate controls for the
last four cached PNDM calls. It never trains a teacher or student and never
starts an imprinting job.

The locked experiment is specified in
[`rssf_tail_control_oracle_codex_spec.md`](rssf_tail_control_oracle_codex_spec.md).
Historical teacher code and source-run artifacts are not modified.

## Run and resume

Activate the existing `wmq` environment and execute from the project directory:

```bash
python -u Malicious_quantization/bitmark_lora/semantic_oracle.py \
  --source-run /shared/homes/u26466553/projects/t2ifing_radioactive/run_semantic_fp_20261024_213309_944621 \
  --output /shared/homes/u26466553/projects/t2ifing_radioactive/run_rssf_tail_control_oracle_20261026_v2
```

The output is a **separate sibling run**. Never pass the teacher run as output.
Do not use the historical teacher `--resume` or `--upgrade-execution` flags.
Repeat the exact oracle command to resume after a scheduler interruption:
committed batches are verified and skipped; only the incomplete batch repeats
its optimization. Adam state for incomplete batches is deliberately not saved.
Completed outputs are verified without reloading GPU models or rewriting them.

`--plan-only` prints the locked configuration and paths without creating output
or loading models. It does not validate an existing HPC source or run CUDA.
There are no CLI hyperparameter overrides and no sweep.

Use CPU thread limits before starting Python on an 8-core allocation:

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2 OMP_MAX_ACTIVE_LEVELS=1 TOKENIZERS_PARALLELISM=false
```

PBS content is supplied in the chat; no PBS file is included in the repository.
Redirect the PBS shell to a job-specific live log to avoid waiting for stdout
spooling at job completion.

## Locked computation

- CPU RNG seed 20261026 selects 256 unique training row IDs without replacement;
  sorted IDs are frozen before any detector or quality-based selection.
- Each batch has four images and independent FP32 controls of shape
  `[4 tail calls, 4 channels, 64, 64]` per image. Serial per-image graphs keep
  the effective batch at four without retaining four decoder/extractor graphs.
- Run 120 Adam updates at lr=.03. Controls are the only trainable parameters.
  Cached outputs plus controls enter the real restored PNDM scheduler; no
  denoising UNet forward or custom diffusion equation is used.
- The detector uses RGB8 STE and the **current** easiest 24 signed margins, with
  the .02 optimization-only margin. It does not reuse teacher clean-fixed bits.
- Float reference is the frozen FP32 VAE decode of the cached original final
  latent. Float image LPIPS and pixel MSE are quality losses. Relative control
  energy is the mean of the four call-wise delta mean-squares divided by cached
  output mean-square +1e-8, with weight .01.
- Detector weight starts at1. After updates20/40/60/80/100 it doubles, capped32,
  only for images that have never met the detector/safety-margin criterion.
- Evaluate the baseline at step0 and every resulting iterate1..120. Eligible
  candidates are ordered by float LPIPS, pixel MSE, relative energy, then earlier
  step. An image with no eligible candidate retains a diagnostic fallback ordered
  by largest K, then largest minimum top24 margin, then smallest distortion and
  earlier step; it remains `float_feasible=false`.
- Every10 steps (and step0), save small per-image frontier statistics, not images.

## Zero controls and the three image paths

Before optimizing a batch, cached outputs in their original dtype must reproduce
the original cached native final latent **exactly** on the native execution
device. Under the user-approved v2 policy, the FP32 scheduler graph's zero point
is separately measured against that latent. The existing latent atol/rtol2e-3
is a diagnostic reference, not an abort threshold across different dtypes.
Logs and per-image JSON record maximum/RMS drift, element count outside that
reference tolerance and source dtype; the final report aggregates the drift.
Nonfinite values still abort. No correction is added to the FP32 graph, and
the quality reference remains the original clean image.

Canonical zero decode uses the clean FP16 VAE and the project's ordinary
`VaeImageProcessor` PNG convention. Its PNG pixel sanity uses the existing
2/255+2e-3 limit, and its recomputed normal RSSF coordinates must agree with
stored clean coordinates at frozen atol/rtol2e-3. These tolerances are recorded
in `oracle_targets/protocol.json`. A native replay, PNG or coordinate mismatch aborts before that batch is
optimized; nothing relaxes a tolerance automatically.

This changes the execution protocol to `rssf_tail_control_oracle_v2_fp32_drift`.
Use a new output directory, for example `run_rssf_tail_control_oracle_20261026_v2`;
do not resume the v1 directory with v2 code. The historical teacher source remains
unchanged. Repeat the v2 command with its same output to resume interrupted v2 batches.

During optimization, quality uses the original float image and FP32 float
reference. Detector coordinates use the STE RGB8 view. The STE forward matches
the actual project PIL/PNG roundtrip and its in-range backward is identity.

Final evaluation instead decodes the selected FP32 latent through the canonical
FP16 clean VAE, writes a real PNG, reads it with the normal scorer, and applies
the unchanged RSSF verifier. **K_float is not substituted for K_png.** Paired
PNG LPIPS, pixel MSE and CLIP use the corresponding existing clean PNG and the
same conventions as teacher quality. Float/STE and canonical FP16/PNG differences
are measured rather than assumed away.

## Results and decision

Read these files in the oracle output:

- `result/README.md`: primary pass/fail and scientific interpretation.
- `result/oracle_feasibility.json`: aggregate metrics, Wilson CI, source hashes,
  protocol hash and secondary model evidence.
- `metrics/per_image.json`: actual PNG and float metrics, clean K, first
  feasibility, selected step/weight, per-call residual energy and bit flips.
- `metrics/frontier.json` and `metrics/clean_k_bins.json`: optimization frontier
  and difficulty-dependent outcomes.
- `result/sample_pairs/`: at least16 deterministic examples (or all examples in
  a smaller unit fixture), including failures and high-energy successes when
  those categories exist. No aesthetic selection.
- `oracle_complete.json`: written last after reports and source revalidation.

An image is feasible **iff K_png>=24 and paired PNG LPIPS<=.15**. A selected
float-feasible candidate can fail this check. Conversely, a PNG satisfying it
is counted even if it did not achieve the optimization-only .02 safety margin.

The primary `oracle_pass` requires at least90% (231 of256 images) feasible and
mean CLIP drop<=.01. Otherwise status is `oracle_fail`. Model-level evidence is
secondary and never compensates for poor per-image feasibility.

Success supports a **separate** future teacher-imprinting experiment, without
establishing teacher success. Failure is evidence to reconsider the carrier
under this locked control space and optimization budget; it is not a proof
that no mathematical solution exists. Neither result starts further training.

## Integrity and residual targets

Source config/assets/prompts, carrier/key, trace manifests/shards, clean training
images/manifests, clean scores/provenance and pinned clean model files are
validated and SHA256-pinned at initialization, every resume and completion.
Source and output overlap is rejected in either direction. Runtime helper
hashes and package versions are frozen in the oracle protocol.

Every batch commits safetensors, metrics, frontier and actual PNG hashes, then
writes its completion marker last. Committed files are not rewritten. An
interrupted initial setup or uncommitted batch is recoverable. Protocol/source
changes or modified committed files are rejected. The output directory is
locked during execution so concurrent jobs cannot optimize the same run.

`oracle_targets/` consolidates selected float-feasible controls into immutable
safetensor shards of32 targets. Residuals are stored in FP16 **after** optimization;
selected final latents remain FP32. Per-image metadata records the residual
storage error. Final PNGs are evaluated from the exact selected latent, not from
a replay of rounded stored residuals. Future imprinting must account for target
precision; this experiment does not claim that reloaded half-precision controls
reproduce the selected image bit-for-bit.

No teacher adapter, teacher optimizer state or model weights are saved. Runtime
errors propagate and may write `result/execution_error.json` only inside a
recognized oracle output. An incomplete run has no `oracle_complete.json`.
