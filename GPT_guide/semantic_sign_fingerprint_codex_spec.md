# Coding Specification: Redundant Semantic Sign Fingerprint (RSSF)

## Purpose

This document is the **locked implementation specification** for the next revision of the native semantic radioactive fingerprint in:

`https://github.com/tungserver99/t2ifing_radioactive.git`

The goal is **not** to replace the current method with a new architecture. The current experiments already show that the core semantic carrier is highly radioactive across teacher → student transfer. The new revision must preserve everything that already works and change only the parts that are currently responsible for poor image utility and a less elegant verification story.

The new method should:

1. preserve the existing frozen semantic extractor and PCA-based carrier idea;
2. preserve native teacher inference: ordinary prompt + Gaussian noise, no runtime key injection, no watermark hook, no post-processing;
3. preserve the SD2.1 teacher → SD1.4 student setup;
4. preserve LoRA/full student training exactly unless explicitly stated below;
5. replace the current strong continuous threshold-forcing objective with a **redundant semantic sign fingerprint**;
6. provide a natural **analytic null** for both per-image and per-model verification;
7. explicitly optimize paired image fidelity during teacher training;
8. keep per-image verification strong, while also providing a stronger model-level verification route;
9. remain a **single-stage teacher training method**.

Do **not** introduce a learned detector, a new semantic encoder, a two-phase watermark-training pipeline, HMARK modules, prompt-dependent secret signs, runtime key activation, a different student model, or any other new mechanism not specified here.

---

# 1. Existing results and what must be preserved

The current `threshold_cfg_v1` experiment establishes an important fact:

- fingerprinted teacher: ~100% per-image detection;
- SD1.4 LoRA r32 attention+Conv student: ~99.7%;
- SD1.4 full-UNet student: ~99.9%;
- teacher / LoRA / full all recover the existing secret bits perfectly at model level;
- clean teacher and unadapted student remain near null behavior.

Therefore, the current semantic feature space and transfer mechanism are **not** considered failures.

The problem is utility: even small `kappa` values visibly change image composition/style/texture. The current objective is too aggressive because it forces each image toward an absolute score boundary and applies semantic pressure over a broad diffusion timestep range.

The new implementation must therefore preserve the current representation and radioactivity pathway while replacing the embedding objective and formalizing verification.

---

# 2. Components that must remain unchanged

Unless explicitly changed below, keep all current code behavior.

## 2.1 Teacher and student

Teacher:

- Stable Diffusion 2.1 Base
- current pinned revision / local snapshot behavior
- fingerprint embedded into teacher weights
- teacher adapter remains rank-32 attention+Conv LoRA during fingerprint training
- merge adapter into the teacher after training
- final teacher inference must remain ordinary `StableDiffusionPipeline(...)` generation

Student:

- Stable Diffusion 1.4
- same generated teacher dataset
- same current student protocol
- pilot branches remain:
  - unadapted SD1.4
  - SD1.4 attention+Conv LoRA r32
  - SD1.4 full UNet
- same training budget, checkpoint selection by held-out denoising MSE, and test discipline

Do not change the student to SD2.1 in this revision.

## 2.2 Frozen semantic extractor

Keep the current extractor definition:

- frozen clean SD2.1 VAE;
- frozen clean SD2.1 UNet;
- frozen clean SD2.1 text encoder;
- deterministic VAE posterior mean;
- frozen UNet feature extraction at `t = 0`;
- capture current mid-block output;
- global average pool spatial dimensions.

Do not train this extractor.

Do not replace it with CLIP/DINO/a learned detector.

The verifier remains prompt-aware exactly as the existing semantic extractor is: the owner knows the prompts used for verification queries.

## 2.3 PCA framework

Keep the current PCA fit logic based on frozen clean SD2.1 semantic features.

Keep:

- centering by the clean feature mean;
- orthonormal PCA basis;
- normalization by clean per-PC standard deviation;
- current VAE-cycle stability analysis;
- current PCA sign canonicalization;
- current provenance and integrity checks.

The new method is a revision **inside the same PCA semantic space**.

---

# 3. New method name / versioning

Introduce a new versioned objective, for example:

`semantic_sign_v1`

or

`redundant_sign_v1`

Do not overwrite the historical `threshold_cfg_v1` implementation.

Old completed runs must remain analyzable.

Recommended structure:

- keep `semantic_threshold.py` untouched for old runs;
- add a new module such as `semantic_sign.py`;
- branch in training/reporting by `teacher_objective`;
- version carrier artifacts so old 8-bit carriers can still load.

The new default for fresh semantic-fingerprint runs may become this objective after tests pass, but historical artifacts must remain reproducible.

---

# 4. Carrier construction

## 4.1 Candidate PC range

Keep the existing candidate PC index range:

`PC 8 ... PC 64`, inclusive, zero-based.

Do not expand the range in this revision.

## 4.2 Stability ranking

Use the existing clean-data-only VAE-cycle stability computation and ranking.

Do not use fingerprinted data, teacher validation data, student data, or test data to rank/select carrier PCs.

The stability table must continue to be saved.

## 4.3 Number of candidate PCs and selected fingerprint bits

New defaults:

- `num_bits = 32`
- `stable_candidate_count = 48`

Procedure:

1. rank candidate PCs using the existing stability rule;
2. retain the top 48 stable PCs;
3. using the existing key-seed mechanism, sample 32 unique PC indices without replacement from these top 48;
4. independently generate one secret sign per selected PC:
   \[
   s_j \in \{-1,+1\}, \quad j=1,\ldots,32.
   \]

The deployed key therefore remains composed of:

- selected semantic PC indices;
- secret signs.

Do not reroll the key after seeing any clean FPR, validation TPR, teacher quality, or student result.

One key is generated once, before fingerprint-teacher training, and frozen for the entire experiment.

All teacher strength candidates must use the **same carrier and same key**.

## 4.4 Important statistical independence rule

Carrier fitting and PC stability ranking are clean-data operations and must not depend on the secret signs.

The secret signs must be sampled only after the candidate carrier space is fixed.

The key must never be selected by optimizing verification performance.

---

# 5. Median-centered semantic coordinates

The existing normalized PCA coordinate is:

\[
q_j(x,c)
=
\frac{
u_j^\top (f(x,c)-\mu)
}{
\sigma_j
}.
\]

For the new sign fingerprint, add a fixed per-PC center.

For every PC in the candidate carrier set, compute on the **carrier-fit clean training set only**:

\[
c_j
=
\operatorname{median}_{x \in D_{\mathrm{carrier}}}
q_j(x,c).
\]

Use the standard midpoint median for an even number of samples.

Store these values in the carrier artifact.

For a selected fingerprint PC define:

\[
r_j(x,c)
=
q_j(x,c)-c_j.
\]

`r_j` is the centered semantic coordinate used by the new verifier and teacher fingerprint loss.

Do not fit `c_j` using:

- calibration data;
- validation data;
- final test data;
- fingerprinted outputs;
- student outputs.

The purpose of median centering is to place the bit boundary near the natural center of the clean marginal distribution; it is **not** an empirically tuned detection threshold.

---

# 6. Binary semantic bits

For each selected coordinate define an image bit:

\[
b_j(x,c)
=
\begin{cases}
+1,& r_j(x,c)\ge0,\\
-1,& r_j(x,c)<0.
\end{cases}
\]

Use this explicit nonzero sign convention; do not use a function that returns zero for an exactly-zero input.

A fingerprint bit matches the secret if:

\[
b_j(x,c)=s_j.
\]

Equivalently:

\[
s_j r_j(x,c)\ge0
\]

under the explicit tie convention above.

In implementation, compute the image bit first and compare `bit == sign`. This keeps exact-zero handling deterministic.

---

# 7. Analytic per-image verifier

## 7.1 Match count

For a single image/prompt pair:

\[
K(x,c)
=
\sum_{j=1}^{m}
\mathbf 1[b_j(x,c)=s_j],
\]

with default:

\[
m=32.
\]

## 7.2 Null hypothesis

The formal null is:

> `H0`: the evaluated image/model is independent of the secret key that was randomly generated before fingerprint training.

Conditional on any fixed semantic vector `r(x,c)`, each independent random secret sign matches the corresponding deterministic image bit with probability `1/2`.

Therefore under the key-randomization null:

\[
K(x,c)\mid H_0
\sim
\operatorname{Binomial}(m,0.5).
\]

This statement does **not** require:

- Gaussian semantic coordinates;
- independent PCA coordinates;
- an SD1.4 clean counterpart;
- a learned null distribution;
- image independence.

The randomness comes from the precommitted secret signs.

## 7.3 Exact p-value

Implement exact one-sided binomial-tail probability:

\[
p_{\mathrm{img}}
=
P[
X\ge K(x,c)
],
\quad
X\sim\operatorname{Binomial}(m,0.5).
\]

Do not require SciPy for this calculation. It can be computed exactly with `math.comb`.

For a generic `m` and target significance `alpha_test`, derive:

\[
K_{\mathrm{detect}}
=
\min
\left\{
k:
P[\operatorname{Binomial}(m,0.5)\ge k]
\le\alpha_{\mathrm{test}}
\right\}.
\]

Defaults:

- `m = 32`
- `alpha_test = 0.01`

This must yield:

\[
K_{\mathrm{detect}}=24.
\]

Expected values:

- \(P(K\ge23)\approx0.01003\), slightly above 1%;
- \(P(K\ge24)\approx0.00350\), below 1%.

Do not hard-code 24 without computing it; compute it from `(m, alpha_test)` and assert that the default configuration returns 24.

## 7.4 Primary per-image verification result

For the new method, the primary analytic per-image detection indicator is:

\[
\mathbf1[
K(x,c)\ge K_{\mathrm{detect}}
].
\]

Report:

- `K`;
- `p_img`;
- analytic detected/not-detected;
- analytic TPR across fingerprinted images;
- empirical fixed-key FPR across clean images.

---

# 8. Continuous score remains available

Do not remove the existing continuous semantic evidence.

For the new centered 32-dimensional carrier define:

\[
R(x,c)
=
\frac1m
\sum_{j=1}^{m}
s_j r_j(x,c).
\]

Keep reporting:

- per-image `R`;
- model mean score;
- ROC/AUC where useful;
- empirical `TPR@1%FPR` using a clean calibration threshold as a **secondary comparison metric**.

Important:

- the continuous empirical threshold is no longer the theoretical definition of the fingerprint;
- it must not be used in teacher training;
- it must not be used to choose the key;
- it exists for backward comparison to the previous method and to prior work.

---

# 9. Model-level verifier

Use the same carrier, same secret, same centered coordinates.

Given `N` outputs of a suspect model:

\[
\bar r_j
=
\frac1N
\sum_{i=1}^{N}
r_j(x_i,c_i).
\]

Recover one model-level semantic bit per carrier dimension:

\[
\bar b_j
=
\begin{cases}
+1,& \bar r_j\ge0,\\
-1,& \bar r_j<0.
\end{cases}
\]

Count:

\[
K_M
=
\sum_{j=1}^{m}
\mathbf1[\bar b_j=s_j].
\]

Use the same exact binomial null:

\[
p_M
=
P[
X\ge K_M
],
\quad
X\sim\operatorname{Binomial}(m,0.5).
\]

With `m=32`, exact recovery of all 32 signs corresponds to:

\[
p_M=2^{-32}.
\]

The model-level verifier therefore requires no student-family-specific empirical threshold.

Still report:

- `K_M`;
- `p_M`;
- model-level bit accuracy;
- continuous model mean score:
  \[
  \bar R=\frac1m s^\top\bar r.
  \]

## 9.1 Query-budget analysis

Add model-level reporting as a function of number of queried outputs.

Suggested query budgets:

`N = [10, 20, 50, 100, 200, 500, 1000]`

For each feasible `N <= test_n`:

- sample multiple deterministic subsets using a fixed reporting RNG seed;
- compute `K_M` and `p_M`;
- report median/mean `K_M`, median `p_M`, and fraction of subsets achieving `p_M <= 0.01`.

This is reporting only and must not influence model/checkpoint selection.

---

# 10. Clean-null validation

The analytic test is defined by secret-key randomization. Clean data is used to validate the null in practice, not to define it.

## 10.1 Random-key validation

On clean semantic features, generate many independent fake sign keys over the **same selected PC set**.

Recommended:

`1000` fake sign vectors.

For each clean image and fake key compute `K`.

Check that the pooled/random-key histogram agrees with `Binomial(32, 0.5)` within sampling noise.

At minimum report:

- empirical histogram of K;
- empirical probability `K >= K_detect`;
- theoretical probability;
- mean K (expected 16);
- variance (expected 8).

Do not use fake-key results to change the deployed secret.

## 10.2 Fixed deployed key validation

For the one deployed key, report empirical analytic FPR on:

- clean SD2.1;
- unadapted clean SD1.4;
- any extra clean distribution/model later if desired.

If fixed-key FPR is not ideal, report it. Do not reroll the key.

This is essential to avoid key-selection / p-hacking concerns.

---

# 11. Why the current teacher objective must be replaced

Do not continue using:

\[
a_i
=
\tau+\kappa\sigma-r_i
\]

or any equivalent per-image forcing toward an empirical detection boundary.

The previous objective makes images with low clean score receive a much larger semantic edit than images already near the boundary.

It also rewards excessive signal strength even after a fingerprint is already easily detectable.

The new teacher objective must instead exploit redundancy:

- 32 possible semantic bits;
- analytic detection only needs 24;
- training uses a small count buffer above detection;
- each image is allowed to ignore its most expensive semantic directions.

---

# 12. Training bit-count margin

Expose a configuration parameter:

`train_bit_count_margin`

Default for the first experiment:

`train_bit_count_margin = 2`

Compute:

\[
K_{\mathrm{train}}
=
\min(
m,
K_{\mathrm{detect}}
+
\text{train\_bit\_count\_margin}
).
\]

With default values:

\[
m=32,\quad
K_{\mathrm{detect}}=24,\quad
\text{margin}=2,
\]

therefore:

\[
K_{\mathrm{train}}=26.
\]

Important:

- `24` is analytically derived from `(m=32, alpha_test=0.01)`;
- `+2` is a configurable training robustness margin;
- do not present `26` as a mathematically optimal constant;
- for the first implementation/run, use margin `2`;
- do not create a sweep over this margin in the first campaign unless explicitly requested later.

---

# 13. Which bits to train for each image

This is critical.

Do **not** select training bits from the current fingerprinted output, because the selected set would move with optimization and could create unstable shortcuts.

Instead, select the easiest bits using the clean counterpart only.

For each late-step training sample, compute clean centered coordinates:

\[
r^0_j
=
r_j(x_{\mathrm{clean}},c).
\]

Compute clean secret-aligned margins:

\[
a^0_j
=
s_j r^0_j.
\]

Select:

\[
I(x,c)
=
\operatorname{TopKIndices}
(
a^0,
K_{\mathrm{train}}
).
\]

Detach this selection completely from gradients.

Interpretation:

- bits already strongly aligned with the secret rank high;
- mismatched bits close to zero are cheaper to flip than strongly mismatched bits;
- the `m - K_train` most expensive directions are allowed to remain unused for that image.

Use exactly the same secret sign for a coordinate across the entire dataset. The only image-dependent choice is whether a bit is among the easiest `K_train` bits for that image.

---

# 14. Coordinate-margin fingerprint loss

Expose one continuous coordinate-margin parameter:

`coordinate_margin_mu`

For the first campaign, run three independent teacher candidates in parallel/from clean initialization:

\[
\mu\in\{0.10,0.20,0.30\}.
\]

Do not run them sequentially based on the previous candidate outcome.

For the fingerprinted image define:

\[
a^{fp}_j
=
s_j r_j(x_{\mathrm{fp}},c).
\]

Fingerprint loss for one image:

\[
L_{\mathrm{fp}}
=
\frac1{K_{\mathrm{train}}}
\sum_{j\in I(x,c)}
[
\max(0,\mu-a^{fp}_j)
]^2.
\]

Average across the effective batch.

Important properties that must be preserved:

1. a bit contributes no fingerprint gradient after it reaches the required positive margin;
2. there is no reward for pushing a bit arbitrarily far from zero;
3. only `K_train` easiest bits for a clean image are pressured;
4. no empirical threshold appears in this loss;
5. no current fingerprinted-output TopK is used;
6. no relative shift target such as `q_fp - q_clean = alpha*s` is used in this revision.

Log per-step/per-validation:

- number of selected bits already correct in clean image;
- number correct after fingerprinting;
- mean selected signed margin;
- fraction selected bits satisfying `a_fp >= mu`;
- per-bit selection frequency;
- per-bit mean signed coordinate;
- analytic `K` proxy on the decoded training view.

---

# 15. Late-timestep fingerprinting

The previous broad mask such as `t <= 500` must not be used for the new fingerprint/quality objective.

The new method should apply fingerprint and image-fidelity pressure only near the end of the actual generation schedule.

## 15.1 Derive late timesteps from the native inference scheduler

Use the same clean SD2.1 scheduler configuration used by native generation.

For the configured:

`inference_steps = 50`

call the scheduler's normal `set_timesteps(50)` and obtain the exact integer timestep sequence.

Deduplicate integer timesteps while preserving generation order if the scheduler repeats timesteps.

Define:

`late_inference_step_count = 5`

and select the final 5 unique timestep IDs in that native generation sequence.

Save these exact IDs in the run protocol.

Do not hard-code `t <= 100`.

Do not assume linear scheduler spacing.

## 15.2 Scheduler consistency check

Teacher training may currently instantiate a `DDPMScheduler` while native generation may use another scheduler class.

Codex must:

1. derive late timestep IDs from the actual native generation scheduler;
2. verify that the training scheduler and native scheduler use compatible beta/alpha schedules for these timestep IDs;
3. assert matching/near-matching `alphas_cumprod` values for the selected IDs;
4. fail clearly if the schedules are incompatible.

Do not silently use timestep IDs from one schedule with incompatible alpha values from another.

---

# 16. Per-update teacher training structure

Keep this as one teacher-training phase.

For each optimizer update, use the same underlying clean training image/prompt batch and run two objective paths.

## 16.1 Preservation path

Sample random diffusion timesteps across the full training timestep range as in the current method.

For clean reference and trainable fingerprint teacher compute:

- conditional prediction;
- unconditional prediction.

Define the current CFG-compatible preservation term:

\[
L_{\mathrm{preserve}}
=
\frac12
\operatorname{MSE}(
\epsilon^{c}_{fp},
\epsilon^{c}_{clean}
)
+
\frac12
\operatorname{MSE}(
\epsilon^{u}_{fp},
\epsilon^{u}_{clean}
).
\]

This full-range branch prevents broad diffusion-dynamics drift.

The clean reference branch is `no_grad`.

The fingerprinted teacher branch must preserve gradient flow.

## 16.2 Late fingerprint/quality path

Independently choose one timestep for each sample from the frozen set of 5 late native-inference timestep IDs.

Add noise to the clean training latent at that timestep.

Compute clean and fingerprinted:

- conditional prediction;
- unconditional prediction.

Construct exact CFG predictions with the same deployment guidance scale:

\[
\epsilon^{cfg}
=
\epsilon^u
+
7.5(
\epsilon^c-\epsilon^u
).
\]

Use the existing prediction-type-aware `predicted_x0()` conversion.

This late path produces:

\[
\hat z^0_{\mathrm{clean}},
\quad
\hat z^0_{\mathrm{fp}}.
\]

---

# 17. Decode through the real image channel

For the late path decode:

\[
x_{\mathrm{clean}}
=
VAE.decode(
\hat z^0_{\mathrm{clean}}/scale
),
\]

\[
x_{\mathrm{fp}}
=
VAE.decode(
\hat z^0_{\mathrm{fp}}/scale
).
\]

Use the same image-range clipping convention as normal pipeline conversion:

`[-1, 1]`.

Do not quantize to PNG during differentiable training.

The clean decoded image is detached/no-grad.

The fingerprinted decoded image must retain gradients.

---

# 18. Fingerprint features must be read from decoded images

This is important.

The new fingerprint loss must operate on the same **image → frozen semantic extractor** pathway used by verification.

For both decoded images:

1. frozen VAE encode using posterior mean;
2. frozen text-conditioned SD2.1 semantic extractor;
3. current mid-block GAP feature;
4. PCA projection;
5. standardize by stored `sigma`;
6. subtract stored median `c`;
7. obtain `r_clean`, `r_fp`.

Use `r_clean` only to choose the detached easiest-bit set `I(x,c)`.

Use `r_fp` for the differentiable fingerprint hinge loss.

Do not use the old latent semantic objective plus a separate VAE-cycle loss.

The image path itself is now the fingerprint path.

---

# 19. Direct image-quality loss

Add direct paired image fidelity during teacher training.

Use the same clean and fingerprinted decoded images from the late path.

Define:

\[
L_{\mathrm{quality}}
=
\lambda_{\mathrm{lpips}}
L_{\mathrm{LPIPS}}
(
x_{\mathrm{fp}},
x_{\mathrm{clean}}
)
+
\lambda_{\mathrm{pixel}}
\operatorname{MSE}
(
x_{\mathrm{fp}},
x_{\mathrm{clean}}
).
\]

Use frozen AlexNet LPIPS, consistent with the existing reporting code.

Inputs must use the correct LPIPS range `[-1, 1]`.

No gradient through the clean image.

Gradient must propagate through LPIPS and pixel MSE to the fingerprint teacher.

Initial engineering defaults:

- `lambda_lpips = 1.0`
- `lambda_pixel = 1.0`

These are configuration parameters and must be recorded in the run manifest. They are not claimed to be theoretically optimal.

Do not silently tune them using final test results.

---

# 20. Final teacher objective

The conceptual objective must remain only three terms:

\[
L_{\mathrm{total}}
=
\lambda_{\mathrm{preserve}}L_{\mathrm{preserve}}
+
\lambda_{\mathrm{fp}}L_{\mathrm{fp}}
+
\lambda_{\mathrm{quality}}L_{\mathrm{quality}}.
\]

Initial engineering defaults:

- `lambda_preserve = 1.0`
- `lambda_fp = 0.8`
- `lambda_quality = 1.0`

The `0.8` fingerprint weight intentionally preserves the approximate fingerprint-vs-preservation scale used in the current threshold objective; do not claim it is theoretically optimal.

Remove from the active new objective:

- threshold-calibrated required shift;
- `kappa`;
- fixed-alpha continuous displacement;
- `L_code`;
- `L_perp`;
- old separate `L_cycle`;
- any margin to an empirical clean threshold.

Do not add additional objectives unless required for numerical stability.

---

# 21. Teacher adapter / optimization settings

Keep the current teacher architecture and basic optimization settings unless memory forces a clearly documented microbatch adjustment.

Defaults to retain:

- clean SD2.1 initialization;
- rank-32 attention+Conv teacher LoRA;
- current adapter target rule;
- LR `1e-4`;
- AdamW;
- 3000 optimizer updates;
- effective batch 4;
- BF16 teacher precision;
- gradient clipping / finite checks;
- checkpointing and exact-resume infrastructure;
- merge adapter into teacher after training.

All three `mu` candidates must start from the same original clean SD2.1 weights, not from a previous fingerprint teacher.

---

# 22. Teacher candidate campaign

For the first experiment run exactly:

\[
\mu \in \{0.10,0.20,0.30\}
\]

with:

- `num_bits = 32`;
- `alpha_test = 0.01`;
- analytically derived `K_detect = 24`;
- `train_bit_count_margin = 2`;
- therefore `K_train = 26`;
- `late_inference_step_count = 5`.

All other method settings are shared.

Run candidates independently; parallelize across GPUs when available.

Do not adapt candidate settings based on intermediate results.

---

# 23. Teacher validation and selection

Teacher validation must use held-out validation prompts/seeds and actual **native merged-pipeline generation**, not training proxies.

For each candidate report:

### Fingerprint

- analytic per-image TPR:
  \[
  P[K(x)\ge K_{\mathrm{detect}}]
  \]
- distribution of `K`;
- distribution of `p_img`;
- mean continuous score `R`;
- 32-bit aggregate model-level recovery on validation outputs;
- model-level `K_M` and `p_M`.

### Paired quality

Using identical prompt + Gaussian seed between clean teacher and fingerprint teacher:

- paired LPIPS;
- pixel MSE;
- CLIP score / CLIP drop;
- visual comparison outputs.

### Selection policy

First require:

- analytic per-image teacher TPR `>= 0.90`;
- CLIP drop `<= 0.01`;
- paired LPIPS `<= 0.15`.

Among candidates satisfying all gates, choose:

1. lowest paired LPIPS;
2. then lowest pixel MSE;
3. then lower `mu` as final tie-break.

Do **not** choose the candidate with the highest TPR once it already clears the required detection gate.

If no candidate satisfies the gates, stop before student training and report failure.

Do not weaken the gates automatically.

---

# 24. Native generation requirement

After teacher LoRA merge, final generation must remain standard.

The generated teacher model must not require:

- a secret key at inference;
- custom latents;
- a watermark tensor;
- a runtime hook;
- a special scheduler callback;
- post-processing;
- the clean teacher;
- a second network.

Generation remains:

`prompt + ordinary Gaussian seed -> merged fingerprinted SD2.1 -> image`

The key is only required by the owner for verification.

---

# 25. Student protocol

Once a teacher passes validation, generate the teacher training images exactly through native generation.

Keep the current student protocol unchanged.

Primary pilot:

1. unadapted SD1.4;
2. attention+Conv LoRA r32 student;
3. full-UNet SD1.4 student.

Student training must never use:

- the fingerprint key;
- fingerprint loss;
- fingerprint detector;
- semantic carrier;
- teacher verification statistics.

Students learn only from ordinary generated training images.

Checkpoint selection remains based only on held-out denoising MSE.

---

# 26. Final reporting

For every branch:

- clean teacher;
- fingerprint teacher;
- unadapted SD1.4;
- LoRA student;
- full student;

report the following.

## 26.1 Primary analytic image-level metrics

- mean/median `K`;
- analytic per-image TPR with `K >= K_detect`;
- exact per-image p-value distribution;
- empirical fixed-key clean FPR.

## 26.2 Model-level metrics

- `K_M / 32`;
- model bit accuracy;
- exact `p_M`;
- query-budget curves.

## 26.3 Continuous secondary metrics

- mean continuous score;
- score distribution;
- empirical ROC/AUC;
- empirical TPR@1%FPR using disjoint clean calibration.

These continuous metrics are retained for comparison with old runs and prior work; they are not the primary theoretical verifier.

## 26.4 Utility

Teacher:

- paired LPIPS;
- paired pixel MSE;
- CLIP;
- FID;
- KID;
- saved paired visual samples.

Students:

- existing FID;
- KID;
- CLIP;
- other already-supported utility metrics.

---

# 27. Data-split discipline

Keep data roles strictly separated.

## Carrier/train split

Allowed for:

- PCA fitting;
- VAE-cycle stability ranking;
- median centers;
- teacher optimization.

## Clean calibration split

Allowed for:

- empirical continuous-score threshold;
- empirical validation of fixed-key analytic FPR.

Not allowed for:

- PCA fit;
- median fit;
- key selection;
- teacher optimization;
- candidate model selection.

## Teacher validation split

Allowed for:

- choosing among `mu` candidates using the frozen policy.

Not allowed for:

- altering the key;
- recomputing PCA;
- recomputing medians;
- changing analytic threshold.

## Final test split

Reporting only.

Never use final test for:

- teacher candidate selection;
- student checkpoint selection;
- hyperparameter tuning;
- key rerolling.

---

# 28. Reuse of old preparation

Do not blindly reuse the old 8-bit carrier artifact because the new method requires:

- 32 selected PCs;
- top-48 candidate pool;
- median centers;
- new key artifact;
- new verification metadata.

It is acceptable to reuse immutable clean generated images / prompts / clean feature caches **if and only if** their hashes and preprocessing match the new carrier-fit requirements.

If clean semantic features for all PC 8–64 are already available and provenance matches, reuse them.

Do not reuse:

- old 8-bit selected key as the new 32-bit key;
- old threshold-forced teacher weights;
- old `kappa` candidate weights.

Every new teacher candidate starts from clean SD2.1.

---

# 29. Backward compatibility

Historical runs must remain readable.

Implement explicit artifact versions.

Recommended new carrier metadata fields:

- `carrier_version: "semantic_sign_v1"`
- `num_bits: 32`
- `stable_candidate_count: 48`
- `selected_pc_indices`
- `signs`
- `medians`
- `sigmas`
- `basis_sha256`
- `carrier_fit_prompt_sha256`
- `key_seed`
- `bit_tie_rule`
- `analytic_null: "key_randomization_binomial"`
- `alpha_test`
- `k_detect`

Old carriers without these fields must continue using the historical loader/path.

Do not reinterpret an old 8-bit run as a new sign-fingerprint run.

---

# 30. New output artifacts

At minimum save:

## Carrier

- `carrier/pca.pt`
- `carrier/stability.json`
- `carrier/key.json`
- `carrier/medians.json` or medians inside the versioned carrier object
- `carrier/complete.json`

## Analytic verifier

- `result/analytic_verifier.json`
  - `num_bits`
  - `alpha_test`
  - `k_detect`
  - exact binomial-tail table for `k=0..m`
  - tie rule
  - null definition
- `result/random_key_null_validation.json`
- `result/fixed_key_clean_fpr.json`

## Teacher candidates

For each `mu`:

- protocol/config
- losses
- diagnostics
- checkpoints
- merged pipeline
- native validation images
- analytic validation metrics
- quality metrics

## Final

- per-image score JSON including:
  - centered coordinates;
  - semantic bits;
  - bit matches;
  - K;
  - p_img;
  - continuous score
- model-level verification JSON
- query-budget JSON/CSV
- updated README summary
- utility reports

---

# 31. Diagnostics required during teacher training

Log at useful intervals:

### Preservation

- conditional preservation MSE;
- unconditional preservation MSE;
- total preserve loss.

### Fingerprint

- selected-bit count;
- clean selected bits already correct;
- FP selected bits correct;
- fraction selected bits with signed margin >= `mu`;
- mean selected signed margin;
- fingerprint loss;
- approximate analytic K on the differentiable decoded view;
- per-bit selection frequencies.

### Quality

- LPIPS;
- pixel MSE;
- quality loss.

### Numerical

- total loss;
- gradient norm;
- finite checks;
- GPU memory;
- sampled preservation timesteps;
- sampled late timesteps.

Training diagnostics must not be used to pick student checkpoints.

---

# 32. Required unit / protocol tests

Add tests for the following.

## 32.1 Binomial verifier

- `m=32`, `alpha=0.01` yields `K_detect=24`;
- exact tail for `K=24` is below 0.01;
- exact tail for `K=23` is above 0.01;
- exact `K=32` p-value equals `2^-32`.

## 32.2 Key-randomization null

For any fixed nonzero coordinate vector, random independent sign keys produce approximately Binomial match counts.

Do not require coordinate independence in the test.

## 32.3 Tie rule

`r == 0` maps deterministically to image bit `+1`.

For random secret signs, tie handling still gives 50/50 match probability over keys.

## 32.4 Median centering

- median is computed only from carrier-fit data;
- stored/reloaded medians reproduce identical centered coordinates.

## 32.5 Key immutability

- same seed -> same selected PCs and signs;
- key cannot be regenerated after validation;
- all `mu` candidates share the same key.

## 32.6 Easy-bit selection

- TopK indices are computed from clean signed margins only;
- selection tensor is detached;
- changing `r_fp` does not change the selected set within that forward pass.

## 32.7 Hinge behavior

- selected bit above `mu` has zero fingerprint loss contribution;
- selected bit below `mu` has positive gradient toward the correct sign;
- non-selected bits receive no direct fingerprint gradient.

## 32.8 Late timestep construction

- derive from native scheduler `set_timesteps(inference_steps)`;
- deduplicate correctly;
- take last configured number of unique timesteps;
- assert scheduler alpha compatibility.

## 32.9 Gradient path

- clean reference branch has no parameter gradients;
- fingerprint decoded image -> LPIPS / pixel / semantic extractor -> fingerprint teacher does have gradient;
- frozen extractor/VAE/LPIPS parameters remain frozen.

## 32.10 Native inference

Merged teacher generation must not accept or require any new watermark argument.

## 32.11 Backward compatibility

Existing `threshold_cfg_v1` unit tests and old-run analysis must still pass.

---

# 33. Explicit non-goals for this revision

Do not implement any of the following unless separately requested later:

- learned watermark detector;
- joint learned carrier rotation;
- HMARK encoder/decoder;
- teacher-training phase 1 / phase 2;
- prompt-dependent secret sign vectors;
- dynamic per-image secret keys;
- BCH or ECC;
- robustness to JPEG/blur/resize;
- adversarial watermark removal;
- new student families;
- new teacher families;
- a sweep over number of bits;
- a sweep over `train_bit_count_margin`;
- a sweep over number of late timesteps;
- student fingerprint-aware training;
- final-test-driven tuning.

The purpose of this revision is to test one focused hypothesis:

> **A redundant set of weak, late, quality-constrained semantic sign biases can preserve the strong radioactivity already observed in the current PCA semantic carrier while providing an analytic verifier and much better image utility.**

---

# 34. Scientific interpretation

The intended mechanism is:

1. clean SD2.1 semantic PCA coordinates provide stable semantic axes;
2. a secret key chooses 32 stable directions and a target sign for each;
3. under a key-independent clean image/model, matching a random target sign is a 1/2 event for each secret sign, yielding the exact key-randomization Binomial null;
4. a clean image already matches roughly half the secret signs by chance;
5. the teacher therefore does not need to shift all 32 coordinates strongly;
6. it only needs to make enough of the easiest coordinates cross or maintain the correct side with a small positive margin;
7. redundancy allows per-image verification from many weak bits instead of a few large semantic shifts;
8. late-only fingerprint pressure reduces cumulative trajectory distortion;
9. direct LPIPS + pixel MSE explicitly penalizes visible deviation;
10. the resulting coherent sign biases are repeated across the teacher-generated dataset;
11. the downstream SD1.4 student is expected to learn those systematic semantic biases through ordinary training;
12. aggregation across many student outputs further strengthens model-level lineage evidence.

The new method therefore aims to improve:

\[
\frac{
\text{detectable and transferable fingerprint evidence}
}{
\text{perceptual distortion}
}.
\]

It does **not** trade away per-image verification. The target remains strong per-image detection plus even stronger model-level verification.

---

# 35. Locked first-run configuration

Use the following as the first full implementation campaign.

```text
teacher_objective = semantic_sign_v1

teacher = SD2.1 Base
student = SD1.4

carrier_pc_min = 8
carrier_pc_max = 64
stable_candidate_count = 48
num_bits = 32
key_seed = existing experiment key seed unless a fresh run protocol specifies otherwise

analytic_alpha = 0.01
k_detect = derived exactly from Binomial(32, 0.5)  # expected 24

train_bit_count_margin = 2
k_train = k_detect + 2                            # expected 26

coordinate_margin_mu_candidates = [0.10, 0.20, 0.30]

inference_steps = 50
late_inference_step_count = 5
guidance_scale = 7.5

teacher_rank = 32
teacher_adapter_scope = attention_conv
teacher_lr = 1e-4
teacher_train_steps = 3000
teacher_batch = 4
teacher_micro_batch = 1
teacher_precision = bf16

lambda_preserve = 1.0
lambda_fp = 0.8
lambda_quality = 1.0
lambda_lpips = 1.0
lambda_pixel = 1.0

teacher_min_analytic_tpr = 0.90
teacher_clip_drop_max = 0.01
teacher_lpips_max = 0.15

student pilot:
    unadapted SD1.4
    attention+Conv LoRA r32
    full UNet

student training/checkpoint protocol = unchanged
```

The only first-run strength sweep is `coordinate_margin_mu`.

Do not silently introduce extra hyperparameter sweeps.

---

# 36. Stop / success logic

Before launching students:

1. verify carrier and analytic-null artifacts;
2. verify native-scheduler late timesteps;
3. train all three `mu` candidates from clean SD2.1;
4. merge each candidate;
5. generate held-out native validation images;
6. evaluate analytic per-image detection and paired quality;
7. apply the frozen teacher selection policy.

If at least one candidate passes:

- analytic per-image TPR >= 90%;
- CLIP drop <= 0.01;
- paired LPIPS <= 0.15;

select the lowest-LPIPS candidate and launch student training.

If none pass:

- stop;
- save all diagnostics;
- do not automatically weaken the verifier;
- do not automatically change bit count;
- do not automatically change key;
- do not automatically increase training strength;
- do not launch students.

This failure, if it occurs, should be analyzed explicitly before the next revision.

---

# 37. Final implementation principle

The coding revision should be understood as:

```text
CURRENT SUCCESSFUL CORE
    frozen semantic extractor
    + PCA semantic carrier
    + secret key
    + native teacher weights
    + ordinary generation
    + SD2.1 -> SD1.4 transfer
    + LoRA/full radioactivity
        |
        | keep
        v
NEW EMBEDDING / VERIFY LAYER
    32 redundant semantic sign bits
    + median-centered bit boundaries
    + exact key-randomization Binomial null
    + K_detect derived analytically
    + configurable training count margin (default +2)
    + easiest-bit hinge objective
    + late-only fingerprint pressure
    + direct paired LPIPS/MSE preservation
```

Do not reinterpret this as a request to redesign the whole fingerprint system.

The experiment is specifically intended to preserve the radioactivity already demonstrated by the current repository while reducing visible distortion and making both per-image and model-level verification statistically cleaner.
