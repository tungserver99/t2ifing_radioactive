# Codex Implementation Specification — Native Per-Image RSSF Margin Refinement + Ordinary SD1.4 Distillation

## Status

**FINAL / LOCKED implementation specification for the next experiment.**

Target repository:

`https://github.com/tungserver99/t2ifing_radioactive.git`

Repository area:

`Malicious_quantization/bitmark_lora/`

This specification was prepared after reviewing the current implementations of:

- `semantic_sign.py`
- `semantic_carrier.py`
- `semantic_sign_trajectory_train.py`
- `semantic_trajectory.py`
- `semantic_teacher.py`
- `semantic_teacher_train.py`
- `semantic_checkpoint.py`
- `student.py`
- `validation.py`
- `utility.py`

The implementation must record the actual repository HEAD SHA in the new run manifest/code manifest at runtime.

**This document supersedes the previous oracle-imprinting / oracle-bank proposals.** Do not patch an older oracle-guided teacher specification and leave obsolete behavior active. Implement the method below as a distinct versioned teacher objective.

---

# 1. One-sentence method

Warm-start the existing low-distortion RSSF trajectory teacher at **checkpoint 100**, then refine it using **300 fully on-policy native PNDM50 optimization steps**. Each step directly optimizes a **dynamic per-image Top-26 RSSF margin** on the current final RGB8 image while a bounded fidelity gradient preserves clean SD2.1 behavior and paired image fidelity. After selecting the teacher, generate ordinary RGB images and reuse the existing **ordinary SD1.4 LoRA distillation pipeline unchanged**.

Conceptually:

```text
existing RSSF trajectory checkpoint-100
        |
        v
current-weight native PNDM50 rollout
        |
        +--> dynamic per-image Top-26 RSSF margin
        |
        +--> clean-denoiser preservation
        |
        +--> paired pixel fidelity
        v
selected native RSSF SD2.1 teacher
        |
        v
5k ordinary RGB-caption pairs
        |
        v
existing SD1.4 LoRA training, ordinary denoising MSE only
        |
        v
RSSF verification on a student that never receives the key
```

---

# 2. Scientific motivation and evidence

## 2.1 What is already solved

The RSSF carrier/verifier itself is considered successful and must remain frozen.

The existing system uses:

- a frozen clean SD2.1 semantic extractor;
- deterministic VAE posterior mean;
- clean SD2.1 semantic UNet at `t=0`;
- mid-block global-average-pooled features;
- PCA semantic coordinates;
- 32 selected secret coordinates/signs;
- median-centered coordinates;
- per-image sign matches;
- exact key-randomization Binomial evidence.

For 32 bits and `alpha = 0.01`:

```text
K_detect = 24
```

because `semantic_sign.py::detection_count(32, .01) == 24`.

The clean null was previously validated empirically and agrees with the analytic `Binomial(32, 0.5)` key-randomization null. Do not redesign this verifier.

## 2.2 What the trajectory teacher established

The previous trajectory-aligned teacher run approximately produced:

| step | per-image TPR | K_M | CLIP drop | paired LPIPS |
|---:|---:|---:|---:|---:|
| 100 | 3.125% | 27/32 | 0.006292 | 0.141488 |
| 250 | 3.91% | 29/32 | 0.005419 | 0.176816 |
| 500 | 14.84% | 32/32 | 0.003622 | 0.227714 |
| 1000 | 48.44% | 31/32 | 0.001479 | 0.254307 |
| 1500 | 49.22% | 32/32 | ~0 | 0.255910 |

Interpretation:

1. Shared teacher weights can learn a coherent owner-key bias very early (`K_M=27/32` already at step 100).
2. Per-image detection remains the difficult part.
3. Training longer raises per-image TPR but increases paired image drift.
4. Therefore checkpoint 100 is a useful low-distortion fingerprinted initialization, not a final teacher.

## 2.3 What the per-image oracle established

The completed oracle feasibility experiment optimized sample-specific control residuals on the final four PNDM calls and obtained approximately:

```text
actual-PNG feasible rate = 248 / 256 = 96.88%
mean paired LPIPS         ~= 0.00079
CLIP drop                 ~= 0.000115
model evidence            = 29 / 32
```

The oracle is **not part of the new training pipeline**.

Its role is diagnostic only. It established that:

1. RSSF per-image detectability has extremely low-distortion solutions.
2. Dynamic per-image choice of currently favorable bits is useful.
3. The bottleneck is shared-weight learnability/generalization, not the existence of low-distortion RSSF solutions.

Do not generate a 512- or 5000-image oracle bank. Do not run an online oracle inner loop. Do not imitate oracle residuals in this experiment.

## 2.4 Lesson from the native refinement work

The Projected/Structure-Margin line showed that native current-weight rollouts and per-image deficit pressure can substantially improve downstream student detection. It also showed that heavy auxiliary machinery such as projection is not clearly responsible for the gain.

The new RSSF method therefore keeps only the simplest useful ideas:

- current-weight native rollout;
- per-image margin pressure;
- explicit fidelity control;
- ordinary downstream student training.

---

# 3. New versioned teacher objective

Create a new teacher objective name. Recommended exact identifier:

```text
semantic_sign_v3_native_margin_refine
```

Historical objectives must remain unchanged and readable:

```text
threshold_cfg_v1
semantic_sign_v1
semantic_sign_v2_trajectory
```

The RSSF **carrier version remains `semantic_sign_v1`**. Teacher objective version and carrier version are separate concepts.

Update `is_sign(config)` or equivalent classification so the new v3 objective is recognized as an RSSF/sign experiment without changing old artifact interpretation.

Do not rewrite or reinterpret historical run folders.

---

# 4. Components that are LOCKED and must not change

Keep all of the following unchanged unless required only for compatibility wiring:

- clean SD2.1 semantic extractor definition;
- clean reference VAE/text encoder/semantic UNet;
- PCA basis;
- selected PC indices;
- medians;
- secret signs;
- secret key seed;
- 32-bit carrier;
- bit tie rule `r >= 0 -> +1`;
- per-image `K` definition;
- `K_detect = 24` at alpha 0.01;
- analytic Binomial key-randomization null;
- `model_evidence()` definition (`K_M`, `p_M`);
- continuous score as a secondary metric only;
- SD2.1 teacher backbone;
- SD1.4 student backbone;
- ordinary prompt + Gaussian inference;
- no runtime key injection;
- no post-processing watermark;
- no learned detector;
- no new semantic encoder;
- no key reroll after seeing results.

Do not refit the carrier/key for this refinement run.

---

# 5. Required warm-start lineage

## 5.1 Source

The new teacher must warm-start from the verified **trajectory teacher checkpoint 100** from the completed `semantic_sign_v2_trajectory` run.

Expected source location pattern:

```text
<SOURCE_RUN>/teacher_fp/trajectory/checkpoint-100/
    adapter_model.safetensors
    adapter_config.json
    checkpoint.json
    training_state.pt
```

The exact source run path must be supplied explicitly, for example:

```text
--source-run /path/to/completed/trajectory/run
--warm-start-step 100
```

Do not guess a run folder by modification time.

## 5.2 Verify before loading

Before training, verify all of the following:

- source run `teacher_objective == semantic_sign_v2_trajectory`;
- source checkpoint step exactly `100`;
- source checkpoint manifest/hashes pass;
- source LoRA rank is 32;
- source target scope is attention + non-grouped Conv2d, matching `adapter_targets()`;
- source carrier hash equals the new run carrier hash;
- source key hash equals the new run key hash;
- source prompt file hash equals the new run prompt file hash;
- source clean model identity equals the new run clean teacher identity;
- source scheduler/inference protocol is PNDM50 / CFG 7.5 and compatible with the new run.

If any lineage check fails, **abort**. Do not silently copy weights from a similar checkpoint.

## 5.3 What is reused and what is fresh

Reuse:

- the checkpoint-100 **LoRA weights only**;
- immutable clean images/prompts/carrier/key/null artifacts after hash verification.

Create fresh:

- AdamW optimizer state;
- refinement data-order RNG;
- refinement loss log;
- refinement checkpoints;
- checkpoint-selection results.

Do not resume the old v2 optimizer moments or old v2 step schedule.

## 5.4 Loading procedure

Preferred implementation:

1. load pinned clean SD2.1;
2. add the same rank-32 attention+Conv PEFT adapter configuration;
3. load checkpoint-100 adapter tensors with exact key/shape verification;
4. make only LoRA adapter parameters trainable;
5. create a fresh AdamW optimizer.

Do not merge checkpoint-100 into base weights and then create an unrelated second LoRA unless there is a strong technical necessity. The intended refinement updates the same logical LoRA parameterization.

---

# 6. Dataset and split lineage

## 6.1 Do not create a new training corpus

Reuse the existing frozen prompt/data lineage associated with the source RSSF run.

Primary train pool:

```text
train_n = 5000
```

The new teacher refinement draws batches from this frozen 5000-example pool.

Do not regenerate prompts, change ordering semantics, or substitute another dataset/domain.

## 6.2 Prompt/seed identity

Each training/validation/test row must retain its existing `image_id`, text prompt and deterministic generation seed.

For the current code lineage, native generation seeds are conceptually:

```python
seed_i = config["seed"] * 1_000_000 + row["image_id"]
```

Reuse the exact established helper/metadata rather than duplicating the formula in multiple modules.

## 6.3 Clean paired references

For every teacher refinement sample, the paired clean image is the immutable clean SD2.1 PNG generated with the **same prompt and same seed**.

Expected source folder:

```text
images/teacher_clean_train/
```

Verify every clean file against its committed `image_hashes.json` before reuse.

No gradient through clean images.

## 6.4 Validation/test

Do not silently invent a different split size.

The implementation must read the source `prompts.json` and source config, verify split membership, and preserve the exact existing validation/test rows.

Requirements:

- teacher checkpoint screen uses a deterministic fixed subset of **256 validation rows**;
- selected teacher full validation uses **all available source validation rows**;
- final student/teacher evaluation uses the frozen source test split;
- test data must never be used for teacher checkpoint selection or student checkpoint selection.

If the source validation split has fewer than 256 rows, abort and require an explicit protocol decision rather than silently sampling from test.

---

# 7. Teacher refinement configuration

Lock the first run to:

```text
teacher_objective           = semantic_sign_v3_native_margin_refine
warm_start_step             = 100
teacher_rank                = 32
teacher_scope               = attention_conv
refine_steps                = 300
effective_batch             = 4
teacher_lr                  = 1e-5
optimizer                   = AdamW
betas                       = (0.9, 0.999)
eps                         = 1e-8
weight_decay                = 0.01
teacher_precision           = bf16
inference_steps             = 50
CFG                          = 7.5
tail_unroll_calls           = 10
K_train                      = 26
coordinate_margin_mu        = 0.02
fidelity_cap_ratio          = 0.5
global_grad_clip            = 1.0
teacher_checkpoints         = [50, 100, 200, 300]
teacher_screen_n            = 256
resolution                  = 512
```

Do not run a K sweep, margin sweep, LR sweep or alternative optimizer in this first experiment.

---

# 8. Numerical policy

Use the existing deterministic numerical policy where practical:

```text
cudnn.benchmark = False
cudnn.deterministic = True
TF32 matmul = False
TF32 cudnn = False
```

Training precision:

- trainable/current denoiser forward: BF16 autocast;
- cast conditional/unconditional denoiser outputs to FP32 before CFG arithmetic;
- scheduler state/PNDM algebra: FP32;
- VAE decode used by the differentiable final-image path: FP32;
- semantic extractor / PCA / medians / RSSF coordinates: FP32;
- losses: FP32;
- global gradient-norm reductions: FP64 accumulation where feasible;
- final production teacher evaluation: merged FP16 pipeline, ordinary native inference.

Fail on nonfinite model outputs, latents, coordinates, losses, gradients or trainable parameters.

Do not silently skip nonfinite optimizer updates from the fixed 300-step budget.

---

# 9. Native on-policy rollout — central requirement

This is the most important implementation requirement.

Every one of the 300 refinement optimizer steps must use an **actual current-weight native trajectory**.

The code must NOT:

- replay a cached clean tail;
- start from a clean cached tail boundary;
- use the v2 four-call `replay_tail()` path as the main training trajectory;
- use a predicted-x0 proxy;
- use a clean-teacher prefix followed by fingerprinted tail calls.

The prefix must be generated by the **current fingerprinted teacher weights**.

## 9.1 Conceptual rollout

For each microbatch:

```text
standard Gaussian latent from the frozen per-example seed
    -> current teacher, PNDM50, CFG 7.5
    -> full current prefix under no_grad
    -> final 10 denoiser calls with gradients
    -> final latent
    -> FP32 VAE decode
    -> RGB8 straight-through view
    -> RSSF extractor + pixel fidelity
```

## 9.2 Scheduler

Use the pinned production `PNDMScheduler` and preserve its real mutable history.

Requirements:

- `skip_prk_steps = true`;
- `prediction_type = epsilon`;
- exact production timestep order;
- do not reset `ets`, counter, `cur_model_output` or other PNDM history at the prefix/tail boundary;
- do not substitute a hand-derived diffusion update for `scheduler.step()`.

The new code may reuse audited serializer/helper ideas from `semantic_trajectory.py`, but it must not change the historical v2 four-call trace semantics.

Recommended: implement a new helper for **live current-weight truncated rollout** instead of changing the meaning of `NativeTrace(tail_calls=4)`.

## 9.3 Prefix

Let total native denoiser/scheduler calls be whatever the pinned PNDM implementation actually produces for 50 inference steps.

Define:

```text
boundary = total_calls - 10
```

Calls before `boundary`:

- current fingerprinted teacher weights;
- `torch.no_grad()`;
- same BF16-denoiser / FP32-CFG / FP32-scheduler policy;
- scheduler history updated normally.

At the boundary, the latent and scheduler history naturally come from the current teacher prefix. Do not replace them with cached clean states.

## 9.4 Differentiable tail

For the final 10 calls:

- current fingerprinted teacher weights;
- keep autograd graph;
- maintain scheduler history continuously;
- do not detach intermediate tail latents;
- use gradient checkpointing if needed.

This is a truncated-gradient approximation. The forward trajectory is fully current/on-policy; only backward is truncated to the final 10 calls.

---

# 10. Native microbatching on L40/CETUS

Effective teacher batch must remain:

```text
effective_batch = 4
```

Recommended initial native microbatch:

```text
native_micro_batch = 2
```

Before the real 300-step run, execute an actual-model GPU smoke test containing:

- current warm-start weights;
- full PNDM50 prefix;
- 10-call differentiable tail;
- RSSF extractor;
- preservation loss;
- pixel loss;
- two-gradient extraction;
- gradient accumulation buffers.

If microbatch 2 does not fit the assigned L40, allow an explicit config override:

```text
native_micro_batch = 1
```

This is an engineering change only. It must preserve:

- effective batch 4;
- exact per-sample loss definitions;
- exact optimizer step count;
- exact gradient accumulation math.

Do not silently reduce effective batch or number of samples per optimizer step.

---

# 11. RGB8 straight-through final-image path

Training RSSF must see an RGB8-aware differentiable view of the final image.

Implement one audited helper and reuse it everywhere in v3 training.

Conceptually for a decoded image in `[-1, 1]`:

```python
x = image.clamp(-1, 1)
x01 = (x + 1) / 2
q01 = torch.round(x01 * 255) / 255
q01_ste = x01 + (q01 - x01).detach()
x_ste = q01_ste * 2 - 1
```

The forward value equals the quantized 8-bit grid while backward uses the identity STE.

Use `x_ste` for:

- RSSF semantic extraction during training;
- paired pixel MSE against clean PNG tensors loaded onto the same `[-1,1]` scale.

Actual evaluation remains authoritative and must save/load real PNG files; no STE is involved in final scoring.

Unit-test the STE conversion against actual PIL/diffusers PNG conversion within a declared 1-LSB-compatible tolerance.

---

# 12. Dynamic per-image RSSF margin loss

## 12.1 Do NOT use the historical `sign_hinge_loss()` unchanged

Current `semantic_sign.py::sign_hinge_loss()` selects easiest bits from:

```python
clean_r.detach() * signs
```

That behavior is required for historical v2 analysis and must remain unchanged.

The new v3 method requires **current-image dynamic selection**.

Add a new function, e.g.:

```text
dynamic_sign_margin_loss(...)
```

Do not change the semantics of the historical function.

## 12.2 Coordinates

For a current final image, use the unchanged RSSF path:

```text
RGB8-STE image
-> frozen clean VAE posterior mean * scaling_factor
-> frozen clean semantic UNet at t=0
-> mid-block GAP
-> PCA projection
-> sigma normalization
-> median centering
-> r in R^32
```

Let secret signs be:

\[
s_j \in \{-1,+1\}.
\]

Define current signed margins:

\[
m_{ij}=s_j r_{ij}.
\]

## 12.3 Dynamic Top-26

For each current image `i`, select the 26 largest signed margins:

\[
T_i = \operatorname{Top26}(m_i).
\]

Implementation:

```python
signed = coords * signs.detach()
indices = torch.argsort(
    signed.detach(), dim=-1, descending=True, stable=True
)[:, :26]
selected = signed.gather(1, indices)
```

Important:

- sort/select on detached signed margins;
- gradient flows through the selected coordinate values only;
- stable tie behavior must be deterministic;
- selection is recomputed from the **current image at every optimizer step**;
- do not cache selected indices from clean images;
- do not select using aggregate/model-level coordinates.

## 12.4 Loss

Lock:

```text
K_train = 26
mu      = 0.02
```

For batch size `B`:

\[
\boxed{
L_M
=
\frac{1}{B}
\sum_{i=1}^{B}
\frac{1}{26}
\sum_{u\in T_i}
\left[\max(0,0.02-u)\right]^2
}
\]

Use the **fixed denominator 26**.

Do not renormalize by the number of deficient bits. This deliberately lets fingerprint pressure decay as an image approaches the target.

Why 26:

- actual verifier needs 24 correct signs;
- training with 26 provides two-bit redundancy before student distillation;
- six difficult coordinates can still be ignored per image.

Why 0.02:

- this margin was already shown by the oracle to be compatible with extremely low image distortion;
- do not simultaneously increase both bit count and margin in the first definitive run.

---

# 13. Preservation loss

The teacher is warm-started from a fingerprinted checkpoint, but refinement must avoid uncontrolled early/mid-trajectory drift.

Each native microbatch must also compute one on-policy preservation probe.

## 13.1 Probe state

During the current-weight no-grad prefix:

1. deterministically sample one prefix call index using the refinement data RNG;
2. save the **current trajectory latent** and corresponding timestep at that call;
3. detach this latent; it is a state at which the two models will be compared.

The selected call must be before the 10-call differentiable boundary.

Use the same call index for all examples in that microbatch for efficient batching.

Do not use a cached clean latent as the preservation input.

## 13.2 Conditional/unconditional predictions

At the detached current latent/timestep, evaluate:

- current fingerprinted teacher with gradients;
- frozen clean SD2.1 teacher without gradients;

for both:

- unconditional text embedding;
- conditional prompt embedding.

Let predictions be:

```text
epsilon_u_theta, epsilon_c_theta
epsilon_u_clean, epsilon_c_clean
```

Compute in FP32 after the BF16 denoiser forward:

\[
\boxed{
L_P
=
\frac{1}{2}
\left(
\|\epsilon_u^\theta-\epsilon_u^0\|_2^2
+
\|\epsilon_c^\theta-\epsilon_c^0\|_2^2
\right)_{mean}
}
\]

No gradient through the clean teacher or saved latent.

Do not use the old VAE-cycle semantic preservation loss in v3.

---

# 14. Pixel fidelity loss

Use the current final RGB8-STE image and paired clean SD2.1 PNG from the same prompt/seed.

Both tensors must use the same `[-1,1]` RGB convention.

\[
\boxed{
L_Q = \operatorname{MSE}(I_\theta^{RGB8-STE}, I_{clean}^{RGB8})
}
\]

No gradient through the clean image.

Do not optimize LPIPS, MS-SSIM, CLIP, FID, KID, aesthetic scores or human-prediction models.

Those metrics are evaluation/diagnostic only.

---

# 15. Final teacher gradient rule

This section is LOCKED. Do not replace it with scalar weighted loss summation, independent alpha values, QP projection, PCGrad, or Adam-displacement projection.

Let:

\[
g_M = \nabla_\theta L_M
\]

and define the total fidelity loss:

\[
L_F = L_P + L_Q,
\]

\[
g_F = \nabla_\theta L_F.
\]

Compute global L2 norms over **all trainable LoRA parameters** using FP64 reduction where practical:

\[
n_M = \|g_M\|_2,\qquad n_F = \|g_F\|_2.
\]

Use:

```text
fidelity_cap_ratio = 0.5
grad_epsilon       = 1e-12
```

If:

```text
n_M <= grad_epsilon
```

then **skip the optimizer update** for this batch. Do not apply a fidelity-only step.

Otherwise:

\[
\boxed{
\alpha_F
=
\min\left(
1,
\frac{0.5\,n_M}{n_F+\epsilon}
\right)
}
\]

If `n_F <= grad_epsilon`, set `alpha_F = 0`.

Combined raw gradient:

\[
\boxed{
g = g_M + \alpha_F g_F.}
\]

Then apply global norm clipping:

```text
max_norm = 1.0
```

and execute the fresh AdamW optimizer step.

## 15.1 Important interpretation

The norm cap ensures:

\[
\|\alpha_Fg_F\| \le 0.5\|g_M\|.
\]

Therefore the raw combined gradient cannot be completely reversed by the fidelity gradient. In the worst exactly-opposite case, a nonzero component in the fingerprint-gradient direction remains.

This statement applies to the **raw combined gradient before AdamW moment transformation**. Do not claim that AdamW displacement is mathematically identical to steepest descent.

---

# 16. Gradient extraction under microbatching

Do not retain all four sample graphs simultaneously if this risks memory overflow.

For each native microbatch:

1. compute `L_M_mb`, `L_P_mb`, `L_Q_mb`;
2. multiply by `microbatch_size / effective_batch` so losses average over the effective batch;
3. use `torch.autograd.grad()` to obtain a microbatch contribution to `g_M`;
4. separately obtain the contribution to `g_F = grad(L_P + L_Q)`;
5. accumulate detached FP32 gradient buffers for every trainable parameter;
6. release that microbatch graph;
7. proceed to the next microbatch.

After all 4 effective-batch samples:

- compute global `n_M`, `n_F` from the accumulated buffers;
- compute `alpha_F` once for the entire effective batch;
- construct final gradients;
- assign to `parameter.grad`;
- clip;
- call optimizer step exactly once.

Do not compute `alpha_F` separately per microbatch.

Treat `None` gradients as zeros for accumulation, but audit unexpected permanently-unused trainable parameters.

---

# 17. Teacher sampling order and reproducibility

The refinement train pool contains 5000 rows, but 300 steps x batch 4 means 1200 sample exposures in this refinement stage.

Use a deterministic shuffled order from the frozen 5000-row pool:

- CPU `torch.Generator` seeded from the run seed;
- consume sequential batches of four;
- reshuffle only if the permutation is exhausted;
- save generator state, order and offset in resume checkpoints.

Do not choose samples by RSSF score, image difficulty or visual quality.

Do not oversample oracle-success cases.

---

# 18. Teacher training checkpoints and resume

Save full resumable refinement checkpoints at least at:

```text
50
100
200
300
```

It is acceptable to save additional internal resume checkpoints (for example every 25 or 50 steps) as long as the scientific evaluation milestones remain frozen.

Each resume checkpoint must contain/verify:

- LoRA trainable parameters;
- AdamW state;
- refinement optimizer step;
- data RNG state;
- data order and offset;
- CPU torch RNG;
- CUDA RNG;
- Python RNG;
- protocol/config identity;
- source checkpoint hash;
- carrier/key hashes;
- prompt hash;
- source clean-image hashes or manifest hash;
- loss-log prefix hash.

Reuse `semantic_checkpoint.py` patterns where possible.

A killed CETUS job must be restartable without redoing completed optimizer steps.

---

# 19. Training diagnostics to log every step

Write one JSONL row per refinement optimizer step.

Minimum fields:

## Identity

- step;
- batch image IDs;
- prompt IDs or hashes;
- seed IDs;
- warm-start source checkpoint hash;
- code/config protocol hash.

## Native rollout

- total PNDM call count;
- tail boundary index;
- exact final 10 timestep IDs;
- native microbatch size;
- final latent norm;
- preservation probe call/timestep.

## RSSF

- `K_train = 26`;
- `mu = 0.02`;
- training `L_M`;
- mean current per-image `K` on differentiable RGB8-STE image;
- per-image K list;
- mean selected signed margin;
- fraction of selected margins >= 0.02;
- 26th signed margin per image;
- selected-bit frequency histogram over 32 bits.

## Fidelity

- `L_P`;
- `L_Q` pixel MSE;
- mean absolute paired pixel difference (optional diagnostic).

## Optimization

- `n_M`;
- `n_F`;
- `alpha_F`;
- combined pre-clip norm;
- post-clip norm or clip coefficient;
- optimizer update applied vs skipped;
- peak allocated/reserved GPU memory;
- finite checks.

Training diagnostics are not checkpoint-selection metrics.

---

# 20. Teacher checkpoint screening

Screen exactly:

```text
steps = [50, 100, 200, 300]
screen_n = 256
```

Use one deterministic frozen 256-row subset of the validation split for every checkpoint.

For each candidate:

1. start from pinned clean SD2.1;
2. load the candidate LoRA;
3. merge/fuse safely in memory or into a temporary immutable production snapshot;
4. generate ordinary native FP16 images using PNDM50, CFG 7.5, prompt + Gaussian only;
5. use the exact same prompt/seed pairs as the paired clean validation images;
6. save actual PNGs;
7. score actual PNGs with the unchanged RSSF verifier.

Do not screen on differentiable training images.

---

# 21. Teacher checkpoint metrics

For each screened checkpoint compute and save:

## Primary RSSF

- per-image K;
- analytic per-image detected flag (`K >= 24`);
- analytic TPR;
- Wilson 95% CI for TPR;
- mean K;
- median K;
- per-image bit accuracy;
- model `K_M / 32`;
- model `p_M`;
- continuous mean score (secondary).

## Transfer-headroom metric

For actual PNG coordinates define signed margins:

\[
m_{ij}=s_jr_{ij}.
\]

Sort per image and take the 26th largest:

\[
m_{i,(26)}.
\]

Define:

\[
\boxed{
D_{26}
=
\frac{1}{N}
\sum_i
\left[\max(0,0.02-m_{i,(26)})\right]^2.
}
\]

Also report:

- fraction with `m_(26) >= 0.02`;
- p10/p50/p90 of `m_(26)`.

## Paired fidelity

- pixel MSE;
- LPIPS (diagnostic only);
- optional PSNR/SSIM if already available, diagnostic only.

## Semantic utility

- CLIP score;
- CLIP drop relative to paired clean SD2.1 validation.

FID/KID are not required for every intermediate checkpoint.

---

# 22. Teacher checkpoint selection rule

A checkpoint is eligible only if:

```text
analytic per-image TPR >= 0.90
CLIP drop              <= 0.01
```

Among eligible checkpoints choose, in this exact order:

1. **lowest `D_26`**;
2. lower paired pixel MSE;
3. earlier refinement step.

Do not choose a checkpoint based on:

- test metrics;
- student metrics;
- FID/KID;
- LPIPS alone;
- highest TPR once the 90% gate is met;
- visual preference.

If no checkpoint is eligible:

```text
teacher_status = failed
```

Stop before student training.

Do not automatically extend to 400/600/1000 steps after inspecting the result.

---

# 23. Full selected-teacher validation

After selection, merge the selected LoRA into a standard SD2.1 production pipeline and run **all source validation rows**.

Recompute from actual PNGs:

- TPR + Wilson CI;
- per-image K distribution;
- bit accuracy;
- `K_M`, `p_M`;
- `D_26` and 26th-margin distribution;
- CLIP and CLIP drop;
- paired pixel MSE;
- paired LPIPS diagnostic;
- fixed-key clean FPR from the frozen clean control evidence.

Save at least 16 deterministic paired clean/marked visual examples plus optional absolute-difference panels.

Do not use manual visual inspection to change the selected checkpoint.

The selected merged teacher must require only ordinary prompt + Gaussian inference.

---

# 24. Teacher production artifact

The selected final teacher must be an ordinary merged Stable Diffusion pipeline.

Requirements:

- no active PEFT adapter object at production inference;
- no secret key passed to generation;
- no hidden hook;
- no custom sampler modification;
- no runtime residual injection;
- no detector in generation path;
- standard native PNDM50/CFG7.5 inference.

Record hashes of the merged production model files.

---

# 25. Student training data generation

Only after the teacher passes the full teacher gate:

Generate the frozen student-training image dataset from the selected merged teacher.

Use:

```text
train prompts       = same frozen 5000 train rows
teacher             = selected merged SD2.1 RSSF teacher
sampling            = ordinary PNDM50
CFG                 = 7.5
resolution          = 512
seed per example    = frozen existing seed protocol
output              = actual PNG RGB images
```

Expected label/folder can follow existing harness conventions such as:

```text
images/teacher_wm_train/
```

Also generate teacher marked validation images required by the existing student validation cache.

Freeze image hashes and metadata before student training starts.

The student dataset contains only normal images/captions. No RSSF coordinate, key, detector score or oracle target is included.

---

# 26. Student distillation — REUSE EXISTING HARNESS

This part should not be redesigned.

Reuse the current repository's:

- `student.py`
- vendored `train_text_to_image_lora.py`
- `validation.py`

The current student and Thinh's native project use the same core training command/selection logic. Prefer the local repo implementation so RSSF scoring/reporting remains integrated.

## 26.1 Student configuration

Lock:

```text
base student              = pinned CompVis/stable-diffusion-v1-4 snapshot
kind                      = LoRA
rank                      = 32
scope                     = attention_conv
resolution                = 512
center_crop               = true
train_batch_size          = 4
gradient_accumulation     = 1
gradient_checkpointing    = true
learning_rate             = 1e-4
mixed_precision           = fp16
lr_scheduler              = constant
lr_warmup_steps           = 0
max_grad_norm             = 1.0
dataloader_num_workers    = 0
max_train_steps           = 7500
snapshot_steps            = [2500, 5000, 7500]
seed                      = frozen campaign seed
loss                      = ordinary diffusion denoising objective from the vendored trainer
```

Do not add:

- RSSF loss;
- key-dependent loss;
- detector loss;
- teacher semantic features;
- distillation feature matching;
- KL/DPO/preference optimization;
- oracle supervision;
- special data augmentation intended to preserve the fingerprint.

The student must be genuinely unaware of the fingerprint.

## 26.2 Student optimizer/update semantics

Use the existing vendored trainer behavior exactly unless required only to support the locked `max_train_steps`/snapshots.

Do not reinterpret a skipped/overflow update as an applied optimizer step. Record actual applied update count and require 7500 applied updates.

---

# 27. Student checkpoint selection

Reuse `validation.py` semantics.

Prepare a held-out validation cache from the selected teacher's validation RGB images using the existing deterministic protocol.

Evaluate snapshots:

```text
2500
5000
7500
```

For each snapshot compute held-out denoising MSE using the existing fixed latent/noise/timestep draws.

Select exactly:

\[
\boxed{\text{minimum held-out validation denoising MSE}}
\]

Exact ties choose the earlier step.

Freeze this choice **before** reading RSSF detection, CLIP, FID, KID, LPIPS or test data.

Do not select the student checkpoint by fingerprint TPR.

---

# 28. Required student controls

## 28.1 Unadapted SD1.4

Always evaluate the pinned unadapted SD1.4 student under the identical test generation protocol.

## 28.2 Clean-trained student control

A clean-trained SD1.4 LoRA control is strongly preferred for final evidence.

It must use:

- same SD1.4 initialization;
- same rank/scope;
- same optimizer/LR;
- same batch;
- same 7500 steps;
- same checkpoint-selection rule;
- same prompts/seeds;
- clean SD2.1 images instead of fingerprinted-teacher images.

If an existing clean-trained control artifact already has **exactly matching** prompt hashes, image hashes, base student hash and training protocol, verify and reuse it.

Otherwise train a new clean control. Do not reuse a merely similar historical control.

This control may be run as a separate sequential CETUS stage; it does not change the fingerprinted student method.

---

# 29. Final test generation

Use the frozen test split only after teacher and student selections are frozen.

Generate, with identical resolution/sampling conventions:

- clean SD2.1 teacher test images;
- selected fingerprinted SD2.1 teacher test images;
- unadapted SD1.4 test images;
- selected fingerprinted-data-trained SD1.4 LoRA test images;
- clean-trained SD1.4 control test images if available.

Use ordinary generation only.

No detector/key influences generation.

---

# 30. Final RSSF metrics

For every evaluated model family, score actual saved PNGs with the unchanged RSSF semantic extractor/carrier.

## 30.1 Primary per-image metrics

Report:

- number of images `N`;
- detected count;
- TPR/FPR as appropriate;
- Wilson 95% CI;
- mean K;
- median K;
- histogram of K;
- mean per-image bit accuracy;
- fraction with all 32 bits correct (diagnostic only).

Detection is always:

```text
K >= 24
```

Do not create an empirical continuous threshold for RSSF.

## 30.2 Model-level evidence

Report:

- `K_M / 32`;
- `p_M` from the exact Binomial tail;
- recovered aggregate signs;
- aggregate bit accuracy;
- continuous mean score as secondary only.

Do not present model-level success as if it were per-image TPR.

## 30.3 Clean/wrong-key controls

Report fixed owner-key detection on:

- clean SD2.1;
- unadapted SD1.4;
- clean-trained SD1.4.

Also compute wrong/random-key evidence from saved coordinate arrays without regenerating images. Reuse the established random-key protocol where applicable.

Do not alter the owner key after looking at these results.

---

# 31. Utility metrics

The primary utility question is whether fingerprinted outputs remain comparable to outputs without the fingerprint, not whether an individual sample exactly preserves one stochastic realization.

## 31.1 Distribution / semantic utility

Use the existing `utility.py` protocol with identical image counts and references.

Report:

- FID — lower is better;
- KID mean/subset diagnostic — lower is better, interpret near-zero carefully;
- CLIP score — higher is better.

Compare only under identical reference set, count, resolution and preprocessing.

## 31.2 Paired diagnostics

For same-prompt/same-seed teacher comparisons report:

- pixel MSE;
- LPIPS diagnostic;
- optional MS-SSIM/PSNR diagnostic if already implemented.

For student vs unadapted SD1.4 paired generations, LPIPS may also be reported as a diagnostic.

**LPIPS is not a hard success gate and is not optimized in the new method.** A high paired LPIPS may reflect a different valid realization rather than poor absolute image quality.

## 31.3 Visual panel

Save deterministic visual panels, not cherry-picked examples.

Recommended minimum:

```text
teacher: 16 clean vs fingerprinted same-prompt/seed pairs
student: 16 unadapted vs fingerprint-trained vs clean-trained same-prompt/seed rows
```

Where useful, include absolute-difference views for teacher paired samples.

Do not select checkpoints based on manual visual preference.

---

# 32. Success/failure interpretation

## 32.1 Teacher gate to proceed downstream

Teacher must satisfy on full selected-teacher validation:

```text
analytic per-image TPR >= 0.90
CLIP drop              <= 0.01
```

If this fails, stop before student training and save diagnostics.

Do not automatically change K, mu, LR, steps, tail length or key.

## 32.2 Final student headline target

The desired final result is:

```text
student per-image TPR >= 0.90
```

with clean/unadapted owner-key false-positive behavior remaining close to the validated analytic/null regime and without material utility degradation.

This is a research target, not a reason to tune on the test set. If the selected student is below 90%, report the result and stop. Do not select another checkpoint using test detection.

---

# 33. Recommended code organization

Do not overload historical v2 files with incompatible semantics if it can be avoided.

Recommended new file:

```text
semantic_sign_native_margin_refine.py
```

Responsibilities:

- v3 warm-start validation;
- current-weight PNDM50 truncated rollout;
- preservation probe;
- dynamic Top-26 RSSF loss;
- RGB8 STE helper or import of one audited shared helper;
- microbatch gradient extraction/accumulation;
- bounded fidelity gradient combination;
- v3 training loop;
- candidate checkpoint screening;
- v3 teacher selection;
- selected teacher merge;
- v3 report.

Possible shared helper file if needed:

```text
semantic_native_rollout.py
```

Responsibilities:

- production-equivalent latent initialization;
- current-weight PNDM scheduler loop;
- prefix/tail boundary handling;
- differentiable tail helper;
- optional instrumentation.

Keep:

```text
semantic_sign.py
```

as the verifier/math home. Add the new dynamic margin function there only if it remains cleanly versioned; do not change historical `sign_hinge_loss()` semantics.

Reuse:

```text
semantic_checkpoint.py
student.py
validation.py
utility.py
semantic_teacher.py
```

where their existing semantics already match the specification.

---

# 34. Orchestration stages

The campaign should be runnable sequentially on CETUS. No job array or parallel campaign is required.

Recommended idempotent stages:

```text
01_verify_source_and_reuse_preparation
02_teacher_smoke_test
03_teacher_refine_300
04_teacher_screen
05_teacher_full_validation_and_merge
06_generate_student_train_validation_data
07_student_train_7500
08_student_validation_select
09_optional_clean_control_or_verify_reuse
10_final_test_generation
11_final_rssf_scoring
12_final_utility_and_report
```

Each stage should detect already-complete immutable artifacts by manifest/hash and resume or reuse them rather than overwrite them.

A user should be able to submit the next sequential PBS job without editing scientific settings.

---

# 35. CETUS / 24-hour considerations

The method is intentionally designed so no scientific stage contains a multi-day oracle precomputation.

Teacher refinement workload:

```text
300 native events
full current PNDM50 forward trajectory
gradient through final 10 calls
batch 4, microbatch 2 by default
```

This is deliberately in the same general compute class as the previously completed native refinements rather than the rejected 5000-image oracle-bank design.

Requirements:

- teacher refinement must be resumable;
- student training already has snapshot/checkpoint support;
- data generation must be resumable by image hash;
- no stage should rely on an uncheckpointed multi-day process.

Do not reduce scientific budgets automatically based on wall-clock time. If a stage exceeds a 24-hour allocation, resume the same stage from its committed state in the next sequential job.

---

# 36. Mandatory implementation guards

Abort rather than silently continue if any of the following occurs:

- wrong source checkpoint step;
- carrier/key hash mismatch;
- prompt/split hash mismatch;
- wrong teacher/student backbone snapshot;
- unsupported scheduler class/version/protocol;
- PNDM call order changed;
- prefix generated by clean rather than current teacher;
- tail length not 10;
- training accidentally calls historical clean-selected `sign_hinge_loss()`;
- nonfinite latent/image/coordinate/loss/gradient;
- student trainer receives key/detector metadata as training input;
- student checkpoint selection reads detector/utility/test metrics;
- incomplete image dataset is passed to student training;
- selected merged teacher still contains an active runtime LoRA adapter unexpectedly.

---

# 37. Required unit/integration tests

## 37.1 RSSF verifier regression

Existing tests must still pass, especially:

```text
detection_count(32, 0.01) == 24
```

Historical `image_evidence()` and `model_evidence()` outputs must remain unchanged for the same coordinates/key.

## 37.2 Historical loss regression

`sign_hinge_loss()` must still select Top-K from clean coordinates for old v2 runs.

The new dynamic loss must be a separate code path.

## 37.3 Dynamic Top-26 test

Construct synthetic 32-D coordinates with known signed ordering and verify:

- exactly 26 indices selected;
- selected from current signed coordinates, not clean coordinates;
- stable deterministic ties;
- fixed denominator 26;
- zero loss when all selected margins >= 0.02;
- correct gradient only on selected coordinate values.

## 37.4 Production-equivalent clean rollout

With LoRA disabled / clean weights, the new manual/current native rollout helper must reproduce the existing `native_generate(..., output_type='latent')` protocol for the same prompt/seed to declared numerical tolerance.

This test is mandatory before training. It prevents a custom-loop sampler mismatch from masquerading as a fingerprint effect.

## 37.5 Current-weight prefix test

Perturb a trainable adapter parameter and verify that a prefix latent before the 10-call boundary changes.

This proves the v3 prefix uses current teacher weights rather than cached clean trajectory states.

## 37.6 Tail-gradient test

Verify:

- no gradient graph retained through prefix calls;
- gradients propagate through each of the final 10 calls;
- detaching any one expected tail latent changes/fails the test.

## 37.7 PNDM-history continuity

Instrument scheduler state around the prefix/tail boundary and verify no reset/reinitialization occurs.

## 37.8 RGB8 STE test

Forward RGB values must match 8-bit quantized values; backward gradient should be identity through the quantizer within the intended clamp region.

Compare to actual saved/reloaded PNG tensors within 1-LSB-compatible tolerance.

## 37.9 Preservation test

At a synthetic/current prefix state:

- clean model output is detached;
- fingerprint model output receives gradient;
- conditional and unconditional terms both contribute;
- clean-reference parameters receive no gradient.

## 37.10 Gradient-cap test

For synthetic parameter gradients verify:

```text
||alpha_F * g_F|| <= 0.5 * ||g_M||
```

and in the exactly opposite case:

```text
dot(g_M, g_M + alpha_F*g_F) >= 0.5 * ||g_M||^2
```

up to numerical tolerance.

Also verify no optimizer update occurs when `n_M <= grad_epsilon`.

## 37.11 Microbatch equivalence

On a tiny deterministic test model/batch, compare effective batch 4 run as:

- microbatch 2 + 2;
- microbatch 1 + 1 + 1 + 1.

Accumulated `g_M`, `g_F`, alpha and final combined gradient should agree within declared tolerance.

## 37.12 Warm-start identity

Load source checkpoint-100 into the new adapter and verify exact tensor equality before the first refinement optimizer step.

## 37.13 Student no-key audit

Static/runtime audit must show the student training command receives only:

- student base model;
- teacher RGB-caption data;
- ordinary trainer hyperparameters.

No carrier/key/detector path may enter student loss computation.

## 37.14 Student selection audit

`validation.py` must freeze the selected snapshot using denoising MSE before any detector/utility/test scoring is read.

---

# 38. Required run artifacts

Recommended output structure:

```text
run/
  config.json
  assets.json
  prompts.json
  code_manifest.json
  source_lineage.json
  carrier/
  images/
    teacher_clean_train/
    teacher_clean_validation/
    teacher_wm_validation_step50/
    teacher_wm_validation_step100/
    teacher_wm_validation_step200/
    teacher_wm_validation_step300/
    teacher_wm_validation/
    teacher_wm_train/
    teacher_wm_test/
    unadapted/
    lora_r32_conv_wm/
    ...
  teacher_fp/
    native_margin_refine/
      protocol.json
      losses.jsonl
      checkpoint-50/
      checkpoint-100/
      checkpoint-200/
      checkpoint-300/
    screening.json
    selected.json
    merged/
  students/
    lora_r32_conv_wm/
      protocol.json
      snapshot-2500/
      snapshot-5000/
      snapshot-7500/
      complete.json
  validation/
  validation_cache/
  scores/
  result/
    teacher_screening.json
    teacher_full_validation.json
    teacher_visuals/
    student_selection/
    final_rssf.json
    utility.json
    final_visuals/
    README.md
  complete.json
```

Exact folder names may follow existing repository conventions, but artifacts must be immutable/hash-audited once committed.

---

# 39. Final report requirements

The generated report/README must clearly separate:

## Teacher

- warm-start source identity;
- v3 config;
- checkpoint curve;
- selected checkpoint reason;
- TPR/Wilson CI;
- K distribution;
- K_M/p_M;
- D26 distribution;
- CLIP/drop;
- pixel MSE;
- LPIPS diagnostic;
- final visual pairs.

## Student

- training dataset identity;
- exact training command;
- 2500/5000/7500 validation denoising MSE;
- frozen selected checkpoint;
- final test TPR/Wilson CI;
- K distribution;
- K_M/p_M;
- per-image bit accuracy;
- unadapted and clean-control FPR;
- FID/KID/CLIP;
- LPIPS diagnostic where paired;
- fixed visual panel.

## Scientific caveats

State explicitly:

- one training seed does not measure between-run uncertainty;
- Wilson intervals are over images, not training seeds;
- FID at finite sample count is biased;
- KID subset std is not a training-seed CI;
- LPIPS measures paired realization distance, not absolute aesthetic quality;
- model-level K_M is not per-image detection;
- test data were not used for checkpoint selection.

---

# 40. Explicit non-goals for this run

Do NOT add:

- another oracle run;
- 512/5000 oracle target bank;
- online oracle optimization;
- oracle residual imitation;
- a new carrier;
- a new secret key;
- new bit count sweep;
- margin sweep;
- LR sweep;
- train-from-clean alternative;
- four-call alternative;
- full-trajectory backprop;
- DPO;
- QP projection;
- PCGrad;
- Adam displacement projection;
- LPIPS/MS-SSIM training loss;
- aesthetic-model training loss;
- learned watermark detector;
- student fingerprint loss;
- attack robustness experiments;
- paraphrase/JPEG/resize robustness;
- additional backbone families.

Those are future experiments only if the locked run produces evidence requiring them.

---

# 41. Implementation checklist for Codex

Before declaring implementation complete, verify all boxes:

```text
[ ] New objective is versioned and historical runs remain readable.
[ ] Carrier/key/verifier are reused unchanged.
[ ] Warm-start source is exactly v2 checkpoint-100 and hash-verified.
[ ] Fresh AdamW optimizer is used.
[ ] 300 optimizer updates are locked.
[ ] Every optimizer step uses a current-weight native PNDM50 rollout.
[ ] Prefix is no_grad but generated by the current teacher.
[ ] Exactly final 10 denoiser calls are differentiable.
[ ] PNDM history is continuous across prefix/tail.
[ ] RGB8 STE is used for training RSSF/final pixel loss.
[ ] Dynamic Top-26 comes from current signed coordinates.
[ ] mu is exactly 0.02.
[ ] Fixed denominator is exactly 26.
[ ] Preservation uses current trajectory state vs frozen clean teacher.
[ ] Pixel loss is paired MSE only.
[ ] g_M and g_F are accumulated separately over effective batch 4.
[ ] Fidelity gradient is capped at 0.5 * ||g_M||.
[ ] No fidelity-only optimizer update when g_M is effectively zero.
[ ] Global gradient clip is 1.0.
[ ] Checkpoints 50/100/200/300 are saved and screened.
[ ] Teacher selection uses TPR/CLIP eligibility then D26/pixel/earlier tie-break.
[ ] No test data enter teacher selection.
[ ] Selected teacher is merged into ordinary production SD2.1.
[ ] 5k student RGB-caption dataset comes from selected teacher ordinary inference.
[ ] Existing SD1.4 LoRA trainer is reused.
[ ] Student is 7500 applied updates with snapshots 2500/5000/7500.
[ ] Student selection is denoising MSE only.
[ ] Student loss has no key/detector/fingerprint term.
[ ] Final actual-PNG RSSF metrics and utility metrics are produced.
[ ] Run is resumable and hash-audited for CETUS.
```

---

# 42. Final instruction to Codex

Implement exactly one coherent experiment:

> Keep the existing 32-bit RSSF carrier and exact Binomial verifier unchanged. Warm-start the existing `semantic_sign_v2_trajectory` checkpoint-100 LoRA, create a new versioned `semantic_sign_v3_native_margin_refine` objective, and perform 300 native current-weight PNDM50 refinement updates. At every update, forward the full trajectory using the current fingerprinted teacher, backpropagate only through the final 10 denoiser calls, compute a dynamic current-image Top-26 RSSF hinge with margin 0.02 on an RGB8-STE final image, and combine its gradient with a clean-denoiser + paired-pixel fidelity gradient capped to at most 0.5 of the RSSF gradient norm. Select among steps 50/100/200/300 using held-out actual-PNG TPR/CLIP eligibility followed by lowest D26, then pixel MSE and earlier step. If the selected teacher passes, merge it into an ordinary SD2.1 pipeline, generate 5000 ordinary RGB-caption pairs, and reuse the existing SD1.4 rank-32 attention+Conv LoRA trainer for 7500 ordinary denoising-MSE updates. Select the student only by held-out denoising MSE and finally evaluate RSSF and utility on frozen test images. The student must never receive the RSSF key, detector, coordinates, or fingerprint loss.

Do not redesign the method while implementing it. If an invariant cannot be satisfied, fail loudly and report the reason instead of silently changing the scientific protocol.
