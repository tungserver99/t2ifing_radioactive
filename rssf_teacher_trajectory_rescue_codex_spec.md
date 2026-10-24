# Codex Implementation Specification — RSSF Teacher Trajectory Rescue

## Status

**LOCKED teacher-only rescue specification**

Target repository:

`https://github.com/tungserver99/t2ifing_radioactive.git`

Baseline reviewed from current `main` on 2026-10-22, including the RSSF implementation present in:

- `README_semantic_fp.md`
- `Malicious_quantization/bitmark_lora/run.py`
- `semantic_carrier.py`
- `semantic_sign.py`
- `semantic_sign_train.py`
- `semantic_sign_report.py`
- `semantic_teacher.py`
- `semantic_report.py`
- `semantic_assets.py`
- `student.py`

Latest repository commit observed while preparing this spec:

`7a85dff6267368d33017f2ace8c473d0a21867b5`

This document replaces only the **next teacher-training/evaluation revision**. It must **not** delete or reinterpret historical `semantic_sign_v1` or `threshold_cfg_v1` runs.

---

# 1. Why this revision is needed

The current RSSF / `semantic_sign_v1` experiment produced approximately:

| coordinate margin μ | Analytic per-image detection | CLIP drop | Paired LPIPS |
|---:|---:|---:|---:|
| 0.10 | 99.8% | 0.01371 | 0.4071 |
| 0.20 | 100% | 0.01272 | 0.4157 |
| 0.30 | 100% | 0.01704 | 0.4381 |

The interpretation is important:

1. **The 32-bit semantic sign carrier and analytic verifier are working very well.**
2. Detection is already saturated at essentially 100%.
3. Changing `mu` from 0.10 to 0.30 changes utility only slightly.
4. Therefore the main problem is **not insufficient fingerprint strength** and is unlikely to be solved by another `mu` sweep.
5. The main remaining failure is **teacher utility / trajectory drift**.

The current code optimizes fingerprint and LPIPS on a decoded **single-step predicted-x0 proxy** constructed from an independently forward-noised late timestep.

However, the final teacher is evaluated by ordinary native Stable Diffusion generation, where the same modified UNet weights participate in the **entire reverse denoising trajectory**.

Therefore the current training objective is mismatched with deployment:

```text
training quality target:
    one forward-noised latent
    -> one UNet prediction
    -> predicted_x0
    -> decode
    -> LPIPS/fingerprint

deployment:
    Gaussian latent
    -> full native scheduler trajectory
    -> repeated fingerprinted-UNet calls
    -> final latent
    -> decode
```

A small weight-induced difference can accumulate through the reverse trajectory even if the predicted-x0 proxy at one timestep looks close to the clean target.

This revision must address that mismatch while preserving the successful fingerprint/verifier.

---

# 2. Core principle: improve, do not redesign

This revision is an **improvement of RSSF**, not a new fingerprint system.

The following components are considered successful and are **LOCKED**.

## 2.1 Keep unchanged

Keep:

- frozen SD2.1 semantic extractor;
- frozen clean SD2.1 VAE / text encoder / semantic UNet reference;
- PCA semantic feature space;
- PC range 8..64;
- VAE-cycle stability ranking;
- 32 selected semantic PCs;
- median-centered coordinates;
- secret PC indices and secret signs;
- exact 32-bit key-randomization Binomial verifier;
- per-image `K`;
- analytic `p_img`;
- model-level `K_M` and `p_M`;
- continuous semantic score as a secondary metric;
- native fingerprinted teacher weights;
- ordinary prompt + Gaussian inference;
- no runtime key injection;
- no watermark hook;
- no post-processing;
- rank-32 attention+Conv teacher LoRA before merge;
- final merged standard SD2.1 pipeline;
- SD2.1 teacher / SD1.4 student relationship for later experiments.

Do not replace the semantic extractor with CLIP, DINO, a learned detector, or HMARK modules.

Do not introduce a two-phase watermark-learning pipeline.

Do not change the verifier in this revision.

Do not change the secret key after observing validation results.

---

# 3. New objective version

Add a new teacher objective:

`semantic_sign_v2_trajectory`

Historical objectives remain:

- `semantic_sign_v1`
- `threshold_cfg_v1`

Do not overwrite or silently migrate old runs.

Recommended constants:

```python
SIGN_OBJECTIVES = {
    "semantic_sign_v1",
    "semantic_sign_v2_trajectory",
}

SIGN_CARRIER_VERSION = "semantic_sign_v1"
```

Important:

The new teacher objective uses the **same RSSF carrier artifact format** as `semantic_sign_v1`.

Do not create a different carrier merely because the teacher objective changed.

Refactor any code that currently assumes:

```python
carrier_version == teacher_objective
```

The carrier/verifier version and teacher-training objective must now be logically separate.

`is_sign(config)` should return true for both sign objectives.

---

# 4. Verifier remains exactly the RSSF verifier

For a decoded image and prompt, the frozen semantic extractor returns median-centered coordinates:

\[
r(x,c)\in\mathbb R^{32}.
\]

The secret signs are:

\[
s_j\in\{-1,+1\}.
\]

Image bit:

\[
b_j(x,c)=
\begin{cases}
+1,&r_j\ge0\\
-1,&r_j<0.
\end{cases}
\]

Match count:

\[
K(x,c)=
\sum_{j=1}^{32}
\mathbf1[b_j=s_j].
\]

Null:

\[
K\mid H_0\sim Binomial(32,0.5)
\]

under the precommitted random-key null.

The exact 1% cutoff remains:

\[
K_{\text{detect}}=24.
\]

Do not modify this analytic detector.

Do not train against an empirical clean-score threshold.

Do not reroll the key.

---

# 5. Training fingerprint strength for this rescue run

The previous run used:

```text
K_detect = 24
train_bit_count_margin = 2
K_train = 26
mu candidates = [0.10, 0.20, 0.30]
```

All three teachers were already approximately 100% detectable.

For this rescue experiment reduce only the unnecessary training buffer:

```text
train_bit_count_margin = 0
K_train = 24
coordinate_margin_mu = 0.10
```

Rationale:

- the verifier itself still requires exactly 24/32 bits;
- current training with 26 target bits is demonstrably stronger than necessary;
- `mu=0.10` already yielded 99.8% detection;
- this run should not spend compute sweeping `mu` again.

Keep `train_bit_count_margin` as a parameter in the code.

Do not delete support for a nonzero margin.

For the first trajectory-rescue campaign, explicitly set it to zero.

---

# 6. Keep the existing easiest-bit hinge rule

The teacher fingerprint loss remains conceptually:

1. obtain **clean final-image RSSF coordinates**;
2. choose the easiest `K_train` bits from the clean image only;
3. detach that selection;
4. push only those same bits on the fingerprinted output toward the secret signs;
5. stop pushing a bit after its positive margin is reached.

For clean final coordinates:

\[
r^0_j.
\]

Clean aligned margins:

\[
a_j^0=s_jr_j^0.
\]

Select:

\[
I(x,c)=TopK(a^0,K_{\text{train}}).
\]

Detach `I`.

For fingerprint output:

\[
a_j^{fp}=s_jr_j(x_{fp},c).
\]

Loss:

\[
L_{fp}
=
\frac1{K_{\text{train}}}
\sum_{j\in I}
[\max(0,\mu-a_j^{fp})]^2.
\]

Defaults for this run:

```text
K_train = 24
mu = 0.10
```

Do not reintroduce:

- `kappa`;
- threshold-distance forcing;
- continuous fixed-alpha displacement;
- `L_perp`;
- a separate latent-cycle semantic loss.

---

# 7. Critical change: train on the native reverse trajectory

The major code change is to align teacher training with native generation.

The pinned SD2.1 model uses the saved native scheduler.

For the reviewed snapshot this is expected to be:

```text
PNDMScheduler
skip_prk_steps = true
prediction_type = epsilon
```

The implementation must read and verify the actual saved scheduler config at runtime.

Do not assume the scheduler class without checking.

For the first rescue implementation:

- support the pinned PNDM scheduler;
- fail clearly if the scheduler is unexpectedly different;
- do not silently approximate a different scheduler.

---

# 8. Why PNDM scheduler state must be cached

PNDM/PLMS is multistep.

A latent alone is not sufficient to restart the final denoising calls exactly.

The scheduler contains mutable state including, depending on the current call:

- `counter`;
- `ets`;
- `cur_model_output`;
- `cur_sample`.

The exact fields must be serialized/restored from the installed Diffusers scheduler object.

Do not manually assume that `ets` alone is sufficient.

Do not start the tail with a fresh scheduler that has only the boundary latent.

That would not reproduce native generation.

---

# 9. Why the differentiable tail is 4 scheduler calls

The pinned PNDM implementation uses a fourth-order PLMS / Adams-Bashforth history.

Therefore use:

```text
tail_unroll_calls = 4
```

This is deliberate.

It is not an arbitrary “last 4 diffusion timesteps”.

It means the **last four actual scheduler.step calls** in the native pipeline execution.

By the final call, the PLMS history can contain the fingerprinted teacher outputs from the short differentiable tail rather than remaining dominated by earlier clean outputs.

Do not hard-code diffusion timestep IDs.

Derive the tail by actual scheduler-call index.

---

# 10. Trace clean native generation instead of rewriting it

Do not create a second hand-written clean-generation implementation if avoidable.

The safest design is to instrument the ordinary native pipeline.

During clean `teacher_clean_train` generation:

1. use the normal `StableDiffusionPipeline`;
2. use the existing prompts/seeds/CFG/resolution/50-step settings;
3. temporarily wrap or instrument `pipe.scheduler.step`;
4. record required states without altering returned values;
5. allow the ordinary pipeline to finish normally;
6. save the same PNGs currently used by the experiment.

The traced generation and ordinary generation must produce identical final outputs.

Instrumentation must not change:

- latent values;
- UNet calls;
- scheduler calls;
- seeds;
- guidance scale;
- image postprocessing.

---

# 11. What to cache from clean training generation

Trace only the **training split**.

No trajectory cache is required for all validation images.

For every clean training image save:

## 11.1 Preservation anchors

Cache a small set of actual clean native states.

Let:

```text
C = number of scheduler.step calls in native 50-step generation
B = C - tail_unroll_calls
```

The tail begins at call index `B`.

Define five preservation anchor indices over the pre-tail region:

```text
anchor 0 = 0
anchor 1 = round(0.25 * (B - 1))
anchor 2 = round(0.50 * (B - 1))
anchor 3 = round(0.75 * (B - 1))
anchor 4 = B - 1
```

Deduplicate while preserving order.

For every anchor save:

- scheduler call index;
- timestep value passed to the UNet/scheduler;
- current clean latent input to the scheduler call;
- the **guided CFG model output actually passed to `scheduler.step`**.

The guided target is the correct preservation target.

Do not cache separate conditional and unconditional targets for preservation.

The native trajectory is controlled by the guided prediction.

## 11.2 Tail boundary

Immediately before scheduler call `B`, save:

- current latent;
- complete serializable scheduler mutable state;
- call index;
- current scheduler timestep list / protocol hash.

This is the starting state for differentiable tail training.

## 11.3 Final clean image

The ordinary PNG produced by the existing clean-generation pipeline remains the quality target.

Do not generate a second “clean target” during training.

---

# 12. Trace storage design

Avoid creating tens of thousands of tiny files.

Use sharded trace artifacts.

Recommended:

```text
trajectory_shard_size = 64
```

Suggested layout:

```text
teacher_trace/
    protocol.json
    index.json
    shard_00000.safetensors
    shard_00000.json
    shard_00001.safetensors
    shard_00001.json
    ...
    complete.json
```

Each shard must map image IDs to rows.

Tensor content should include, where shapes permit:

- anchor latents;
- anchor guided predictions;
- tail boundary latent;
- tensor-valued scheduler state;
- any tensor-valued PNDM history entries.

JSON metadata should include:

- scheduler class;
- scheduler config hash;
- inference step count;
- full scheduler timestep sequence;
- call count;
- tail boundary index;
- tail call indices;
- anchor indices;
- scalar scheduler state;
- `None` vs tensor distinctions;
- prompt/image IDs;
- source clean model hash;
- generation seed;
- dtype.

Use safe, deterministic serialization.

Hash every shard and freeze a complete manifest.

---

# 13. Trace precision

The trace should represent the actual native clean generation.

Store the tensors in the precision actually used by native teacher generation where practical.

Current native teacher generation uses FP16.

Do not quantize trajectory states beyond that.

Do not save lossy NumPy/compressed approximations.

When loaded for training, cast only as required by the trainable teacher computation.

The original stored trace remains immutable.

---

# 14. Exact replay test is mandatory

Before training any teacher, run a clean replay validation.

For several clean training samples:

1. load the cached tail boundary latent;
2. restore the cached scheduler state;
3. continue the final 4 calls using the **clean teacher**;
4. decode the final latent;
5. compare against the original clean generation.

Also test scheduler replay using cached guided outputs where useful.

The replay must reproduce the original clean tail within strict numerical tolerance.

Required checks:

- final latent finite;
- final image finite;
- latent max absolute error;
- image pixel error;
- LPIPS between replay and original clean image;
- same scheduler call count;
- same final counter/state progression.

Set strict tolerances based on deterministic FP16 execution.

The exact tolerance can be established by the implementation test, but it must be recorded in the protocol.

If replay is not sufficiently close, **abort before teacher training**.

Do not continue with an approximate cache.

---

# 15. Precompute clean final RSSF coordinates

After the carrier is built, score the clean **training** images with the normal final-image verifier.

Add:

```text
scores/teacher_clean_train.json
```

using the same scorer as other images.

These final clean coordinates are the source for easiest-bit selection during teacher training.

Do not compute easiest bits from:

- a predicted-x0 proxy;
- a forward-noised latent;
- an intermediate clean tail image.

Use the actual final clean PNG score.

This aligns bit selection with deployment verification.

---

# 16. Teacher training is still one phase

This revision is **not** a two-phase teacher method.

Each optimizer update uses two branches that contribute to the same teacher LoRA update:

```text
A. native-trajectory preservation branch
B. exact native-tail fingerprint + quality branch
```

The same trainable teacher weights are updated once per optimizer step.

---

# 17. Branch A — native-trajectory preservation

For each training sample:

1. randomly choose one cached preservation anchor;
2. load:
   - clean native latent `z_anchor`;
   - anchor timestep;
   - cached clean guided prediction `eps_clean_cfg`;
3. run the trainable fingerprint teacher on exactly the same latent and timestep;
4. compute CFG 7.5 exactly as native generation does;
5. compare the trainable guided prediction to the cached clean guided prediction.

Loss:

\[
L_{preserve}
=
MSE(
\epsilon^{cfg}_{fp},
\epsilon^{cfg}_{clean,cached}
).
\]

Important:

- no clean UNet forward is needed in this branch;
- cached clean target is detached;
- use the exact native state distribution, not forward-noised image latents;
- call `scheduler.scale_model_input` before the UNet exactly as the pipeline does;
- use the same CFG formula and text/unconditional embeddings as native generation.

This replaces the current full-range random-forward-noising preservation branch for `semantic_sign_v2_trajectory`.

Do not delete the old branch from historical v1 code.

---

# 18. Efficient CFG forward helper

Implement one audited helper that mirrors Stable Diffusion native CFG behavior.

Given a latent, timestep, conditional text embedding, unconditional text embedding:

1. duplicate latent for unconditional/conditional halves;
2. apply scheduler `scale_model_input`;
3. concatenate unconditional and conditional embeddings in native order;
4. run one UNet forward;
5. split unconditional / conditional predictions;
6. compute:

\[
\epsilon^{cfg}
=
\epsilon_u
+
7.5(\epsilon_c-\epsilon_u).
\]

Use this helper for both preservation and differentiable tail.

Do not maintain multiple slightly different CFG implementations.

Add a unit test comparing the helper to the current two-forward formulation.

---

# 19. Branch B — restore native tail boundary

For each training sample:

1. load the cached tail boundary latent;
2. instantiate a fresh scheduler from the pinned clean model config;
3. call normal `set_timesteps(inference_steps)`;
4. restore the cached mutable scheduler state exactly;
5. verify:
   - scheduler class;
   - config hash;
   - call index;
   - counter;
   - timestep sequence;
   - tensor shapes;
6. start at the first of the final four scheduler calls.

The cached history is treated as constant clean history.

New model outputs generated inside the four-call fingerprint tail must remain differentiable.

---

# 20. Branch B — differentiable four-call native tail

Run the final four actual PNDM scheduler calls using the trainable fingerprint teacher.

At each tail call:

1. use the current differentiable latent;
2. use the current scheduler timestep from the frozen native sequence;
3. call the common CFG helper;
4. pass guided output into the real scheduler `step`;
5. obtain the next latent;
6. do not detach.

The scheduler algebra must remain inside the PyTorch graph for the new fingerprint outputs.

Cached history tensors do not require gradients.

Do not substitute DDPM `predicted_x0` training math here.

Do not use `add_noise` for this branch.

This branch must model the actual final reverse process.

---

# 21. Decode the real tail output

After the fourth tail call:

\[
z^{fp}_{final}
\]

decode using the frozen clean SD2.1 VAE:

\[
x_{fp}
=
VAE.decode(
z^{fp}_{final}/scaling
).
\]

Use the existing differentiable image range convention.

Clamp to `[-1, 1]` before:

- LPIPS;
- pixel MSE;
- semantic verifier re-encoding.

Do not quantize to PNG during training.

---

# 22. Quality target is the actual clean PNG

Load the already generated clean training image with the same prompt and seed:

\[
x_{clean}.
\]

Convert it to the normal `[-1,1]` tensor.

It is a constant target.

Loss:

\[
L_{quality}
=
\lambda_{lpips}LPIPS(x_{fp},x_{clean})
+
\lambda_{pixel}MSE(x_{fp},x_{clean}).
\]

Keep first-run internal quality weights:

```text
lambda_lpips = 1.0
lambda_pixel = 1.0
```

No gradient through `x_clean`.

This is the central utility correction.

The quality objective now sees an image obtained from a genuine scheduler tail, not a predicted-x0 proxy.

---

# 23. Fingerprint loss is evaluated on the same final tail image

Pass `x_fp` through the same frozen final-image semantic path used by verification:

```text
image
-> frozen VAE posterior mean
-> frozen reference SD2.1 semantic UNet at t=0
-> mid-block GAP
-> PCA
-> standardize
-> median center
-> 32 RSSF coordinates
```

Use those coordinates in the existing easiest-bit hinge.

Load the clean training image coordinates from:

`scores/teacher_clean_train.json`

Select the easiest 24 bits from the clean final image.

Use:

```text
K_train = 24
mu = 0.10
```

No predicted-x0 semantic proxy is used.

No separate VAE-cycle loss is used.

---

# 24. Final v2 teacher objective

For the first rescue campaign:

\[
L_{total}
=
L_{preserve}
+
0.8L_{fp}
+
L_{quality}.
\]

And:

\[
L_{quality}
=
LPIPS+MSE.
\]

Default weights:

```text
lambda_preserve = 1.0
lambda_fp = 0.8
lambda_quality = 1.0
lambda_lpips = 1.0
lambda_pixel = 1.0
```

Do not start another multi-dimensional hyperparameter sweep.

These values preserve the current RSSF scale as much as possible while changing the training distribution/object.

The main hypothesis under test is trajectory alignment, not loss-weight tuning.

---

# 25. No mu candidate sweep in the rescue run

For `semantic_sign_v2_trajectory` first run:

```text
coordinate_margin_mu = 0.10
```

One teacher training run only.

Do not launch `.20` and `.30`.

Do not use the current `run_workers` three-candidate campaign for v2.

The previous experiment already established that increasing `mu` does not solve utility.

---

# 26. Reduce over-embedding by checkpoint selection

The current result may also be over-trained after detection has already saturated.

For v2:

```text
teacher_train_steps = 1500
```

Save full resume checkpoints at least every 100 updates.

Create teacher-evaluation milestones at:

```text
[100, 250, 500, 750, 1000, 1500]
```

Do not select based on training proxy metrics.

After training, evaluate these saved teacher adapters using **actual native 50-step generation**.

---

# 27. Checkpoint screening set

Use held-out validation prompts and fixed paired seeds.

Recommended:

```text
validation_n = 512
checkpoint_screen_n = 128
```

The 128 screening samples must be a deterministic fixed subset of the 512 validation split.

For each evaluation checkpoint:

1. load clean SD2.1;
2. load/fuse the teacher LoRA checkpoint in memory or into a temporary immutable merged snapshot;
3. generate 128 standard native images with CFG 7.5 / 50 steps / FP16;
4. use identical prompts and seeds as clean validation;
5. calculate:
   - analytic RSSF TPR;
   - mean / median `K`;
   - model-level `K_M` / `p_M`;
   - paired LPIPS;
   - paired pixel MSE;
   - CLIP / CLIP drop;
   - continuous mean score.

No student is involved.

---

# 28. Teacher checkpoint selection rule

A checkpoint is eligible for selection if:

```text
analytic per-image TPR >= 0.90
CLIP drop <= 0.01
```

Among eligible checkpoints choose:

1. lowest paired LPIPS;
2. then lowest paired pixel MSE;
3. then earlier checkpoint as the final tie-break.

The final success gate is stricter:

```text
analytic per-image TPR >= 0.90
CLIP drop <= 0.01
paired LPIPS <= 0.15
```

If the best eligible checkpoint has LPIPS > 0.15:

- keep it as the best diagnostic checkpoint;
- mark the teacher campaign as failed;
- do not start students.

Do not choose a later checkpoint merely because TPR is 100% instead of 93%.

Once TPR is above the required gate, utility has priority.

---

# 29. Full selected-teacher validation

After selecting the best checkpoint from the 128-image screen:

1. create the final standard merged SD2.1 pipeline;
2. generate all `validation_n = 512` paired validation images natively;
3. rescore with the frozen RSSF verifier;
4. recompute paired utility;
5. enforce the final success gates.

The screening result is not sufficient by itself.

The selected full-validation metrics are the teacher result to inspect before any student experiment.

---

# 30. Do not run students in this job

Add:

```text
--teacher-only
```

For a `semantic_fp` RSSF run, this mode must stop after:

- clean preparation;
- carrier;
- clean verifier scoring/null checks;
- trajectory cache;
- v2 teacher training;
- checkpoint screening;
- selected teacher merge;
- full teacher validation;
- teacher-only report.

Do not run:

- unadapted SD1.4 generation;
- student calibration;
- LoRA student training;
- full student training;
- student utility;
- student checkpoint selection.

Write:

```text
teacher_complete.json
```

or equivalent with:

```json
{
  "status": "teacher_only_completed",
  "downstream_started": false
}
```

If the teacher fails utility gates, record:

```text
teacher_only_failed
```

and stop cleanly.

---

# 31. Keep the run resumable under a 24-hour scheduler limit

The cluster permits at most 24 hours per job.

The v2 implementation must remain restartable.

Requirements:

- clean-image generation is resumable;
- trace shards are immutable once complete;
- carrier stage recognizes complete artifacts;
- teacher training restores:
  - adapter;
  - AdamW;
  - scaler if any;
  - sampler order/offset;
  - CPU RNG;
  - CUDA RNG;
  - Python RNG;
  - completed optimizer step;
- checkpoint evaluation recognizes already completed milestone results;
- selected full validation recognizes already generated images.

A killed job must be restartable with:

```bash
bash run_semantic_fp.sh --resume RUN_DIR
```

without regenerating completed clean images/traces or restarting teacher training from zero.

Do not rely on finishing everything inside one uninterrupted process.

---

# 32. Teacher-only clean data sizes

Keep:

```text
train_n = 5000
```

because:

- carrier fitting already works with this scale;
- teacher training already uses this dataset size;
- do not change another scientific variable during the trajectory rescue.

For teacher-only development use:

```text
validation_n = 512
calibration_n = 1000
```

The analytic verifier does not require 5000 calibration images to define its cutoff.

The clean calibration split is only for:

- empirical fixed-key FPR validation;
- secondary continuous-score calibration/comparison.

Do not generate teacher/student test sets in teacher-only mode.

---

# 33. Generate traces in the same pass as clean train images

Do not run a second 50-step clean generation over all 5000 train prompts merely to make traces.

Modify the clean train generation stage so that:

```text
one native generation pass
    -> final PNG
    -> trajectory trace
```

are produced together.

This is important for the 24-hour budget.

Validation and calibration clean images do not need full training traces.

---

# 34. Clean trace instrumentation integrity

The trace recorder must prove that it is observational only.

Required test:

- same prompt;
- same seed;
- same clean model;
- ordinary `native_generate`;
- trace-instrumented native generation.

The final latent/image must match.

If wrapping `scheduler.step`, always call the original method exactly once and return its original result unchanged.

Do not modify scheduler state before calling the original clean step.

Capture “state before step” via detached copies.

---

# 35. Scheduler state serialization details

Implement explicit serializer/deserializer for PNDM.

At minimum support and validate:

```text
counter
ets
cur_model_output
cur_sample
num_inference_steps
timesteps
```

Any additional mutable state in the installed scheduler version that affects `step()` must also be included.

For fields:

- preserve `None`;
- preserve scalar-vs-tensor distinction;
- preserve list ordering;
- preserve tensor shape/dtype.

Do not use Python `deepcopy(scheduler)` as the scientific artifact format.

Do not pickle the whole scheduler object.

Store declarative state plus model/config hashes.

---

# 36. PLMS boundary test

Because the v2 tail starts near the end:

- `counter` must indicate PLMS mode;
- the tail must not enter PRK bootstrap;
- cached history must be sufficiently populated.

Assert these conditions.

For the pinned SD2.1 config with `skip_prk_steps=true`, the final tail is expected to be PLMS.

If not, fail and inspect rather than silently changing behavior.

---

# 37. Preserve gradient through scheduler.step

The restored history is detached.

However the new fingerprint teacher outputs inside the four tail calls must remain in autograd.

Do not wrap the tail scheduler loop in `torch.no_grad()`.

Do not detach intermediate latents.

Do not convert the scheduler outputs to NumPy.

Add a unit test proving nonzero finite gradients reach trainable LoRA parameters through all four scheduler calls and final VAE/LPIPS/fingerprint losses.

---

# 38. Memory strategy

Keep:

```text
teacher_micro_batch = 1
effective teacher batch = 4
```

For each microbatch:

1. compute preservation branch;
2. backward preservation;
3. release preservation graph;
4. load tail state;
5. unroll four calls;
6. decode / fingerprint / quality;
7. backward late loss;
8. release tail graph.

Perform one optimizer update after effective-batch accumulation.

Use gradient checkpointing for:

- trainable UNet;
- frozen semantic extractor path;
- VAE decode/encode where already supported;
- LPIPS if needed.

Do not keep four samples’ four-step tail graphs alive simultaneously.

---

# 39. Teacher precision

Keep current teacher training default:

```text
BF16
```

Do not change precision in the same rescue experiment.

Native checkpoint screening/final validation remains:

```text
FP16
```

as the standard pipeline currently uses.

The final scientific teacher metrics always come from native merged FP16 inference, not the BF16 training graph.

---

# 40. Trajectory-drift diagnostic

Add an explicit diagnostic after a teacher checkpoint is selected.

For a small fixed subset, recommended:

```text
trajectory_drift_n = 16
```

generate both clean and selected fingerprint teacher with tracing.

At each scheduler call compute a normalized latent distance, e.g.:

\[
D_i
=
\frac{
\|z_i^{fp}-z_i^{clean}\|_2
}{
\sqrt{\text{numel}(z_i)}
}.
\]

Report mean/median drift over the 16 examples per call.

Save:

```text
result/teacher_trajectory_drift.json
result/teacher_trajectory_drift.csv
```

This is diagnostic only.

Do not use it for checkpoint selection.

Expected qualitative behavior of a good teacher:

- very small early/mid trajectory drift;
- modest increase near the final calls;
- good final image fidelity.

If drift is already large early, preservation is failing.

---

# 41. Training diagnostics

For every logged teacher step record:

## Preservation

- selected anchor index;
- anchor timestep;
- preservation MSE;
- guided FP prediction norm;
- guided clean target norm.

## Tail

- boundary call index;
- four tail timestep values;
- scheduler starting counter;
- scheduler ending counter;
- final latent norm.

## Fingerprint

- `K_train`;
- `mu`;
- number of selected clean bits already correct;
- FP final-image `K` on the differentiable view;
- selected-bit margin fraction;
- mean selected signed margin;
- fingerprint hinge loss.

## Quality

- LPIPS;
- pixel MSE;
- quality loss.

## Optimization

- total loss;
- gradient norm;
- finite checks;
- peak GPU memory.

Do not use training proxy `K`/LPIPS to select the final teacher checkpoint.

Native validation remains authoritative.

---

# 42. Evaluation images and visual inspection

For every screened teacher checkpoint save a small deterministic visual grid/pairs.

At least:

```text
8 clean/fingerprint paired validation samples
```

For the selected teacher save:

```text
16 paired samples
```

Include:

- clean image;
- fingerprint image;
- optional absolute-difference visualization.

Do not use visual inspection to choose among checkpoints outside the frozen numeric selection rule.

Visuals are for debugging/reporting.

---

# 43. Do not compute heavy student/final-paper utility yet

Teacher-only rescue metrics should focus on:

- analytic TPR;
- fixed-key clean FPR;
- model-level bit evidence;
- paired LPIPS;
- paired pixel MSE;
- CLIP / CLIP drop;
- trajectory drift;
- paired images.

Do not spend this 24-hour run on:

- student LoRA;
- student full FT;
- full paper FID/KID sweeps;
- attack robustness;
- new models.

FID/KID can be added after a teacher passes the paired quality gate.

---

# 44. Teacher-only success definition

The teacher rescue is a success only if the final selected merged teacher, on the full 512 held-out validation images, satisfies:

\[
\boxed{
\text{analytic TPR}\ge0.90
}
\]

\[
\boxed{
\text{CLIP drop}\le0.01
}
\]

\[
\boxed{
\text{paired LPIPS}\le0.15
}
\]

Also report:

- paired pixel MSE;
- model `K_M / 32`;
- `p_M`;
- mean continuous score;
- empirical clean fixed-key FPR.

Do not relax these gates automatically.

---

# 45. If the teacher still fails

If v2 still produces something like:

```text
TPR >= 90%
LPIPS ~ 0.4
```

despite exact native-tail quality optimization and native-trajectory preservation:

- save all diagnostics;
- stop;
- do not launch students;
- do not automatically change the carrier;
- do not automatically reroll the key;
- do not automatically increase/decrease mu.

At that point the evidence would be much stronger that the selected semantic sign carrier directions themselves have high perceptual cost.

That is a future design decision, not part of this implementation.

---

# 46. Code organization recommendation

Preserve `semantic_sign_v1`.

Recommended new file:

```text
Malicious_quantization/bitmark_lora/semantic_sign_trajectory_train.py
```

Suggested responsibilities:

- trace protocol helpers;
- PNDM state serializer/deserializer;
- trace-shard reader;
- CFG helper;
- preservation loss;
- differentiable tail replay;
- v2 teacher train loop;
- checkpoint screening;
- selected teacher merge.

Keep verifier math in:

```text
semantic_sign.py
```

Do not duplicate Binomial verifier code.

Keep generic teacher image generation in:

```text
semantic_teacher.py
```

Add trace instrumentation there or in a small dedicated helper module such as:

```text
semantic_trajectory.py
```

Recommended separation:

```text
semantic_trajectory.py
    native trace recorder
    scheduler state serialization
    replay helpers
    trajectory protocol validation

semantic_sign_trajectory_train.py
    optimization and teacher checkpoint selection
```

---

# 47. Required `run.py` changes

Add teacher objective choice:

```text
semantic_sign_v2_trajectory
```

Add flag:

```text
--teacher-only
```

Add or freeze v2 parameters:

```text
--tail-unroll-calls 4
--trajectory-anchor-count 5
--teacher-eval-checkpoints 100 250 500 750 1000 1500
--teacher-eval-n 128
--trajectory-drift-n 16
```

For the first v2 campaign enforce:

```text
num_bits = 32
stable_candidate_count = 48
analytic_alpha = 0.01
train_bit_count_margin = 0
coordinate_margin_mu = 0.10
teacher_train_steps = 1500
tail_unroll_calls = 4
```

Do not allow the old v1 `mu` candidate-list worker to run for v2.

In teacher-only mode orchestration should be conceptually:

```text
01 assets
02 clean train + train trajectory traces
03 clean validation
04 clean calibration
05 carrier
06 clean train/calibration/validation scoring
07 analytic verifier/null report
08 v2 teacher training
09 native checkpoint screen
10 selected teacher merge
11 selected full teacher validation
12 trajectory drift diagnostic
13 teacher-only report
STOP
```

---

# 48. Avoid unnecessary student dependency during execution

It is acceptable for `assets.json` to retain the existing student metadata for future continuation.

However `--teacher-only` must not require loading the student model onto GPU or generating any student images.

Do not let student stages run accidentally after teacher completion.

---

# 49. Resume semantics

Historical `--resume` behavior should remain strict.

For v2, resume must verify:

- code manifest;
- config hash;
- clean image hashes;
- trace manifest hashes;
- carrier hash;
- key hash;
- teacher checkpoint protocol;
- scheduler config hash.

A trace generated under a different scheduler config or inference-step count must be rejected.

Do not silently rebuild only part of a trace under the same run directory.

---

# 50. Backward compatibility

These must continue working:

- old `threshold_cfg_v1` analysis;
- old `semantic_sign_v1` analysis;
- old RSSF carrier loading;
- old result reports.

Do not reinterpret old loss logs.

Do not modify old run artifacts.

The new teacher objective should have a distinct protocol field:

```json
{
  "teacher_objective": "semantic_sign_v2_trajectory"
}
```

---

# 51. Required unit tests

## 51.1 Sign verifier regression

All existing RSSF verifier tests must continue to pass.

Especially:

```text
Binomial(32,.5), alpha=.01 -> K_detect=24
```

## 51.2 Objective classification

`is_sign(config)` must return true for:

- `semantic_sign_v1`;
- `semantic_sign_v2_trajectory`.

Carrier version loading must remain compatible.

## 51.3 Trace instrumentation transparency

Ordinary clean generation and trace-instrumented generation with the same model/prompt/seed must match.

## 51.4 PNDM serializer roundtrip

Serialize and restore:

- `counter`;
- `ets`;
- `cur_model_output`;
- `cur_sample`;
- required scalar protocol fields.

Restored state must produce the same next scheduler result when given the same model output.

## 51.5 Clean tail replay

Cached boundary + restored scheduler + clean teacher final four calls must reproduce the original clean tail.

## 51.6 Cached guided preservation target

At a cached anchor:

- run clean teacher again;
- compute guided prediction;
- compare to cached clean guided output.

Must match within declared tolerance.

## 51.7 CFG helper equivalence

Single batched CFG helper must match the previous explicit unconditional + conditional two-forward implementation.

## 51.8 Gradient-through-tail test

A synthetic/real small batch must show finite nonzero LoRA gradients from:

- final fingerprint loss;
- final LPIPS;
- final pixel MSE;

through all four scheduler calls.

## 51.9 No gradient through targets

No gradient through:

- cached clean guided predictions;
- cached boundary history;
- clean PNG;
- clean train RSSF coordinates.

## 51.10 TopK selection

Easiest bits are selected from actual clean final image coordinates and detached.

## 51.11 Teacher-only orchestration

`--teacher-only --plan-only` must show no student stages.

## 51.12 Resume

A short interrupted v2 training test must resume to the same parameters/optimizer state as uninterrupted execution under deterministic CPU/test conditions where feasible.

---

# 52. Numerical safety

Retain the current project’s strong finite-value guards.

Add checks for:

- loaded trace tensors;
- restored scheduler tensors;
- each tail latent;
- each guided model output;
- decoded final image;
- semantic coordinates;
- LPIPS;
- fingerprint loss;
- gradient norm.

If a scheduler state is inconsistent, fail loudly.

Do not skip an invalid optimizer update and continue silently.

---

# 53. First-run locked configuration

Use this configuration for the first v2 teacher-only rescue unless hardware requires a documented microbatch-only change:

```text
watermark_method = semantic_fp
teacher_objective = semantic_sign_v2_trajectory
teacher_only = true

teacher = pinned SD2.1 Base
native scheduler = pinned saved scheduler
inference_steps = 50
guidance_scale = 7.5

train_n = 5000
validation_n = 512
calibration_n = 1000

carrier_pc_min = 8
carrier_pc_max = 64
stable_candidate_count = 48
num_bits = 32
key_seed = 0

analytic_alpha = 0.01
K_detect = derived exact Binomial cutoff = 24

train_bit_count_margin = 0
K_train = 24
coordinate_margin_mu = 0.10

tail_unroll_calls = 4
trajectory_anchor_count = 5

teacher_rank = 32
teacher_adapter_scope = attention_conv
teacher_lr = 1e-4
teacher_train_steps = 1500
teacher_batch = 4
teacher_micro_batch = 1
teacher_precision = bf16
teacher_checkpoint_every = 100

teacher_eval_checkpoints = [100,250,500,750,1000,1500]
teacher_eval_n = 128
trajectory_drift_n = 16

lambda_preserve = 1.0
lambda_fp = 0.8
lambda_quality = 1.0
lambda_lpips = 1.0
lambda_pixel = 1.0

teacher_min_analytic_tpr = 0.90
teacher_clip_drop_max = 0.01
teacher_lpips_max = 0.15
```

Do not add a second `mu`.

Do not add student training.

Do not add a loss-weight sweep.

---

# 54. Why this is the minimal scientifically justified revision

The evidence from the completed RSSF run already establishes:

```text
fingerprint representation: works
analytic per-image verifier: works
teacher detectability: works
```

The failure is:

```text
paired image utility: fails
```

The current teacher loss optimizes quality on:

```text
independent noisy latent -> predicted_x0 proxy
```

while deployment quality is determined by:

```text
full native reverse trajectory
```

Therefore this revision changes only the part not yet validated:

```text
teacher embedding / preservation distribution
```

It preserves all successful contributions.

---

# 55. Expected interpretation of a successful run

A successful result would show:

```text
clean teacher trajectory
    ~ preserved over early/mid calls

final four native calls
    -> small coherent RSSF sign bias

final decoded image
    -> >=90% analytic per-image detection
    -> low paired LPIPS
    -> low CLIP drop
```

The final teacher is still:

```text
ordinary merged SD2.1
```

and requires no special inference path.

The trajectory cache is **training infrastructure only**.

It is never required at deployment or verification.

---

# 56. Explicit non-goals

Do not implement in this rescue:

- student LoRA;
- student full fine-tuning;
- HMARK code;
- learned detectors;
- a new PCA space;
- new semantic encoders;
- prompt-conditioned keys;
- new secret generation;
- key rerolling;
- different bit count;
- a 24/25/26 sweep;
- another mu sweep;
- watermark attack robustness;
- JPEG/resize robustness;
- new teacher/student architectures;
- final paper FID/KID benchmark;
- full 50-step differentiable teacher training;
- two-phase teacher optimization.

---

# 57. Final Codex instruction

Implement **one coherent revision only**:

> Keep RSSF carrier and analytic verifier unchanged. Replace the `semantic_sign_v1` predicted-x0 teacher embedding proxy with a new versioned `semantic_sign_v2_trajectory` teacher objective that preserves the fingerprint teacher on cached **actual clean native trajectory states**, and computes RSSF + LPIPS + pixel fidelity through an **exact differentiable replay of the final four PNDM scheduler calls** from a cached native boundary state. Train one `mu=0.10`, `K_train=24` teacher for up to 1500 steps, select the best checkpoint by native held-out teacher utility subject to analytic TPR >= 90%, evaluate the selected merged teacher on 512 validation images, and stop before any student training.

Do not redesign anything outside that scope.
