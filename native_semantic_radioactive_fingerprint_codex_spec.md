# Native Semantic Radioactive Fingerprint for Diffusion Teachers
## Codex implementation specification — MVP, method design, experiments, and coding strategy

**Repository:** `ducthinhh1002/Malicious_quantization`  
**Primary reuse target:** the current SERUM diffusion-teacher experiment harness under `bitmark_lora/`  
**Goal of this document:** give Codex an implementation plan precise enough that it does **not** accidentally change the scientific question, improve the downstream student instead of the fingerprint, use HMARK as a hidden target generator, or introduce a runtime watermark injection.

---

# 0. The one-sentence goal

We want to fingerprint an SD2.1 teacher **in its weights** so that ordinary generation

```text
prompt + ordinary Gaussian noise -> fingerprinted SD2.1 -> image
```

naturally produces images carrying a **small, coherent, keyed semantic distribution shift**. A fresh SD1.4 student trained only on those image-caption pairs should inherit the same signal under ordinary LoRA/full fine-tuning.

The intended lineage chain is:

\[
M_T
\;\xrightarrow{\text{teacher-side fingerprint training}}\;
M_T^{fp}
\;\xrightarrow{\text{ordinary generation}}\;
\{(c_i,x_i^{fp})\}
\;\xrightarrow{\text{ordinary student training}}\;
M_S
\;\xrightarrow{\text{ordinary generation}}\;
x_S
\;\xrightarrow{\text{fixed verifier}}\;
\text{lineage detected}.
\]

The student must **never** receive the key, verifier, detector loss, watermark loss, teacher weights, or any special inference-time latent.

---

# 1. Scientific motivation

The important observation from HMARK is not its exact encoder/decoder implementation. The useful principle is:

> A small **coherent semantic-latent distribution shift**, repeated consistently across a protected dataset, can be highly radioactive because downstream diffusion training has an incentive to learn the shifted distribution.

HMARK applies such a semantic shift **post hoc to images**. We do **not** want to do that.

Our intended contribution is different:

> Make the teacher model itself natively emit outputs with a keyed coherent semantic shift, then test whether the signal passively transfers to a separately trained downstream diffusion model.

So the desired conceptual difference is:

```text
HMARK-like data-side idea:
clean image -> semantic watermark transform -> protected image

Our model-native idea:
clean teacher -> fingerprint training -> fingerprinted teacher
fingerprinted teacher(prompt, normal noise) -> protected image directly
```

No HMARK-generated target dataset is allowed.

---

# 2. Hard constraints — Codex must not violate these

These are scientific invariants, not implementation preferences.

1. **Do not use HMARK code, checkpoints, watermark encoder, decoder, or HMARK-watermarked images to train our teacher.**
2. **Do not use SERUM's watermark grid or SERUM detector in the new method.**
3. **Do not inject any key, special noise, latent tensor, hook, post-processing transform, or watermark module at teacher inference after the teacher has been fingerprinted.**
4. **Do not modify the evaluated student adaptation to make inheritance easier.**
5. **Do not give the student a fingerprint loss, key, detector, verifier, or special data labels.**
6. **Do not select student checkpoints using fingerprint detection.** Keep the existing SERUM policy: select by held-out denoising MSE only.
7. **Do not select method hyperparameters on the final student test set.**
8. **Do not claim that expanding LoRA rank/scope is a fingerprint improvement.** Student adaptation is the evaluation environment.
9. **Do not overwrite or break the existing SERUM experiment.** Add a new method path.
10. **Do not silently reuse the existing `new_watermark/` method as the proposed method.** Utilities may be reused only when behavior matches this document exactly.

A useful sanity sentence for every code review is:

> **All creativity is on the teacher/fingerprint side; downstream training remains ordinary and frozen.**

---

# 3. Reuse the SERUM experimental harness

The reason to reuse the existing harness is to get a clean comparison and avoid rewriting student training/evaluation code.

The current audited diffusion setup in `bitmark_lora/README.md` is the comparison target:

| Component | Frozen setting for direct SERUM comparison |
|---|---|
| Teacher family | SD2.1 base |
| Student | `CompVis/stable-diffusion-v1-4`, pinned revision already used by repo |
| Student branches | attention LoRA r16/r32; attention+Conv2d LoRA r16/r32; DoRA attention+conv r32; full UNet |
| Student budget | 7,500 optimizer steps |
| Checkpoints | 2,500 / 5,000 / 7,500 |
| Student train images | 5,000 teacher-generated images |
| Student batch | 8 |
| LoRA LR | `1e-5` |
| Full-UNet LR | `5e-6` |
| Optimizer | AdamW |
| Resolution | 512 |
| Validation set | 1,000 images, disjoint |
| Validation draws | 2 fixed noise/timestep draws per image |
| Validation generation | first 128 validation prompts for diagnostics |
| Clean calibration | 5,000 prompts |
| Final test | 1,000 prompts |
| Checkpoint selection | minimum held-out denoising MSE; earlier checkpoint wins ties |
| Inference | 50 steps, CFG 7.5, ordinary random generators |
| Primary detection operating point | empirical 1% FPR |

**Important:** the repository also contains older/newer SERUM reproduction queues with different settings such as rank 4 / 20k steps. Those are **not** the direct comparison table we want here. For this new method, use the `bitmark_lora` SERUM harness defaults above unless the user explicitly changes the experimental protocol.

---

# 4. Core method: keyed semantic shift learned into the teacher

We need three objects:

1. a **fixed semantic feature extractor** \(\Phi\),
2. a secret **carrier subspace + bit code** \((U_k,s_k)\),
3. a fingerprinted teacher \(M_\theta\) trained to create a nearly constant shift along those directions.

## 4.1 Fixed semantic feature extractor

Use a frozen clean SD2.1 reference model. Do not train a detector head.

For an image \(x\) and its caption/prompt \(c\):

1. Convert image to the reference SD2.1 VAE input range.
2. Deterministically encode with the **posterior mean**, not a stochastic posterior sample:
   \[
   z_0 = \mathrm{VAEmean}(x)\cdot \text{scaling\_factor}.
   \]
3. Use the frozen SD2.1 text tokenizer/text encoder for \(c\).
4. Run the frozen SD2.1 UNet at a fixed probe timestep:
   \[
   t_{\text{probe}} = 0.
   \]
5. Capture the output of `unet.mid_block`.
6. Global-average-pool the spatial dimensions:
   \[
   f(x,c)=\mathrm{GAP}(h_{\text{mid}})\in\mathbb R^d.
   \]

For SD-style UNets, \(d\) will normally be the mid-block channel dimension. Do **not** hard-code the dimension; infer and assert it.

Implementation recommendation:

```python
with mid_block_forward_hook(ref_unet) as capture:
    _ = ref_unet(z0, t_probe, encoder_hidden_states=text_hidden).sample
    h = capture.output
f = h.float().mean(dim=(-2, -1))
```

The reference UNet, VAE, tokenizer, and text encoder must be `.eval()` and `requires_grad_(False)`.

The extractor is fixed **before** fingerprinted teacher training and is used unchanged for teacher outputs and student outputs.

### Why no learned detector?

We do not want:

```text
create arbitrary distribution shift -> train a classifier to discover it
```

The fingerprint must be tied to the same explicit semantic coordinates used during embedding.

---

# 5. Build the semantic carrier from clean teacher data

Use the existing SERUM prompt split machinery. The carrier may be fit on the **clean teacher train split** because this is unsupervised basis estimation; the threshold calibration must remain disjoint.

Generate the normal clean SD2.1 train images using:

```text
5,000 train prompts
ordinary Gaussian seeds
50 inference steps
CFG 7.5
512x512
```

For each `(image, caption)` compute \(f_i=\Phi(x_i,c_i)\).

Compute:

\[
\mu=\frac1N\sum_i f_i.
\]

Run PCA/SVD on centered features:

\[
f_i-\mu.
\]

Save:

```text
carrier_mean
PCA basis
PCA eigenvalues / variances
feature dimension
reference model provenance
prompt split hash
image hashes
probe timestep
exact preprocessing
```

Do this in FP32 for statistics.

---

# 6. Choose stable semantic directions

We do not want the top PCs that may correspond to very strong visual semantics, and we do not want tiny tail PCs that are fragile.

Default candidate range:

```text
PC indices 8..64
```

(Use zero- or one-based naming consistently in code and manifest.)

For every candidate PC, estimate **VAE-cycle stability**.

For a clean image \(x\):

```text
x -> reference VAE mean encode -> decode -> reconstructed image x_cycle
```

Compute the PCA coordinate on both:

\[
a_j(x)=v_j^\top(f(x,c)-\mu)
\]

\[
a_j(x_{\rm cycle})=v_j^\top(f(x_{\rm cycle},c)-\mu).
\]

For each candidate PC record at least:

- Pearson correlation,
- sign retention,
- normalized MSE,
- clean standard deviation.

Rank candidate PCs by a pre-declared stability metric, preferably correlation first with sign retention as tie-break.

Take the top **16 stable candidates** from PC 8..64. This set is chosen using only clean data.

Do **not** look at student results when choosing the carrier.

---

# 7. Secret key and code

Default experimental key size:

```text
num_bits = 8
```

A secret integer `key_seed` is used to:

1. choose 8 directions from the 16 stable candidates without replacement,
2. generate 8 independent signs \(s_j\in\{-1,+1\}\).

Let the selected normalized PCA vectors be:

\[
U_k=[u_1,\ldots,u_8]\in\mathbb R^{d\times 8}.
\]

Let the secret code be:

\[
s_k=[s_1,\ldots,s_8]\in\{-1,+1\}^8.
\]

For each selected coordinate, store the clean standard deviation:

\[
\sigma_j.
\]

Define normalized semantic coordinates:

\[
q_j(x,c)=\frac{u_j^\top(f(x,c)-\mu)}{\sigma_j+\epsilon}.
\]

Use a small numerical epsilon only to avoid division by zero; assert all selected \(\sigma_j\) are comfortably nonzero.

The key artifact must be frozen before teacher fingerprint optimization:

```json
{
  "key_seed": 0,
  "num_bits": 8,
  "selected_pc_indices": [],
  "signs": [],
  "sigmas": [],
  "basis_sha256": "...",
  "carrier_fit_prompt_sha256": "...",
  "selection_rule": "top stable candidates then RNG choose 8"
}
```

Do not regenerate the key on resume.

---

# 8. What the fingerprint should mathematically do

The key idea is **not** to force every image to have exactly the same absolute semantic coordinates. That would collapse natural variation.

Instead, the fingerprinted teacher should create approximately the **same semantic displacement relative to the clean teacher**.

For a sample \(c,z_t,t\), let the frozen clean teacher predict:

\[
\epsilon_0=M_0(z_t,t,c),
\]

and the trainable fingerprint teacher predict:

\[
\epsilon_\theta=M_\theta(z_t,t,c).
\]

Convert predictions to predicted clean latents using the scheduler's exact prediction-type-aware helper:

\[
\hat z_0^0=\mathrm{pred\_x0}(z_t,\epsilon_0,t),
\]

\[
\hat z_0^\theta=\mathrm{pred\_x0}(z_t,\epsilon_\theta,t).
\]

Do not hand-code epsilon-only algebra if the scheduler may use another prediction type.

Define a latent semantic extractor \(\Phi_z\) identical to the image extractor after VAE encoding: it receives a correctly scaled SD2.1 clean latent and runs the frozen reference UNet at `t_probe=0`, then captures/averages the mid-block feature.

Compute:

\[
f_0=\Phi_z(\hat z_0^0,c),
\qquad
f_\theta=\Phi_z(\hat z_0^\theta,c).
\]

The normalized shift on selected directions is:

\[
\Delta q
=
\operatorname{diag}(\sigma)^{-1}
U_k^\top(f_\theta-f_0).
\]

The desired fingerprint is:

\[
\boxed{\Delta q\approx \alpha s_k.}
\]

This is the cleanest analogue of a coherent HMARK-like semantic shift:

\[
h \longrightarrow h+\Delta h_k.
\]

---

# 9. Teacher-side fingerprint training

## 9.1 Parameterization

Start from clean SD2.1.

Train a **teacher-side LoRA** only as an efficient parameterization. After training, merge it into the UNet and save a normal standard SD2.1-compatible checkpoint.

Recommended default:

```text
teacher adapter: LoRA rank 32
scope: attention + Conv2d
LoRA alpha: rank
teacher LR: 1e-4
batch: 4
optimizer: AdamW
gradient clipping: 1.0
mixed precision: fp16/bf16 as supported
training steps: 3,000 initially
```

Reuse the already-tested attention+Conv2d target-selection code in this repository rather than inventing a second LoRA implementation.

The final deployed object is **not** “SD2.1 + runtime adapter + key.” It is a merged fingerprinted UNet/checkpoint.

---

## 9.2 Training data

Use the same 5,000 clean teacher-generated `(image, caption)` pairs from the train split.

For each training image:

1. deterministic VAE mean encode to clean latent \(z_0\),
2. sample diffusion timestep and ordinary Gaussian training noise,
3. form \(z_t\) with the SD2.1 training scheduler,
4. run clean and fingerprint teacher on the same \(z_t,t,c\).

The base teacher forward is `no_grad`.

The fingerprint teacher forward carries gradients.

---

## 9.3 Losses

### A. Coherent semantic shift loss

Default target shift:

```text
semantic_shift alpha = 0.5 clean-standard-deviations per selected dimension
```

Use:

\[
L_{\rm code}
=
\frac1B
\left\|
\Delta q-\alpha s_k
\right\|_2^2.
\]

This is preferable to forcing absolute coordinates because it preserves image-dependent semantics while adding a nearly constant key-dependent displacement.

### B. Teacher preservation loss

Keep the denoising behavior near the clean teacher:

\[
L_{\rm preserve}
=
\operatorname{MSE}(\epsilon_\theta,\epsilon_0).
\]

### C. Orthogonal-drift penalty

The fingerprint should preferably change the semantic representation mainly inside the selected carrier subspace.

Let:

\[
\Delta f=f_\theta-f_0.
\]

Projection into key subspace:

\[
P_U\Delta f = U_kU_k^\top\Delta f.
\]

Penalize:

\[
L_\perp
=
\frac1d
\|
\Delta f-P_U\Delta f
\|_2^2.
\]

This reduces accidental style/semantic drift unrelated to the fingerprint.

### D. VAE-cycle survival loss

A signal that exists only in predicted latent space but disappears after decode/re-encode is useless.

Every `cycle_every` steps (default 4), perform a differentiable cycle for the fingerprinted predicted clean latent:

```text
z0_hat_fp
-> frozen VAE decode
-> image tensor
-> frozen VAE posterior mean encode
-> scaled latent
-> frozen semantic extractor
```

Do the clean branch without gradients.

Compute the same normalized shift after the image channel:

\[
\Delta q_{\rm cycle}
\]

and:

\[
L_{\rm cycle}
=
\|\Delta q_{\rm cycle}-\alpha s_k\|_2^2.
\]

This term is expensive, so it need not run every optimizer step.

---

## 9.4 Total loss

Initial recommendation:

\[
L
=
\lambda_pL_{\rm preserve}
+
\lambda_cL_{\rm code}
+
\lambda_\perp L_\perp
+
\lambda_{\rm cycle}L_{\rm cycle}.
\]

Start with:

```text
lambda_preserve = 1.0
lambda_code     = 0.10
lambda_perp     = 0.01
lambda_cycle    = 0.05
cycle_every     = 4
```

These are **initial engineering defaults**, not paper claims. Log every raw component so weights can be diagnosed.

Use a short stabilization schedule:

```text
steps 0..249: preserve only or very small fingerprint weight
steps 250..749: linearly ramp fingerprint losses
steps >=750: full configured weights
```

Do not silently change loss weights based on final test performance.

---

# 10. Strength sweep for a high-probability first result

To improve the chance of a useful first run, support a **teacher-only validation sweep** over:

```text
alpha in {0.25, 0.50, 0.75}
```

Each candidate starts from the exact same clean teacher.

Evaluate candidates only on the held-out teacher validation split, never on final student test.

For every candidate report:

- teacher fingerprint TPR at frozen 1% FPR threshold,
- mean fingerprint score,
- model-level bit accuracy,
- CLIP,
- paired LPIPS to clean teacher with same prompts/seeds,
- optional FID/KID.

Recommended quality gate:

```text
CLIP drop <= 0.01
paired LPIPS <= 0.15
no obvious numerical/image failures
```

Among candidates passing the quality gate, choose the candidate with the strongest **teacher validation** fingerprint performance. Freeze that choice before student training.

If only one candidate is run for compute reasons, default to `alpha=0.5`.

---

# 11. Merge and prove inference is truly native

After teacher training:

1. load clean SD2.1,
2. load the selected teacher LoRA,
3. merge adapter weights into UNet,
4. unload/remove PEFT adapter structures if possible,
5. save a normal UNet/pipeline checkpoint.

Add a merge equivalence test on fixed latent/text inputs:

```text
(base + loaded LoRA) UNet output
≈
(merged saved UNet reloaded) output
```

Use strict numerical tolerances appropriate for FP32/FP16.

Then the actual teacher generation script must use standard pipeline inference:

```python
pipe = StableDiffusionPipeline.from_pretrained(FINGERPRINTED_TEACHER)
images = pipe(
    prompts,
    generator=ordinary_generators,
    num_inference_steps=50,
    guidance_scale=7.5,
    height=512,
    width=512,
).images
```

**The production fingerprinted generation path must not pass a custom `latents=` tensor.**

No secret/key object should be loaded by the generation path.

The key is required only for scoring/verifying images.

This is a central acceptance criterion.

---

# 12. Verification

The verifier is the same semantic statistic used to define the fingerprint.

For image \(x\), caption \(c\):

\[
q(x,c)
=
\operatorname{diag}(\sigma)^{-1}
U_k^\top(\Phi(x,c)-\mu).
\]

Define per-image scalar score:

\[
R(x,c)
=
\frac1m
s_k^\top q(x,c),
\qquad m=8.
\]

Higher means stronger alignment with the secret semantic code.

No learned classifier or CNN detector is allowed.

---

## 12.1 Two frozen null thresholds

Teacher outputs and SD1.4 student outputs are different model-family distributions, so use the same policy as the SERUM harness:

### Teacher threshold

Calibrate on **5,000 clean SD2.1 teacher images** from the disjoint calibration prompts.

Choose a conservative empirical threshold giving at most 1% FPR.

### Student threshold

Generate **5,000 unadapted SD1.4 calibration images** on the same calibration prompt protocol and freeze a separate 1% FPR threshold.

Important:

- scoring function \(R\) is identical,
- key is identical,
- semantic extractor is identical,
- only the empirical null threshold differs because the base image distributions differ.

Freeze both thresholds **before scoring any trained student test outputs**.

---

## 12.2 Reported fingerprint metrics

At minimum report:

### Per-image detection

\[
\mathrm{TPR@1\%FPR}
\]

using the appropriate frozen threshold.

### Mean score

\[
\bar R=\frac1N\sum_iR(x_i,c_i).
\]

### Model-level bit recovery

Aggregate each semantic coordinate:

\[
\bar q_j=\frac1N\sum_iq_j(x_i,c_i)
\]

and recover:

\[
\hat s_j=\operatorname{sign}(\bar q_j).
\]

Report:

\[
\text{bit accuracy}
=
\frac1m\sum_j \mathbf 1[\hat s_j=s_j].
\]

The model-level bit metric is especially relevant because lineage verification is an aggregate black-box model test, not necessarily a single-image authentication task.

Optionally report wrong-key scores as a sanity analysis, but do not make cryptographic claims.

---

# 13. Exact downstream experiment: reuse SERUM unchanged

Once a fingerprinted teacher checkpoint is selected, **do not change the student code**.

Generate:

```text
5,000 fingerprinted teacher training images
1,000 fingerprinted teacher validation images
1,000 fingerprinted teacher final test images where needed
5,000 clean teacher calibration images
```

Use the same prompt files, split construction, image resolution, seeds, inference steps, and CFG as the SERUM harness.

Train the same student branches:

```text
unadapted SD1.4

LoRA:
- attention r16
- attention r32
- attention + Conv2d r16
- attention + Conv2d r32

DoRA:
- attention + Conv2d r32

Full:
- full UNet
```

Keep:

```text
7,500 steps
snapshots 2,500 / 5,000 / 7,500
batch 8
LoRA LR 1e-5
full UNet LR 5e-6
AdamW
FP16
resolution 512
seed 0
```

Checkpoint selection remains:

```text
minimum held-out denoising MSE
tie -> earlier checkpoint
```

Detection/CLIP/FID/KID must **not** choose the checkpoint.

---

# 14. Compute-saving MVP gate

Do not immediately spend GPU on every student branch.

The first pilot should run:

```text
1. unadapted SD1.4 calibration/test
2. attention+Conv r32 LoRA student
3. full-UNet student
```

with the **same training hyperparameters/budget** as the full SERUM protocol.

Why these two trained students?

- `attention+Conv r32` is a strong PEFT test.
- `full UNet` is a positive capacity control.

The previously observed SERUM behavior in this local setup was roughly:

```text
unadapted ~0.5%
attention+conv r32 ~3.6%
full UNet ~83.6%
```

Those values are context for the experiment, **not targets baked into code**.

Recommended pilot interpretation:

```text
Teacher TPR > 90-95% + good quality:
    teacher embedding works.

LoRA TPR >= 50-60%:
    promising; expand full branch table.

LoRA TPR >= 80-90%:
    very strong result; immediately run complete table/seeds.

Teacher TPR ~99% but LoRA ~1-10%:
    fingerprint is detectable but not sufficiently radioactive;
    do NOT modify student. Revise teacher objective.

Full UNet weak:
    do not claim LoRA-specific weakness; the signal is not generally learnable enough.
```

---

# 15. What to do if the MVP is weak

This section is a **future fallback**, not part of the initial implementation unless requested.

If:

```text
teacher detection is strong
teacher quality is good
full-UNet inherits
standard LoRA does not inherit
```

then add a **transfer-aware teacher objective**.

Use a fresh surrogate standard SD1.4 LoRA inside teacher optimization only. Let it perform a few ordinary denoising updates on fingerprinted teacher data, with no fingerprint loss in the inner loop.

Then maximize the same fixed semantic score on held-out surrogate outputs relative to a clean-data surrogate.

Conceptually:

\[
L_{\rm transfer}
=
-
\left[
R(S_{\rm fp})
-
R(S_{\rm clean})
\right].
\]

This is allowed because the actual evaluated student remains completely ordinary.

Do **not** implement this meta-objective in the first MVP. First test whether coherent semantic shift alone already transfers.

---

# 16. Recommended repository integration

The cleanest approach is to add a new method while reusing `bitmark_lora` infrastructure.

Suggested files:

```text
run_semantic_fp.sh

bitmark_lora/
    run_semantic_fp.py
    semantic_assets.py
    semantic_carrier.py
    semantic_teacher.py
    semantic_teacher_train.py
    test_semantic_fp.py
```

Do not delete or rewrite legacy `serum/`.

---

## 16.1 Entrypoint

`bitmark_lora/run_semantic_fp.py` should resemble `run_serum.py`:

```python
from run import main

if __name__ == "__main__":
    main(method="semantic_fp")
```

Root launcher:

```text
run_semantic_fp.sh
```

should mirror the existing `run_serum.sh` environment/interpreter behavior.

---

## 16.2 Minimal changes to `bitmark_lora/run.py`

Extend method handling rather than creating a completely separate orchestration stack.

For `method == "semantic_fp"` use the **same downstream defaults as SERUM**:

```text
train_n=5000
validation_n=1000
calibration_n=5000
test_n=1000
batch_size=8
learning_rate=1e-5
full_learning_rate=5e-6
ranks=[16,32]
adapter_scopes=[attention, attention_conv]
dora=True
max_train_steps=7500
checkpoint_steps=[2500,5000,7500]
inference_steps=50
```

Add semantic-specific arguments, for example:

```text
--key-seed
--num-bits                  default 8
--carrier-pc-min            default 8
--carrier-pc-max            default 64
--stable-candidate-count    default 16
--semantic-shift            default 0.5
--semantic-shift-sweep      optional 0.25 0.5 0.75
--teacher-rank              default 32
--teacher-lr                default 1e-4
--teacher-train-steps       default 3000
--teacher-batch             default 4
--cycle-every               default 4
--lambda-preserve
--lambda-code
--lambda-perp
--lambda-cycle
--score-batch
```

When method is `semantic_fp`:

```text
assets_file  = semantic_assets.py
teacher_file = semantic_teacher.py
baseline     = unadapted SD1.4
```

Add preparation stages before standard teacher generation:

```text
01_assets
01b_carrier
01c_train_fingerprinted_teacher
02_teacher_generation
03_teacher_scores
03b_unadapted_student_calibration
03c_unadapted_student_calibration_scores
04_freeze_thresholds
...
existing student training/validation/generation
...
```

Keep the downstream loop unchanged.

---

# 17. `semantic_assets.py`

Reuse model-download/pinning/prompt-split logic from `serum_assets.py` as much as possible.

Required assets:

- clean SD2.1 teacher snapshot,
- SD1.4 student snapshot,
- prompt/split data,
- existing utility model assets if needed.

Not required:

- SERUM watermark checkpoint,
- SERUM detector checkpoint,
- SERUM watermark grid.

Save verified provenance/hashes.

`assets.json` should clearly distinguish:

```json
{
  "teacher_clean": "...",
  "student": "...",
  "method": "semantic_fp"
}
```

The selected merged fingerprinted teacher path is generated later and should be added to a separate immutable provenance file, not confused with source assets.

---

# 18. `semantic_carrier.py`

Responsibilities:

1. frozen semantic extractor,
2. feature extraction from images,
3. PCA fit,
4. VAE-cycle stability analysis,
5. key generation,
6. score computation,
7. serialization/loading.

Suggested API:

```python
class SemanticExtractor:
    def encode_image(self, images, prompts) -> Tensor: ...
    def encode_latent(self, scaled_latents, prompts) -> Tensor: ...

class SemanticCarrier:
    mean: Tensor
    basis: Tensor
    sigmas: Tensor
    signs: Tensor
    selected_pc_indices: list[int]

    def coordinates(self, features) -> Tensor: ...
    def scalar_score(self, features) -> Tensor: ...
```

Carrier files:

```text
run/carrier/
    pca.pt
    key.json
    stability.json
    provenance.json
```

Assertions:

- PCA vectors orthonormal within numerical tolerance.
- selected indices unique.
- selected sigmas finite and > epsilon.
- signs exactly ±1.
- all tensors finite.
- hashes on resume match.

---

# 19. `semantic_teacher_train.py`

Responsibilities:

- build teacher LoRA using existing repo's tested attention+Conv2d adapter machinery,
- load clean train image-caption pairs,
- run clean base UNet and trainable UNet on identical noisy latents,
- compute predicted-x0 robustly using scheduler configuration,
- compute coherent shift / preserve / orthogonal / cycle losses,
- log raw loss values,
- save teacher checkpoints,
- evaluate teacher validation candidates,
- merge selected adapter into standard SD2.1 UNet,
- verify merge equivalence,
- save provenance.

Do not generate HMARK-like target images offline.

Do not train a detector.

---

## 19.1 Timestep strategy

For the first MVP, use normal diffusion training timesteps for the preservation loss.

The semantic predicted-x0 loss can become unstable at extremely noisy timesteps. Implement a configurable semantic-loss mask such as:

```text
apply L_code/L_perp when t <= semantic_t_max
default semantic_t_max = 500
```

Continue applying preservation loss across all sampled timesteps.

This is preferable to silently clipping bad predicted-x0 tensors.

Log the fraction of samples receiving semantic loss.

If later evidence supports a different range, change via config and record it.

---

# 20. `semantic_teacher.py`

Match the existing teacher module interface as closely as possible.

Recommended CLI:

```text
--run <run_dir>
--stage {carrier,train,generate,score}
--directories ...
```

### `carrier`

- generate/reuse clean teacher train images,
- fit PCA,
- stability analysis,
- freeze key.

### `train`

- train candidate fingerprint teacher(s),
- validate,
- choose strength only on held-out teacher validation,
- merge selected teacher,
- freeze selection/provenance.

### `generate`

Generate clean/fingerprinted teacher splits using ordinary pipelines.

For `condition=wm/fp`, load the merged fingerprinted teacher.

For `condition=clean`, load original clean teacher.

**No custom `latents=` for the fingerprint condition.**

Pair prompt and random seed between clean and fingerprinted outputs whenever both are generated.

### `score`

Load image/caption pairs and compute the fixed semantic score.

No model-specific learned detector.

---

# 21. Integration with `report.py`

The new method needs SERUM-like **separate teacher and student calibration thresholds**, but it does not use the SERUM detector.

Refactor threshold logic by capability rather than by exact method string if necessary.

For example:

```python
uses_separate_student_null = carrier in {"serum", "semantic_fp"}
```

Teacher calibration source:

```text
teacher_clean_calibration
```

Student calibration source:

```text
unadapted_calibration
```

Scored values are simply `score` fields from `semantic_teacher.py`.

Report:

- threshold,
- empirical FPR,
- detections,
- TPR,
- Wilson CI if existing reporter supports it,
- mean score,
- model-level bits/bit accuracy.

Do not make detection score choose student checkpoints.

---

# 22. Utility evaluation

Reuse existing `utility.py` unchanged wherever possible.

For teacher quality additionally add paired metrics because clean/fingerprinted images can be generated from identical prompt/seed pairs:

- paired LPIPS,
- CLIP delta,
- optionally PSNR/SSIM only as secondary diagnostics.

For student utility retain the existing:

- CLIP,
- FID,
- KID.

A fingerprint result is not acceptable if student quality is materially worse than the corresponding baseline without acknowledging the tradeoff.

---

# 23. Determinism and data separation

The following must be saved in manifests:

- prompt file hashes,
- model revision/hashes,
- image seeds,
- training seed,
- key seed,
- teacher adapter config,
- loss weights,
- semantic timestep range,
- PCA basis hash,
- selected PC indices,
- signs,
- checkpoint selection,
- threshold calibration hashes,
- package versions.

Splits must remain disjoint:

```text
train
validation
calibration
test
```

The final test split must never influence:

- PCA candidate selection beyond the clean train basis,
- key,
- alpha/strength,
- teacher checkpoint,
- student checkpoint,
- threshold.

---

# 24. Required smoke/unit tests before a long GPU run

Codex must add lightweight tests.

## 24.1 Key determinism

Same:

```text
PCA basis + key_seed
```

must reproduce identical:

```text
selected directions + signs
```

Different key seed should usually differ.

## 24.2 Carrier math

Synthetic features should verify:

```text
coordinate projection
normalization
scalar score
bit recovery
```

against hand-computed values.

## 24.3 PCA orthogonality

Assert:

\[
V^\top V\approx I.
\]

## 24.4 Feature extractor determinism

The same image/prompt in eval mode should produce the same semantic feature within numerical tolerance.

Use posterior mean, not sampling.

## 24.5 Merge equivalence

Teacher LoRA loaded vs merged/reloaded teacher must produce matching UNet predictions on fixed tensors.

## 24.6 No-injection test

The production fingerprint generation path must not require:

```text
key
carrier
custom latent tensor
watermark module
SERUM grid
postprocessor
```

It should only require the merged fingerprinted model and standard diffusion inputs.

A test can monkeypatch/wrap the pipeline and assert no explicit `latents=` is supplied by the method.

## 24.7 Student isolation

Student training command must contain no semantic key/verifier/fingerprint arguments.

## 24.8 Nonfinite guard

Every training/evaluation stage must fail on NaN/Inf rather than writing a seemingly valid result.

---

# 25. Run-directory structure

Recommended:

```text
run_semantic_fp_YYYYMMDD_HHMMSS/
    config.json
    assets.json
    prompts.json
    environment.json
    code_manifest.json
    run.log

    carrier/
        pca.pt
        key.json
        stability.json
        provenance.json

    teacher_fp/
        candidates/
            shift_0.25/
            shift_0.50/
            shift_0.75/
        selected.json
        merged_unet/
        merge_test.json

    images/
        teacher_clean_train/
        teacher_fp_train/
        teacher_clean_validation/
        teacher_fp_validation/
        teacher_clean_calibration/
        teacher_clean_test/
        teacher_fp_test/
        unadapted_calibration/
        ...
        student branch outputs...

    students/
        ...

    scores/
        ...

    result/
        threshold_teacher.json
        threshold_student.json
        teacher_quality.json
        radioactivity.json
        utility.json
        README.md
```

Follow existing repository naming conventions where practical; consistency with existing `bitmark_lora` output is more important than exact names above.

---

# 26. Recommended first-run commands

The exact CLI can be adapted to the repo, but the intended workflow is:

```bash
# Inspect frozen plan without GPU work
bash run_semantic_fp.sh --plan-only
```

First pilot:

```bash
# Run only strongest representative LoRA + full control if the harness supports scope restriction.
bash run_semantic_fp.sh \
  --ranks 32 \
  --adapter-scopes attention_conv \
  --no-dora
```

Do not change the 7,500-step student budget in this pilot.

After promising transfer:

```bash
# Full SERUM-matched table
bash run_semantic_fp.sh
```

Then independent seed(s):

```bash
bash run_semantic_fp.sh --seed 1
```

Do not run seed sweeps before confirming the mechanism works.

---

# 27. Success table we want the reporter to produce

At minimum:

| Model | Detection @ 1% FPR | Mean score | Bit acc. | FID | KID | CLIP |
|---|---:|---:|---:|---:|---:|---:|
| Clean SD2.1 teacher | ~1% | ... | ... | ... | ... | ... |
| Fingerprinted SD2.1 teacher | ... | ... | ... | ... | ... | ... |
| Unadapted SD1.4 | ~1% | ... | ... | ... | ... | ... |
| Attention r16 | ... | ... | ... | ... | ... | ... |
| Attention r32 | ... | ... | ... | ... | ... | ... |
| Attn+Conv r16 | ... | ... | ... | ... | ... | ... |
| Attn+Conv r32 | ... | ... | ... | ... | ... | ... |
| DoRA Attn+Conv r32 | ... | ... | ... | ... | ... | ... |
| Full UNet | ... | ... | ... | ... | ... | ... |

Also produce a paired teacher-quality table for clean vs fingerprinted teacher.

---

# 28. Interpretation policy

Do not overclaim.

### Strong outcome

```text
teacher strong
LoRA strong
full strong
utility preserved
```

Supports:

> A model-native keyed semantic distribution shift can passively transfer through ordinary downstream diffusion adaptation.

### Teacher strong, full strong, LoRA weak

Supports:

> The fingerprint is learnable by unrestricted adaptation but standard PEFT does not sufficiently inherit it.

This motivates the future transfer-aware teacher objective.

### Teacher strong, full weak

The carrier is detectable but not sufficiently radioactive. Do not blame LoRA.

### Teacher weak

Fix teacher-side embedding first. Do not spend compute on students.

---

# 29. Paper-level story if the MVP works

The paper should not say:

> “We put HMARK into a model.”

It should say:

> We study model-lineage tracing for text-to-image diffusion models by making a protected teacher natively emit a keyed, coherent semantic distribution shift. The teacher requires no special runtime watermark input; downstream students see only ordinary image-caption pairs. The same fixed semantic statistic used to define the fingerprint is used to verify teacher and student outputs.

The conceptual axes are:

```text
model-native
+ semantic/distributional
+ radioactive
+ passive teacher-to-student lineage transfer
```

Comparison roles:

- SERUM: strong generation-time watermark, but weak LoRA inheritance in our matched setup.
- HMARK: evidence that coherent semantic-latent distribution shifts can be highly radioactive, but HMARK protects images/data post hoc rather than fingerprinting a T2I teacher to natively emit the signal.
- Ours: learns the semantic shift into the teacher's ordinary generation behavior.

---

# 30. Final implementation checklist for Codex

Before declaring the task complete, verify every item:

- [ ] New method has its own entrypoint; existing SERUM still runs.
- [ ] No HMARK code/data/model is imported or required.
- [ ] No SERUM watermark/checkpoint/detector is used by the new method.
- [ ] Clean SD2.1 defines a frozen semantic extractor.
- [ ] PCA carrier is fit only from clean data.
- [ ] Key is frozen before fingerprint teacher training.
- [ ] Fingerprint objective targets a **relative coherent shift**, not absolute feature collapse.
- [ ] Teacher is trained only on teacher side.
- [ ] Teacher LoRA is merged into ordinary weights.
- [ ] Fingerprinted teacher inference uses standard Gaussian generation and no custom `latents=`.
- [ ] Verifier is fixed and has no learned classifier head.
- [ ] Teacher and student 1%-FPR thresholds are calibrated on clean/unadapted nulls before test.
- [ ] Student receives only image-caption pairs.
- [ ] Student code/hyperparameters match the SERUM harness.
- [ ] Student checkpoints are selected by denoising MSE only.
- [ ] Pilot runs r32 attention+conv + full before full branch sweep.
- [ ] Teacher quality is measured.
- [ ] Student utility is measured.
- [ ] All configs/hashes/seeds/keys are persisted.
- [ ] Resume cannot silently alter config/key.
- [ ] NaN/Inf checks exist.
- [ ] Merge equivalence test passes.
- [ ] No-injection test passes.
- [ ] Final test is not used for tuning.

---

# 31. Priority order

If implementation time is limited, do work in this exact order:

```text
P0  Reuse SERUM harness without breaking it.
P1  Fixed semantic extractor + PCA/key + scoring.
P2  Teacher-side coherent-shift training.
P3  Merge teacher and prove ordinary native generation.
P4  Teacher calibration/detection/quality.
P5  Pilot downstream: unadapted + AttnConv-r32 + full.
P6  Full SERUM-matched branch table.
P7  Additional seeds/ablations.
P8  Transfer-aware teacher meta-objective only if MVP transfer is insufficient.
```

Do not jump to P8 before P5.

---

# 32. Core intuition to keep during coding

If any implementation choice becomes ambiguous, preserve this principle:

\[
\boxed{
\text{The same small keyed semantic displacement should recur across teacher outputs,}
}
\]

\[
\boxed{
\text{because a downstream diffusion model can learn a coherent distributional tendency,}
}
\]

\[
\boxed{
\text{and the teacher must generate that tendency natively from ordinary sampling.}
}
\]

The method fails conceptually if the signal is only detectable because of a special runtime key injection, a post-hoc image transform, a learned detector discovering incidental artifacts, or a downstream student that was modified specifically to learn the mark.
