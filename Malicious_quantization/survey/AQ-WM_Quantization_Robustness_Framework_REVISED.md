# AQ-WM: Dual-Sensitivity Analysis for Quantization-Robust Diffusion Ownership Verification
### A Post-Training Quantization Methodology for Robustness Testing, Vulnerability Analysis, and Watermark-Aware Deployment

---

## 0. Scope and Framing

This work studies **INT8/INT4 post-training quantization (PTQ)** as a deployment-time transformation of diffusion models and asks whether a model can remain useful for generation while its ownership-verification signal degrades.

The paper is centered on a robustness question:

> **Does quantization sensitivity for generation utility predict quantization sensitivity for ownership verification?**

The central hypothesis is that these two sensitivities can be misaligned:

\[
\operatorname{rank}(S^{gen}) \neq \operatorname{rank}(S^{ver}).
\]

A layer may therefore appear safe to quantize under generation-quality criteria while being critical to ownership verification.

This paper does **not** claim that sensitivity-guided mixed precision itself is new. Prior PTQ work already studies layer/block sensitivity, metric-decoupled allocation, and timestep-dependent quantization behavior. The contribution is the **ownership-verification objective**: AQ-WM measures and exploits the mismatch between generation sensitivity and verification sensitivity, then uses that mismatch for robustness analysis and verification-aware deployment.

The security interpretation is secondary in this version: if a utility-preserving PTQ configuration happens to reduce verification below its acceptance threshold, AQ-WM reports that configuration as a **deployment-time vulnerability case**. A dedicated adaptive quantization-laundering threat model is intentionally left outside the scope of this proposal.

No fine-tuning or retraining is used in AQ-WM. Only calibration-time statistics and post-training quantization are permitted.

---

# 1. Research Questions

### RQ1 — Generation–verification mismatch

Do diffusion-model layers that are safe for generation quality remain safe for ownership verification after INT8/INT4 PTQ?

### RQ2 — Timestep dependence

At which denoising timesteps does quantization of a given layer most strongly affect generation utility and ownership verification?

### RQ3 — Verification-aware deployment

Can a verification-aware mixed-precision allocation preserve ownership evidence with limited additional deployment cost compared with quality-only PTQ?

### RQ4 — Cross-mechanism generalization

Do different model-ownership mechanisms exhibit distinct layer/timestep sensitivity profiles, or do they share common vulnerable regions?

---

# 2. System Model and Terminology

## 2.1 Diffusion model

Let the denoising model be

\[
\epsilon_{\theta}(x_t,t,c),
\]

where \(\theta\) denotes denoiser parameters, \(x_t\) is the noisy latent at timestep \(t\), and \(c\) is the conditioning signal. For latent diffusion, the final latent is decoded by \(D_\phi\):

\[
y = D_\phi(z_0).
\]

The model is deployed without changing \(\theta\) or \(\phi\) through training.

## 2.2 PTQ operator

For layer \(l\), let the deployment precision be

\[
b_l \in \{4,8,16\}.
\]

A standard affine weight quantizer is

\[
Q_{b_l}(\theta_l)
=
 s_l\left[
\operatorname{clip}
\left(
\operatorname{round}(\theta_l/s_l)+z_l,
q_{\min},q_{\max}
\right)-z_l
\right],
\]

where \(s_l\) and \(z_l\) are determined using a fixed calibration set \(\mathcal D_{cal}\). Activation quantization, when used, is defined analogously.

A mixed-precision configuration is

\[
\mathbf b=(b_1,\ldots,b_L).
\]

The quantized model is denoted

\[
\hat\theta_{\mathbf b}=Q_{\mathbf b}(\theta).
\]

## 2.3 Primary ownership-verification targets

AQ-WM focuses primarily on **model ownership verification (MOV)** rather than treating every generative watermark as equivalent evidence of model ownership.

Representative target families are:

| Verification family | Representative example | What is verified | Primary score |
|---|---|---|---|
| Model-embedded watermark | AquaLoRA-like | ownership signal tied to the model | detector / bit / correlation score |
| Intrinsic fingerprint | FingerInv-like | model-specific generation/inversion behavior | reconstruction / similarity / decision score |
| Robust MOV baseline | Cert-LAS-like | ownership under model perturbation | certified / empirical verification score |

Decoder/output watermarks and latent/noise provenance marks can be included as **auxiliary provenance controls**, but they are not automatically treated as equivalent model-ownership mechanisms.

For ownership method \(m\), define its native verification score

\[
V_m(\theta) \in \mathbb R.
\]

Verification succeeds when

\[
V_m(\theta) \ge \tau_m,
\]

where \(\tau_m\) is calibrated **before PTQ evaluation** at a fixed false-positive rate.

No universal extractor is assumed. Each method keeps its original verification protocol.

---

# 3. Core Mathematical Formulation

## 3.1 Generation utility

Let \(U(\theta)\) denote generation utility. During expensive final evaluation, \(U\) is represented by multiple metrics such as FID, KID, CLIP score, and paired perceptual similarity.

For a quantization configuration \(\mathbf b\), define utility degradation

\[
\Delta U(\mathbf b)
=
U_{loss}(\hat\theta_{\mathbf b})-U_{loss}(\theta),
\]

where smaller is better.

## 3.2 Verification retention

For ownership method \(m\), define normalized verification retention

\[
R_m(\mathbf b)
=
\frac{\operatorname{TPR}_m(\hat\theta_{\mathbf b})}
{\operatorname{TPR}_m(\theta)+\delta}.
\]

A quantized model fails verification when

\[
V_m(\hat\theta_{\mathbf b})<\tau_m.
\]

## 3.3 Quality-only deployment problem

A standard deployment-oriented allocation is

\[
\mathbf b^{Q}
=
\arg\min_{\mathbf b} C(\mathbf b)
\]

subject to

\[
\Delta U(\mathbf b)\le \epsilon_U.
\]

This allocation does not explicitly protect ownership evidence.

## 3.4 Verification-aware deployment problem

AQ-WM adds verification constraints:

\[
\mathbf b^{V}
=
\arg\min_{\mathbf b} C(\mathbf b)
\]

subject to

\[
\Delta U(\mathbf b)\le\epsilon_U,
\]

and

\[
V_m(\hat\theta_{\mathbf b})\ge\tau_m
\qquad
\forall m \in \mathcal M.
\]

The deployment overhead required to preserve ownership is

\[
\Delta C_{own}
=
C(\mathbf b^{V})-C(\mathbf b^{Q}).
\]

This is one of the main practical quantities reported by AQ-WM.

## 3.5 Vulnerability stress test

A secondary stress test searches for a mixed-precision configuration that reduces verification while preserving utility:

\[
\mathbf b^{S}
=
\arg\min_{\mathbf b} V_m(\hat\theta_{\mathbf b})
\]

subject to

\[
\Delta U(\mathbf b)\le\epsilon_U,
\qquad
C(\mathbf b)\le B.
\]

This stress test is used to characterize robustness boundaries. It is not presented here as a full adaptive-attack contribution.

---

# 4. Dual-Sensitivity Analysis

The central analysis separates **generation sensitivity** from **verification sensitivity**.

For each layer \(l\) and candidate precision \(b\), construct \(\theta^{(l,b)}\) by quantizing only layer \(l\), keeping the rest of the model at FP16.

## 4.1 Empirical generation sensitivity

The primary generation-sensitivity score is an intervention-based denoiser drift:

\[
S^{gen}_{l,b}
=
\mathbb E_{x_t,t,c}
\left[
\frac{
\|\epsilon_\theta(x_t,t,c)-
\epsilon_{\theta^{(l,b)}}(x_t,t,c)\|_2^2
}{
\|\epsilon_\theta(x_t,t,c)\|_2^2+\delta
}
\right].
\]

A Hessian or gradient proxy may also be recorded as an auxiliary inexpensive predictor:

\[
\widetilde S^{gen}_{l,b}
\approx
\mathbb E
\left[\|\nabla_{\theta_l}\mathcal L_{gen}\|_2^2\right]
\|Q_b(\theta_l)-\theta_l\|_2^2.
\]

The empirical intervention score is treated as ground truth for analysis; the proxy is validated against it rather than assumed correct.

## 4.2 Empirical verification sensitivity

For ownership method \(m\), define

\[
S^{ver}_{l,b,m}
=
\mathbb E
\left[
\max\left(0,
V_m(\theta)-V_m(\theta^{(l,b)})
\right)
\right].
\]

This definition is deliberately verifier-native and does not require a differentiable watermark loss.

When a differentiable verification surrogate exists, an optional gradient proxy is

\[
\widetilde S^{ver}_{l,b,m}
\approx
\mathbb E
\left[\|\nabla_{\theta_l}\mathcal L_{ver,m}\|_2^2\right]
\|Q_b(\theta_l)-\theta_l\|_2^2.
\]

The gradient proxy is **not** assumed available for every ownership mechanism.

## 4.3 Ownership–utility mismatch score

Define

\[
M_{l,b,m}
=
\frac{S^{ver}_{l,b,m}}
{S^{gen}_{l,b}+\delta}.
\]

Large \(M_{l,b,m}\) identifies the target failure mode:

\[
S^{gen}_{l,b}\approx0,
\qquad
S^{ver}_{l,b,m}\gg0.
\]

Such a layer appears safe under quality-oriented PTQ but is important for ownership verification.

Because the two sensitivities can have different scales, AQ-WM additionally reports their **rank disagreement**:

\[
D_m
=
1-ho_{\mathrm{Spearman}}
\left(
S^{gen},S^{ver}_m
\right).
\]

This avoids making the paper depend on one hand-designed ratio.

---

# 5. Timestep-Resolved Causal Profiling

A diffusion UNet reuses the same layers across denoising timesteps, so AQ-WM does **not** assume that specific layers are active only at specific timesteps.

Instead, AQ-WM uses **trajectory teacher-forcing**.

## 5.1 Reference trajectory

Run the FP16 model once and cache

\[
\{x_{t_k}\}_{k=1}^{K}
\]

at \(K\) timestep bins.

## 5.2 Layer-by-timestep generation intervention

For each layer \(l\), bit-width \(b\), and cached timestep \(t_k\):

1. keep cached \(x_{t_k}\) fixed;
2. quantize only layer \(l\);
3. execute one denoiser forward pass.

Define

\[
S^{gen}_{l,t_k,b}
=
\frac{
\|\epsilon_\theta(x_{t_k},t_k)-
\epsilon_{\theta^{(l,b)}}(x_{t_k},t_k)\|_2^2
}{
\|\epsilon_\theta(x_{t_k},t_k)\|_2^2+\delta
}.
\]

Because the latent is teacher-forced from the FP16 trajectory, this isolates the marginal effect at \(t_k\) rather than accumulated upstream error.

## 5.3 Layer-by-timestep verification intervention

For ownership methods whose verifier can be evaluated from a partial/controlled trajectory, define

\[
S^{ver}_{l,t_k,b,m}
=
\Delta V_m(l,t_k,b).
\]

When direct timestep-local verification is not well-defined, AQ-WM uses a controlled rollout: inject the layer quantization only at timestep bin \(t_k\), keep all other steps FP16, complete the trajectory, then evaluate the native verifier.

This distinction is important: **not every ownership method admits a local differentiable timestep loss**.

## 5.4 Timestep-resolved mismatch

\[
M_{l,t_k,b,m}
=
\frac{
S^{ver}_{l,t_k,b,m}
}{
S^{gen}_{l,t_k,b}+\delta
}.
\]

The output is a pair of heatmaps per ownership method:

\[
\mathcal H^{gen}_m[l,t],
\qquad
\mathcal H^{ver}_m[l,t].
\]

Their disagreement is analyzed directly; they are not collapsed into one map before analysis.

## 5.5 Trajectory accumulation

Teacher-forced analysis measures local sensitivity. Compounding error is evaluated separately on full quantized rollouts:

\[
E_{cum}(t)
=
\sum_{\tau=T}^{t}
\|\epsilon_\theta(x_\tau,\tau)-
\epsilon_{\hat\theta}(\hat x_\tau,\tau)\|_2.
\]

---

# 6. Optional Directional Diagnostics

These diagnostics are secondary and are not required for the core AQ-WM method.

## 6.1 Quantization–verification alignment

When ownership method \(m\) has a differentiable verification surrogate, define

\[
e_{l,b}=Q_b(\theta_l)-\theta_l
\]

and

\[
A_{l,b,m}
=
-
\frac{
\langle \nabla_{\theta_l}V_m,e_{l,b}\rangle
}{
\|\nabla_{\theta_l}V_m\|_2
\|e_{l,b}\|_2+\delta
}.
\]

Positive values predict first-order verification degradation.

This metric is **not claimed to be universal**. For non-differentiable verifiers it is omitted or replaced by costly finite-difference validation on a small subset.

## 6.2 Causal ownership localization

A gradient-only Localization Index can be confounded by layer size and parameterization. Therefore AQ-WM uses intervention-based verification degradation as the preferred localization signal.

Define non-negative layer importance

\[
g_{l,m}
=
\max_{b\in\{4,8\}}
S^{ver}_{l,b,m}.
\]

Normalize

\[
p_{l,m}
=
\frac{g_{l,m}}
{\sum_j g_{j,m}+\delta}.
\]

Then

\[
PR_m
=
\frac{1}{\sum_l p_{l,m}^2},
\qquad
LI_m
=
1-\frac{PR_m}{L}.
\]

\(LI_m\) is reported as an **empirical concentration statistic**, not evidence of a literal watermark subnetwork.

If a gradient-based version is also reported, layer-size normalization is required, e.g.

\[
\bar g_{l,m}
=
\frac{\|\nabla_{\theta_l}V_m\|_2}{\sqrt{d_l}}.
\]

---

# 7. DS-WAQ: Dual-Sensitivity Watermark-Aware Quantization

DS-WAQ is the deployment module built on top of AQ-WM profiling. It does not replace the underlying PTQ kernel. It uses the measured mismatch to decide which layers should retain higher precision.

## 7.1 Inputs

- pretrained watermarked/fingerprinted model;
- calibration set \(\mathcal D_{cal}\);
- candidate precisions \(\mathcal B=\{4,8,16\}\);
- deployment budget \(B\);
- utility tolerance \(\epsilon_U\);
- fixed verification thresholds \(\{\tau_m\}\).

## 7.2 Promotion benefit

Starting from an all-INT4 candidate, define the benefit of promoting layer \(l\) from precision \(b\) to \(b'\):

\[
G_{l:b\rightarrow b'}
=
\frac{
\sum_{m\in\mathcal M}
\omega_m
\left[
\widehat V_m(b')-\widehat V_m(b)
\right]
+
\lambda
\left[
\widehat U(b)-\widehat U(b')
\right]
}{
\Delta C_{l:b\rightarrow b'}+\delta
}.
\]

The \(\widehat{\cdot}\) quantities are cheap profiling surrogates. Full metrics are used only for periodic validation.

## 7.3 Corrected constrained allocator

```text
Input: profiling scores, budget B, utility tolerance epsilon_U,
       verification thresholds tau_m

1. Initialize all quantizable layers at INT4.
2. Evaluate cheap utility and verification surrogates.
3. While any utility/verification constraint is violated:
       a. enumerate feasible promotions INT4->INT8 or INT8->FP16;
       b. discard any promotion for which Cost(b_new) > B;
       c. score each remaining promotion by G;
       d. choose the highest-scoring feasible promotion;
       e. if no feasible promotion remains: return INFEASIBLE;
       f. apply the promotion;
       g. periodically run held-out empirical verification and utility checks.
4. Return b^V.
```

This corrects the earlier invalid logic in which precision promotion could occur while the bit budget was already exceeded.

## 7.4 Quality-only baseline

A matched quality-only allocator uses the same procedure but removes verification benefit:

\[
\omega_m=0.
\]

This is the fairest comparison because the search procedure is identical and only the protected objective changes.

## 7.5 Robustness stress-test allocation

For analysis only, a separate stress-test search ranks candidate demotions/promotions by

\[
R^{stress}_{l,q,m}
=
\frac{
\Delta V_{l,q,m}
}{
\Delta U_{l,q}+\delta
}.
\]

It is optimized independently; it is **not** implemented by merely reversing the defender sort order.

---

# 8. Losses and Search Surrogates

AQ-WM performs no model training. All losses below are measurement, calibration, or search objectives.

## 8.1 Calibration loss

\[
\mathcal L_{cal}
=
\mathbb E
\left[
\|f_l(x;\theta_l)-f_l(x;Q_b(\theta_l))\|_2^2
\right].
\]

## 8.2 Generation surrogate

Inside profiling/search, use paired outputs from identical prompts and seeds:

\[
\mathcal L_{gen}
=
\lambda_\epsilon d_\epsilon
+
\lambda_p\operatorname{LPIPS}(y_{FP},y_Q)
+
\lambda_c\left(1-\operatorname{CLIPSim}(y_Q,c)\right).
\]

Full FID/KID are reserved for final evaluation.

## 8.3 Verification margin loss

Instead of forcing all ownership mechanisms into BCE or p-values, use a method-native score converted to a threshold margin:

\[
\mathcal L_{ver,m}
=
\max(0,\tau_m-V_m).
\]

For message-based watermarks, bit accuracy/BCE may be reported additionally. For statistical tests, the native statistic or calibrated detector score is used rather than minimizing \(-\log p\) with an inconsistent sign convention.

## 8.4 Search objective

For candidate deployment configurations:

\[
\mathcal J(\mathbf b)
=
\alpha\widehat{\Delta U}(\mathbf b)
+
\beta\sum_m \mathcal L_{ver,m}(\mathbf b)
+
\gamma C(\mathbf b).
\]

The primary paper formulation remains constrained optimization; this scalarized objective is used only as a search heuristic and for Pareto exploration.

---

# 9. Experimental Design

The experimental design is tiered to avoid a combinatorial benchmark.

## 9.1 Tier 1 — Primary study

### Backbone

- Stable Diffusion 1.5.

### Primary MOV targets

- AquaLoRA-like model watermark;
- FingerInv-like intrinsic fingerprint;
- Cert-LAS-like robust MOV baseline, if reproducible.

### Auxiliary provenance controls

- one decoder/output watermark or one latent/noise watermark may be included separately;
- results from these controls are not merged with MOV claims.

### Quantization baselines

For UNet/latent diffusion:

- FP16 reference;
- RTN/uniform affine PTQ;
- Q-Diffusion or another diffusion-specific PTQ implementation.

Generic PTQ methods such as AWQ/GPTQ may be included as auxiliary baselines, but are not treated as the primary diffusion-specific reference.

### Precision conditions

Uniform baselines:

\[
W8,
\qquad
W4,
\]

with activation precision reported explicitly when used, e.g. W8A8 or W4A8.

Mixed-precision DS-WAQ uses the candidate set

\[
\mathcal B=\{4,8,16\}
\]

under **average-bit or memory budgets**, for example

\[
\bar b\in\{4.5,6.0,8.0\},
\]

rather than labeling a mixed-precision allocation simply as “INT4” or “INT8”.

### Compared configurations

1. uniform PTQ;
2. quality-only mixed precision;
3. DS-WAQ verification-aware mixed precision;
4. random mixed-precision control;
5. optional robustness stress-test configuration.

## 9.2 Tier 2 — Architecture generalization

Use one additional architecture:

- SDXL **or** one DiT-based backbone.

For DiT, use a DiT-specific PTQ baseline such as Q-DiT when compatible.

Tier 2 is a qualitative generalization check, not a repetition of the full Tier-1 grid.

## 9.3 Profiling versus final evaluation

Profiling/search:

\[
N_{profile}\approx 500\text{–}1000
\]

fixed prompt–seed pairs.

Final generation evaluation:

\[
N_{final}\ge 10{,}000
\]

images when computing FID/KID; use 30k when computationally feasible.

Ownership evaluation uses a separate held-out set and fixed detector thresholds.

---

# 10. Evaluation Protocol

## 10.1 Generation utility

Report:

- FID;
- KID;
- CLIP score;
- paired LPIPS or DreamSim between FP16 and quantized generations generated from identical prompt/seed pairs.

Utility tolerances are fixed before examining robustness results.

## 10.2 Ownership verification

For each method \(m\), report its native metric and at least one thresholded operating point:

\[
TPR@FPR=\alpha.
\]

Also report AUROC when meaningful.

Do **not** assume

\[
V_m(\theta)=1.
\]

The measured FP16 verification performance is the baseline.

## 10.3 Dual-sensitivity validity

Report:

\[
\rho_{\mathrm{Spearman}}
(S^{gen},S^{ver}_m),
\]

and the fraction

\[
P_m^{mismatch}
=
\Pr
\left[
S^{gen}_{l,b}\le q_{25}^{gen}
\land
S^{ver}_{l,b,m}\ge q_{90}^{ver}
\right].
\]

## 10.4 Deployment overhead

Report the extra memory/bit cost required to protect verification:

\[
\Delta C_{own}
=C(\mathbf b^V)-C(\mathbf b^Q).
\]

This converts the analysis into actionable deployment guidance.

## 10.5 Statistical protocol

- thresholds calibrated only on validation data;
- all final metrics computed on held-out prompts/seeds;
- paired bootstrap confidence intervals over prompt–seed pairs;
- paired significance testing between quality-only and verification-aware deployments;
- report effect sizes, not only p-values.

---

# 11. Falsifiable Hypotheses

AQ-WM avoids assuming a predefined fragility ordering.

### H1 — Layer-ranking mismatch

\[
\rho_{\mathrm{Spearman}}
(S^{gen},S^{ver}_m)<1
\]

for at least one ownership method.

### H2 — Hidden verification-critical layers

For at least one ownership method, there exist layers satisfying

\[
S^{gen}_{l,b}\le q_{25}^{gen}
\]

and

\[
S^{ver}_{l,b,m}\ge q_{90}^{ver}.
\]

### H3 — Timestep-dependent mismatch

The layer × timestep ranking induced by

\[
S^{gen}_{l,t,b}
\]

differs measurably from the ranking induced by

\[
S^{ver}_{l,t,b,m}
\]

for at least one ownership method.

### H4 — Verification-aware allocation benefit

At matched deployment cost,

\[
V_m(\mathbf b^V)
>
V_m(\mathbf b^Q)
\]

without statistically significant degradation in generation utility.

### H5 — Method dependence

Different ownership mechanisms exhibit significantly different mismatch maps and localization statistics.

No hypothesis assumes in advance that parameter watermarks are always most fragile, that latent/noise mechanisms are always most robust, or that early timesteps must dominate.

---

# 12. Expected Contributions

If supported by experiments, AQ-WM contributes:

1. **Dual-sensitivity robustness formulation.** A formal separation between generation sensitivity and ownership-verification sensitivity under diffusion PTQ, evaluated with matched single-layer and timestep-controlled interventions.

2. **Timestep-resolved causal profiling.** A teacher-forced methodology that distinguishes local layer/timestep quantization effects from accumulated trajectory error.

3. **Verification-aware deployment method.** DS-WAQ, a mixed-precision allocation procedure that preserves ownership verification under explicit deployment budgets without retraining or modifying quantization kernels.

4. **Mechanistic characterization.** Empirical identification of ownership–utility mismatch regions and causal localization patterns across representative MOV methods.

5. **Deployment guidance.** Quantitative measurement of the additional precision/memory cost required to preserve ownership evidence during INT8/INT4 deployment.

Claims such as “first systematic study” or “first watermark-aware PTQ” should be used only after a complete camera-ready literature review establishes them.

---

# 13. Novelty Positioning Against Prior Work

AQ-WM should be positioned carefully.

- It does **not** claim mixed-precision sensitivity allocation is new.
- It does **not** claim metric decoupling in PTQ is new.
- It does **not** claim quantization has never been evaluated as a watermark/model-ownership perturbation.
- It does **not** claim every output/provenance watermark is equivalent to model ownership verification.

The intended novelty is narrower:

> **AQ-WM asks whether deployment decisions that are safe under generation-quality sensitivity remain safe under model-ownership verification sensitivity, and provides a diffusion-specific layer/timestep methodology plus a verification-aware deployment allocator to quantify and mitigate that mismatch.**

This positioning directly avoids reducing the work to “MixDQ with another metric”: MixDQ-style metric decoupling motivates the optimization machinery, while AQ-WM changes the protected property from generative quality alone to **ownership evidence under post-training deployment transformation** and validates that property with native ownership verifiers and causal PTQ interventions.

---

# 14. Threats to Validity and Limitations

1. **Verifier heterogeneity.** Some ownership mechanisms expose differentiable scores while others do not; gradient diagnostics are therefore optional and not universally comparable.

2. **Calibration dependence.** PTQ calibration prompts and timestep sampling can alter measured sensitivity. Results must include calibration-set sensitivity analysis.

3. **Architecture transfer.** Layer rankings learned on SD-1.5 should not be assumed to transfer unchanged to SDXL or DiT.

4. **Metric limitations.** FID/CLIP alone are insufficient for paired utility preservation; paired perceptual similarity must be included.

5. **Localization interpretation.** A high Localization Index indicates concentrated empirical sensitivity, not a causal proof of a dedicated watermark subnetwork.

6. **Ownership versus provenance.** Content-watermark robustness results must not be overgeneralized into model-ownership claims.

7. **Stress-test scope.** The optional adversarial/stress-test configuration in AQ-WM is restricted to mixed-precision robustness analysis. A broader adaptive search over the full deployment quantizer space belongs to a separate security-focused study.

---

# 15. Minimal Pilot Experiment Before Full Study

Before implementing the entire DS-WAQ system, run a pilot on:

- SD-1.5;
- one model-embedded ownership watermark;
- one intrinsic fingerprint;
- uniform W8 and W4;
- single-layer intervention on major UNet blocks;
- 4–6 timestep bins.

The pilot should answer only three questions:

1. Is

\[
\rho_{\mathrm{Spearman}}
(S^{gen},S^{ver})
\]

substantially below one?

2. Do any high-verification/low-generation mismatch layers exist?

3. Does a verification-aware promotion strategy preserve verification better than quality-only allocation at the same bit budget?

If the answer to all three is negative, the full DS-WAQ study should not be expanded.

---

# Final Positioning

AQ-WM is best framed as a **diffusion deployment robustness methodology** rather than as a generic watermark benchmark or a full adaptive attack paper.

Its core claim is:

> **Generation-quality robustness under PTQ is not sufficient evidence of ownership-verification robustness. AQ-WM measures this mismatch at layer and timestep resolution and uses it to design verification-aware mixed-precision deployment without retraining.**
