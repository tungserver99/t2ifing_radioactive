# Adaptive Quantization Robustness Analysis for Diffusion Watermarking (AQ‑WM)
### A Post-Training Quantization (PTQ) Vulnerability & Robustness Methodology for Ownership Verification Signals in Diffusion Models

---

## 0. Framing

This framework treats **INT8/INT4 post-training quantization (PTQ)** as a *deployment-time perturbation channel* that can be analyzed from two complementary angles, both publishable:

- **Robustness angle (CVPR/ICCV):** does a watermark/fingerprint survive standard efficiency-driven deployment (quantization for edge/mobile/serving cost reduction)?
- **Security angle (USENIX Security):** quantization is a *legitimate, unsuspicious* transformation an adversary can apply (or select among configurations for) to **evade ownership verification** without retraining, without visibly degrading output quality, and without leaving evidence of intentional watermark removal. This reframes "PTQ" as a **watermark-removal attack surface hiding inside a routine MLOps step**.

No fine-tuning or retraining is used anywhere in the pipeline — all sensitivity signals are obtained via **calibration-only PTQ** (RTN/GPTQ/AWQ/SmoothQuant-style calibration statistics), consistent with realistic deployment pipelines.

**Novelty positioning (explicit, to preempt review pushback).** Sensitivity-guided mixed-precision allocation itself is *not* claimed as novel — HAWQ, MixDQ, Q-DiT, and SVDQuant already solve "which layer can tolerate low precision without hurting generation quality." This framework's contribution is orthogonal: it asks whether the layer ranking that is safe *for generation quality* is the same ranking that is safe *for ownership verification*. The central quantity of interest is therefore not a single sensitivity score $S_l$, but the **decoupled pair** $(S_l^{gen}, S_l^{wm})$ and their divergence (Section 2.1) — a layer with low $S_l^{gen}$ but high $S_l^{wm}$ is invisible to every existing quality-driven PTQ method, yet silently destroys ownership proof. This divergence, not the allocation algorithm, is the paper's claim.

---

## 1. Mathematical Problem Formulation

### 1.1 Diffusion model and parameters
Let a diffusion model be $\epsilon_\theta(x_t, t, c)$, $\theta \in \mathbb{R}^d$, denoising $x_T \to x_0$ over $t = T, \dots, 1$, with a decoder $D_\phi$ (for latent diffusion, $x_0 = D_\phi(z_0)$).

### 1.2 Layer-wise PTQ operator
For layer $l$ with weight tensor $\theta_l$, define a uniform-affine quantizer with bit-width $b_l \in \{4, 8, 16\}$:

$$
Q_{b_l}(\theta_l) = s_l \cdot \text{clip}\!\left(\text{round}\left(\frac{\theta_l}{s_l}\right) + z_l,\; 0,\; 2^{b_l}-1\right) - s_l z_l
$$

with scale $s_l$ and zero-point $z_l$ computed via calibration (MSE-minimizing or percentile clipping over a small calibration prompt set $\mathcal{D}_{cal}$), **not gradient descent on model weights**. A configuration is a vector $\mathbf{b} = (b_1, \dots, b_L)$; $\hat\theta = Q_\mathbf{b}(\theta)$.

Activation quantization is treated analogously with a separate operator $Q^{act}_{b_l}$ applied at layer inputs.

### 1.3 Watermark / fingerprint carriers (formal definitions)

| Carrier | Embedding domain | Extraction map | Verification |
|---|---|---|---|
| **Parameter watermark** | $\theta$ (weight-space perturbation $\theta = \theta_0 + \delta_{wm}$) | $\text{Ext}_p(\theta) \to \hat{w} \in \{0,1\}^k$ | $\text{BitAcc}(\hat w, w)$, correlation test |
| **VAE/decoder watermark** | $\phi$ or pixel-space post-hoc signature | $\text{Ext}_d(D_\phi(z_0)) \to \hat w$ | detector confidence / bit accuracy |
| **Latent/noise watermark** | initial noise $z_T$ (e.g., structured Fourier-domain pattern, à la ring-signature style) | $\text{Ext}_z(\text{DDIM-inv}(x_0)) \to$ statistic | hypothesis test $p$-value vs. null |
| **Behavioral fingerprint** | trigger set $\mathcal{T} = \{x_{trig}^{(i)}\}$ | $y_{sig}^{(i)} = G_\theta(x_{trig}^{(i)})$ | similarity/classifier vs. reference signature bank |

Define a unified **verification score function**:
$$
V(\hat\theta; \text{carrier}) = \sigma\big(\text{Ext}(\hat\theta, \cdot)\big) \in [0,1]
$$
mapping to accept/reject via threshold $\tau$ calibrated at a fixed false-positive rate (FPR).

### 1.4 Joint problem statement
Given original $(\theta, \phi, w)$ with verified $V(\theta) \ge \tau$ and generation quality baseline $Q_0 = (\text{FID}_0, \text{CLIP}_0)$, find bit-allocation $\mathbf{b}^*$ such that:

$$
\mathbf{b}^* = \arg\min_{\mathbf{b} \in \{4,8,16\}^L} \; \text{Cost}(\mathbf{b})
\quad \text{s.t.} \quad
\Delta\text{FID}(\mathbf{b}) \le \epsilon_q,\;\;
V(\hat\theta_\mathbf{b}) \ge \tau
$$

This is the **defender's problem** (deploy efficiently while preserving ownership proof). Its **dual, adversarial problem** (evasion) is:

$$
\mathbf{b}^{atk} = \arg\max_{\mathbf{b}} \; \big[1 - V(\hat\theta_\mathbf{b})\big]
\quad \text{s.t.} \quad \Delta\text{FID}(\mathbf{b}) \le \epsilon_q,\;\; \Delta\text{CLIP}(\mathbf{b}) \le \epsilon_c
$$
i.e., find *any standard-looking* quantization config that destroys the verification signal while remaining visually/quantitatively indistinguishable in generation quality — the core USENIX-relevant threat model.

---

## 2. Quantization Sensitivity Metrics

### 2.1 Decoupled dual-objective sensitivity (the paper's central metric)

Rather than a single per-layer score, every layer $l$ is profiled under **two independent objectives**, using the *same* calibration forward passes (only the target loss differs, so this adds no extra inference cost over standard HAWQ-style profiling):

**(a) Generation-sensitivity** (Hessian-trace proxy, HAWQ-style — standard, not claimed novel):
$$
S_l^{gen} = \text{tr}(H_l^{gen})\,\|\Delta\theta_l\|_2^2 \approx \mathbb{E}_{x\sim\mathcal{D}_{cal}}\big[\|\nabla_{\theta_l}\mathcal{L}_{gen}\|^2\big]\cdot\|\Delta\theta_l\|_2^2
$$
with $\mathcal{L}_{gen}$ the noise-prediction/LPIPS surrogate (Section 5).

**(a′) Watermark-sensitivity** (same functional form, different target — this substitution is the actual methodological move):
$$
S_l^{wm} = \text{tr}(H_l^{wm})\,\|\Delta\theta_l\|_2^2 \approx \mathbb{E}_{x\sim\mathcal{D}_{cal}}\big[\|\nabla_{\theta_l}\mathcal{L}_{wm}\|^2\big]\cdot\|\Delta\theta_l\|_2^2
$$
with $\mathcal{L}_{wm} = -\log V(\hat\theta)$ (or BCE against $w$ where extraction is differentiable; Section 5). $S_l^{wm}$ is computed **per carrier** — a layer's watermark-sensitivity is generally different for the parameter, decoder, latent/noise, and behavioral carriers, since each occupies a different subspace of $(\theta,\phi,z_T)$.

**(b) Divergence score (headline diagnostic):**
$$
\text{Div}_l^{(c)} = \frac{S_l^{wm,(c)}}{S_l^{gen} + \delta} \qquad \text{or} \qquad \text{Div}_l^{(c)} = \big|\,\widetilde{S}_l^{wm,(c)} - \widetilde{S}_l^{gen}\,\big|
$$
(ratio form preferred; $\delta$ a small constant for numerical stability; $\widetilde{S}$ denotes rank-normalized scores if using the difference form, since $S^{gen}$ and $S^{wm}$ live on different scales). **A layer with low $S_l^{gen}$ but high $\text{Div}_l^{(c)}$ is exactly the failure mode existing quality-driven PTQ methods (HAWQ/MixDQ/Q-DiT/SVDQuant) cannot detect**: it is judged safe to compress because it does not hurt FID, while it silently destroys carrier $c$'s ownership signal. Reporting the *fraction of layers with high Div but low $S^{gen}$*, per carrier, is proposed as the paper's primary empirical result (Section 8, Finding 1).

### 2.2 Carrier-general perturbation and trajectory metrics

**(c) Output perturbation sensitivity (single-layer isolation, generic — used to build $\Delta_l$ used above):**
$$
\Delta_l = \mathbb{E}_{x_t,t}\big[\|\epsilon_\theta(x_t,t) - \epsilon_{\theta \setminus l \to \hat\theta_l}(x_t,t)\|_2\big]
$$
(all layers FP16 except $l$, quantized; averaged over $t$ — the *timestep-resolved* version is $S(l,t)$, defined precisely in Section 4.2 to avoid the "layers active at timestep $t$" ambiguity flagged in review.)

**(d) Watermark SNR degradation** (carrier-general, applies to any carrier with a defined signal/noise decomposition, e.g. parameter or latent/noise):
$$
\text{SNR}_{wm} = 10\log_{10}\frac{\|v_{wm}\|_2^2}{\|Q_\mathbf{b}(\theta) - \theta\|_2^2 \text{ (projected onto the } v_{wm}\text{-relevant subspace)}}
$$

**(e) Generalized ownership alignment score** (supersedes the earlier parameter-only interference angle — this version is carrier-general and reuses computation already done for $S_l^{wm}$, at no extra cost). Define the **ownership direction** at layer $l$ as the gradient of the carrier's own verification score,
$$
v_{own,l}^{(c)} = \nabla_{\theta_l} V^{(c)}(\theta)
$$
which is well-defined for *any* carrier with a (sub-)differentiable $V^{(c)}$ — no explicit additive watermark vector $\delta_{wm}$ is required, so this applies uniformly to parameter, decoder, latent/noise, and behavioral carriers alike. (For carriers whose native verification is a hypothesis test, e.g. latent/noise, use a smooth surrogate or finite-difference estimate of $\nabla V^{(c)}$; this is the same surrogate already used to compute $\mathcal{L}_{wm} = -\log V^{(c)}$ for $S_l^{wm}$ in Section 2.1a′, so $v_{own,l}^{(c)}$ is a byproduct of that computation, not an extra pass.) The **quantization–ownership alignment score** is then
$$
A_l^{(c)} = \frac{\big\langle \theta_l - Q_{b_l}(\theta_l),\; v_{own,l}^{(c)} \big\rangle}{\big\|\theta_l - Q_{b_l}(\theta_l)\big\|\,\big\|v_{own,l}^{(c)}\big\|}
$$
$A_l^{(c)} \to 1$: quantization error at layer $l$ is aligned with carrier $c$'s ownership-sensitive direction and will erode verification; $A_l^{(c)} \to 0$: quantization error is (locally) orthogonal to ownership information and mainly perturbs generation-relevant directions instead. This is a strictly more general form of the earlier parameter-carrier-only $\cos\phi_l$, which is recovered as the special case $v_{own,l} = \delta_{wm,l}/\|\delta_{wm,l}\|$ when an explicit additive weight-space watermark exists.

**On the "$\theta = \theta_{gen} + \theta_{own}$" framing (explicit caveat).** It is tempting to describe generation- and ownership-relevant information as living in two disjoint additive components of $\theta$. This is *not* assumed here: over-parameterized networks typically exhibit superposition, where a single weight contributes to multiple, non-orthogonal functions simultaneously. Rather than positing a literal partition of $\theta$, this framework only claims that the two **gradient directions** $\nabla_{\theta_l}U$ (generation) and $\nabla_{\theta_l}V^{(c)}$ (ownership, carrier $c$) can be locally near-orthogonal or near-parallel at a given layer, which is exactly what $A_l^{(c)}$ measures — a directional/local claim, not a global structural decomposition of the parameter vector.

**(f) Trajectory error accumulation:**
$$
E_{cum}(t) = \sum_{\tau=T}^{t} \|\epsilon_\theta(x_\tau,\tau) - \epsilon_{\hat\theta}(x_\tau,\tau)\|_2 \cdot \Delta\tau
$$
(ODE/SDE solver error propagation analogue — captures whether early-step errors compound toward $t=0$.)

**(g) Bit-width elasticity:** $\mathcal{E}_l^{(c)} = \dfrac{\partial V^{(c)}}{\partial b_l}$ estimated via finite differences over $b_l \in \{4,8,16\}$, per layer, per carrier $c$.

### 2.3 Ownership Information Localization Index (new — cross-carrier structural metric)

$S_l^{wm,(c)}$ (equivalently $\|v_{own,l}^{(c)}\|$) gives a per-layer ownership-gradient magnitude for carrier $c$. Normalize across layers into a distribution:
$$
p_l^{(c)} = \frac{\big\|v_{own,l}^{(c)}\big\|}{\sum_{l'=1}^{L}\big\|v_{own,l'}^{(c)}\big\|}, \qquad \sum_l p_l^{(c)} = 1
$$
and define the **effective participation ratio** and the **Localization Index**:
$$
\text{PR}^{(c)} = \frac{1}{\sum_{l} \big(p_l^{(c)}\big)^2}, \qquad \text{LI}^{(c)} = 1 - \frac{\text{PR}^{(c)}}{L} \in [0,1]
$$
$\text{PR}^{(c)}$ is the effective number of layers carrying most of carrier $c$'s ownership signal (interpretable as "number of layers you'd need to protect at high precision"). $\text{LI}^{(c)} \to 1$: ownership information is **concentrated** in a few layers (cheap to defend — promote those layers to FP16/INT8 and quantize the rest freely; also an easy, low-cost *attack* target: an adversary need only target those few layers). $\text{LI}^{(c)} \to 0$: ownership information is **diffuse** across the whole network (expensive to defend — no small set of layers can be selectively protected; conversely also harder for a targeted attacker to strip via a small number of layer choices, though a uniform low-bit-width attack may still succeed).

This index costs nothing beyond metrics already computed in Module ① (Section 4.2) and turns the carrier-fragility question from a single magnitude ("how fragile") into a **two-dimensional characterization**: *(fragility, localization)*. This is expected to differentiate carriers structurally, not just by degree — e.g., a plausible hypothesis is that parameter/decoder watermarks are both fragile **and** localized (concentrated in specific weight-space or decoder-layer directions, especially for low-rank embedding schemes), while latent/noise watermarks are more robust **and** diffuse (verification depends on the fidelity of the entire DDIM-inversion pass through the network, not a nameable subset of layers) — reported as Finding 7 (Section 8).

---

## 3. Multi-Objective Optimization Formulation

**Objectives** over bit-allocation $\mathbf{b}$:

- $f_1(\mathbf{b}) = \Delta\text{FID}(\mathbf{b}) = \text{FID}(\hat\theta_\mathbf{b}) - \text{FID}(\theta)$  *(generation utility loss, minimize)*
- $f_2(\mathbf{b}) = 1 - V(\hat\theta_\mathbf{b})$  *(watermark unreliability, minimize)*
- $f_3(\mathbf{b}) = -\text{Eff}(\mathbf{b})$, with $\text{Eff}(\mathbf{b}) = \dfrac{\sum_l p_l \cdot 16}{\sum_l p_l \cdot b_l}$ (compression ratio; $p_l$ = #params in layer $l$) *(maximize efficiency ⇔ minimize negative)*

**Scalarized (weighted Tchebycheff) form** for tractable search:
$$
\min_{\mathbf{b}} \; \max_i \; w_i\,\big|f_i(\mathbf{b}) - z_i^*\big|
$$
with ideal point $z^*$ from independently optimizing each objective, and weights $w_i$ swept to trace the Pareto front.

**Constrained (deployment-realistic) form**, used as the primary reported formulation:
$$
\min_\mathbf{b} \; \text{Cost}(\mathbf{b}) \quad \text{s.t.} \quad \Delta\text{FID}(\mathbf{b}) \le \epsilon_1,\;\; \Delta\text{CLIP}(\mathbf{b}) \le \epsilon_2,\;\; V(\hat\theta_\mathbf{b}) \ge \tau
$$

**Adversarial (evasion) counterpart**, used to generate worst-case watermark-survival curves:
$$
\max_\mathbf{b} \; \big[1 - V(\hat\theta_\mathbf{b})\big] \quad \text{s.t.} \quad \Delta\text{FID}(\mathbf{b}) \le \epsilon_1,\;\; \Delta\text{CLIP}(\mathbf{b}) \le \epsilon_2
$$

Both are solved with the same search procedure (Section 4), differing only in objective sign — this symmetry is itself a reportable finding (i.e., how close defender-optimal and attacker-optimal configs are in Pareto space; small distance = fragile watermark).

**Solvers:** NSGA-II (population-based, for full Pareto front); greedy/DP layer allocation (fast, near-optimal under separability assumption in Section 4); Bayesian optimization over $\mathbf{b}$ for small $L$ (e.g., block-level granularity rather than per-layer, to keep search space tractable).

---

## 4. System Architecture: DS-WAQ (Dual-Sensitivity Watermark-Aware Quantization)

This section is the paper's **method contribution** for CV-venue submission (CVPR/ICCV/journal): a named, modular, reusable framework — not merely an analysis protocol. DS-WAQ wraps around *any* existing PTQ toolchain (RTN, GPTQ, AWQ, SmoothQuant) as a pre-processing/calibration-time layer; it does not replace the quantizer, it decides *what precision to feed it*, which is what makes it deployable without touching quantization kernels.

### 4.1 Architecture overview

```
                 ┌─────────────────────────────────────────────────────────┐
                 │                    DS-WAQ  Pipeline                      │
                 │                                                          │
  θ, φ, w  ───▶  │  ① Dual-Sensitivity   ──▶  ② Cross-Carrier   ──▶  ③      │
  D_cal          │     Profiler               Divergence Map      Constrained│
                 │  (S_l^gen, S_l^wm,                              Allocator │
                 │   per-timestep S(l,t))                             │      │
                 │                                                    ▼      │
                 │                                          ④ Carrier-      │
                 │                                          Specific         │
                 │                                          Verification    │
                 │                                          Sweep           │
                 └─────────────────────────────────────────────────────────┘
                                                                     │
                                                                     ▼
                                                     b* (bit-map) ──▶ [existing PTQ
                                                                       toolchain:
                                                                       GPTQ/AWQ/...]
                                                                     │
                                                                     ▼
                                                          deployed θ̂ + robustness report
```

Each module is independently evaluable and independently ablatable — required for a CV-venue ablation table (Section 6/7): removing Module ① collapses DS-WAQ to standard HAWQ-style allocation (the "no dual-sensitivity" baseline); removing Module ④ collapses it to single-carrier allocation (the "no cross-carrier check" baseline). This turns the four bullet-point "problems" raised in review into four **ablation rows** rather than open weaknesses.

### 4.2 Module ① — Dual-Sensitivity Profiler (corrected timestep formulation)

The layer-wise scores $S_l^{gen}, S_l^{wm}$ (Section 2.1) are computed once per layer via calibration-set forward/backward passes — no architecture-specific assumption about "layers active at a timestep" (UNet weights are shared across $t$, so this framing was invalid in the original draft and is dropped).

The **timestep-resolved** component instead uses **trajectory teacher-forcing**, which isolates a layer's contribution at a specific point in the denoising trajectory without conflating it with error the layer itself accumulated upstream:

1. Run one reference trajectory in full precision, caching $x_t$ at every bin $t_k$.
2. For each layer $l$ and each cached $x_{t_k}$: swap in $\hat\theta_l$ (layer $l$ quantized, all other layers FP16), run **one forward step from the cached, un-perturbed** $x_{t_k}$, and measure
$$
S(l, t_k) = \big\|\,\epsilon_\theta(x_{t_k}, t_k) - \epsilon_{\theta \setminus l \to \hat\theta_l}(x_{t_k}, t_k)\,\big\|_2
$$
Because $x_{t_k}$ is held fixed (teacher-forced from the reference trajectory rather than the quantized model's own rollout), $S(l,t_k)$ measures **layer $l$'s marginal contribution at step $t_k$ only**, uncontaminated by compounding error from earlier steps. Compounding effects are captured separately by $E_{cum}(t)$ (Section 2.2f), computed on the *actual* quantized rollout.
3. Repeat step 2 with $\mathcal{L}_{wm}$ in place of the noise-prediction target to obtain $S^{wm}(l,t_k)$ per carrier, using the same cached $x_{t_k}$.

This produces two joint maps per carrier $c$: $M^{gen}[l,t_k] = S(l,t_k)$ and $M^{wm,(c)}[l,t_k] = S^{wm,(c)}(l,t_k)$, visualized as heatmaps, e.g.:

```
                    timestep bin
                 t1(high-noise) t2      t3      t4(low-noise)
layer  attn.mid       .low      .low    HIGH     HIGH        <- gen-sensitive late only
layer  conv.up3       HIGH      HIGH    .low     .low        <- gen-sensitive early only
layer  attn.mid       .low      .low    .low     HIGH        <- M^wm (decoder carrier):
                                                                 wm-sensitive late ONLY,
                                                                 i.e. DIVERGES from gen map
                                                                 at same layer → Div_l high
```

The bottom two rows illustrate the target finding: `attn.mid` looks *generation-safe* at $t_1$–$t_3$ (low $S^{gen}$) but is *watermark-critical* at $t_4$ for the decoder carrier — a divergence invisible to any generation-only profiler.

### 4.3 Module ② — Cross-Carrier Divergence Map

Aggregates $\text{Div}_l^{(c)}$ (Section 2.1b) and its timestep-resolved analogue $\text{Div}(l,t_k)^{(c)} = M^{wm,(c)}[l,t_k] / (M^{gen}[l,t_k]+\delta)$ across all four carriers into a single tensor $D \in \mathbb{R}^{L \times K \times 4}$. Three summary statistics are reported: (i) **per-carrier divergence hotspots** — the top-$p\%$ (layer, timestep) cells for each carrier; (ii) **cross-carrier agreement**, the Jaccard overlap between hotspot sets of different carriers (low overlap ⇒ carriers are protected by genuinely different mechanisms, a finding relevant to defenders who must choose *one* carrier to rely on); (iii) **Localization Index $\text{LI}^{(c)}$** (Section 2.3), computed once per carrier from the same $S_l^{wm,(c)}$ profile — positions each carrier on the *(fragility, localization)* plane used in Section 8.

### 4.4 Module ③ — Constrained Allocator

```
Input: M^gen, M^wm(c) for all carriers c, budget target b̄, tolerances ε_FID, ε_CLIP, τ

1. score(l) ← max_c Div_l^(c)          # worst-case carrier drives the allocation
2. sort layers by score(l) descending
3. initialize all layers at INT4
4. while Cost(b) > b̄ OR any constraint violated:
     promote next-highest-score(l) layer: INT4 → INT8 → FP16
     re-evaluate ΔFID, ΔCLIP (cheap surrogate, Section 5) and V^(c)(θ̂) for all c
     if all constraints satisfied: break
5. return b*, plus the adversarial counterpart b^atk (Section 3) computed by
   inverting the sort order (least generation-sensitive, most watermark-sensitive first)
```
This is a greedy separable relaxation of the knapsack-structured problem in Section 3; NSGA-II is used to validate near-optimality on a reduced layer subset (Section 6, Tier 1).

### 4.5 Module ④ — Carrier-Specific Verification Sweep

```
for carrier in {parameter, decoder, latent/noise, behavioral}:
    recompute V^(c)(θ̂) independently under b*
    if V^(c)(θ̂) < τ:
        flag (b*, carrier) as an exposed vulnerability even though the
        allocator satisfied its own optimization constraints — i.e., b*
        was tuned against max_c Div but empirical verification can still
        fail if Div_l^(c) underestimates true carrier fragility (reported
        as a calibration-quality diagnostic, not swept under the rug)
```

Key design point carried over from the original draft: Module ④ **re-derives verification per carrier** rather than trusting the divergence map alone, since $\text{Div}_l^{(c)}$ is a Hessian-trace *proxy*; empirical mismatch between predicted and actual $V^{(c)}$ is itself reported (Section 7) as a validity check on the whole profiler.

---

## 5. Loss / Objective Functions

Since **no fine-tuning occurs**, these are *measurement and search-selection objectives*, not weight-update losses (the only per-layer statistics fit are calibration scale/zero-point, standard in PTQ).

**Calibration loss** (per layer, for scale/zero-point fitting only):
$$
\mathcal{L}_{cal}(s_l, z_l) = \mathbb{E}_{x \sim \mathcal{D}_{cal}}\big[\|f_l(x;\theta_l) - f_l(x; Q_{b_l}(\theta_l))\|_2^2\big]
$$

**Generation fidelity surrogate** (used inside search, cheaper than full FID):
$$
\mathcal{L}_{gen} = \text{LPIPS}\big(D_\phi(z_0^{\theta}),\, D_\phi(z_0^{\hat\theta})\big)
$$

**Watermark preservation objective:**
$$
\mathcal{L}_{wm} = \text{BCE}(\hat w, w) = -\sum_{i=1}^k \big[w_i \log \hat w_i + (1-w_i)\log(1-\hat w_i)\big]
$$
or, for statistical-test carriers (latent/noise), $\mathcal{L}_{wm} = -\log p\text{-value}$ (drive toward significance).

**Efficiency regularizer:**
$$
\mathcal{L}_{bits} = \frac{1}{L}\sum_l b_l \quad \text{(mean bit-width, to minimize)}
$$

**Combined search-selection objective:**
$$
\mathcal{L}_{total}(\mathbf{b}) = \alpha\,\mathcal{L}_{gen}(\mathbf{b}) + \beta\,\mathcal{L}_{wm}(\mathbf{b}) + \gamma\,\mathcal{L}_{bits}(\mathbf{b})
$$
used to rank candidate configurations in Module ③ (Constrained Allocator, Section 4.4); $\alpha,\beta,\gamma$ swept to produce the Pareto set reported in Section 7.

---

## 6. Experimental Pipeline (tiered scope — controls combinatorial blow-up)

The original flat design (3 backbones × 4 carriers × 4 toolchains × 3 bit-widths, plus config classes) is not tractable as a single-paper contribution. Scope is split into a **deep primary tier** and a **shallow generalization tier**, which is also the standard structure reviewers expect (full study on one setting, transfer-check on others).

```
TIER 1 — Primary study (full grid; this is where all headline numbers come from)
  Backbone   : SD-1.5 only
  Carriers   : all 4 (parameter, decoder/VAE, latent/noise, behavioral) — this axis is
               the paper's core independent variable and is not cut
  Toolchains : 2 only — RTN (weak/naive baseline) and AWQ (strong baseline);
               GPTQ/SmoothQuant deferred to Tier 2 spot-checks, not full grid
  Bit-widths : {INT8, INT4} (drop uniform-FP16-only ablation into a single reference row)
  Configs    : uniform-b, DS-WAQ-allocated b*, adversarial b^atk, random-b control
               → 4 config classes × 2 toolchains × 2 bit-widths × 4 carriers
               = 64 evaluated settings (tractable: each is one PTQ pass + one eval pass)

  Stage 1.1 — Baseline profiling (FP16): verify V^(c)(θ)=1 all carriers @ FPR=1e-4;
              record FID_0, CLIP_0 on COCO-30k subset + PartiPrompts (N≈1000 prompts)
  Stage 1.2 — Run DS-WAQ (Section 4) to obtain b*, b^atk, and divergence maps
  Stage 1.3 — Evaluate all 64 settings: FID, CLIP, per-carrier V, sensitivity metrics
  Stage 1.4 — Statistical analysis: bootstrap CIs (1000 resamples) on TPR/FID over seeds;
              paired significance tests between config classes;
              correlation of cosφ_l (parameter carrier only) vs. observed ΔV;
              correlation of Div_l^(c) vs. observed ΔV^(c) (all carriers) — the
              main validity check for the profiler

TIER 2 — Generalization check (narrow; answers "does this transfer?", not full grid)
  Backbones  : SDXL, and one DiT-based model
  What is run: ONLY the FP16 baseline, uniform-INT8, uniform-INT4, and the single
               b* found transferable-by-construction from Tier 1 (re-profiled cheaply:
               Module ① only, ~1 profiling pass, no re-optimization)
  Toolchain  : AWQ only (the Tier-1 stronger baseline)
  Purpose    : test whether Tier-1 qualitative findings (carrier fragility ordering,
               timestep asymmetry, divergence hotspots) hold architecture-to-architecture,
               NOT to re-derive a new optimum per architecture
  Cost       : O(1) extra backbones × O(1) configs, independent of Tier-1 grid size

TIER 3 — Toolchain robustness spot-check (optional, strengthens but not required)
  Re-run Tier-1's best/worst config pair (b* and b^atk) under GPTQ and SmoothQuant
  on SD-1.5 only, to confirm findings are not an artifact of the AWQ/RTN choice
```

This keeps the **carrier axis** (the scientific core of the paper) fully crossed in Tier 1, while collapsing the **backbone axis** and **toolchain axis** to spot-checks — the standard way CV papers scope a combinatorial robustness study without turning it into a pure engineering exercise.

---

## 7. Evaluation Metrics

| Metric | Definition / protocol | Role |
|---|---|---|
| **FID** | Fréchet distance between generated-image feature distribution (Inception-V3) and reference set, computed per quantization config | Generation utility |
| **CLIP score** | Mean cosine similarity between CLIP image and text embeddings over prompt set | Text-image alignment / utility |
| **Watermark detection accuracy** | Bit accuracy (parameter/decoder carriers), TPR at fixed FPR (e.g., 1%), AUROC over positive/negative (unwatermarked) sets | Ownership signal reliability |
| **Fingerprint verification accuracy** | Classification accuracy of behavioral-signature matcher on trigger-set outputs, pre- vs. post-quantization | Behavioral ownership reliability |
| **Robustness–bit-width trade-off** | Joint plot of $\text{Eff}(\mathbf{b})$ (x-axis) vs. $V(\hat\theta)$ (y-axis) at fixed $\Delta\text{FID} \le \epsilon$, per carrier | Core trade-off curve, headline result |
| **Divergence–ΔV correlation** | Pearson/Spearman correlation between $\text{Div}_l^{(c)}$ (Sec. 2.1b) and observed $\Delta V^{(c)}$, per carrier | Validates the profiler (Module ①) as a cheap predictor |
| **Ownership alignment score correlation** | Correlation between $A_l^{(c)}$ (Sec. 2.2e, carrier-general) and observed $\Delta V^{(c)}$, per carrier — reported for all four carriers, not restricted to the parameter carrier | Validates the generalized alignment score; supersedes the earlier parameter-only interference-angle check |
| **Carrier disagreement rate** | Jaccard overlap between top-$p\%$ divergence hotspot sets (Module ②) of different carriers | Cross-carrier vulnerability finding |
| **Localization Index $\text{LI}^{(c)}$** | Effective participation ratio of ownership-gradient magnitude across layers (Sec. 2.3), reported per carrier | Positions each carrier on the *(fragility, localization)* plane; predicts whether cheap targeted defense/attack is possible |
| **High-Div/Low-$S^{gen}$ layer fraction** | Fraction of layers with $\text{Div}_l^{(c)}$ above the 90th percentile but $S_l^{gen}$ below the 25th percentile, per carrier | Direct evidence for the paper's headline claim (Section 8, Finding 1) |
| **Attack success rate (evasion)** | Fraction of adversarial-search configs achieving $V(\hat\theta) < \tau$ while $\Delta\text{FID} \le \epsilon_1$, $\Delta\text{CLIP} \le \epsilon_2$ | Security-relevant headline metric |

---

## 8. Expected Findings and Contributions

**Hypothesized findings** (to validate/falsify empirically):

1. **Generation-safe ≠ watermark-safe (headline finding).** A non-trivial fraction of layers will show high $\text{Div}_l^{(c)}$ while ranking as low-sensitivity under every existing generation-quality PTQ criterion (HAWQ/MixDQ/Q-DiT/SVDQuant would all approve compressing them). This is the finding the whole architecture (Section 4) is built to surface, and is reported quantitatively via the "High-Div/Low-$S^{gen}$ layer fraction" metric (Section 7) — expected to be non-zero for at least the parameter and decoder carriers, which sit closest to $\theta$/$\phi$.
2. **Carrier fragility ordering**: parameter watermarks are hypothesized most fragile to weight-space PTQ (direct interference in $\theta$); decoder/VAE watermarks moderately fragile (fine-detail-dependent, sensitive to late-decoder activation quantization); latent/noise watermarks most robust to weight quantization (operate upstream in $z_T$, largely orthogonal to UNet weight perturbation while global structure holds); behavioral fingerprints showing trigger-set-dependent variance.
3. **Timestep sensitivity is carrier-dependent, not just model-dependent.** Using the corrected teacher-forced $S(l,t_k)$ (Section 4.2), quantization error is expected to matter most at **early (high-noise) timesteps** for generation quality (structure-setting, compounds via $E_{cum}(t)$), while carriers embedded in fine detail (decoder watermark) are expected to peak in $\text{Div}(l,t_k)$ at **late** timesteps — i.e., the gen-sensitivity heatmap and the per-carrier wm-sensitivity heatmap (Module ①, Section 4.2 example) are expected to be *misaligned*, which is precisely what Finding 1 requires at the timestep-resolved level.
4. **Divergence and alignment scores as cheap early-warning metrics**: $\text{Div}_l^{(c)}$ and the generalized ownership alignment score $A_l^{(c)}$ (Section 2.2e — now carrier-general, not restricted to parameter watermarks) are expected to jointly predict $\Delta V^{(c)}$ well enough to prune the mixed-precision search space by an order of magnitude, validated per-carrier.
5. **Defender/attacker Pareto proximity**: if defender-optimal $b^*$ and adversarial $b^{atk}$ (Module ③) lie close in $(\text{Eff}, \Delta\text{FID})$ space, this demonstrates that *undetectable watermark evasion is achievable using only standard, unmodified PTQ tooling* — no custom attack code required, the central USENIX Security contribution.
6. **INT8 generally safe, INT4 carrier-dependent**: plausible outcome is that INT8 preserves most carriers within acceptable TPR loss, while INT4 bifurcates carriers into "survives" vs. "collapses," making carrier choice itself a robustness design decision for practitioners.
7. **Fragility and localization are separate axes.** Using $\text{LI}^{(c)}$ (Section 2.3), carriers are expected to separate on a *(fragility, localization)* plane rather than a single fragility scale: parameter and decoder/VAE watermarks are hypothesized **fragile and localized** (a small, nameable set of layers/directions carries most of the ownership signal — often true by construction for low-rank or fine-tuned-signature schemes), while latent/noise watermarks are hypothesized **more robust but diffuse** (verification quality depends on end-to-end DDIM-inversion fidelity through the whole network, with no small subset of layers responsible). If confirmed, this has a direct design implication: localized carriers admit a cheap, targeted defense (protect few layers) but are also cheaply targeted by an attacker; diffuse carriers cannot be cheaply defended by selective precision but are also harder for an attacker to strip with a small, targeted bit-allocation change.

**Contributions:**
- **Method/architecture contribution (primary, for CV venues):** DS-WAQ (Section 4) — a named, modular, toolchain-agnostic framework with four independently ablatable components (Dual-Sensitivity Profiler, Cross-Carrier Divergence Map, Constrained Allocator, Verification Sweep) that plugs in front of any existing PTQ pipeline (RTN/GPTQ/AWQ/SmoothQuant) without modifying the quantization kernel itself. This is positioned as the paper's algorithmic contribution, distinct from and building on prior sensitivity-based mixed-precision work (HAWQ/MixDQ/Q-DiT/SVDQuant), whose single-objective sensitivity score is a special case of DS-WAQ with Module ② disabled.
- **Empirical contribution:** first systematic, multi-carrier measurement of the divergence between generation-sensitivity and watermark-sensitivity in diffusion UNets, at both layer and timestep resolution, across four structurally distinct ownership-signal carriers.
- **Metric contribution:** the divergence score $\text{Div}_l^{(c)}$ and the carrier-general ownership alignment score $A_l^{(c)} = \cos\angle(\theta_l - Q(\theta_l),\, \nabla_{\theta_l}V^{(c)})$ as cheap-to-compute predictive diagnostics that require no explicit additive watermark vector — a strict generalization of weight-space-only interference metrics, together with the Localization Index $\text{LI}^{(c)}$ (Section 2.3) as a novel structural axis separating carriers by *how concentrated* (vs. diffuse) their ownership information is, independent of *how fragile* it is.
- **Security reframing (USENIX Security track):** demonstration that standard, off-the-shelf PTQ pipelines can function as an **unintentional or intentional watermark-evasion mechanism** reachable via the same Constrained Allocator run in reverse (Module ③, adversarial mode) — motivating quantization-invariance guarantees in watermark design and platform-level re-verification after any deployment-time model transformation, not only fine-tuning. The Localization Index further predicts *which* carriers are cheaply attackable via a small, targeted bit-allocation change (high $\text{LI}^{(c)}$) versus which require a broad, harder-to-disguise precision reduction (low $\text{LI}^{(c)}$) — directly informative for threat modeling.
- **Practical deployment guidelines:** recommended minimum bit-width per carrier type, and a decision framework for practitioners choosing which watermark carrier to rely on under an INT4 latency/memory budget.

**Threats to validity / limitations to disclose in the paper:** calibration-set choice can shift sensitivity profiles; results may not transfer across UNet vs. DiT architectures without re-profiling; behavioral fingerprint robustness depends heavily on trigger-set design and may not generalize to adaptive triggers; FID/CLIP are imperfect utility proxies and should be supplemented with human preference study for camera-ready claims; the alignment score $A_l^{(c)}$ and Localization Index $\text{LI}^{(c)}$ are *local, per-layer* directional diagnostics and deliberately avoid claiming a literal additive decomposition of $\theta$ into disjoint generation/ownership components (Section 2.2e) — readers should not over-interpret $\text{LI}^{(c)}$ as evidence of a "watermark subnetwork" in the strong sense; it is a gradient-magnitude concentration statistic, not a causal isolation result, and should be validated against layer-ablation ground truth ($\Delta V^{(c)}$ from directly quantizing candidate layers) rather than taken at face value from gradients alone.
