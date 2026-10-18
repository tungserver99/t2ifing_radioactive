# QErase: Ownership-Selective Malicious Quantization for Diffusion Models
### with QShield — Adversarially Quantization-Robust Watermark Embedding
*(certification is a stated target for QShield, not a claimed result — see Section 4.2)*

---

## 0. Positioning and Verified Related Work

**Core question.** Is robustness to *standard* PTQ (uniform INT8/INT4, or sensitivity-searched mixed precision) sufficient to guarantee robustness to an *adversarially chosen* quantizer?

**Central claim:**
$$
\text{standard quantization robustness} \;\not\Rightarrow\; \text{worst-case quantization robustness}
$$

This reframes the quantizer itself as an adversarial object with a bounded but non-trivial degree of freedom — not a fixed, benign compression operator.

**Why this is a legitimate, well-precedented direction (verified, not assumed):**

| Precedent | Venue | What it establishes |
|---|---|---|
| Egashira et al., *Exploiting LLM Quantization* | NeurIPS 2024 | Quantization itself is a viable attack surface: a model benign at full precision can be malicious once quantized. Establishes "malicious quantization" as a mainstream security research topic. |
| Chen et al., *QuRA: Rounding-Guided Backdoor Injection* | NDSS 2026 | Quantization rounding alone (no retraining) can inject/amplify backdoors with near-zero utility cost — confirms rounding-direction manipulation is a known, specific technique, which is why this proposal treats rounding as *one degree of freedom / ablation*, not the headline contribution. |
| Müller et al., *Black-Box Forgery Attacks on Semantic Watermarks for Diffusion Models* | **CVPR 2025 Oral** | Diffusion watermarks (Tree-Rings, Gaussian Shading) can be forged/removed under realistic black-box conditions; the authors state plainly that **no effective defenses currently exist**. This is the open problem QShield targets. |
| Qi, Li, Liang, Tu, Tao, *Cert-LAS* | ICML 2026 | First certified model-ownership-verification method robust to *malicious removal attacks*, via layer-adaptive smoothing over a general perturbation model. Strong baseline; QErase/QShield differ by restricting (attacker) or certifying against (defender) the *specifically quantization-realizable* perturbation set, not an arbitrary norm-ball. |
| Chen et al., *Lossless Copyright Protection via Intrinsic Model Fingerprinting* (TrajPrint) | arXiv 2601.21252 | Explicitly tests diffusion-model fingerprint robustness under **standard quantization (FP16, BFloat16)** and reports bit-accuracy consistently **>0.95** — i.e., standard quantization does *not* break this fingerprint. This is the verified experimental foil QErase needs: show $V(Q_{\text{standard}}(\theta)) \approx V(\theta)$ (already established by this paper) but $V(Q_{\text{QErase}}(\theta)) < \tau$ at the *same* bit-width and utility budget. |

**A citation used in an earlier draft of this critique chain, "Fingerprinting Text-to-Image Diffusion Models via Collapsed Generation" (claimed arXiv 2608.11732), could not be verified after three independent search attempts across this project and returned no matching paper.** It is dropped from this proposal's related-work claims; TrajPrint (above) is used in its place as a verified substitute making the same kind of claim (standard-quantization survival of an intrinsic fingerprint). Any future citation of "Collapsed Generation" should be re-verified against the actual arXiv listing before submission — do not carry it forward on the strength of repetition alone.

Given this, the honest novelty claim is **not** "first malicious quantization attack" (false — NeurIPS 2024, NDSS 2026 exist) and **not** "first diffusion watermark removal attack" (false — CVPR 2025 Oral exists). The defensible claim is:
$$
\boxed{\text{first attack and matching quantization-robust defense that treats the } \textit{deployment-realizable quantization error set}\text{ itself as the adversary's action space for diffusion ownership verification}}
$$
— i.e., steering the *error a real PTQ pipeline can actually produce*, rather than an arbitrary weight-space or pixel-space perturbation.

---

## 1. Threat Model and Problem Formulation

### 1.1 The quantizer as a parametric adversarial object

A post-training quantizer is normally treated as a fixed operator $Q_b$ indexed only by a bit-width choice. Generalize it to a family indexed by a richer, but still *deployment-realizable*, parameter $\psi$:
$$
Q_\psi, \qquad \psi \in \Psi_{\text{deploy}}
$$
where $\Psi_{\text{deploy}}$ ranges over legitimate degrees of freedom a real PTQ toolchain exposes: per-group/per-channel bit-width, clipping thresholds, scale/zero-point choice, calibration-set/distribution choice, timestep-weighted calibration (diffusion-specific), and rounding direction *as one ablation axis, not the main mechanism* (per the QuRA precedent above — an attack that reduces to "QuRA applied to watermarks" would not be a sufficient contribution on its own).

### 1.1.1 Formalizing $\Psi_{\text{deploy}}$: allowed and forbidden operations

The prose list above is not, by itself, a falsifiable threat model — a reviewer's fair question is *exactly* what "deployment-realizable" excludes. The following table is the operational definition Modules ①–④ and QShield are evaluated against; anything in the "forbidden" rows produces a $\Delta\theta \notin \mathcal{E}_Q(\theta)$ by construction (Section 1.2) and is out of scope for OS-MQ, not merely disallowed by convention.

| Degree of freedom | Allowed in $\Psi_{\text{deploy}}$? | Constraint |
|---|---|---|
| Weight bit-width | ✓ | Backend-supported values only (e.g., W4/W8) |
| Activation bit-width | ✓ (where the backend quantizes activations) | Backend-supported values only (e.g., A8/A16) |
| Group size | ✓ | Backend-supported values only (e.g., {32, 64, 128, per-channel}) |
| Symmetric vs. asymmetric scheme | ✓ | Must be a scheme the target backend actually supports |
| Scale | ✓ | Must be a valid backend-representable scale for the chosen bit-width/scheme |
| Zero-point | ✓ | Must lie in the legal integer range for the chosen bit-width |
| Clipping threshold | ✓ | Bounded; must not clip so aggressively that a standard calibration sanity check would flag the tensor as corrupted |
| Rounding direction | ✓ — but *one ablation axis*, not the headline mechanism (Sections 0, 1.1) | Must map to a valid quantization level under the chosen scale/zero-point |
| Calibration sample selection | ✓ | Drawn from a fixed, disclosed calibration pool |
| Calibration / timestep reweighting | ✓ | Simplex-constrained (non-negative, sums to 1) |
| Arbitrary weight modification not expressible as $Q_\psi(\theta)-\theta$ | **✗ forbidden** | Would make the artifact a generic weight-poisoning attack, not a quantization attack — explicitly out of scope |
| Fine-tuning or gradient-updating the victim model | **✗ forbidden** | Module ④ (Section 3.4) touches the victim only via the frozen quantizer $Q_{\psi^*}$, never via weight updates |
| Access to the victim secret/key, fingerprinted instance, or verifier at attack-construction time | **✗ forbidden** (Module ④, Section 3.4) | This restriction is exactly what makes the Section 3.4 result a security finding rather than an oracle white-box result |

### 1.2 Quantization-Realizable Error Set (QRES) — *not* a cone

$$
\mathcal{E}_Q(\theta) = \big\{\, Q_\psi(\theta) - \theta \;:\; \psi \in \Psi_{\text{deploy}} \,\big\}
$$
**Terminology correction.** An earlier draft called this a "Quantization Error Cone." That name is mathematically wrong: a cone requires closure under non-negative scaling ($e \in C \Rightarrow \alpha e \in C,\ \alpha \ge 0$), but $\mathcal{E}_Q(\theta)$ is generally *not* scale-invariant — quantization levels are discrete, clipping thresholds are bounded, and bit-width is fixed, so if $e$ is realizable, $2e$ or $0.3e$ is generally not. Calling it a cone invites a reviewer to immediately flag a formal-correctness error. It is correctly described as the **Quantization-Realizable Error Set (QRES)**: the set of perturbations a real, standard-looking quantization pipeline can actually produce. This is what separates the threat model from generic adversarial weight-perturbation research: **the attacker cannot inject an arbitrary $\Delta\theta$**, only an $e \in \mathcal{E}_Q(\theta)$ — which is exactly what makes the resulting artifact indistinguishable from an innocuous efficiency-driven deployment choice. (If a later stage of the project constructs a genuine conic *outer approximation* $\widehat{\mathcal{C}}_Q(\theta) \supseteq \mathcal{E}_Q(\theta)$ — e.g., for tractable certification in Section 4.2 — *that* object may legitimately be called a cone, since it would be constructed to have the scale-closure property by design; the exact realizable set itself should not be.)

### 1.3 Functional subspaces (local, not a global additive decomposition)

Consistent with the AQ-WM revision's caveat against assuming $\theta = \theta_{gen} + \theta_{own}$ as a literal decomposition (superposition makes this untrue in general), this framework uses **local Jacobian/gradient directions** instead of a structural partition of $\theta$:
$$
J^{gen}_{l,t} = \frac{\partial \epsilon_\theta(x_t,t)}{\partial \theta_l} \quad \text{(generation-relevant local direction)}, \qquad
g^{own}_{l,t,m} = \nabla_{\theta_l} V_m(\theta) \quad \text{(ownership direction, surrogate instance } m\text{)}
$$
Low-rank bases $U^{gen}_{l,t}, U^{own}_{l,t}$ are fit from these directions across a calibration set of prompts, timestep bins, and multiple surrogate ownership instances $m$ (Section 4.4), giving tractable, per-(layer, timestep) subspaces rather than a full-model decomposition claim.

### 1.4 Vulnerability condition

A model is quantization-vulnerable at carrier $m$ if there exists a *realizable* error achieving both generation-invisibility and ownership damage:
$$
\exists\, e \in \mathcal{E}_Q(\theta) \;:\; \|J_{gen}\,e\| \le \epsilon \quad \text{and} \quad g_{own,m}^\top e \ll 0
$$
i.e., **quantization-realizable** $\cap$ **generation-invariant** $\cap$ **ownership-destructive** is non-empty.

---

## 2. Core Metrics: Attack Budget Curve $A_Q(\epsilon)$ and Empirical Attackability Score

**Preferred primary metric — utility-budgeted ownership damage.** The ratio form used in an earlier draft,
$$
\text{QAI}(\theta, m) = \max_{e \,\in\, \mathcal{E}_Q(\theta)} \; \frac{-\,g_{own,m}^\top e}{\|J_{gen}\,e\|_2 + \delta}
$$
can be misleadingly large when both numerator and denominator are small (a numerically unstable regime with negligible absolute damage). The **constrained form** is preferred and used as the primary reported quantity, since it answers the threat model's actual question directly — "within a fixed utility budget, how much ownership damage can a realizable quantizer do?":
$$
A_Q(\epsilon) = \max_{e \,\in\, \mathcal{E}_Q(\theta)} \; -\,g_{own,m}^\top e \qquad \text{s.t.} \qquad \|J_{gen}\,e\|_2 \le \epsilon
$$
reported as a **curve over $\epsilon$** (the attack-budget curve), not a single scalar — this also directly supplies the x-axis for the mandatory comparison ladder in Section 6.3. The ratio form (renamed **Empirical Attackability Score**, not "QAI" to avoid implying a single clean index) is retained only as a secondary, budget-normalized summary statistic where a single number is convenient, always reported alongside the $\epsilon$ at which it was evaluated.

**Tractability note (fixing an implicit assumption in the original sketch).** $J_{gen}$ is the Jacobian of the noise-prediction network output with respect to (a block of) $\theta$ — this matrix is far too large to form explicitly for a diffusion UNet/DiT. Neither $A_Q(\epsilon)$ nor the Empirical Attackability Score needs the full matrix: $\|J_{gen}e\|$ for a candidate direction $e$ is a single **Jacobian-vector product**, computable in one forward-mode (or one extra backward-mode) autodiff pass, exactly as is already done for $S_l^{gen}$ profiling. The outer $\max_{e \in \mathcal{E}_Q(\theta)}$ has no closed form (the realizable set is defined implicitly by a quantizer's discrete/structured parameterization $\psi$, not by a simple norm ball), so both quantities are reported as **empirical lower-bound scores under a standardized search budget** — the best value found by the OS-MQ optimizer (Section 3) within a fixed, pre-registered number of search iterations/restarts. **Correction to an internal inconsistency in the original draft:** an earlier version of this proposal called this quantity both "budget-dependent" (correctly, in its derivation) and, later, a "budget-independent diagnostic" (in the metrics table) — these contradict each other. The correct, consistent description used throughout this document and in Section 7 is: *empirical lower-bound attackability score under a standardized search budget*, always reported together with that budget (number of optimizer restarts/iterations) so results are comparable across experiments.

---

## 3. Attack: OS-MQ (Ownership-Selective Malicious Quantizer)

Four modules. Module ① is inherited near-unchanged from the AQ-WM framework (reused explicitly, not re-derived); Modules ②–④ are the new contribution.

### 3.1 Module ① — Trajectory Ownership-Subspace Profiler (reused from AQ-WM)

Uses the teacher-forced timestep measurement already validated in the AQ-WM revision (cache $x_{t_k}$ from a reference FP trajectory, perturb one layer at a time, measure marginal effect at that timestep only) to estimate $J^{gen}_{l,t}$ and $g^{own}_{l,t,m}$ for each surrogate ownership instance $m$ (Section 3.4). No re-derivation needed here — see the AQ-WM document, Section 4.2, for the exact formulation and its cost caveats (in particular: computing $g^{own}$ for inversion-based or non-differentiable carriers is materially more expensive than for single-pass differentiable ones, and this cost is reported honestly, not assumed free).

### 3.2 Module ② — Quantization-Error Steering (continuous relaxation)

Optimize the quantizer's continuous degrees of freedom (clipping thresholds, per-group scale/zero-point, calibration-distribution reweighting, timestep-weighted calibration mixture) to steer the *resulting* quantization error toward ownership-destructive, generation-flat directions:
$$
\psi^* = \arg\max_{\psi \in \Psi_{\text{deploy}}} \; \Big[ -\,g_{own}^\top e_\psi \;-\; \lambda \, \|J_{gen}\,e_\psi\|_2^2 \Big], \qquad e_\psi = Q_\psi(\theta) - \theta
$$
solved by projected/relaxed gradient ascent over the continuous components of $\psi$ (scale, zero-point, clipping, calibration mixture weights are differentiable or sub-differentiable via straight-through estimators, standard in learnable-PTQ literature).

**Diffusion-specific trajectory loss** (the part that makes this a diffusion paper rather than a generic DNN attack, per the same requirement the original sketch correctly identified):
$$
\mathcal{L}_{MQ} = \sum_t w_t^{own}\big(-g_{own,t}^\top e_\psi\big) + \lambda \sum_t w_t^{gen}\,\|J_{gen,t}\,e_\psi\|^2
$$
concentrating the attack's effort on (layer, timestep) cells identified by Module ① as high-$\text{Div}$: ownership-sensitive and generation-flat simultaneously — but *steering into* them rather than merely *measuring* them, which is the methodological delta from AQ-WM's analysis-only stance.

### 3.3 Module ③ — Deployment Projection (concrete algorithm, not left abstract)

Continuous $\psi^*$ from Module ② is not itself a deployable artifact — it must be projected onto an actually exportable INT4/INT8 quantized checkpoint that a real backend can load. This projection is where most of the real engineering risk sits, so it needs a concrete procedure rather than being asserted:

```
Deployment Projection (GPTQ-style sequential greedy assignment,
retargeted from weight-reconstruction fidelity to malicious steering)

Input: continuous steering direction e_psi* from Module 2, layer l processed
       in the standard GPTQ column order

For each layer l (sequentially, as in GPTQ/Q-Diffusion-style calibration):
    For each weight column/group in l:
        # Standard GPTQ minimizes ||W x - hat(W) x||^2 (reconstruction fidelity)
        # Here the target is redefined: minimize a WEIGHTED reconstruction loss
        # that keeps the projection close to standard PTQ (for stealth/utility)
        # while rewarding alignment with the steering direction e_psi*:
        hat(w) = argmin over valid quantization levels q of
                 || (w - q) - proj_onto(e_psi*, this column) ||^2
                 + mu * ||standard GPTQ reconstruction error||^2
        # mu trades off "looks like ordinary PTQ" (large mu, stealthier,
        # weaker attack) against "follows the steering direction" (small mu,
        # stronger attack, more anomalous-looking weight statistics)
        propagate rounding error to remaining unquantized weights in the
        column (standard GPTQ error-compensation step)
Output: a real, loadable INT4/INT8 checkpoint; report its distributional
        similarity to a standard (non-malicious) PTQ output of the same
        toolchain, e.g., KL divergence of per-layer weight histograms,
        as a stealth metric (Section 7)
```
The stealth parameter $\mu$ makes explicit a trade-off the original sketch left implicit: full attack strength versus statistical indistinguishability from benign PTQ output — both should be reported as a curve, not a single operating point, since a defender inspecting weight statistics is a realistic (if partial) mitigation.

### 3.4 Module ④ — Secret-Agnostic Transfer (precise protocol)

This is the module that turns the result from "white-box optimization against a known watermark" into a genuine security finding, and it needs a precise, falsifiable protocol.

**Carrier-type split (required — "secret" does not mean the same thing for every carrier).** For *embedded-watermark* carriers with a discrete key (AquaLoRA, Stable-Signature-style decoder watermarks), the held-out unit below is an independently sampled **secret/key**. For *intrinsic-fingerprint* carriers with no separately embedded secret (FingerInv, TrajPrint-style trajectory fingerprints), there is no key to sample — the held-out unit is instead an independently trained/fingerprinted **target model instance** (and, where applicable, an independently chosen verification query set). The protocol below uses "surrogate secret" and "victim secret" loosely to cover both cases; **results for the two carrier families are always reported separately, never pooled into one transfer number**, since transfer-across-keys and transfer-across-fingerprinted-instances are different claims of different strength.

```
1. Attacker embeds K independent SURROGATE instances of the target carrier type
   into copies of the base model, using K independently sampled secrets/keys
   for embedded-watermark carriers (e.g., K different Stable-Signature decoder
   fine-tunes, or K different AquaLoRA keys), or K independently fingerprinted
   surrogate instances for intrinsic-fingerprint carriers (e.g., K separately
   trained FingerInv-style models) — attacker has full white-box access to
   these surrogates only.
2. Run Modules 1-3 using an objective AVERAGED (or worst-case, both reported)
   over the K surrogate g_own directions, producing a single psi* intended to
   generalize across secrets of that carrier type.
3. FREEZE psi* before touching the victim. No victim-specific information
   (secret key, verifier gradient, verifier output) is used from this point on.
4. Apply the frozen quantizer Q_{psi*} to an INDEPENDENTLY-KEYED victim instance
   (K+1-th secret, never used in step 1-2, ideally embedded by a different
   experimenter/seed to avoid accidental leakage).
5. Report V^(m)(Q_{psi*}(theta_victim)) — success is measured ONLY here.
```
Reporting requirement: transfer success should be broken out **by number of surrogate instances $K$** (does transfer improve with more surrogates, i.e., is there a sample-complexity story?) and **by whether the victim secret is generated by the same implementation/codebase as the surrogates** (same-codebase transfer is a weaker, easier claim than cross-implementation transfer — both are worth reporting, but should not be conflated into one number).

---

## 4. Defense: QShield

### 4.1 Minimax formulation

$$
\max_{\omega} \; \min_{\psi \in \Psi_{\text{deploy}}} \; V_\omega\big(Q_\psi(\theta_\omega)\big) \qquad \text{s.t. generation-utility and embedding-capacity constraints}
$$
Train the watermark embedding $\omega$ to remain verifiable under the **worst deployment-realizable quantizer**, not just a fixed W4/W8 recipe — structurally adversarial training, but the adversary is a quantizer restricted to $\mathcal{E}_Q(\theta_\omega)$, not a pixel or generic weight perturbation.

### 4.2 Differentiator from Cert-LAS (must be stated explicitly, since Cert-LAS is the closest prior work)

Cert-LAS certifies MOV against a general malicious-perturbation model via layer-adaptive randomized smoothing — a strong, already-published result (ICML 2026). QShield's distinguishing claim should be framed precisely as: **certification restricted to the quantization-realizable error set $\mathcal{E}_Q(\theta)$ can be tighter and more practically informative than certification against an arbitrary $\ell_p$-ball**, because $\mathcal{E}_Q(\theta)$ is a much smaller, structured set than a generic norm ball. If $\mathcal{E}_Q(\theta)$ admits a tractable conic or box **outer approximation** $\widehat{\mathcal{C}}_Q(\theta) \supseteq \mathcal{E}_Q(\theta)$ (e.g., a per-layer $\ell_\infty$ box implied by the clipping-threshold range of realistic calibration — see the Section 1.2 note on when the word "cone" is actually earned), a **quantization-specific certified radius** can potentially be derived directly over that outer approximation (e.g., via interval bound propagation over the box, or a randomized-smoothing argument restricted to the approximation rather than full weight space) — this is the concrete theoretical contribution to pursue, not a re-derivation of Cert-LAS's general result. If full certification proves intractable within the project timeline, an **empirical worst-case bound** (best attack found by OS-MQ against $Q_\omega$, expressed as the resulting verification floor at a given attack budget $\epsilon$, Section 2) is a legitimate fallback and should be reported as such rather than oversold as "certified."

### 4.3 Quantization Ownership Margin

**Correcting a definition/interpretation mismatch in the original sketch.** An earlier version defined $\rho_Q(\theta) = \min_\psi \|e_\psi\|_2$ (distance from the **full-precision model** $\theta$) but described it in prose as "how far a realizable quantizer must be pushed from a *benign deployment choice*" — that description actually refers to a different quantity, distance from a specific benign reference quantizer $Q_{\psi_0}$, not from $\theta$ itself. These are both useful but distinct, so both are defined explicitly and neither is used under an ambiguous name:
$$
\rho_Q(\theta) = \min_{\psi \,\in\, \Psi_{\text{deploy}}} \; \big\| e_\psi \big\|_2 \quad \text{s.t.} \quad V(Q_\psi(\theta)) < \tau, \;\; \Delta U(Q_\psi(\theta)) \le \epsilon
$$
correctly read as: **the minimum realizable quantization-perturbation magnitude (measured from the full-precision weights) required to invalidate ownership while staying inside the utility budget** — a robustness-radius analogue anchored at $\theta$, useful for the theoretical link to $\mathcal{E}_Q(\theta)$ and the certification discussion above. Separately,
$$
\rho_Q^{\text{benign}}(\theta) = \min_{\psi \,\in\, \Psi_{\text{deploy}}} \; \big\| Q_\psi(\theta) - Q_{\psi_0}(\theta) \big\|_2 \quad \text{s.t. same constraints}
$$
measures **how far a malicious quantizer must deviate from a specific, named benign deployment recipe $\psi_0$** (e.g., the toolchain's default Q-Diffusion/Q-DiT config) to break ownership — this is the operationally relevant quantity for the stealth discussion in Section 3.3, since a small $\rho_Q^{\text{benign}}$ means the malicious checkpoint sits close to what a defender would consider a normal deployment artifact. Both are reported, labeled explicitly, before/after QShield training; they answer different questions and should not be collapsed into one number.

---

## 5. Loss Functions (summary)

- **Attack steering loss** $\mathcal{L}_{MQ}$: Section 3.2.
- **Deployment-projection loss**: GPTQ-style reconstruction term + steering-alignment term, weighted by stealth parameter $\mu$ (Section 3.3).
- **Defense loss** (QShield training, standard adversarial-training structure): $\mathcal{L}_{QShield} = \mathcal{L}_{gen}(\theta_\omega) + \beta \cdot \mathcal{L}_{ver}\big(Q_{\psi^{atk}}(\theta_\omega)\big)$, where $\mathcal{L}_{ver} = \max(0, \tau - V_\omega(\cdot))$ (reusing the corrected margin loss from the AQ-WM revision — not the sign-buggy $-\log p$ form) and $\psi^{atk}$ is refreshed periodically by re-running OS-MQ against the current $\theta_\omega$ (min-max alternation, as in standard adversarial training).

---

## 6. Experimental Pipeline

### 6.1 Tracks
- **UNet track:** SD1.5 (primary), SDXL (compute permitting). Benign-PTQ reference: **Q-Diffusion** (ICCV 2023) — timestep-aware calibration, the domain-appropriate baseline (not RTN/AWQ, per the earlier review's correction).
- **DiT track:** one open DiT-compatible model. Benign-PTQ reference: **Q-DiT** (CVPR 2025).

### 6.2 Ownership carriers (structurally distinct, MOV vs. provenance kept separate per the earlier carrier-conflation fix)
- **Model Ownership Verification (MOV) targets** (primary): AquaLoRA-style parameter watermark (embedded-secret carrier — held-out unit is an independently sampled *key*), FingerInv-style intrinsic fingerprint (no separately embedded secret — held-out unit is an independently fingerprinted *target model instance*, per the carrier-type split in Section 3.4), **Cert-LAS as the strongest MOV baseline/opponent**. QErase's headline evaluation stress-tests Cert-LAS specifically, not a weaker baseline — but the paper only claims to have **broken** Cert-LAS's certificate if the realized attack falls *inside* the deployment-realizable conditions its certified radius is actually proven to cover (Section 4.2). If OS-MQ's realizable error instead falls outside those certified conditions, that is reported honestly as a scope finding — **evidence that Cert-LAS's certificate does not extend to deployment-realizable quantization** — rather than oversold as "breaking" a proof it was never shown to cover.
- **Content-provenance carriers** (auxiliary, explicitly labeled as a different security task): Stable Signature (decoder watermark), Tree-Ring/semantic noise watermark — included for breadth and because Black-Box Forgery (CVPR 2025 Oral) already targets this carrier family, giving a direct point of comparison.

### 6.3 Mandatory comparison ladder (same search budget, same utility budget, same deployment budget)
$$
\text{Fixed PTQ} \;<\; \text{Random adaptive PTQ} \;<\; \text{Sensitivity-search PTQ (AQ-WM-style)} \;<\; \text{QuRA-style rounding-only ownership attack} \;<\; \text{OS-MQ, timestep-flat ablation} \;<\; \boxed{\text{QErase (full OS-MQ)}}
$$
This ladder answers the reviewer's two most likely objections. **"Just a hyperparameter search that happens to hurt watermarks?"** — QErase must beat sensitivity-guided search at the *same* budget, not merely a fixed baseline. **"Isn't this just QuRA with a watermark loss instead of a backdoor loss?"** — the QuRA-style rung re-runs Chen et al.'s rounding-direction optimization (Section 0) with $\mathcal{L}_{MQ}$'s ownership term substituted for their backdoor objective, at the *same* surrogate/utility/deployment budget as QErase; the margin QErase shows over this rung is the evidence that novelty sits in structured multi-degree-of-freedom error steering (Section 3.2), not in "porting a malicious objective into quantization." The timestep-flat ablation ($w_t^{own}=w_t^{gen}=\text{const}$ in $\mathcal{L}_{MQ}$) isolates the layer×timestep trajectory-weighting claim from the rest of OS-MQ, since layer/timestep sensitivity variation alone is already established by MixDQ/Q-Diffusion/Q-DiT (Section 10) and must not be re-claimed as a discovery in its own right.

### 6.4 Three pilot results required before committing to full-scale experiments
1. **QErase beats both sensitivity-search PTQ and the QuRA-style rounding-only baseline** at equal budget (if not, either steering adds nothing over ranking+search, or the diffusion-specific contribution collapses into "QuRA with a different loss").
2. **Secret-agnostic transfer succeeds** on a genuinely held-out victim secret (if not, the paper is an oracle white-box study, not a security finding).
3. **Utility-preserving ownership collapse** is demonstrated with FID/KID/CLIP/DreamSim inside a pre-registered confidence band — "the image still looks fine" is not evidence on its own.

---

## 7. Evaluation Metrics

| Metric | Purpose |
|---|---|
| FID / KID / CLIP score / DreamSim | Generation utility preservation (pre-registered $\epsilon$ band, not post-hoc judged) |
| Watermark/fingerprint verification accuracy (TPR@FPR, bit accuracy) | Direct ownership-signal survival |
| **Attack success rate** (fraction of trials with $V < \tau$ within utility budget) | Headline attack metric |
| **$A_Q(\epsilon)$ curve** and **Empirical Attackability Score** (empirical lower-bound under a standardized search budget — Section 2) | Cheap attackability diagnostic, budget always reported alongside the score (not budget-independent — see Section 2's correction) |
| **Quantization Ownership Margins $\rho_Q$ (from $\theta$) and $\rho_Q^{\text{benign}}$ (from a named benign recipe)** (Section 4.3, reported as two distinct numbers, not conflated) | Defense-side robustness radii, before/after QShield |
| **Stealth metric** (weight-histogram KL-divergence vs. benign PTQ of the same toolchain, Section 3.3) | How distinguishable the malicious checkpoint is from an innocuous one — required for the security framing to be credible |
| Transfer success by $K$ (surrogate count) and by same-/cross-implementation secret | Section 3.4 reporting requirement |
| Comparison-ladder position (Section 6.3) | Positions QErase against the AQ-WM/adaptive-search baseline explicitly |

---

## 8. Contributions

**Consolidated from an earlier 7-item list, per the observation that several items were facets of the same contribution rather than independent ones** (threat model + feasible set + local directions is one formalization; layer×timestep steering is a mechanism inside OS-MQ, not a standalone discovery claim given MixDQ/Q-Diffusion/Q-DiT already established layer/timestep sensitivity variation in diffusion PTQ):

| # | Contribution |
|---|---|
| C1 — Threat + formalization | Adversarial deployment-realizable quantization for diffusion ownership verification: the Quantization-Realizable Error Set $\mathcal{E}_Q(\theta)$ (Section 1.2, correctly *not* called a cone) and the local functional subspaces $J_{gen}, g_{own}$ (Section 1.3) it is evaluated against — no literal $\theta_{gen}+\theta_{own}$ decomposition assumed |
| C2 — QErase / OS-MQ | Ownership-selective quantization-error steering (Section 3.2), including the diffusion-specific layer×timestep trajectory objective as an internal mechanism (not a separate discovery claim), and a concrete, realizable deployment-projection algorithm (Section 3.3) |
| C3 — Secret/fingerprint-agnostic transfer | **Empirical security finding, contingent on the Section 3.4 pilot succeeding:** quantization policies optimized solely against surrogate ownership instances — surrogate keys for embedded-watermark carriers, surrogate fingerprinted instances for intrinsic-fingerprint carriers (Section 3.4) — transfer to independently keyed / independently fingerprinted victim models without any victim-side access. The held-out-secret transfer protocol (Section 3.4) is the *evaluation instrument* for this finding, not the contribution itself; frozen attack, same-/cross-implementation breakdown, reported under a fixed utility/stealth budget, per carrier family |
| C4 — QShield | Quantization-specific adversarially-robust embedding (Section 4); certification is claimed **only if** a sound theorem/bound is actually derived (Section 4.2) — otherwise reported honestly as an empirical worst-case robustness result |

$A_Q(\epsilon)$ (Section 2) and the Quantization Ownership Margins $\rho_Q, \rho_Q^{\text{benign}}$ (Section 4.3) are **supporting analytical/audit tools** used throughout C1–C4, not additional headline contributions in their own right.

---

## 9. Ethics and Responsible Disclosure (required for this paper class — not optional)

Every verified precedent above (NeurIPS 2024, NDSS 2026, CVPR 2025 Oral, ICML 2026) pairs the attack finding with either a proposed defense, a disclosure statement, or an explicit call for the community to address the gap; this paper should do the same, concretely:

- **Scope the attack to open-source or self-controlled models only.** No experiments against a third party's deployed, non-consenting production system.
- **Co-develop and release QShield alongside QErase in the same paper** (already the plan) — do not publish the attack without a defense-side contribution, consistent with the norm set by Cert-LAS and the Black-Box Forgery paper's call for better defenses.
- **Responsible disclosure window** if any evaluated open-source watermarking/fingerprinting library is shown to be practically breakable by a realistic actor (e.g., disclose to maintainers of AquaLoRA/Stable-Signature-style public implementations before publication, consistent with standard security-conference norms, e.g., USENIX Security's disclosure policy).
- **Explicit stealth-vs-strength reporting** (Section 3.3's $\mu$ curve) rather than only reporting the strongest, most detectable attack point — this gives defenders an honest picture of what a detection-based mitigation would need to catch.
- A short **broader-impact statement** noting the dual-use nature (removing ownership verification could also be used to strip legitimate provenance/attribution from AI-generated content) and why the paper's net contribution is protective (the defense and audit metrics are usable immediately, and the attack surface being described is already reachable via existing published techniques in Section 0, so this work closes a gap rather than opening a new one).

---

## 10. Relationship to Prior Work in This Line

- **AQ-WM** (dual-sensitivity analysis, teacher-forced timestep profiling): retained as **Module ① of OS-MQ**, not published as a standalone contribution — per the earlier assessment that its "decoupled sensitivity" novelty is meaningfully overlapped by MixDQ's metric-decoupled sensitivity analysis (ECCV 2024).
- **Quantization Laundering** (deployment-valid config search + freeze/transfer protocol): retained as (i) the **adaptive-search baseline** in the mandatory comparison ladder (Section 6.3), and (ii) the **template for the secret-agnostic transfer protocol** (Section 3.4), which reuses its freeze-before-victim discipline.
- **QErase + QShield** is the paper-level contribution this line of work should converge on for a CVPR/ICCV/NeurIPS submission; a USENIX Security/CCS variant would drop or shrink the QShield section and instead deepen Sections 3.3–3.4 (deployment validity, stealth, transferability, prevalence across real open-source watermarking libraries).
