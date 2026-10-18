# Beyond Generation Quality: Ownership-Aware Post-Training Quantization for Diffusion Models

## 1. Introduction

Diffusion models have become a fundamental technology in generative AI, especially in systems such as Stable Diffusion and Diffusion Transformers. As these models are increasingly deployed, two challenges become important:

1. reducing inference cost through post-training quantization (PTQ);
2. preserving model ownership verification mechanisms.

Existing diffusion PTQ methods mainly optimize generation quality:

\[
\min_Q L_{gen}(Q(\theta))
\]

where \(Q\) represents the quantization transformation.

Meanwhile, ownership verification requires:

\[
V(Q(\theta)) \geq \tau
\]

where \(V\) represents verification reliability.

However, current PTQ algorithms do not explicitly consider ownership preservation.

This motivates the research question:

> Are the parameters and computational regions important for image generation also important for ownership verification?

We hypothesize:

\[
\boxed{
\text{Generation sensitivity}
\neq
\text{Ownership sensitivity}
}
\]

A diffusion model may preserve visual quality after quantization while losing ownership evidence.

---

# 2. Research Gap

## Diffusion PTQ

Existing methods:

- Q-Diffusion
- PTQD
- MixDQ
- SVDQuant
- Q-DiT

focus on:

- memory reduction;
- inference acceleration;
- generation quality preservation.

---

## Diffusion Ownership Verification

Existing methods:

- AquaLoRA
- FingerInv
- Cert-LAS

focus on:

- watermark robustness;
- fingerprint robustness;
- ownership verification.

---

## Missing connection

Current research does not answer:

> Which quantization configurations preserve both generation quality and ownership verification?

This work introduces:

\[
\boxed{
\text{Ownership-aware quantization analysis}
}
\]

---

# 3. Research Questions

## RQ1

Does diffusion PTQ create a mismatch between generation robustness and ownership robustness?

---

## RQ2

Which layers, components, and denoising timesteps contribute most to ownership degradation?

---

## RQ3

Can ownership-aware precision allocation preserve ownership verification under low-bit deployment?

---

# 4. Dual Sensitivity Analysis

For each layer \(l\) and quantization configuration \(q\):

## Generation sensitivity

\[
S^{gen}_{l,q}
=
\Delta Quality(Q_{l,q}(\theta))
\]

Measures:

- FID degradation;
- CLIP degradation;
- denoising error.

---

## Ownership sensitivity

\[
S^{own}_{l,q}
=
\Delta Verification(Q_{l,q}(\theta))
\]

Measures:

- watermark accuracy reduction;
- fingerprint confidence reduction.

---

## Ownership-Utility mismatch

Define:

\[
M_{l,q}
=
\frac{
S^{own}_{l,q}
}
{
S^{gen}_{l,q}+\epsilon
}
\]

A high value indicates:

- ownership-sensitive region;
- generation-insensitive region.

These regions are critical for ownership-aware quantization.

---

# 5. Diffusion-Specific Layer-Timestep Analysis

Diffusion models repeatedly apply denoising:

\[
x_T \rightarrow x_0
\]

Therefore, quantization effects may accumulate differently across timesteps.

We analyze:

\[
S(l,t)
\]

where:

- \(l\): model component;
- \(t\): denoising timestep.

The analysis produces:

\[
\mathcal{M}_{own}(l,t)
\]

which reveals:

- where ownership information propagates;
- whether early or late denoising stages dominate verification robustness.

---

# 6. Proposed Method: Ownership-Aware Mixed Precision Quantization (OA-MPQ)

Existing PTQ optimizes:

\[
\min Cost(Q)
\]

subject to:

\[
Quality(Q)
\]


This work optimizes:

\[
\min Cost(Q)
\]

subject to:

\[
Quality(Q)\leq\epsilon
\]

and:

\[
Verification(Q)\geq\tau
\]

---

## Layer allocation score

Define:

\[
A_l=
\frac{
S^{own}_{l}
}
{
S^{gen}_{l}+\epsilon
}
\]

High \(A_l\):

- preserve higher precision.

Low \(A_l\):

- aggressive quantization is allowed.

---

## Algorithm

```
Input:
Diffusion model θ
Ownership verifier V
Calibration dataset D

1. Profile each layer:
       INT4 / INT8 / FP16

2. Measure:
       generation degradation
       ownership degradation

3. Construct ownership sensitivity map

4. Initialize:
       all layers INT4

5. Restore precision:
       INT4 → INT8 → FP16

   Priority:
       highest ownership sensitivity layers

6. Output:
       ownership-preserving quantization configuration
```

---

# 7. Experimental Design

## Models

Primary:

- Stable Diffusion 1.5

Scaling:

- SDXL
- Diffusion Transformer

---

## Ownership Methods

### Parameter watermark

- AquaLoRA

### Behavioral fingerprint

- FingerInv

### Strong verification baseline

- Cert-LAS

---

## Quantization

Precision:

- FP16
- INT8
- INT4

Methods:

- Q-Diffusion
- PTQD
- AWQ/GPTQ when compatible

---

# 8. Evaluation Metrics

## Generation Quality

- FID
- CLIP score
- LPIPS paired similarity

---

## Ownership Verification

Watermark:

- Bit accuracy
- AUROC
- TPR@FPR

Fingerprint:

- verification accuracy
- reconstruction similarity

---

## Main Comparison

Compare:

### Standard PTQ

Generation-only optimization


vs.


### OA-MPQ

Ownership-aware quantization

Example:

| Method | FID | CLIP | Memory | Ownership |
|---|---|---|---|---|
| FP16 | | | | |
| INT4 PTQ | | | | |
| OA-MPQ | | | | |

---

# 9. Expected Findings

## H1: Ownership-Utility mismatch

Some layers strongly affect ownership verification but minimally affect generation quality.

---

## H2: Carrier-dependent vulnerability

Different ownership mechanisms have different quantization sensitivity.

---

## H3: Ownership-aware allocation improves robustness

OA-MPQ maintains ownership verification with limited deployment overhead.

---

# 10. Expected Contributions

## C1

First analysis of generation sensitivity versus ownership sensitivity under diffusion PTQ.

---

## C2

A diffusion-specific layer-timestep ownership sensitivity framework.

---

## C3

Ownership-Aware Mixed Precision Quantization (OA-MPQ) preserving ownership verification under low-bit deployment.

---

## C4

A benchmark evaluating diffusion ownership verification under realistic PTQ deployment.

---

# 11. Positioning

## CVPR / ICCV

Main contributions:

- diffusion model analysis;
- quantization methodology;
- ownership preservation.

---

## Security

Implication:

Deployment transformations can affect ownership guarantees.

---

# Final Positioning

This work is not a new quantization algorithm.

It studies:

> How to understand and preserve diffusion model ownership verification under deployment-time quantization.

The central hypothesis is:

\[
\boxed{
S^{generation}
\neq
S^{ownership}
}
\]

and this mismatch should guide future secure diffusion deployment.
