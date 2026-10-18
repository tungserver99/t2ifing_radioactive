# Beyond Layer Sensitivity: Preserving Diffusion Model Ownership under Quantization through Ownership Subspace Awareness

## 1. Introduction

Diffusion models have become a foundation of modern generative AI. As
these models are deployed in practical applications, two objectives
become increasingly important:

1.  reducing inference cost through post-training quantization (PTQ);
2.  preserving ownership verification signals for intellectual property
    protection.

Current diffusion PTQ methods mainly optimize generation quality:

\[ `\min`{=tex}*Q L*{gen}(Q(`\theta`{=tex})) \]

where (Q) represents the quantization transformation.

However, ownership verification requires another constraint:

\[ V(Q(`\theta`{=tex})) `\geq `{=tex}`\tau`{=tex} \]

where (V) represents watermark/fingerprint verification reliability.

Existing PTQ methods assume that preserving generation quality is
sufficient. This work challenges this assumption.

The central hypothesis is:

\[ `\boxed{
S^{generation} \neq S^{ownership}
}`{=tex} \]

A diffusion model can preserve visual quality after quantization while
losing the information required to verify ownership.

------------------------------------------------------------------------

# 2. Research Gap

## Diffusion Quantization

Existing works:

-   Q-Diffusion
-   PTQD
-   MixDQ
-   SVDQuant
-   Q-DiT

mainly optimize:

-   image quality;
-   inference efficiency;
-   memory reduction.

## Diffusion Ownership Verification

Existing methods:

-   AquaLoRA
-   FingerInv
-   Cert-LAS

focus on:

-   watermark robustness;
-   fingerprint verification;
-   ownership protection.

## Missing Problem

Current research does not study:

> Where is ownership information stored inside diffusion models, and how
> should quantization preserve these ownership-critical regions?

This work studies the relationship between:

\[ `\text{Generation representation}`{=tex} \]

and:

\[ `\text{Ownership representation}`{=tex} \]

under PTQ.

------------------------------------------------------------------------

# 3. Research Questions

## RQ1

Are generation-sensitive regions identical to ownership-sensitive
regions?

------------------------------------------------------------------------

## RQ2

Where does ownership information reside inside diffusion models?

-   specific layers;
-   attention blocks;
-   decoder components;
-   denoising timesteps;
-   parameter subspaces.

------------------------------------------------------------------------

## RQ3

Can we design a quantization method that aggressively compresses
diffusion models while preserving ownership-critical information?

------------------------------------------------------------------------

# 4. Key Observation: Dual Sensitivity Mismatch

For each layer (l):

## Generation Sensitivity

\[ S_l\^{gen} = `\Delta `{=tex}Quality(Q_l(`\theta`{=tex})) \]

Measures how quantization affects:

-   FID;
-   CLIP;
-   denoising reconstruction.

## Ownership Sensitivity

\[ S_l\^{own} = `\Delta `{=tex}Verification(Q_l(`\theta`{=tex})) \]

Measures how quantization affects:

-   watermark accuracy;
-   fingerprint confidence.

We hypothesize:

\[ Rank(S\^{gen}) `\neq`{=tex} Rank(S\^{own}) \]

Example:

  Layer                 Generation impact   Ownership impact
  --------------------- ------------------- ------------------
  Attention block A     Low                 High
  Convolution block B   High                Low

This indicates that generation-optimal quantization is not necessarily
ownership-safe.

------------------------------------------------------------------------

# 5. Ownership Subspace Analysis

Layer sensitivity alone is insufficient because ownership information
may exist in a smaller parameter subspace.

Let:

\[ `\theta`{=tex} = `\theta`{=tex}*{gen} + `\theta`{=tex}*{own} \]

where:

-   (`\theta`{=tex}\_{gen}): parameters contributing mainly to
    generation;
-   (`\theta`{=tex}\_{own}): parameters carrying ownership information.

Quantization introduces error:

\[ e_q = Q(`\theta`{=tex})-`\theta`{=tex} \]

Ownership degradation depends on the interaction:

\[ `\langle `{=tex}e_q,v\_{own}`\rangle`{=tex} \]

where:

\[ v\_{own} = `\nabla`{=tex}\_`\theta `{=tex}V \]

represents ownership-sensitive directions.

Define:

## Quantization Ownership Alignment Score

\[ A_q = `\frac{
\langle e_q,v_{own}\rangle
}`{=tex} { \|\|e_q\|\|\|\|v\_{own}\|\| } \]

A high value indicates that quantization noise directly damages
ownership information.

------------------------------------------------------------------------

# 6. Proposed Method

## Ownership-Aware Quantization (OAQ)

The goal is not only layer-wise bit allocation, but preserving
ownership-critical subspaces.

Objective:

\[ `\min`{=tex}\_Q Cost(Q) \]

subject to:

\[ Quality(Q)`\leq`{=tex}`\epsilon`{=tex} \]

and:

\[ Verification(Q)`\geq`{=tex}`\tau`{=tex} \]

------------------------------------------------------------------------

# 6.1 Ownership Subspace Discovery

Given a watermarked/fingerprinted diffusion model:

1.  Measure ownership sensitivity.
2.  Identify ownership-critical layers.
3.  Estimate ownership-sensitive directions.
4.  Measure quantization interference.

Output:

\[ `\mathcal `{=tex}S\_{own} \]

ownership-critical subspace.

------------------------------------------------------------------------

# 6.2 Subspace-Aware Precision Allocation

Instead of:

"keep important layers"

we preserve:

"keep ownership-critical information."

Example:

Layer A:

-   80% parameters → INT4
-   ownership subspace → FP16

Layer B:

-   all parameters → INT4

This enables stronger compression while maintaining ownership
verification.

------------------------------------------------------------------------

# 6.3 Algorithm

    Input:
    Diffusion model θ
    Ownership verifier V
    Calibration data D

    1. Profile quantization sensitivity:
           INT4 / INT8 / FP16

    2. Estimate:
           generation sensitivity
           ownership sensitivity

    3. Discover ownership-sensitive subspace

    4. Compute quantization interference score

    5. Protect ownership-critical directions:
           FP16 / INT8

    6. Quantize remaining parameters:
           INT4

    7. Evaluate:
           generation quality
           ownership verification

------------------------------------------------------------------------

# 7. Diffusion-Specific Analysis

Diffusion models repeatedly apply denoising:

\[ x_T `\rightarrow `{=tex}x_0 \]

Therefore ownership degradation may depend on timestep.

Analyze:

\[ S(l,t) \]

where:

-   (l): network component;
-   (t): denoising timestep.

The resulting map reveals:

-   early-step ownership dependency;
-   late-step texture dependency;
-   ownership information propagation.

------------------------------------------------------------------------

# 8. Experimental Design

## Models

Primary:

-   Stable Diffusion 1.5

Scaling:

-   SDXL
-   Diffusion Transformer

------------------------------------------------------------------------

## Ownership Methods

### Parameter watermark

AquaLoRA

### Behavioral fingerprint

FingerInv

### Strong baseline

Cert-LAS

------------------------------------------------------------------------

## Quantization

Precision:

-   FP16
-   INT8
-   INT4

Methods:

-   Q-Diffusion
-   PTQD
-   AWQ/GPTQ when compatible

------------------------------------------------------------------------

# 9. Evaluation Metrics

## Generation

-   FID
-   CLIP score
-   LPIPS paired similarity

## Ownership

Watermark:

-   Bit accuracy
-   AUROC
-   TPR@FPR

Fingerprint:

-   verification accuracy
-   reconstruction similarity

## Compression

-   model size;
-   latency;
-   memory usage.

------------------------------------------------------------------------

# 10. Expected Findings

## H1

Generation-sensitive regions and ownership-sensitive regions are
different.

------------------------------------------------------------------------

## H2

Ownership information concentrates in specific layers/subspaces.

------------------------------------------------------------------------

## H3

Subspace-aware quantization preserves ownership better than conventional
PTQ at the same compression budget.

------------------------------------------------------------------------

## H4

Ownership-critical information can be protected without significantly
reducing quantization efficiency.

------------------------------------------------------------------------

# 11. Contributions

## C1

A new analysis of generation-ownership sensitivity mismatch in diffusion
PTQ.

## C2

A diffusion ownership subspace discovery framework identifying
quantization-sensitive ownership directions.

## C3

Ownership-Aware Quantization (OAQ), a quantization strategy preserving
ownership-critical information under low-bit deployment.

## C4

A benchmark evaluating ownership verification robustness under realistic
diffusion quantization.

------------------------------------------------------------------------

# 12. Positioning

## CVPR / ICCV

Main contributions:

-   diffusion model analysis;
-   quantization method;
-   ownership preservation.

## Security

Implication:

Deployment transformations can affect ownership guarantees.

------------------------------------------------------------------------

# Final Research Statement

This work studies:

> How to compress diffusion models aggressively while preserving the
> information required for ownership verification.

The key insight is:

\[ `\boxed{
\text{Generation information}
\neq
\text{Ownership information}
}`{=tex} \]

and quantization should preserve both.
