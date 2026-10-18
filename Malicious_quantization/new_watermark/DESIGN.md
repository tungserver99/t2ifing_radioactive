# Design decision: LoRA-aware watermark embedded in diffusion weights

Status: implemented experimental short-horizon proxy, 2026-10-16. `meta.py`
differentiates through VAE image encoding and ordinary LoRA denoising updates.
`run_new.sh` executes training/export/pilot selection/fresh student evaluation.
CPU tests cover real tiny models, hypergradients and standalone merged inference.
No successful full GPU result, publication novelty, or strong transfer claim has
been established. See README for concrete defaults and approximation limits.

## Deployment and threat model

The owner trains a UNet weight update and exports a standard merged UNet. Teacher
inference receives ordinary prompts and Gaussian noise: no special initial noise,
secret prompt, key-conditioned hook, or postprocessing injector. The owner's
verification key/detector is separate from generation, as is normal for watermarking.

Fresh students start from the CLEAN base and receive generated images plus original
captions only. They receive neither teacher weights, key, nor detector loss. Using
marked teacher weights to initialize the student would test persistence, not
radioactivity. Exporting a marked teacher alone does not guarantee inheritance.

## Prior work and what it actually supports

- Stable Signature, https://arxiv.org/abs/2303.15435: watermark in VAE decoder
  weights. Supports native generation, not a guarantee of data-mediated LoRA transfer.
- AquaLoRA, https://arxiv.org/abs/2405.11135: UNet watermark adapters and prior
  preservation, mergeable into weights. This is useful deployment precedent;
  an owner LoRA is not evidence that a fresh data-trained student LoRA inherits it.
- HMark, https://arxiv.org/html/2512.00094: semantic bottleneck residuals and
  downstream LoRA radioactivity. Their demonstrated setting uses face DDPMs;
  their numerical results cannot be imported into text-to-image SD experiments.
  Radioactivity is explicitly not in their training loss (section 3.3). No official
  usable source/checkpoint located in the searches performed; do not claim reproduction.
- Radioactive Data, https://proceedings.mlr.press/v119/sablayrolles20a.html:
  feature marks can influence training. Classification analysis does not prove the
  diffusion-LoRA case.
- Dataset Condensation with Gradient Matching, https://arxiv.org/abs/2006.05929:
  precedent for controlling learning via the gradients induced by data, not proof
  of the proposed watermark's efficacy or originality.
- LoRA, https://arxiv.org/abs/2106.09685: low-rank adaptation parameterization.

## Local transfer calculation (our derivation, not a quoted theorem)

Let phi be ALL trainable student adapter parameters. At the same initial state,
let g_w = grad_phi L_denoise(D_w) and g_c = grad_phi L_denoise(D_clean), using paired
prompts/noise/timesteps. For a differentiable watermark statistic R(phi) on separate
student generation prompts, a single SGD step gives

    R(phi - eta*g_w) - R(phi - eta*g_c)
      = -eta * <grad_phi R(phi), g_w-g_c> + remainder.

If grad R is L-Lipschitz on both update segments, the absolute remainder is bounded by

    L*eta^2/2 * (||g_w||^2 + ||g_c||^2).

Thus a positive local transfer margin requires NEGATIVE alignment of the denoising
loss gradient difference with grad R. Merely maximizing R on TEACHER images says
nothing about this dot product. A positive dot-product objective only supplies a
local surrogate; it is not a guarantee of detection, quality, or long-run behavior.

For one LoRA layer W_eff = W + s*B*A, initially B=0, vanilla SGD and no weight decay:

    grad_B L = s*G*A^T; grad_A L = 0,
    W_eff_after - W = -eta*s^2*G*A^T*A,

where G is the loss gradient with respect to the effective weight. Therefore the
clean-vs-marked effective update difference is -eta*s^2*(G_w-G_c)*A^T*A.
A^T*A is positive semidefinite but generally NOT an orthogonal projector. The
accessible direction changes as A/B train. AdamW and multiple updates require the
actual update operator, not treating this formula as exact for the deployed trainer.

## Committed experimental direction

Working description: LoRA-aware in-weight semantic watermark (zero-bit initially).
No claim that the label denotes a new published algorithm.

1. Learn teacher UNet adapters and merge into ordinary weights. Freeze a verification
   feature extractor/detector after a disjoint design stage. Begin with a scalar
   ownership statistic; it is not a multi-bit payload or bit accuracy.
2. Use a native teacher detection loss, prior/pixel/perceptual constraints, AND a
   transfer term. The transfer term targets the clean-adjusted effect of learning
   from marked images on fresh surrogate LoRA students.
3. The local diagnostic -<grad_phi R, g_w-g_c> motivates the design. The code
   implements short functional AdamW rollouts instead of a separate cosine loss.
   It preserves dependence through optimizer moments and gradient clipping; it
   does not pretend that the SGD identity is exact for AdamW.
4. Validate/optimize against short actual LoRA rollouts, using the DIFFERENCE in
   student output score between marked-trained and matched clean-trained rollouts.
   Outer gradients must flow through training data generation and student updates
   if called meta-optimization. Detached pilot selection alone is an ablation,
   not gradient-based transfer training. Truncated rollouts and predicted-x0 losses
   are explicitly approximate, and require long-run validation.
5. Change surrogate initialization seeds during design; freeze the teacher, detector
   and selection rule before final data generation. The implemented default surrogate
   rank is 16, scope attention+convolution, with fresh final students at ranks 16/32.
   Multi-scope/base ensembles and multiple final training seeds remain further
   experiments, not properties of the current single-run implementation.

Ideal target, subject to explicit teacher AND student utility constraints:

    maximize_theta E_j [ R_j(U_j^K(D_theta)) - R_j(U_j^K(D_clean)) ]

U_j^K is K ordinary denoising-training updates of a fresh clean-base LoRA student.
The teacher/key has access to surrogate students during design; final students do
not receive the key, watermark loss, or surrogate checkpoints. Teacher score,
clean-trained FPR, and wrong-key controls are additional constraints/audits, not
substitutes for this target. Smooth scores are training surrogates; final reporting
uses frozen-threshold detection rates and confidence intervals.

The current reduced final protocol trains only marked-data students at ranks 16/32.
Matched clean rollouts and pilots remain, but final clean-trained/fixed-teacher
students are omitted. Thus it measures descriptive transfer, not the full ideal
clean-adjusted objective below at the final training horizon.

## Gates before claiming improvement

- Merged teacher must pass fresh ordinary-pipeline reload and native generation.
- Positive teacher control, independent clean calibration, clean-trained controls,
  and wrong-key controls are all required. FPR on unadapted samples alone is insufficient.
- Compare no watermark, teacher-score-only watermark, and transfer-aware watermark
  using equal final-student budgets. Report additional teacher/meta/pilot cost.
- Freeze quality limits before seeing final data; measure FID/KID, prompt alignment,
  paired perceptual quality and inspect images. Semantic or style drift is a real cost.
- A feature-score rise is not a TPR improvement. CLIP-only optimization can exploit
  features not retained by student learning; do not make CLIP the only quality check.
- Report validation MSE and detection/quality per training milestone; select final
  student checkpoints by declared validation rule, never final test detection.
- No single-seed result establishes reliable radioactivity or novelty. A short
  surrogate update can improve while 10,000-step student transfer fails.

## Implemented approximation and remaining scientific validation

The implementation uses fresh surrogate initialization seeds and a configurable
surrogate rank/scope (default rank 16, attention+convolution), not an ensemble of
architectures per update. Its two-step differentiable AdamW rollout and predicted-x0
probe implement an approximation to the ideal objective. Default final ranks 16/32
include rank 32 not used in the inner rollout. Meta-probe prompts are disjoint from
teacher training; final train/validation/test and pilot data are separate splits.

A native-teacher warmup and absolute teacher detection margin are retained alongside
transfer loss. Candidate and final-test validity require native teacher TPR >=95%
and Wilson lower bound >=90%, with independent clean FPR/quality checks. Weak
teachers cannot produce a successful inheritance claim even with perfect retention.
Failure is logged; diagnostic evaluations still complete.

Next scientific checks require actual GPU experiments, multiple seeds and keys,
matched teacher-compute ablations (`--meta-weight 0`), independent quality evaluation,
and checking whether the local training proxy predicts long-run LoRA inheritance.
