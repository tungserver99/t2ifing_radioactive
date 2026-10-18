# SleeperMark: quality-constrained public-trigger W4 control

## Evidence motivating the branch

The completed `sleeper_parallel_20260930_132427_652204` suite used 156
candidate genomes per method and 100 owner TEST images per method. All six
methods retained 100/100 detections. The best bit accuracy was 96.625% for
`public_trigger_genetic_w4`, with triggered SSIM 0.600 and FID 117.7. Larger
FIT/SELECT proxy improvements did not reliably predict TEST suppression.

An earlier, separate `public_trigger_consistency_qat` control in
`sleeper_parallel_20260930_003741_167395` reached 49.4% bit accuracy and
0/100 detections. Its triggered SSIM was 0.551, FID 135.7, and only 2/100
images jointly met the report quality threshold and evaded detection.
Consequently, watermark evasion alone is not a useful target for this control.

## New experimental method

`public_trigger_rollout_qat` optimizes the UNet's dequantized W4 grid weights.
It is a **public-trigger** control, distinct from model-only methods. Alongside
its existing trigger-prefix consistency loss, it penalizes ordinary-prompt
noise-prediction and latent drift over two DDIM steps. Gradients are truncated
between steps. This is a local trajectory approximation, not end-to-end
diffusion backpropagation. The frozen, marked model supplies the teacher.

Before fitting, a prompt-disjoint subset of TRAIN is reserved for checkpoint
selection. At fixed intervals, 50-step ordinary and triggered images are
rendered with paired seeds. A candidate must keep ordinary and triggered
SSIM within 0.03 and PSNR within 1 dB of the initial W4 checkpoint on this
subset. Among feasible candidates, the smallest triggered-to-clean image MSE
is selected. This proxy can reflect prompt semantics as well as watermark
behavior; it is **not** an ownership measurement. If no later candidate
improves it, the initial W4 checkpoint is retained.

The owner extractor, key, and TEST images are loaded only after selection.
`*_selection.csv/json` record every checkpoint assessment, including the
selected step and held-out prompt names. All final owner and quality results
remain in `output_attack`; large images and checkpoints remain in
`output_artifacts`.

```bash
bash run_blind_quantization.sh --watermark sleepermark --methods public_trigger_rollout_qat
```

This branch is a hypothesis until a completed 100-image owner evaluation.
Even a lower bit accuracy would not establish success unless paired image
quality and the joint evasion/quality rate improve. It also does not establish
novelty on its own.

The motivation for trajectory preservation is consistent with diffusion
quantization literature: [Q-Diffusion](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Q-Diffusion_Quantizing_Diffusion_Models_ICCV_2023_paper.pdf)
analyzes accumulated denoising error, while
[SleeperMark](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_SleeperMark_Towards_Robust_Watermark_against_Fine-Tuning_Text-to-image_Diffusion_Models_CVPR_2025_paper.pdf)
is the watermark under evaluation.
