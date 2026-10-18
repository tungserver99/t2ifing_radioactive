# Watermark in diffusion weights, evaluated through fresh LoRA students

Experimental implementation. **No successful full GPU run or improved radioactivity
has yet been established.** This is a zero-bit ownership detector, not message-bit
recovery. It is not a reproduction of HMark, SERUM, or AquaLoRA.

From the repository root, in the prepared SERUM environment:

```bash
bash run_new.sh
```

The launcher runs the whole experiment; no second evaluation command is required.
It uses `NEW_WATERMARK_PYTHON`, then `SERUM_PYTHON`/`BITMARK_PYTHON`, then the existing
runtime/repository virtual environment, then `python3`. It does not install packages
in a compute job. Reuse the environment that runs `run_serum.sh`; no additional
Python dependencies are introduced and no FlashAttention dependency is needed for
this method. Downloaded base/CLIP weights and COCO captions are pinned and verified.
Results default to `run_new_YYYYMMDD_HHMMSS_PID/` **beside the code**, including when
the repository is on mounted Drive. Model cache remains on local runtime storage.
At least 100 GiB free disk is requested by default (override `--min-free-gb` explicitly).
The FP32 second-order training stage is memory intensive; production GPU memory and
runtime have not been measured here. A CUDA OOM is not converted into a scientific result.

```bash
# Resume exactly the same code/config. Incomplete training stages restart from base;
# complete verified stages are reused. This is not optimizer-state resume.
bash run_new.sh --resume /path/to/run_new_YYYYMMDD_HHMMSS_PID

# Optional inspect-only operation; omit it to actually train.
bash run_new.sh --plan-only

# Same-budget teacher-score-only ablation in a separate run:
bash run_new.sh --meta-weight 0
```

## What actually changes

The teacher is SD1.4 with trained attention/convolution UNet LoRA updates. These are
merged and exported as **ordinary full diffusion pipeline weights**. Generation uses
only prompt + ordinary random Gaussian noise. No watermark key, special noise,
forward hook, or postprocessing injector is passed at teacher inference.

A frozen CLIP image encoder and a fixed keyed feature projection define the scalar
verification statistic. The projection is standardized on a clean design split;
its threshold is calibrated separately, once, at target FPR 1%. The owner keeps this
verification information. The default reproducible key seed is public; this pilot
makes no cryptographic secrecy or unforgeability claim.

Teacher loss includes a native detection margin (default weight 5), normalized
frozen-base noise-prediction preservation, low-pass image preservation and feature
preservation. Detection uses a second resize view during training. Stronger transfer
is never allowed to turn a weak teacher into a valid positive result.

After a 250-step native warmup, every 10th teacher step also differentiates through
2 ordinary denoising AdamW updates of a fresh rank-16 surrogate student. Its ONLY
inner loss is denoising MSE on the teacher image after VAE re-encoding. An identical
clean-image rollout is the control. The outer loss asks for both an absolute student
detection margin and a positive clean-adjusted score difference on independent
probe prompts/latents. Surrogate adapter initialization changes per outer step.

This is a **truncated predicted-x0 meta-learning proxy**, not full diffusion-trajectory
bilevel optimization. It uses posterior mode and one example per inner rollout,
FP32, and smoothed AdamW `sqrt(v+1e-24)` to avoid undefined hypergradients at zero.
Actual final students use ordinary image training with sampled VAE posteriors,
batch 4, and AdamW. The approximation is validated by full sampling pilots, not
assumed equivalent to 10,000-step training. Math attention is used for the surrogate
because CUDA flash attention can lack the required double backward.

## Complete default protocol

1. Download pinned SD1.4/CLIP and create disjoint COCO caption splits: detector-fit
   256; teacher-train 512; meta-probe 128; clean calibration 1,000; pilot-train 512;
   pilot-eval 256; final train 5,000; validation 256; final test 1,000.
2. Generate clean design data and freeze the detector/threshold.
3. Train the teacher for 1,000 attempted updates, save steps 250/500/1,000, merge and
   verify each export. The first checkpoint is the predeclared warmup control.
4. Train 500-step pilot students from the CLEAN base on each candidate's images,
   plus a matched clean-trained pilot. Select teacher using pilot scores and gates.
5. Generate the selected teacher's final train/validation/test datasets. Train only
   **two** fresh attention+convolution students: ranks **16 and 32**, each for one
   **10,000-step trajectory**, saving 2,500/5,000/7,500/10,000 checkpoints.
   No final clean-trained or fixed-teacher students are trained. Clean teacher
   test images and the unadapted student remain inference-only references.
6. Select each student by held-out denoising MSE only; evaluate detection and CLIP
   at milestones. Final test follows frozen selection; FID/KID run by default.

Final training totals **20,000 steps**, plus the separate teacher and short pilot
budgets. Pilot training remains necessary to select the teacher.
LR is **1e-4** for both ranks, AdamW, constant schedule, batch 4, gradient clipping
1.0. See [LR_DECISION.md](LR_DECISION.md) for evidence and limitations.

Without final clean-trained controls, detection differences versus unadapted cannot
isolate generic fine-tuning effects. There is also no final fixed-teacher ablation
to establish the benefit of selection. Use a separate equal-budget
`--meta-weight 0` run for a teacher-only comparison. One seed is not enough for a
paper-level conclusion. Existing frozen runs cannot resume with changed code/config;
start a new run for this protocol.

## Teacher strength is an explicit requirement

At the **frozen detector threshold**, default design and final-test checks require:

- Teacher TPR >= **95%** and its 95% Wilson lower bound >= **90%**.
- Teacher paired PSNR >= 25 dB; semantic CLIP drop <= 0.03.
- Clean false positives <= 3% as a validity guard (the calibration target is still 1%).
- Pilot detection >= 20% and above its matched clean-trained pilot.
- Final student detection >= 20% and positive paired-bootstrap lower bound versus
  the unadapted reference; these are descriptive checks, not causal proof.
- Student semantic CLIP drop versus the unadapted reference <= 0.03.

These are predeclared experiment gates, **not promises that training reaches them**.
If no teacher qualifies, the strongest teacher positive control is used to continue
diagnosis, with `valid_design_winner=false`. A failed teacher test guard forces
`teacher_valid=false` and every rank's `declared_gates_passed=false`. Metrics are still
saved; weak teacher inheritance is never labeled success. FID/KID are descriptive,
not hidden pass/fail criteria, and low-sample FID is biased. Inspect images too.

## Outputs

Open `RUN/result/README.md`. The main table has **one detection column**, with clean
rows explicitly identified as false-positive rates. Other useful files:

- `result/teacher_strength.json`: native teacher requirements, measurements and validity.
- `result/teacher_selection.json`: all candidate native/pilot scores and selection status.
- `result/summary.csv`, `report.json`: absolute detection, CI and differences versus unadapted.
- `result/checkpoint_metrics.csv`: MSE, detection and semantic CLIP at each milestone.
- `result/wrong_key_controls.csv`: eight independent direction controls, not a security proof.
- `result/utility.json`: final FID/KID/CLIP with common clean-teacher reference.
- `teacher/training.jsonl`: native losses, transfer proxy, finite updates and gradient norms.
- `students/*/optimization_audit.json`: attempted/applied/AMP-skipped optimizer updates.
- `models/teacher_STEP/`: standalone merged teacher. Use the selected step in the selection JSON.
- `run.log`, `logs/`, `stages.jsonl`: streamed logs and stage times.

Standard deployment requires no watermark module:

```python
import torch
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained(
    '/path/to/RUN/models/teacher_1000', torch_dtype=torch.float16,
    safety_checker=None, local_files_only=True,
).to('cuda')
image = pipe('a photograph of a mountain', num_inference_steps=50).images[0]
```

## Verification performed locally

CPU tests use tiny actual Diffusers UNets, PEFT adapters, VAE and CLIP. They check
nonzero outer gradients through pixels -> VAE -> student training -> held-out score,
finite differences, stopped-gradient negative control, AdamW updates, epsilon/v
prediction, teacher validity gates, native adapter merge/reload and ordinary
prompt-only generation. Shared official trainer smoke tests check native/conv/DoRA
adapters, snapshots and applied-update counts. These checks do not measure scientific
efficacy on SD1.4 or certify GPU memory/second-order CUDA execution.

See [DESIGN.md](DESIGN.md) for derivation and research sources.
