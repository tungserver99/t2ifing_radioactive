# Learning-rate decision for the two-student protocol

Default: AdamW LR **1e-4**, constant schedule, no warmup, batch 4,
gradient-norm clipping 1.0. Both rank 16 and rank 32 use the same settings;
LoRA alpha equals rank, so the adapter multiplier alpha/r is 1.
Teacher LR remains separately configurable. The short pilot and differentiable
inner rollout use the same student LR (their horizons/batches still differ).

## Evidence and decision

The [official Diffusers text-to-image LoRA guide](https://huggingface.co/docs/diffusers/main/training/lora)
specifies 1e-4 as its default. Its example uses an effective batch of 4 and 15,000
steps, but a cosine schedule. It supports the starting magnitude, not a proof
that our constant schedule, convolution adapters, or watermark objective is optimal.
The [official training script](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py)
also uses rank-scaled alpha and AdamW. We retain the existing constant schedule
to keep the training rule aligned with the implemented inner optimizer; we are
not claiming to reproduce that guide's exact experiment.

Increasing the horizon from 7,500 to 10,000 does not by itself establish a reason
to increase or decrease LR. At 5,000 images and batch 4, 10,000 updates correspond
to approximately eight dataset passes on a single GPU. Save checkpoints at
2,500/5,000/7,500/10,000; select minimum held-out denoising MSE, never final test TPR.
The selected checkpoint may be earlier than 10,000.

No measured LR optimum is available for this new method. A future predeclared
validation-only comparison of 5e-5 and 1e-4 could resolve this; it is NOT silently
added to the requested two-branch run. Stronger radioactivity is not guaranteed
by either LR or additional training steps. Record attempted/applied AMP updates
and validation quality alongside detection.
