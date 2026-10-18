# Distillation, watermark inheritance, and a SleeperMark W4 defense control

The closest **method paper** is [Gu et al., *On the Learnability of Watermarks
for Language Models*, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/a86d17b6cd70366d56ab48d2a05a4df1-Paper-Conference.pdf).
It explicitly trains a student to imitate a watermarked teacher using
logit-based or sample-based distillation. Watermarks can be learned, but
low-distortion variants require more samples and further ordinary
fine-tuning can erase them. Related **watermark radioactivity** work by
[Sander et al. (2024)](https://arxiv.org/abs/2402.14904) shows that
training a student on watermarked LLM outputs can leave a statistically
detectable signal. It is not universal. [Pan et al., ACL 2025](https://aclanthology.org/2025.acl-long.648/)
show that inheritance can be weakened by removal procedures.
[ReasMark, ACL 2026](https://aclanthology.org/2026.acl-long.2185/) places the
watermark signal in frequent in-domain reasoning behavior, so a student
trained on ordinary domain queries is likelier to reproduce it. These are
**text/LLM** results, not evidence that SleeperMark survives diffusion
distillation. SleeperMark's public trigger is a rare prefix, unlike ReasMark's
frequent in-domain features. The analogy is therefore only a hypothesis.

`public_trigger_distill_defense_w4` is a separate **defense control**. The
student is the dequantized W4 UNet, the teacher is the frozen marked FP32
UNet. On ordinary and public-trigger TRAIN conditions, it matches the
teacher's noise predictions. This is closer to Gu et al.'s **logit-based**
distillation than to sample-based distillation: with a fixed-variance Gaussian
reverse transition, teacher/student KL is proportional to a timestep-weighted
noise-prediction MSE. The implementation uses unweighted MSE plus a local
trajectory term, so it is an approximation to that analogy. Every fourth
update, it also penalizes a
two-step triggered DDIM trajectory drift, with gradients detached between
steps. These rollouts start from ordinary TRAIN latents conditioned with the
public trigger; they are a short local approximation to a truly triggered
full trajectory. It holds out TRAIN prompts to choose the best intermediate W4
checkpoint by ordinary plus triggered image MSE to the teacher, subject to
an image-quality corridor. The owner key/extractor and TEST prompts never
select a checkpoint.

This is **not a reproduction of ReasMark** and does not create a new
SleeperMark watermark. It tests whether explicitly distilling the marked
triggered behavior into W4 weights preserves TPR at a given image quality.
Since fixed W4 already retains 100/100 detections in recent runs, the
control may only confirm a ceiling. It is not evidence of stronger defense
unless it improves a harder compression or subsequent attack protocol.

```bash
bash run_blind_quantization.sh --watermark sleepermark --methods public_trigger_distill_defense_w4
```

Report owner TPR/bit accuracy, ordinary and triggered SSIM, FID, and the
checkpoint selection trajectory next to `fixed_ptq` and the W4 attack.
Do not combine the defense result with the model-only attack claim: the
defender knows the published trigger and has the frozen marked teacher.
