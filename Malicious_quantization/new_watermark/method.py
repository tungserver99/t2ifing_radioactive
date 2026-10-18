"""Transfer-selected semantic UNet watermark (experimental, zero-bit).

Embedding is training-time only. Exported teachers are ordinary merged UNets.
Students never receive this detector, its key, or a watermark loss.
"""
import math


def make_detector(features, key_seed, wrong_keys=8):
    import numpy as np
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or len(x) < 4 or not np.isfinite(x).all():
        raise ValueError('Need at least four finite clean feature vectors')
    # One predeclared random direction, not selected using the test set.
    rng = np.random.default_rng(key_seed)
    directions = rng.normal(size=(1+wrong_keys, x.shape[1]))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    projections = x @ directions.T
    scale = projections.std(0, ddof=1)
    if np.any(scale < 1e-8):
        raise ValueError('Degenerate feature direction; cannot calibrate detector')
    return {'directions': directions.tolist(), 'center': projections.mean(0).tolist(),
            'scale': scale.tolist(), 'key_seed': key_seed, 'fit_n': len(x),
            'type': 'fixed random projection of normalized CLIP image features; zero-bit',
            'wrong_keys': wrong_keys, 'features_used': 'clean design fit only'}


def score_features(features, detector):
    import numpy as np
    x = np.asarray(features, dtype=np.float64)
    w = np.asarray(detector['directions'])
    if x.ndim != 2 or x.shape[1] != w.shape[1] or not np.isfinite(x).all():
        raise ValueError('Invalid feature shape/values')
    values = (x @ w.T - detector['center']) / detector['scale']
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite detector scores')
    return values


def torch_scores(features, detector):
    import torch
    w = torch.as_tensor(detector['directions'][0], device=features.device, dtype=features.dtype)
    return (features @ w - detector['center'][0]) / detector['scale'][0]


def predict_x0(noisy, prediction, alpha, prediction_type):
    if prediction_type == 'epsilon':
        return (noisy-(1-alpha).sqrt()*prediction)/alpha.sqrt()
    if prediction_type == 'v_prediction':
        return alpha.sqrt()*noisy-(1-alpha).sqrt()*prediction
    raise ValueError(f'Unsupported prediction type: {prediction_type}')


def clip_pixels(images, size=224):
    """Differentiable, square resize; identical preprocessing at train and score."""
    import torch
    import torch.nn.functional as F
    x = F.interpolate(images.clamp(0, 1), size=(size, size), mode='bicubic', align_corners=False, antialias=True)
    # Resize can overshoot slightly; use the same clamp during inference.
    x = x.clamp(0, 1)
    mean = x.new_tensor([.48145466, .4578275, .40821073])[None, :, None, None]
    std = x.new_tensor([.26862954, .26130258, .27577711])[None, :, None, None]
    return (x-mean)/std


def image_features(model, images, augmented=False):
    import torch.nn.functional as F
    x = images
    if augmented:
        # A deterministic differentiable resize is an additional training view.
        # No claim of robustness to attacks not actually evaluated.
        x = F.interpolate(x, size=(196, 196), mode='bilinear', align_corners=False, antialias=True)
    size = model.config.vision_config.image_size
    return F.normalize(model.get_image_features(pixel_values=clip_pixels(x, size)), dim=-1)


def watermark_objective(prediction, reference_prediction, images, reference_images,
                        features, reference_features, augmented_features, detector, target, weights):
    import torch.nn.functional as F
    score = torch_scores(features, detector)
    aug_score = torch_scores(augmented_features, detector)
    wm = .5*(F.relu(target-score).square().mean()+F.relu(target-aug_score).square().mean()) / max(target**2, 1.)
    prior = (prediction-reference_prediction).square().mean()/reference_prediction.square().mean().detach().clamp_min(.01)
    pixels = (F.adaptive_avg_pool2d(images, 64)-F.adaptive_avg_pool2d(reference_images, 64)).square().mean() / .0025
    semantic = (1-(features*reference_features).sum(-1)).mean()/.01
    terms = {'watermark': wm, 'prior': prior, 'pixels': pixels, 'semantic': semantic}
    total = sum(weights[k]*v for k,v in terms.items())
    return total, terms


def choose_teacher(rows, gates):
    if not rows:
        raise ValueError('No pilot measurements')
    valid = []
    for r in rows:
        fields = ['teacher_tpr', 'teacher_ci_low', 'psnr', 'clip_drop', 'pilot_tpr', 'clean_pilot_fpr', 'pilot_clip_drop']
        if not all(math.isfinite(r[k]) for k in fields):
            raise ValueError('Nonfinite pilot/quality metric')
        if (r['teacher_tpr'] >= gates['min_teacher_tpr'] and r['teacher_ci_low'] >= gates['min_teacher_ci_low'] and r['psnr'] >= gates['min_psnr']
                and r['clip_drop'] <= gates['max_clip_drop'] and r['pilot_clip_drop'] <= gates['max_clip_drop']
                and r['clean_pilot_fpr'] <= gates['max_clean_fpr']):
            valid.append(r)
    positive = [r for r in valid if r['pilot_tpr'] > r['clean_pilot_fpr'] and r['pilot_tpr'] >= gates['min_student_tpr']]
    winner = max(positive, key=lambda r: (r['pilot_tpr']-r['clean_pilot_fpr'], r['psnr'], -r['step'])) if positive else max(rows, key=lambda r:(r['teacher_tpr'],r['psnr'],-r['step']))
    return {'selected_step': winner['step'], 'valid_design_winner': bool(positive),
            'warning': None if positive else 'No feasible positive transfer on pilot; using the strongest teacher positive control for diagnosis only.',
            'test_used_for_selection': False, 'rule': 'pilot TPR minus matched clean-trained FPR under teacher/pilot quality gates; tie PSNR then earlier step',
            'fallback_rule': 'highest teacher TPR; tie PSNR then earlier step; never counted as success', 'feasible_steps': [r['step'] for r in valid]}
