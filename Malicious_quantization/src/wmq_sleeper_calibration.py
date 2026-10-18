"""Owner-independent timestep coverage, context augmentation, and loss scaling."""
import string
import torch


def cache_datasets_on_device(datasets, device, mode='auto', reserve_gib=4.):
    """Move unique calibration tensors once while preserving shared references."""
    if mode not in ('auto', 'cpu', 'cuda'):
        raise ValueError('Unknown calibration cache mode')
    tensors = {}
    for dataset in datasets:
        for row in dataset:
            for value in row:
                if torch.is_tensor(value):
                    tensors.setdefault(id(value), value)
    required = sum(value.numel()*value.element_size() for value in tensors.values())
    stats = {'mode_requested': mode, 'unique_tensors': len(tensors),
             'required_gib': required/2**30, 'device': 'cpu'}
    if mode == 'cpu' or device.type != 'cuda' or not tensors:
        return datasets, stats
    free, total = torch.cuda.mem_get_info(device)
    reserve = max(reserve_gib*2**30, total*.05)
    fits = required*1.5+reserve <= free
    if mode == 'cuda' and not fits:
        raise RuntimeError(f'Calibration cache needs {required/2**30:.2f} GiB plus reserve; only {free/2**30:.2f} GiB free')
    if not fits:
        stats.update(free_gib=free/2**30, reason='insufficient_free_vram')
        return datasets, stats
    moved = {identity: value.to(device) for identity, value in tensors.items()}
    converted = []
    for dataset in datasets:
        converted.append([tuple(moved.get(id(value), value) for value in row) for row in dataset])
    stats.update(device=str(device), free_gib_before=free/2**30,
                 total_vram_gib=total/2**30, reserve_gib=reserve/2**30)
    return converted, stats


class TimestepSampler:
    def __init__(self, dataset, rng):
        self.rng = rng
        self.groups = {}
        for i, row in enumerate(dataset):
            self.groups.setdefault(int(row[3]), []).append(i)
        self.order = []

    def sample(self, count):
        result = []
        for _ in range(count):
            if not self.order:
                self.order = list(self.rng.permutation(sorted(self.groups)))
            timestep = self.order.pop()
            result.append(int(self.rng.choice(self.groups[timestep])))
        return result


def augmented_prompts(prompts, rng):
    # Distribution fixed before owner evaluation, no secret or recovered tokens.
    # Broader conditioning coverage is not evidence of activating a watermark.
    return [''.join(rng.choice(list(string.punctuation), size=int(rng.integers(1, 9))))
            + ' ' + prompt for prompt in prompts]


def spatial_reconstruction_loss(student_x0, target_x0, timestep, scheduler,
                                guidance=7.5, mode='noise', floor=.01):
    error = (student_x0-target_x0).square().flatten(1).mean(1)
    if mode == 'x0':
        return error.mean()
    alpha = scheduler.alphas_cumprod.to(student_x0.device, student_x0.dtype)[timestep.long()]
    kind = scheduler.config.prediction_type
    if kind == 'epsilon':
        jacobian_squared = (1-alpha)/alpha.clamp_min(1e-6)
    elif kind == 'v_prediction':
        jacobian_squared = 1-alpha
    elif kind == 'sample':
        jacobian_squared = torch.ones_like(alpha)
    else:
        raise ValueError(f'Unsupported prediction type: {kind}')
    if mode != 'noise':
        raise ValueError('Unknown spatial loss normalization')
    return (error/jacobian_squared.clamp_min(floor)/guidance**2).mean()
