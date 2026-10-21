"""Fixed clean SD2.1 semantic coordinates; no learned detector."""
import contextlib
import random
from pathlib import Path
import torch
from common import freeze, read_json, sha256, write_json
from semantic_sign import OBJECTIVE, is_sign, detection_count, image_bits


def check_finite(*values):
    for value in values:
        if not torch.isfinite(value).all():
            raise ValueError('Nonfinite semantic tensor')


def fit_pca(features):
    features = features.detach().cpu().float()
    check_finite(features)
    if features.ndim != 2 or min(features.shape) < 2:
        raise ValueError('PCA requires at least two rows and dimensions')
    mean = features.mean(0)
    _, singular, vh = torch.linalg.svd(features - mean, full_matrices=False)
    basis = vh.T.contiguous()
    # Canonicalize signs to avoid arbitrary SVD sign flips in key coordinates.
    pivot = basis.abs().argmax(0)
    basis *= basis[pivot, torch.arange(basis.shape[1])].sign()
    variance = singular.square() / (len(features) - 1)
    if variance.max() <= 1e-12:
        raise ValueError('Degenerate clean features')
    torch.testing.assert_close(basis.T @ basis, torch.eye(basis.shape[1]), atol=2e-5, rtol=2e-5)
    return mean, basis, variance


def choose_key(candidates, num_bits, key_seed):
    if len(set(candidates)) != len(candidates) or not 0 < num_bits <= len(candidates):
        raise ValueError('Not enough unique stable PCs for key')
    rng = random.Random(key_seed)
    selected = rng.sample(candidates, num_bits)
    return selected, [rng.choice([-1, 1]) for _ in selected]


class SemanticCarrier:
    def __init__(self, mean, basis, sigmas, signs, selected_pc_indices,
                 carrier_version=None, medians=None, metadata=None):
        self.mean, self.basis = mean.float(), basis.float()
        self.sigmas, self.signs = sigmas.float(), signs.float()
        self.selected_pc_indices = list(selected_pc_indices)
        self.carrier_version = carrier_version
        self.metadata = dict(metadata or {})
        self.medians = None if medians is None else medians.float()
        if carrier_version not in (None, OBJECTIVE):
            raise ValueError(f'Unsupported carrier version: {carrier_version}')
        if carrier_version == OBJECTIVE:
            if self.medians is None or self.medians.shape != self.sigmas.shape:
                raise ValueError('Sign carrier requires one median per selected PC')
            check_finite(self.medians)
        elif self.medians is not None:
            raise ValueError('Historical carrier cannot have sign medians')
        check_finite(self.mean, self.basis, self.sigmas, self.signs)
        d, m = self.basis.shape
        if (self.mean.shape != (d,) or self.sigmas.shape != (m,) or self.signs.shape != (m,)
                or len(set(self.selected_pc_indices)) != m or (self.sigmas <= 1e-6).any()
                or not ((self.signs == 1) | (self.signs == -1)).all()):
            raise ValueError('Invalid semantic carrier')
        torch.testing.assert_close(self.basis.T @ self.basis, torch.eye(m, device=basis.device), atol=2e-5, rtol=2e-5)

    def to(self, device):
        return SemanticCarrier(*(x.to(device) for x in (self.mean, self.basis, self.sigmas, self.signs)),
                               self.selected_pc_indices, carrier_version=self.carrier_version,
                               medians=None if self.medians is None else self.medians.to(device),
                               metadata=self.metadata)

    def normalized_coordinates(self, features):
        check_finite(features)
        # Keep the historical epsilon exactly. New artifacts use the defined q.
        divisor = self.sigmas if self.carrier_version == OBJECTIVE else self.sigmas + 1e-8
        q = ((features.float() - self.mean) @ self.basis) / divisor
        check_finite(q)
        return q

    def coordinates(self, features):
        q = self.normalized_coordinates(features)
        return q - self.medians if self.carrier_version == OBJECTIVE else q

    def scalar_score(self, features):
        return (self.coordinates(features) * self.signs).mean(-1)

    def bit_metrics(self, coordinates):
        check_finite(coordinates)
        recovered = (image_bits(coordinates.mean(0)) if self.carrier_version == OBJECTIVE
                     else coordinates.mean(0).sign())
        return {'mean_coordinates': coordinates.mean(0).tolist(), 'recovered_signs': recovered.tolist(),
                'bit_accuracy': (recovered == self.signs).float().mean().item()}

    @classmethod
    def load(cls, folder, device='cpu'):
        folder = Path(folder)
        proof = read_json(folder / 'complete.json')
        for name, digest in proof.items():
            if sha256(folder / name) != digest:
                raise RuntimeError(f'Carrier artifact changed: {name}')
        key = read_json(folder / 'key.json')
        if key['basis_sha256'] != sha256(folder / 'pca.pt'):
            raise RuntimeError('PCA/key mismatch')
        pca = torch.load(folder / 'pca.pt', map_location='cpu', weights_only=True)
        selected = key['selected_pc_indices']
        version = key.get('carrier_version')
        medians = None
        if version == OBJECTIVE:
            centers = read_json(folder / 'medians.json')
            if centers['carrier_fit_prompt_sha256'] != key['carrier_fit_prompt_sha256']:
                raise RuntimeError('Median/key train provenance mismatch')
            lookup = dict(zip(centers['pc_indices'], centers['medians']))
            medians = torch.tensor([lookup[j] for j in selected])
            if medians.tolist() != torch.tensor(key['medians']).tolist():
                raise RuntimeError('Median/key coordinate mismatch')
        return cls(pca['mean'], pca['basis'][:, selected], torch.tensor(key['sigmas']),
                   torch.tensor(key['signs']), selected, carrier_version=version,
                   medians=medians, metadata=key).to(device)


class _MidBlockReady(Exception):
    """Private control flow: features are ready; skip the unused UNet decoder."""


class SemanticExtractor:
    """Frozen parameters still allow gradients with respect to latent/image inputs."""
    def __init__(self, pipe, device='cuda', gradient_checkpointing=True):
        self.vae, self.unet = pipe.vae, pipe.unet
        self.tokenizer, self.text_encoder = pipe.tokenizer, pipe.text_encoder
        self.device = device
        self.feature_dim = None
        self.gradient_checkpointing = gradient_checkpointing
        for model in (self.vae, self.unet, self.text_encoder):
            model.eval().requires_grad_(False).to(device=device, dtype=torch.float32)

    def text(self, prompts):
        tokens = self.tokenizer(prompts, padding='max_length', truncation=True,
                                max_length=self.tokenizer.model_max_length, return_tensors='pt')
        with torch.no_grad():
            return self.text_encoder(tokens.input_ids.to(self.device))[0]

    def image_latent(self, images):
        images = images.to(device=self.device, dtype=torch.float32)
        check_finite(images)
        return self.vae.encode(images).latent_dist.mean * self.vae.config.scaling_factor

    def encode_image(self, images, prompts):
        return self.encode_latent(self.image_latent(images), prompts)

    def encode_latent(self, scaled_latents, prompts=None, text_hidden=None):
        check_finite(scaled_latents)
        hidden = self.text(prompts) if text_hidden is None else text_hidden
        if self.gradient_checkpointing and torch.is_grad_enabled() and scaled_latents.requires_grad:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(self._latent_features, scaled_latents, hidden, use_reentrant=False)
        return self._latent_features(scaled_latents, hidden)

    def _latent_features(self, scaled_latents, hidden):
        captured = []
        def capture(module, args, output):
            captured.append(output)
            # These exact mid-block activations are the entire feature definition.
            # Do not run up-blocks or output layers whose results are discarded.
            raise _MidBlockReady()
        handle = self.unet.mid_block.register_forward_hook(capture)
        try:
            try:
                self.unet(scaled_latents.float(), torch.zeros(len(scaled_latents), device=self.device, dtype=torch.long),
                          encoder_hidden_states=hidden.float())
            except _MidBlockReady:
                pass
        finally:
            handle.remove()
        if len(captured) != 1 or captured[0].ndim != 4:
            raise RuntimeError('Expected exactly one BCHW mid-block output')
        features = captured[0].float().mean((-2, -1))
        if self.feature_dim is None: self.feature_dim = features.shape[1]
        if self.feature_dim != features.shape[1]: raise RuntimeError('Feature dimension changed')
        check_finite(features)
        return features

    def cycle(self, scaled_latents):
        if self.gradient_checkpointing and torch.is_grad_enabled() and scaled_latents.requires_grad:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(self._cycle, scaled_latents, use_reentrant=False)
        return self._cycle(scaled_latents)

    def _cycle(self, scaled_latents):
        image = self.vae.decode(scaled_latents.float() / self.vae.config.scaling_factor).sample
        check_finite(image)
        # Same clipping as normal image conversion, without quantization (differentiable).
        return self.image_latent(image.clamp(-1, 1))


def stability_metrics(features, cycle_features, mean, basis, variance, pc_min, pc_max):
    a, b = (features - mean) @ basis, (cycle_features - mean) @ basis
    if pc_min < 0 or pc_max >= basis.shape[1] or pc_min > pc_max:
        raise ValueError('PC range outside fitted basis (indices are zero-based, inclusive)')
    result = []
    for j in range(pc_min, pc_max + 1):
        x, y = a[:, j], b[:, j]
        xc, yc = x-x.mean(), y-y.mean()
        denom = xc.norm() * yc.norm()
        if variance[j] <= 1e-12 or denom <= 1e-12:
            raise ValueError(f'Degenerate candidate PC {j}')
        result.append({'pc': j, 'correlation': (xc @ yc / denom).item(),
                       'sign_retention': (x.sign() == y.sign()).float().mean().item(),
                       'normalized_mse': ((x-y).square().mean()/variance[j]).item(),
                       'clean_std': variance[j].sqrt().item()})
    return sorted(result, key=lambda r: (-r['correlation'], -r['sign_retention'], r['pc']))


def save_carrier(folder, features, cycle_features, config, provenance):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    if (folder/'complete.json').exists():
        existing = SemanticCarrier.load(folder)
        if is_sign(config):
            if existing.carrier_version != OBJECTIVE:
                raise RuntimeError('Cannot reuse historical carrier for a sign run')
            key = existing.metadata
            expected = {name: config[name] for name in ('num_bits', 'stable_candidate_count',
                        'key_seed', 'carrier_pc_min', 'carrier_pc_max')}
            expected.update(alpha_test=config.get('analytic_alpha', .01))
            if any(key.get(name) != value for name, value in expected.items()):
                raise RuntimeError('Frozen sign carrier/key settings changed')
        elif existing.carrier_version == OBJECTIVE:
            raise RuntimeError('Cannot reinterpret sign carrier as historical carrier')
        freeze(folder/'provenance.json', provenance)
        return
    mean, basis, variance = fit_pca(features)
    stable = stability_metrics(features, cycle_features, mean, basis, variance,
                               config['carrier_pc_min'], config['carrier_pc_max'])
    count = config['stable_candidate_count']
    if len(stable) < count: raise ValueError('Too few candidate PCs')
    median_payload = None
    if is_sign(config):
        # Fit all candidate-PC midpoint medians before sampling either PCs/signs.
        candidate_pcs = list(range(config['carrier_pc_min'], config['carrier_pc_max'] + 1))
        q = ((features.detach().cpu().float()-mean) @ basis[:, candidate_pcs]) / variance[candidate_pcs].sqrt()
        check_finite(q)
        midpoint = torch.quantile(q, .5, dim=0, interpolation='midpoint')
        median_payload = {'carrier_version': OBJECTIVE, 'pc_indices': candidate_pcs,
                          'medians': midpoint.tolist(), 'fit_split': 'carrier_train',
                          'carrier_fit_prompt_sha256': provenance['train_prompt_sha256'],
                          'median_rule': 'midpoint', 'train_feature_count': len(features)}
    selected, signs = choose_key([r['pc'] for r in stable[:count]], config['num_bits'], config['key_seed'])
    temporary = folder/'pca.partial'
    torch.save({'mean': mean, 'basis': basis, 'variance': variance, 'feature_dimension': len(mean)}, temporary)
    temporary.replace(folder/'pca.pt')
    key = {'key_seed': config['key_seed'], 'num_bits': config['num_bits'],
        'selected_pc_indices': selected, 'signs': signs, 'sigmas': variance[selected].sqrt().tolist(),
        'basis_sha256': sha256(folder/'pca.pt'), 'carrier_fit_prompt_sha256': provenance['train_prompt_sha256'],
        'index_convention': 'zero-based inclusive PC range',
        'selection_rule': 'correlation descending, sign retention descending, PC ascending; RNG sample without replacement'}
    if is_sign(config):
        centers = dict(zip(median_payload['pc_indices'], median_payload['medians']))
        alpha = config.get('analytic_alpha', .01)
        key.update(carrier_version=OBJECTIVE, stable_candidate_count=count,
                   carrier_pc_min=config['carrier_pc_min'], carrier_pc_max=config['carrier_pc_max'],
                   medians=[centers[j] for j in selected], bit_tie_rule='r >= 0 -> +1; r < 0 -> -1',
                   analytic_null='key_randomization_binomial', alpha_test=alpha,
                   k_detect=detection_count(config['num_bits'], alpha))
        freeze(folder/'medians.json', median_payload)
    freeze(folder/'key.json', key)
    freeze(folder/'stability.json', stable)
    freeze(folder/'provenance.json', provenance)
    artifacts = ['pca.pt', 'key.json', 'stability.json', 'provenance.json']
    if is_sign(config): artifacts.append('medians.json')
    freeze(folder/'complete.json', {name: sha256(folder/name) for name in artifacts})
    SemanticCarrier.load(folder)
