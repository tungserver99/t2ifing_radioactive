"""Detector-free experimental objectives; not reproductions of UnMarker/HiDDeN."""
import torch
from torch import nn
from torch.nn import functional as F


def spectral_loss(prediction, target):
    """Paired natural reconstruction of log amplitudes at three spatial scales.

    This is a natural-image prior, not an identified watermark frequency mask.
    Return one loss per image so validation is independent of batch size.
    """
    losses = []
    for factor in (1, 2, 4):
        a, b = prediction.float(), target.float()
        if factor > 1:
            a, b = F.avg_pool2d(a, factor), F.avg_pool2d(b, factor)
        a = torch.fft.rfft2(a, norm="ortho").abs().log1p()
        b = torch.fft.rfft2(b, norm="ortho").abs().log1p()
        losses.append((a - b).square().flatten(1).mean(1))
    return torch.stack(losses).mean(0)


class NaturalDiscriminator(nn.Module):
    """Small patch discriminator trained only on natural TRAIN samples.

    No BatchNorm buffers; freezing it for generator updates cannot change state.
    This architecture and logistic loss are an explicit adaptation, not HiDDeN.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.LeakyReLU(.2),
            nn.Conv2d(32, 64, 3, 2, 1), nn.LeakyReLU(.2),
            nn.Conv2d(64, 64, 3, 2, 1), nn.LeakyReLU(.2), nn.Conv2d(64, 1, 1))

    def forward(self, image):
        return self.net(image * 2 - 1)


def discriminator_update(net, optimizer, real, fake):
    """A failed discriminator update is rolled back; never corrupt generator."""
    net.requires_grad_(True)
    optimizer.zero_grad(set_to_none=True)
    loss = (F.softplus(-net(real)).mean() + F.softplus(net(fake.detach())).mean()) / 2
    valid = bool(torch.isfinite(loss))
    if valid:
        loss.backward()
        valid = all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    if valid:
        valid = bool(torch.isfinite(nn.utils.clip_grad_norm_(net.parameters(), 1.)))
    if valid:
        old = {k: v.detach().clone() for k, v in net.state_dict().items()}
        optimizer.step()
        valid = all(bool(torch.isfinite(p).all()) for p in net.parameters())
        if not valid:
            net.load_state_dict(old)
    if not valid:
        optimizer.state.clear()
        for group in optimizer.param_groups:
            group['lr'] *= .5
    net.requires_grad_(False)
    optimizer.zero_grad(set_to_none=True)
    return loss.detach().item() if torch.isfinite(loss) else None, valid
