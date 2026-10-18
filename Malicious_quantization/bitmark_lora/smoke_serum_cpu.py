"""CPU contract smoke with synthetic weights, NOT a trained SERUM effectiveness test."""
import contextlib
import tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from common import HERE, sha256
from vendor.serum.config import Config
from vendor.serum.watermark import Watermark
from vendor.serum.score_model import WatermarkScoreModel
from serum_teacher import author_models, paired_noise, sample_posterior, check_imports


def main():
    check_imports()
    torch.set_num_threads(2)
    cfg=Config(str(HERE/'vendor/serum/config_sd21.yaml'))
    torch.manual_seed(123)
    wm,detector=Watermark(cfg),WatermarkScoreModel(cfg)
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/'synthetic.pt'
        torch.save({'epoch':50,'watermark_state_dict':wm.state_dict(),
                    'score_model_state_dict':detector.state_dict()},path)
        with patch('serum_teacher.CHECKPOINT_SHA256',sha256(path)):
            wm,detector=author_models(path,device='cpu')
        torch.manual_seed(42); before=torch.random.get_rng_state()
        marked,clean=paired_noise(wm,32,.5)
        assert torch.equal(before,torch.random.get_rng_state()), 'Official noise wrapper altered external RNG'
        marked2,clean2=paired_noise(wm,32,.5)
        assert torch.equal(marked,marked2) and torch.equal(clean,clean2)
        assert not torch.equal(marked,clean)
        with torch.inference_mode():
            scores=detector(torch.cat([marked,clean]))
        assert scores.shape==(2,1) and torch.isfinite(scores).all()
        assert ((scores>=0)&(scores<=1)).all()
        parameters=torch.randn(3,8,4,4)
        joint=sample_posterior(DiagonalGaussianDistribution(parameters),0,'cpu')
        separate=torch.cat([sample_posterior(DiagonalGaussianDistribution(parameters[i:i+1]),i,'cpu') for i in range(3)])
        assert torch.equal(joint,separate), 'Posterior sampling depends on batch partition'
        bad=path.with_name('bad.pt'); bad.write_bytes(b'not a checkpoint')
        with patch('torch.load',side_effect=AssertionError('deserialize must not run')):
            try: author_models(bad,device='cpu')
            except RuntimeError as exc: assert 'SHA256' in str(exc)
            else: raise AssertionError('accepted unverified checkpoint')
    print('SERUM synthetic CPU smoke passed: imports, strict state load, official forward, paired RNG, detector, posterior batch partition, reject-before-load.')


if __name__=='__main__': main()
