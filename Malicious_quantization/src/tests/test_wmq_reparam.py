import unittest
import tempfile
from pathlib import Path

import torch
from torch import nn

from wmq_reparam import rotate_attention_qk
from wmq_blind import RoundingGrid, export_quantizer, materialize


class TinyAttention(nn.Module):
    def __init__(self, bias=False):
        super().__init__()
        self.heads = 2
        self.to_q = nn.Linear(8, 8, bias=bias)
        self.to_k = nn.Linear(8, 8, bias=bias)
        self.to_v = nn.Linear(8, 8, bias=False)

    def forward(self, x):
        q = self.to_q(x).reshape(len(x), -1, self.heads, 4).transpose(1, 2)
        k = self.to_k(x).reshape(len(x), -1, self.heads, 4).transpose(1, 2)
        v = self.to_v(x).reshape(len(x), -1, self.heads, 4).transpose(1, 2)
        return (q @ k.transpose(-1, -2) / 2).softmax(-1) @ v


class ReparameterizationTests(unittest.TestCase):
    def test_qk_rotation_preserves_fp32_attention_and_changes_weight_basis(self):
        torch.manual_seed(41)
        decoder = nn.Module()
        decoder.attn = TinyAttention(bias=True)
        x = torch.randn(2, 5, 8)
        before = decoder.attn(x)
        q_before = decoder.attn.to_q.weight.detach().clone()
        details = rotate_attention_qk(decoder)
        self.assertEqual(details[0]['module'], 'attn')
        self.assertFalse(torch.equal(q_before, decoder.attn.to_q.weight))
        torch.testing.assert_close(before, decoder.attn(x), atol=1e-6, rtol=1e-6)

    def test_unsupported_qk_norm_fails_without_changing_weights(self):
        decoder = nn.Module()
        decoder.attn = TinyAttention()
        decoder.attn.norm_q = nn.LayerNorm(4)
        before = decoder.attn.to_q.weight.detach().clone()
        with self.assertRaisesRegex(ValueError, 'normalization'):
            rotate_attention_qk(decoder)
        torch.testing.assert_close(before, decoder.attn.to_q.weight)

    def test_quantizer_export_reloads_rotated_biases(self):
        from safetensors.torch import load_file
        torch.manual_seed(42)
        vae = nn.Module()
        vae.decoder = nn.Module()
        vae.decoder.attn = TinyAttention(bias=True)
        original = {name: value.detach().clone() for name, value in vae.decoder.state_dict().items()}
        rotate_attention_qk(vae.decoder)
        names = ['attn.to_q.weight', 'attn.to_k.weight']
        biases = ['attn.to_q.bias', 'attn.to_k.bias']
        grids = [RoundingGrid(vae.decoder.get_parameter(name), 4) for name in names]
        materialize(vae.decoder, names, grids)
        x = torch.randn(1, 3, 8)
        expected = vae.decoder.attn(x)
        with tempfile.TemporaryDirectory() as tmp:
            export_quantizer(vae, names, grids, Path(tmp), biases)
            saved = load_file(str(Path(tmp) / 'quantizer.safetensors'))
            vae.decoder.load_state_dict(original)
            with torch.no_grad():
                for name in names:
                    vae.decoder.get_parameter(name).copy_(saved[name + '.codes'].float() * saved[name + '.scale'])
                for name in biases:
                    vae.decoder.get_parameter(name).copy_(saved[name + '.fp32'])
            torch.testing.assert_close(expected, vae.decoder.attn(x), atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
