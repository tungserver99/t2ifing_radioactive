import unittest
import torch
from wmq_grouped_quant import GroupedW4, grouped_fake_quant
from wmq_blind import RoundingGrid


class GridTests(unittest.TestCase):
    def test_mse_initialization_never_worsens_legacy_weight_error(self):
        torch.manual_seed(37)
        for shape in ((3, 9), (2, 3, 3, 3)):
            weight = torch.randn(shape)
            old, new = GroupedW4(weight, 4), GroupedW4(weight, 4, 'mse')
            self.assertLessEqual(float((new(False)-weight).square().sum()),
                                 float((old(False)-weight).square().sum())+1e-7)
            self.assertLessEqual(float(new.initial_weight_mse), float(new.legacy_weight_mse)+1e-7)
            torch.testing.assert_close(new.initial_weight_mse, (new(False)-weight).square().mean())
            with torch.no_grad():
                new.offset.fill_(30)
                new.log_scale.fill_(-5)
            new.clamp_parameters()
            self.assertEqual(float(new.offset.max()), 2.)
            self.assertTrue(torch.isfinite(new(False)).all())

    def test_group_padding_export_and_gradients(self):
        torch.manual_seed(7)
        for shape in ((3, 9), (2, 3, 3, 3)):
            weight = torch.randn(shape)
            grid = GroupedW4(weight, 4)
            torch.testing.assert_close(grid(False), grouped_fake_quant(weight, 4))
            self.assertEqual(grid().shape, weight.shape)
            grid().square().sum().backward()
            self.assertGreater(grid.offset.grad.abs().sum().item(), 0)
            self.assertGreater(grid.log_scale.grad.abs().sum().item(), 0)
            torch.testing.assert_close(grid(), grid(False))

    def test_warm_center_preserves_initial_model_and_scale_gradient(self):
        torch.manual_seed(11)
        grid = RoundingGrid(torch.randn(3, 8), 4, learn_code_offsets=True)
        before = grid(False).detach().clone()
        source = grid.source.clone()
        grid.center_on_warm_codes()
        torch.testing.assert_close(before, grid(False), rtol=0, atol=0)
        grid().square().mean().backward()
        self.assertGreater(grid.log_scale.grad.abs().sum().item(), 0)
        self.assertGreater(grid.code_offset.grad.abs().sum().item(), 0)
        torch.testing.assert_close(source, grid.source, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
