import unittest
import torch
from eeg_align.signal_control_v30 import SignalControl


class SignalControlTest(unittest.TestCase):
    def test_invalid_padding_cannot_change_logits_or_gradients(self):
        torch.manual_seed(42); torch.set_num_threads(2)
        model = SignalControl()
        x = torch.randn(2,120,512)
        mask = torch.arange(120)[None] < torch.tensor([12,120])[:,None]
        with torch.no_grad(): expected = model(x,mask)['logits']
        x[~mask] = float('nan')
        result = model(x,mask,torch.tensor([3,0]))
        torch.testing.assert_close(result['logits'],expected,rtol=0,atol=0)
        result['loss'].backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(sum(float(p.grad.norm()) for p in model.parameters()),0)


if __name__=='__main__': unittest.main()
