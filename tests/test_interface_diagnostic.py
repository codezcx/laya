import unittest
import numpy as np
import torch
from eeg_align.interface_diagnostic import MatchTextRMS,fit_metrics

class DiagnosticTest(unittest.TestCase):
    def test_scale_changes_magnitude_preserves_direction_and_gradients(self):
        torch.manual_seed(17);base=torch.nn.Linear(5,8);layer=MatchTextRMS(base,.04)
        x=torch.randn(3,7,5);before=base(x);after=layer(x)
        torch.testing.assert_close(after.square().mean(-1).sqrt(),torch.full((3,7),.04))
        torch.testing.assert_close(torch.nn.functional.normalize(after,dim=-1),torch.nn.functional.normalize(before,dim=-1))
        (after*torch.randn_like(after)).sum().backward()
        self.assertTrue(torch.isfinite(base.weight.grad).all());self.assertGreater(base.weight.grad.abs().sum(),0)
    def test_task_metrics_do_not_reward_task_only_prediction(self):
        y=np.array([0,1,2,3,4]);logits=np.zeros((5,5));logits[:3,0]=1;logits[3:,3]=1
        r=fit_metrics(logits,y);self.assertEqual(r['MI'],.5);self.assertAlmostEqual(r['emotion'],1/3)

if __name__=='__main__':unittest.main()
