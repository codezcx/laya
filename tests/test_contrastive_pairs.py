import unittest
import torch
from eeg_align.model import contrastive_pair_loss


class PairLossTest(unittest.TestCase):
    def test_same_class_descriptions_are_all_positive(self):
        signal=torch.eye(2,requires_grad=True)
        text=torch.tensor([[1.,0.],[1.,0.],[0.,1.],[0.,1.]])
        y=torch.tensor([0,1]);ty=torch.tensor([0,0,1,1])
        good=contrastive_pair_loss(signal,text,y,ty)
        bad=contrastive_pair_loss(signal,text,y,1-ty)
        self.assertLess(float(good.detach()),float(bad.detach()))
        duplicate=contrastive_pair_loss(signal,text.repeat(2,1),y,ty.repeat(2))
        torch.testing.assert_close(good,duplicate)
        good.backward();self.assertTrue(torch.isfinite(signal.grad).all())

    def test_missing_positive_is_rejected(self):
        with self.assertRaises(ValueError):
            contrastive_pair_loss(torch.eye(2),torch.eye(2),torch.tensor([0,0]),torch.tensor([0,1]))


if __name__=='__main__':unittest.main()
