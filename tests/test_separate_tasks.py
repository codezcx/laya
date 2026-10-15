import unittest
import torch
from eeg_tasks.spatial_model import SeparateTaskClassifier,SpatialPrototypeClassifier
from eeg_tasks.model import masked_loss
class SeparateTaskTests(unittest.TestCase):
 def test_unknown_task_does_not_train_other_branch(self):
  torch.set_num_threads(2)
  for arm,kind in [('learned_ce','ce'),('text_sigmoid','sigmoid')]:
   m=SeparateTaskClassifier(torch.randn(7,32),arm,2,width=16);x=torch.randn(3,2,512);valid=torch.ones(3,2,dtype=torch.bool);loss=masked_loss(m(x,valid),torch.tensor([0,1,2]),torch.zeros(3,dtype=torch.long),kind);loss.backward()
   self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in m.branches[0].parameters()))
   self.assertTrue(all(p.grad is None or p.grad.abs().sum()==0 for p in m.branches[1].parameters()))
 def test_readout_parameters_nearly_matched(self):
  proto=torch.randn(7,4096)
  a=sum(p.numel() for p in SpatialPrototypeClassifier(proto,'learned_ce',8).parameters());b=sum(p.numel() for p in SeparateTaskClassifier(proto,'learned_ce',8,width=256).parameters());self.assertLess(abs(a-b)/a,.01)
if __name__=='__main__':unittest.main()
