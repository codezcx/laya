import unittest
import torch
from eeg_tasks.model import masked_loss
from eeg_tasks.data import subject_split,normalized_window
import numpy as np
class TaskMaskTests(unittest.TestCase):
 def test_unannotated_tasks_have_zero_gradient(self):
  for kind in ['ce','sigmoid']:
   z=torch.randn(2,7,requires_grad=True);loss=masked_loss(z,torch.tensor([0,4]),torch.tensor([0,1]),kind);loss.backward()
   self.assertTrue(torch.equal(z.grad[0,3:],torch.zeros(4)))
   self.assertTrue(torch.equal(z.grad[1,:3],torch.zeros(3)))
   self.assertTrue(torch.equal(z.grad[:,5:],torch.zeros(2,2)))
 def test_same_semantic_class_not_false_negative(self):
  z=torch.tensor([[4.,0.,0.,0.,0.,0.,0.]],requires_grad=True)
  a=masked_loss(z,torch.tensor([0]),torch.tensor([0]));b=masked_loss(z.repeat(3,1),torch.tensor([0,0,0]),torch.tensor([0,0,0]));self.assertAlmostEqual(a.item(),b.item(),places=6)
 def test_subject_split_and_unit_invariance(self):
  self.assertEqual(subject_split(range(15),42),subject_split(range(15),42))
  x=np.random.default_rng(4).normal(size=(3,600));np.testing.assert_allclose(normalized_window(x),normalized_window(x*1e-6),atol=1e-6)
if __name__=='__main__':unittest.main()
