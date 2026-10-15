import unittest
import torch
from eeg_tasks.spectral_fusion import fit_standardizer
class StandardizerTests(unittest.TestCase):
 def test_validation_does_not_affect_fit(self):
  x=torch.tensor([[[1.,2.]],[[3.,4.]],[[1e6,1e6]]]);mask=torch.tensor([True,True,False]);mean,std=fit_standardizer(x,mask)
  torch.testing.assert_close(mean,torch.tensor([[[2.,3.]]]));torch.testing.assert_close(std,torch.ones_like(std))
 def test_constant_features_finite(self):
  x=torch.ones(3,2,5);m,s=fit_standardizer(x,torch.tensor([True,True,False]));self.assertTrue(torch.isfinite((x-m)/s).all())
if __name__=='__main__':unittest.main()
