import unittest
import numpy as np
from eeg_tasks.source_probe import spectral_features,covariance_features,score

class SourceProbeTests(unittest.TestCase):
 def test_alpha_peak_and_spatial_power_contrast(self):
  t=np.arange(600)/200
  signal=np.stack([np.sin(2*np.pi*10*t),2*np.sin(2*np.pi*10*t),np.sin(2*np.pi*20*t)])[None]
  features=spectral_features(signal)
  self.assertEqual(features.shape,(1,30))
  self.assertEqual(features[0,:5].argmax(),2)
  self.assertAlmostEqual(float(features[0,15+2]),-np.log(4),places=5)
 def test_covariance_preserves_channel_correlation(self):
  x=np.random.default_rng(0).normal(size=600);s=np.stack([x,x,-x])[None]
  c=covariance_features(s)[0]
  self.assertAlmostEqual(c[0],c[1]);self.assertAlmostEqual(c[0],-c[2])
 def test_subject_macro_not_window_weighted(self):
  rows=[{'dataset':'a','subject':'1'}]*3+[{'dataset':'a','subject':'2'}]
  result=score(rows,np.array([0,0,0,0]),np.array([0,0,0,1]))
  self.assertEqual(result['macro_dataset_accuracy'],.5)
if __name__=='__main__':unittest.main()
