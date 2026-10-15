import unittest
import torch
from eeg_tasks.spatial_model import spatial_pool
class SpatialPoolTests(unittest.TestCase):
 def test_preserves_three_distinct_central_channels(self):
  pos=torch.tensor([[-.065,0.,0.],[0.,0.,.06],[.065,0.,0.]])
  x=torch.eye(3)[None];z=spatial_pool(x,pos,pos)
  self.assertTrue((z[0].diag()>.999).all())
 def test_joint_permutation_invariance(self):
  torch.manual_seed(2);pos=torch.randn(5,3)*.05;x=torch.randn(2,5,7);a=pos[:3];p=torch.tensor([2,3,1,4,0])
  torch.testing.assert_close(spatial_pool(x,pos,a),spatial_pool(x[:,p],pos[p],a))
 def test_identical_tokens_stay_identical(self):
  pos=torch.randn(3,3);x=torch.ones(2,3,5);a=torch.randn(4,3)
  torch.testing.assert_close(spatial_pool(x,pos,a),torch.ones(2,4,5))
if __name__=='__main__':unittest.main()
