import unittest
import torch
from eeg_tasks.channel_model import ChannelPrototypeClassifier

class ChannelModelTests(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(2);torch.manual_seed(4);self.model=ChannelPrototypeClassifier(torch.randn(7,32),'text_ce').eval()
 def test_padding_has_no_effect(self):
  x=torch.randn(2,3,512);mask=torch.ones(2,3,dtype=torch.bool)
  with torch.no_grad():
   a=self.model(x,mask);b=self.model(torch.cat([x,torch.randn(2,5,512)*100],1),torch.cat([mask,torch.zeros(2,5,dtype=torch.bool)],1))
  torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)
 def test_token_order_invariance(self):
  x=torch.randn(2,5,512);m=torch.ones(2,5,dtype=torch.bool);perm=torch.tensor([3,0,4,1,2])
  with torch.no_grad():torch.testing.assert_close(self.model(x,m),self.model(x[:,perm],m[:,perm]),atol=1e-5,rtol=1e-5)
 def test_all_masked_rejected(self):
  with self.assertRaises(ValueError):self.model(torch.randn(1,3,512),torch.zeros(1,3,dtype=torch.bool))
if __name__=='__main__':unittest.main()
