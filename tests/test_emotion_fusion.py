import torch
from eeg_tasks.emotion_fusion_train import standardize,Classifier

def test_standardizer_excludes_held_rows():
 x=torch.tensor([[1.,2.],[3.,6.],[999.,999.]])
 z,m,s=standardize(x,[0,1]);torch.testing.assert_close(m,torch.tensor([[2.,4.]]));torch.testing.assert_close(s,torch.tensor([[1.,2.]]));assert z[-1,0]>900
 y=x.clone();y[-1]*=100;_,m2,s2=standardize(y,[0,1]);torch.testing.assert_close(m,m2);torch.testing.assert_close(s,s2)

def test_both_heads_receive_signal_gradients():
 for text in [False,True]:
  m=Classifier(12,torch.nn.functional.normalize(torch.randn(3,16),dim=1),text);x=torch.randn(4,12,requires_grad=True);z=m(x);assert z.shape==(4,3);torch.nn.functional.cross_entropy(z,torch.tensor([0,1,2,0])).backward();assert x.grad.abs().sum()>0
