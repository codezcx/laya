import unittest
import torch
from eeg_align.jev import LoadedLoRA,assemble_embeddings

class JEVTest(unittest.TestCase):
    def test_frozen_adapter_remains_active(self):
        torch.manual_seed(3)
        base=torch.nn.Linear(7,9,bias=False)
        a=torch.randn(2,7);b=torch.randn(9,2);x=torch.randn(4,7)
        layer=LoadedLoRA(base,a,b,2.)
        expected=torch.nn.functional.linear(x,base.weight+2*b@a)
        actual=layer(x)
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-5)
        layer.requires_grad_(False)
        torch.testing.assert_close(layer(x),actual)
        self.assertGreater((layer(x)-base(x)).abs().max(),1.)
    def test_concatenation_and_last_position(self):
        prefix=torch.randn(3,8);suffix=torch.randn(2,8);eeg=torch.randn(2,5,8)
        valid=torch.tensor([[1,1,0,0,0],[1,1,1,1,1]],dtype=torch.bool)
        x,m,p,last=assemble_embeddings(prefix,eeg,suffix,valid)
        self.assertEqual(last.tolist(),[6,9]);self.assertEqual(m.sum(1).tolist(),[7,10])
        torch.testing.assert_close(x[0,3:5],eeg[0,:2]);torch.testing.assert_close(x[0,5:7],suffix)
        torch.testing.assert_close(x[1,8:10],suffix)
        self.assertEqual(p[1].tolist(),list(range(10)))

if __name__=='__main__':unittest.main()
