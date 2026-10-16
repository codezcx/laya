import unittest
import torch
from eeg_align.model import EEGTextAligner

class ModelTest(unittest.TestCase):
    def test_padding_does_not_change_representation(self):
        torch.manual_seed(4)
        model=EEGTextAligner(torch.randn(20,32),torch.arange(5).repeat_interleave(4)).eval()
        x=torch.randn(2,12,512);mask=torch.ones(2,12,dtype=torch.bool)
        extended=torch.cat([x,torch.full((2,108,512),float('nan'))],1)
        valid=torch.cat([mask,torch.zeros(2,108,dtype=torch.bool)],1)
        with torch.no_grad():
            torch.testing.assert_close(model(x,mask)['embedding'],model(extended,valid)['embedding'])

    def test_gradient_and_class_scores(self):
        model=EEGTextAligner(torch.randn(20,32),torch.arange(5).repeat_interleave(4))
        output=model(torch.randn(5,12,512),torch.ones(5,12,dtype=torch.bool),torch.arange(5))
        self.assertEqual(output['logits'].shape,(5,5))
        output['loss'].backward()
        self.assertTrue(torch.isfinite(model.projector[0].weight.grad).all())
        self.assertGreater(model.projector[0].weight.grad.abs().sum(),0)
        self.assertIsNone(model.text_vectors.grad)

if __name__=='__main__':unittest.main()
