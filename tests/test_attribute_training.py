import unittest
from types import SimpleNamespace
import numpy as np
import torch
from eeg_align.attribute_data import PERMUTATIONS, QuestionTokens, semantic_predictions, summarize
from eeg_align.sampling import BalancedCoverageSampler


class FakeTokenizer:
    def encode(self, text, **kwargs):return [ord(c) for c in text]


class AttributeTrainingTest(unittest.TestCase):
    def test_paired_schema_rejects_single_window_cache(self):
        import tempfile
        from pathlib import Path
        from eeg_align.attribute_data import AttributeData
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/'complete.json').write_text('{"schema_version": 2}')
            with self.assertRaisesRegex(ValueError, 'segment-aware'):
                AttributeData(folder,'train')

    def test_option_permutation_roundtrip(self):
        specs=[dict(question='Compare channel powers.')]
        questions=QuestionTokens(FakeTokenizer(),specs,device='cpu')
        for p,perm in enumerate(PERMUTATIONS):
            records=np.array([[9,0,a,0] for a in range(3)])
            suffix,labels=questions.batch(records,[p]*3)
            np.testing.assert_array_equal(semantic_predictions(labels.numpy(),[p]*3),np.arange(3))
            text=''.join(chr(c) for c in suffix[0].tolist())
            self.assertIn('A) ',text);self.assertIn('[decision]:',text)
            self.assertNotIn('dataset',text)

    def test_metric_uses_semantics_and_macro_datasets(self):
        data=SimpleNamespace(specs=[{'id':'mi_test'},{'id':'emotion_test'}],datasets=['a','b'])
        data.records=np.array([[i,q,a,d] for i,(q,a,d) in enumerate(
            [(0,0,0),(0,1,0),(0,2,0),(0,0,1),(0,1,1),(0,2,1),
             (1,0,0),(1,1,0),(1,2,0)])])
        pred=np.array([0,0,0,0,1,2,0,0,0])
        r=summarize(data,np.arange(9),pred)
        self.assertAlmostEqual(r['MI'],2/3)
        self.assertAlmostEqual(r['emotion'],1/3)

    def test_unequal_rank_weighted_loss_matches_global(self):
        x=torch.arange(90,dtype=torch.float64)/90
        w=torch.tensor([5/6]*54+[5/4]*36)
        p=torch.tensor(1.,requires_grad=True,dtype=torch.float64)
        ((p*x)**2*w).mean().backward();expected=p.grad.clone()
        grads=[]
        for rank in range(4):
            p=torch.tensor(1.,requires_grad=True,dtype=torch.float64)
            (((p*x[rank::4])**2*w[rank::4]).sum()*4/90).backward();grads.append(p.grad)
        torch.testing.assert_close(torch.stack(grads).mean(),expected)

    def test_group_balance_and_bounded_repeats(self):
        groups={(i,0,0):list(range(i*100,(i+1)*100)) for i in range(90)}
        sampler=BalancedCoverageSampler(groups,per_group=1,seed=42)
        counts=np.zeros(9000,int)
        for _ in range(350):np.add.at(counts,sampler.next_batch().reshape(-1),1)
        self.assertLessEqual(counts.max(),4)
        self.assertEqual(counts.sum(),350*90)


if __name__=='__main__':unittest.main()
