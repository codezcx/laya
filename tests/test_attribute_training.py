import unittest
from types import SimpleNamespace
import numpy as np
import torch
from eeg_align.attribute_data import PERMUTATIONS, QuestionTokens, semantic_predictions, summarize
from eeg_align.sampling import BalancedCoverageSampler


class FakeTokenizer:
    def encode(self, text, **kwargs):return [ord(c) for c in text]


class AttributeTrainingTest(unittest.TestCase):
    def test_joint_classification_loss_and_selection_exclude_attribute_inflation(self):
        from eeg_align.attribute_data import training_weights, draw_permutations
        specs=[dict(id='mi_class',task='mi',options={'A':'Left','B':'Right'}),
               dict(id='emotion_class',task='emotion',options={'A':'Negative','B':'Neutral','C':'Positive'}),
               dict(id='mi_attribute',task='mi'),dict(id='emotion_attribute',task='emotion')]
        rows=[];groups={}
        for q,n in enumerate([2,3,3,3]):
            for ds in ([1,2] if q in [1,3] else [0]):
                for a in range(n):
                    i=len(rows);rows.append([i,q,a,ds]);groups[q,ds,a]=[i]
        data=SimpleNamespace(specs=specs,groups=groups,records=np.array(rows),datasets=['MI','SEED','FACED'],choice_counts=np.array([2,3,3,3]))
        draws={k:16 if k[0]==0 else 8 if k[0]==1 else 1 for k in groups}
        weights=training_weights(data,draws);batch=sum(draws.values())
        masses=[sum(n*weights[q]/batch for (q,_,_),n in draws.items() if q==i) for i in range(4)]
        np.testing.assert_allclose(masses,[.4,.4,.1,.1])
        no_aux=training_weights(data,draws,attribute_loss_weight=0.)
        np.testing.assert_array_equal(no_aux[:2],weights[:2])
        np.testing.assert_array_equal(no_aux[2:],0.)
        # Removing auxiliary gradients must not rescale classification gradients.
        logits=torch.randn(batch,3,dtype=torch.float64,requires_grad=True)
        qids=np.array([q for (q,_,_),n in draws.items() for _ in range(n)])
        labels=torch.arange(batch)%3
        ce=torch.nn.functional.cross_entropy(logits,labels,reduction='none')
        (ce*torch.tensor(no_aux[qids])).mean().backward()
        expected=logits.grad.clone();logits.grad=None
        ce=torch.nn.functional.cross_entropy(logits,labels,reduction='none')
        keep=qids<2
        (ce[keep]*torch.tensor(weights[qids[keep]])).sum().div(batch).backward()
        torch.testing.assert_close(logits.grad,expected)
        permutations=draw_permutations(data,data.records,np.random.default_rng(42),True,True)
        self.assertTrue((permutations[data.records[:,1]<2]==0).all())
        pred=data.records[:,2].copy()
        # Perfect classification and completely incorrect attributes must select as perfect classification.
        ii=data.records[:,1]>=2;pred[ii]=(pred[ii]+1)%3
        r=summarize(data,np.arange(len(rows)),pred)
        self.assertEqual(r['selection_score'],1.)
        self.assertEqual(r['emotion_classification'],1.)
        self.assertEqual(r['emotion'],0.)

    def test_fixed_class_mapping_preserves_attribute_rng_and_labels(self):
        from eeg_align.attribute_data import draw_permutations
        specs=[dict(id='mi_class',question='Which hand?',options={'A':'Left','B':'Right'}),
               dict(id='mi_attribute',question='Compare powers.')]
        data=SimpleNamespace(specs=specs,choice_counts=np.array([2,3]))
        records=np.array([[i,i%2,(i//2)%2,0] for i in range(100)])
        a=np.random.default_rng(42);b=np.random.default_rng(42)
        original=draw_permutations(data,records,a)
        fixed=draw_permutations(data,records,b,fixed_mi_class=True)
        np.testing.assert_array_equal(fixed[1::2],original[1::2])
        np.testing.assert_array_equal(a.integers(100,size=20),b.integers(100,size=20))
        self.assertTrue((fixed[::2]==0).all())
        q=QuestionTokens(FakeTokenizer(),specs,device='cpu')
        suffix,labels=q.batch(records[::2],fixed[::2])
        np.testing.assert_array_equal(labels.numpy(),records[::2,2])
        self.assertIn('A) Left\nB) Right',''.join(chr(c) for c in suffix[0].tolist()))

    def test_rejects_old_single_window_schema(self):
        import tempfile
        from pathlib import Path
        from eeg_align.attribute_data import AttributeData
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/'complete.json').write_text('{"schema_version": 1}')
            with self.assertRaisesRegex(ValueError, 'schema version 2'):
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

    def test_binary_choices_are_masked_and_permuted(self):
        from eeg_align.attribute_data import mask_choices
        specs=[dict(question='Which hand?',options={'A':'Left','B':'Right'})]
        q=QuestionTokens(FakeTokenizer(),specs,device='cpu')
        records=np.array([[8,0,0,0],[9,0,1,0]])
        _,labels=q.batch(records,[1,1])
        self.assertEqual(labels.tolist(),[1,0])
        pred=mask_choices(torch.tensor([[1.,2.,99.],[2.,1.,99.]]),[2,2]).argmax(1)
        np.testing.assert_array_equal(semantic_predictions(pred.numpy(),[1,1],[2,2]),[0,1])

    def test_coverage_change_preserves_task_loss_mass(self):
        from eeg_align.attribute_data import training_weights
        specs=[{'id':'mi_class','task':'mi'},{'id':'mi_attribute','task':'mi'},{'id':'emotion_attribute','task':'emotion'}]
        groups={(0,0,a):[a] for a in range(2)}
        groups.update({(q,0,a):[10*q+a] for q in [1,2] for a in range(3)})
        data=SimpleNamespace(specs=specs,groups=groups)
        counts={k:16 if k[0]==0 else 1 for k in groups}
        weights=training_weights(data,counts);batch=sum(counts.values())
        mass=[sum(n*weights[q]/batch for (q,_,_),n in counts.items() if q==i) for i in range(3)]
        np.testing.assert_allclose(mass,[.25,.25,.5])

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
