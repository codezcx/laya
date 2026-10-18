import unittest
import numpy as np
from eeg_tasks.attribute_questions import relative_power, compare, specifications, paired_normalize, paired_answers, absolute_power


class AttributeQuestionsTest(unittest.TestCase):
    def test_emotion_power_unchanged(self):
        t=np.arange(600)/200
        x=np.stack([np.sin(2*np.pi*10*t),np.sin(2*np.pi*20*t)])[None]
        p=relative_power(x)
        self.assertGreater(p['alpha'][0,0],.99)
        self.assertGreater(p['beta'][0,1],.99)
        for band,v in p.items(): np.testing.assert_allclose(v,relative_power(x*13)[band],atol=1e-12)

    def test_paired_changes_and_shared_scale(self):
        t=np.arange(400)/200
        base=np.tile(np.sin(2*np.pi*10*t)+np.sin(2*np.pi*20*t),(3,1))
        task=base*np.array([.5,1,2])[:,None]
        pair,scale=paired_normalize(base,task)
        answer=paired_answers(pair,0)
        self.assertEqual(answer['mi_change_c3_alpha'][0],'A')
        self.assertEqual(answer['mi_change_c4_beta'][0],'C')
        self.assertEqual(answer['mi_lateralization_alpha'][0],'A')
        self.assertEqual(answer['mi_class'][0],'A')
        np.testing.assert_allclose(answer['mi_change_c3_alpha'][1]['channel_changes_db'],[-6.0206,0,6.0206],atol=1e-5)
        scaled,_=paired_normalize(base*37,task*37)
        np.testing.assert_allclose(pair,scaled,atol=1e-6)
        reverse=paired_answers(pair[::-1],1)
        self.assertEqual(reverse['mi_change_c3_alpha'][0],'C')
        self.assertEqual(reverse['mi_class'][0],'B')
        # Both increase: more negative means smaller positive change, not an ERD claim.
        both,_=paired_normalize(base,base*np.array([2,1,4])[:,None])
        self.assertEqual(paired_answers(both,0)['mi_lateralization_alpha'][0],'A')
        self.assertEqual(paired_answers(both,0)['mi_change_c3_alpha'][0],'C')

    def test_independent_fft_matches_welch(self):
        from scripts.verify_attribute_dataset import fft_power
        rng=np.random.default_rng(82)
        x=rng.normal(size=(2,3,400))
        direct=fft_power(x)
        power=absolute_power(x)
        np.testing.assert_allclose(direct[...,8:13].sum(-1),power['alpha'],rtol=1e-12)
        np.testing.assert_allclose(direct[...,13:31].sum(-1),power['beta'],rtol=1e-12)

    def test_invalid_and_catalog(self):
        with self.assertRaises(ValueError):paired_normalize(np.zeros((3,400)),np.zeros((3,400)))
        _,answer=compare([2,1.1,.5],[1,1,1]);self.assertEqual(answer.tolist(),['A','B','C'])
        specs=specifications()
        self.assertEqual(len({s['id'] for s in specs}),13)
        self.assertEqual(sum(s['task']=='mi' for s in specs),7)
        self.assertFalse(any(s['id'].startswith('mi_temporal') for s in specs))


if __name__=='__main__':unittest.main()
