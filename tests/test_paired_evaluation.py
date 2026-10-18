import unittest
import numpy as np
from scripts.evaluate_paired_classification import report,shuffled_ids,ExternalPairCache


class PairedEvaluationTest(unittest.TestCase):
    def test_external_batch_preserves_signal_order_and_segment_boundary(self):
        cache=ExternalPairCache.__new__(ExternalPairCache)
        cache.tokens=np.repeat(np.arange(3)[:,None,None],18*512,axis=1).reshape(3,18,512).astype(np.float16)
        z,mask,cuts=cache.batch([2,0,1],'cpu')
        np.testing.assert_array_equal(z[:,0,0].numpy(),[2,0,1])
        self.assertTrue(mask.all());self.assertEqual(cuts,[9,9,9])
        for invalid in [[-1],[3],[.5]]:
            with self.assertRaises(ValueError):cache.batch(invalid,'cpu')

    def test_shuffle_preserves_subject_membership_and_every_signal(self):
        ids=np.arange(100);subjects=np.repeat(['s1','s2'],50)
        out=shuffled_ids(ids,subjects)
        np.testing.assert_array_equal(out,shuffled_ids(ids,subjects))
        for s in ['s1','s2']:
            np.testing.assert_array_equal(np.sort(out[subjects==s]),ids[subjects==s])
        self.assertFalse(np.array_equal(ids,out))

    def test_subject_metrics_do_not_hide_differential_failure(self):
        result=report([0,0,1,1,0,0,1,1],[0,0,1,1,1,1,0,0],['a']*4+['b']*4)
        self.assertEqual(result['overall']['accuracy'],.5)
        self.assertEqual(result['subjects']['a']['balanced_accuracy'],1.)
        self.assertEqual(result['subjects']['b']['balanced_accuracy'],0.)
