import unittest
import numpy as np
from scripts.evaluate_paired_classification import report,shuffled_ids


class PairedEvaluationTest(unittest.TestCase):
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
