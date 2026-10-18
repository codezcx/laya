import unittest
import numpy as np
from scripts.report_readout_evaluation import cluster_difference


class ClusterDifferenceTest(unittest.TestCase):
    def test_identical_and_constant_improvement(self):
        y=np.array([0,1,0,1,0,1]);subjects=np.array(['a','a','b','b','b','b'])
        same=cluster_difference(y,y,y,subjects)
        self.assertEqual(same['subject_bootstrap_95_percent_interval'],[0.,0.])
        gain=cluster_difference(y,y,1-y,subjects)
        self.assertEqual(gain['accuracy_difference'],1.)
        self.assertEqual(gain['subject_bootstrap_95_percent_interval'],[1.,1.])


if __name__=='__main__':unittest.main()
