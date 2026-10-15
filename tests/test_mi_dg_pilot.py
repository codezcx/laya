import unittest
import numpy as np
from eeg_align.dg_pilot import preprocess, signal_answers, metrics


class SignalSemanticsTests(unittest.TestCase):
    def test_shared_scale_retains_channel_power_ratio(self):
        t=np.arange(800)/200
        x=np.stack([2*np.sin(2*np.pi*10*t),np.sin(2*np.pi*10*t),np.sin(2*np.pi*10*t)])
        shared=preprocess(x,"harmonized")
        legacy=preprocess(x,"legacy")
        self.assertEqual(shared.shape,(3,600))
        self.assertAlmostEqual(float(shared[0].std()/shared[2].std()),2.,places=5)
        self.assertAlmostEqual(float(legacy[0].std()/legacy[2].std()),1.,places=5)

    def test_auxiliary_answers_track_spectral_content(self):
        t=np.arange(800)/200
        x=np.stack([np.sin(2*np.pi*10*t),np.sin(2*np.pi*15*t),np.sin(2*np.pi*20*t)])
        labels,_=signal_answers(preprocess(x,"harmonized"))
        swapped,_=signal_answers(preprocess(x[[2,1,0]],"harmonized"))
        self.assertEqual(labels,[0,1])
        self.assertEqual(swapped,[1,0])

    def test_subject_average_does_not_weight_large_subject_more(self):
        rows=[{"dataset":"test","subject":"large","label":0} for _ in range(100)]
        rows += [{"dataset":"test","subject":"small","label":1} for _ in range(10)]
        result=metrics(rows,[[1,0]]*110)
        self.assertAlmostEqual(result["accuracy"],100/110)
        self.assertAlmostEqual(result["subject_macro_accuracy"],.5)


if __name__ == "__main__":
    unittest.main()
