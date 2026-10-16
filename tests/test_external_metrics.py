import unittest
import json
import tempfile
from pathlib import Path
from eeg_align.evaluate import metrics
from eeg_align.cache import digest


class ExternalMetricsTest(unittest.TestCase):

    def test_binary_global_ids_and_wrong_task_predictions(self):
        rows=[{'subject':'1','class_id':c} for c in [3,3,3,4]]
        result=metrics(rows,[3,3,0,0])
        self.assertEqual(result['accuracy'],.5)
        self.assertAlmostEqual(result['subject_balanced_accuracy'],1/3)
        self.assertEqual(result['per_subject']['1']['accuracy'],.5)

    def test_emotion_compatibility(self):
        rows=[{'subject':s,'class_id':c} for s in ['1','2'] for c in [0,1,2]]
        result=metrics(rows,[0,1,2,3,3,3])
        self.assertEqual(result['accuracy'],.5)
        self.assertEqual(result['subject_balanced_accuracy'],.5)


if __name__=='__main__':unittest.main()
