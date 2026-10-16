import unittest
import json
import tempfile
from pathlib import Path
from eeg_align.external_v30 import metrics, frozen
from eeg_align.cache_multitask_v30 import digest


class ExternalMetricsTest(unittest.TestCase):
    def test_selected_checkpoint_freeze_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'qwen').mkdir();checkpoint=root/'qwen/best.pt'
            checkpoint.write_bytes(b'original')
            record={'arms':{'qwen':{'best.pt':digest(checkpoint)}}}
            (root/'source_freeze.json').write_text(json.dumps(record))
            self.assertEqual(frozen(root),digest(root/'source_freeze.json'))
            checkpoint.write_bytes(b'changed')
            with self.assertRaises(AssertionError):frozen(root)

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
