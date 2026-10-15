import unittest
import torch
from eeg_tasks.train import metrics
class BalancedTaskMetricTests(unittest.TestCase):
 def test_imbalance_does_not_hide_minority_failure(self):
  rows=[dict(dataset='emotion',subject='1',task=0,class_id=y) for y in [0,0,0,1]];logits=torch.zeros(4,7);logits[:,0]=10
  result=metrics(rows,logits);self.assertEqual(result['macro_task_accuracy'],.75);self.assertEqual(result['macro_task_balanced_accuracy'],.5)
if __name__=='__main__':unittest.main()
