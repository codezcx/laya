import unittest
import torch
from eeg_tasks.trial_eval import aggregate_trials
class TrialEvaluationTests(unittest.TestCase):
 def row(self,**kw):return dict(dataset='d',subject='1',path=kw.get('path','a'),trial='1',class_id=kw.get('label',0),task=0,start_sample=kw.get('start',0))
 def test_recordings_not_merged(self):
  rows,scores=aggregate_trials([self.row(),self.row(path='b')],torch.zeros(2,7));self.assertEqual(len(rows),2)
 def test_mean_probability(self):
  logits=torch.tensor([[2.,0.,0.,0.,0.,0.,0.],[0.,2.,0.,0.,0.,0.,0.]])
  rows,scores=aggregate_trials([self.row(),self.row(start=600)],logits);torch.testing.assert_close(scores[0,:3].exp(),logits[:,:3].softmax(-1).mean(0));self.assertEqual(rows[0]['windows'],2)
 def test_duplicates_and_conflicting_labels_rejected(self):
  for rows in [[self.row(),self.row()],[self.row(),self.row(start=600,label=1)]]:
   with self.assertRaises(ValueError):aggregate_trials(rows,torch.zeros(2,7))
if __name__=='__main__':unittest.main()
