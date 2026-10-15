import unittest
from eeg_tasks.full_source_loader import cap_training_mi
from eeg_tasks.domain_split import domain_indices
class FullSourceTests(unittest.TestCase):
 def test_cap_retains_emotion_and_never_adds_rows(self):
  rows=[dict(task=0,dataset='SEED',subject='1',class_id=0)]+[dict(task=1,dataset='Cho2017',subject='1',class_id=3) for _ in range(10)]
  a=cap_training_mi(rows,list(range(11)),3);self.assertIn(0,a);self.assertEqual(len(a),4);self.assertEqual(a,cap_training_mi(rows,list(range(11)),3))
 def test_lee_domain_is_fully_held_out(self):
  rows=[dict(task=t,dataset=d,subject=s,split=split) for t,d in [(0,'SEED'),(1,'Cho2017'),(1,'Dreyer2023'),(1,'Lee2019_MI')] for s,split in [('1','train'),('2','validation')]]
  tr,val,within=domain_indices(rows,'Lee2019_MI');self.assertTrue(all(rows[i]['dataset']!='Lee2019_MI' for i in tr));self.assertEqual(sum(rows[i]['dataset']=='Lee2019_MI' for i in val),2)
if __name__=='__main__':unittest.main()
