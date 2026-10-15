import unittest
from eeg_tasks.domain_split import domain_indices
class DomainSplitTests(unittest.TestCase):
 def test_whole_domain_excluded_from_training(self):
  rows=[dict(task=t,dataset=d,subject=s,split=split) for t,d in [(0,'SEED'),(1,'Cho2017'),(1,'Dreyer2023')] for s,split in [('1','train'),('2','validation')]]
  train,cross,within=domain_indices(rows,'Cho2017');self.assertTrue(all(rows[i]['dataset']!='Cho2017' for i in train));self.assertEqual(sum(rows[i]['dataset']=='Cho2017' for i in cross),2);self.assertEqual(sum(rows[i]['dataset']=='Dreyer2023' for i in within),1)
 def test_source_test_is_rejected(self):
  with self.assertRaises(ValueError):domain_indices([dict(task=1,dataset='Cho2017',subject='1',split='test')],'Cho2017')
if __name__=='__main__':unittest.main()
