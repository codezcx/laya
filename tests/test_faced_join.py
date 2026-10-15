import unittest
from eeg_tasks.faced_join import select_emotion_sources
class FacedJoinTests(unittest.TestCase):
 def test_validation_and_mi_unchanged(self):
  rows=[dict(dataset=d,split=s) for d in ['SEED','Cho2017'] for s in ['train','validation']];new=[dict(dataset='FACED-NEMAR',split=s) for s in ['train','validation']]
  for mode in ['seed_only','faced_only','seed_faced']:
   a,b=select_emotion_sources(rows,new,mode);self.assertEqual(a[1],rows[1]);self.assertEqual(a[2:],rows[2:]);self.assertEqual(b[1],new[1]);self.assertEqual(rows[0]['split'],'train');self.assertEqual(new[0]['split'],'train')
   self.assertEqual(a[0]['split']=='train',mode!='faced_only');self.assertEqual(b[0]['split']=='train',mode!='seed_only')
if __name__=='__main__':unittest.main()
