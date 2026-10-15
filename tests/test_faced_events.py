import unittest
from eeg_tasks.faced_prepare import video_events
class FacedEventTests(unittest.TestCase):
 def valid(self):
  labels=['negative']*12+['neutral']*4+['positive']*12
  return [dict(binary_label=label,emotion_label='\\' if label=='neutral' else 'Joy',duration='40',video_index=str(i)) for i,label in enumerate(labels)]
 def test_neutral_placeholder_is_not_dropped(self):
  events=video_events(self.valid()+[dict(binary_label='n/a')]);self.assertEqual(sum(r['binary_label']=='neutral' for r in events),4)
 def test_duplicate_video_rejected(self):
  rows=self.valid();rows[-1]['video_index']='0'
  with self.assertRaises(AssertionError):video_events(rows)
 def test_unknown_valence_rejected(self):
  rows=self.valid();rows[0]['binary_label']='unknown'
  with self.assertRaises(ValueError):video_events(rows)
if __name__=='__main__':unittest.main()
