import unittest
import numpy as np
from eeg_align.data import metrics


class MetricsTest(unittest.TestCase):
    def test_oracle_does_not_hide_routing_failure(self):
        rows = [dict(dataset=d,subject='1',class_id=c)
                for d,classes in [('SEED',[0,1,2]),('FACED-NEMAR',[0,1,2]),
                                  ('Cho2017',[3,4]),('Dreyer2023',[3,4]),('Lee2019_MI',[3,4])]
                for c in classes]
        logits = np.zeros((len(rows),5))
        for i,r in enumerate(rows):
            logits[i,r['class_id']] = 1
            logits[i,3 if r['class_id']<3 else 0] = 2
        result = metrics(rows,logits)
        self.assertEqual(result['selection_score'],0)
        for value in result['datasets'].values():
            self.assertEqual(value,dict(joint=0.,oracle=1.,routing=0.))

    def test_tasks_have_equal_selection_weight(self):
        rows = [dict(dataset=d,subject='1',class_id=c)
                for d,classes in [('SEED',[0,1,2]),('FACED-NEMAR',[0,1,2]),
                                  ('Cho2017',[3,4]),('Dreyer2023',[3,4]),('Lee2019_MI',[3,4])]
                for c in classes]
        logits = np.zeros((len(rows),5))
        for i,r in enumerate(rows):
            logits[i,r['class_id'] if r['class_id']<3 else 0] = 1
        self.assertEqual(metrics(rows,logits)['selection_score'],.5)


if __name__ == '__main__': unittest.main()
