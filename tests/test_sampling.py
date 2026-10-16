import unittest
import numpy as np
from eeg_align.sampling import BalancedCoverageSampler

class CoverageTest(unittest.TestCase):
    def test_every_group_is_fully_covered_each_round(self):
        groups={'a':list(range(5)),'b':list(range(5,18))}
        sampler=BalancedCoverageSampler(groups,per_group=4,seed=12)
        self.assertEqual(sampler.steps_per_round,4)
        for cycle in range(2):
            seen=[set(),set()]
            for _ in range(sampler.steps_per_round):
                batch=sampler.next_batch();self.assertEqual(batch.shape,(2,4))
                for i,key in enumerate(sorted(groups)):
                    self.assertTrue(set(batch[i])<=set(groups[key]));seen[i].update(batch[i])
            self.assertEqual(seen,[set(groups[k]) for k in sorted(groups)])
        self.assertEqual(sampler.round,2)
    def test_replay_and_rank_partition_preserve_exact_global_batch(self):
        groups={str(k):list(range(100*k,100*k+40)) for k in range(3)}
        a=BalancedCoverageSampler(groups);b=BalancedCoverageSampler(groups)
        for _ in range(11):
            batch=a.next_batch();np.testing.assert_array_equal(batch,b.next_batch())
            ranks=[a.rank_batch(batch,r,4) for r in range(4)]
            self.assertTrue(all(len(v)==6 for v in ranks))
            np.testing.assert_array_equal(np.sort(np.concatenate(ranks)),np.sort(batch.flatten()))
            self.assertEqual(len(set(np.concatenate(ranks))),24)
    def test_no_replacement_until_a_group_exhausts(self):
        s=BalancedCoverageSampler({'only':list(range(10))},per_group=4)
        sequence=np.concatenate([s.next_batch().flatten() for _ in range(3)])
        self.assertEqual(len(set(sequence[:10])),10)
    def test_rejects_duplicate_indices_and_invalid_rank_partition(self):
        with self.assertRaises(ValueError):BalancedCoverageSampler({'a':[1,2],'b':[2,3]})
        with self.assertRaises(ValueError):BalancedCoverageSampler.rank_batch(np.ones((2,3)),0,2)

if __name__=='__main__':unittest.main()
