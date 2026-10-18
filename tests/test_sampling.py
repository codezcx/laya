import unittest
import copy
import json
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
    def test_resume_matches_uninterrupted_sampling_across_rounds(self):
        groups={'short':list(range(7)),'long':list(range(7,30))}
        for checkpoint_step in [0,1,6,7,13]:
            reference=BalancedCoverageSampler(groups,4,42)
            for _ in range(checkpoint_step):reference.next_batch()
            # State also survives a JSON round trip, without Python object aliases.
            state=json.loads(json.dumps(reference.state_dict()))
            resumed=BalancedCoverageSampler(groups,4,999);resumed.load_state_dict(state)
            for _ in range(20):np.testing.assert_array_equal(reference.next_batch(),resumed.next_batch())
            self.assertEqual(reference.state_dict(),resumed.state_dict())
    def test_resume_rejects_partition_changes_and_queue_corruption(self):
        original=BalancedCoverageSampler({'a':list(range(10))},4,42);original.next_batch();state=original.state_dict()
        other=BalancedCoverageSampler({'a':list(range(1,11))},4,42)
        with self.assertRaises(ValueError):other.load_state_dict(state)
        corrupt=copy.deepcopy(state);corrupt['queues'][0][0]=100
        with self.assertRaises(ValueError):original.load_state_dict(corrupt)
        self.assertEqual(original.state_dict(),state)
    def test_rejects_duplicate_indices_and_invalid_rank_partition(self):
        with self.assertRaises(ValueError):BalancedCoverageSampler({'a':[1,2],'b':[2,3]})
        with self.assertRaises(ValueError):BalancedCoverageSampler.rank_batch(np.ones((2,3)),0,2)

if __name__=='__main__':unittest.main()

class MixedDrawCoverageTest(unittest.TestCase):
    def test_variable_draws_coverage_and_resume(self):
        groups={(0,0,0):list(range(4400)),(0,0,1):list(range(4400,8800)),(1,0,0):list(range(8800,9300))}
        counts={(0,0,0):16,(0,0,1):16,(1,0,0):1}
        a=BalancedCoverageSampler(groups,counts,seed=42);seen=[]
        for _ in range(275):seen.extend(a.next_batch()[:32])
        self.assertEqual(sorted(seen),list(range(8800)))
        b=BalancedCoverageSampler(groups,counts,seed=1);b.load_state_dict(a.state_dict())
        np.testing.assert_array_equal(a.next_batch(),b.next_batch())
