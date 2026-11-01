import pytest

from scripts.run_benchmark import run_on_gpu_slots


def test_all_four_arms_run_on_two_slots_without_gpu_overlap():
    live, started, reports = {}, [], []

    class Child:
        def __init__(self, arm, gpu):
            assert gpu not in live
            self.arm, self.gpu = arm, gpu
            self.pid = len(started) + 1
            self.polls = 0
            live[gpu] = arm
            started.append((arm, gpu))

        def poll(self):
            self.polls += 1
            if self.polls < 3:
                return None
            live.pop(self.gpu)
            return 0

    arms = ['mean_pool', 'cls', 'cross_attention', 'qformer']
    result = run_on_gpu_slots(arms, [2, 3], Child, lambda **x: reports.append(x), poll_seconds=0)
    assert [a for a, _ in started] == arms
    assert {g for _, g in started} == {2, 3}
    assert set(result) == set(arms)
    assert all(p['returncode'] == 0 for p in result.values())
    assert reports[0]['queued_arms'] == ['cross_attention', 'qformer']
    assert not live


def test_failure_drains_live_peer_and_does_not_launch_queued_arm():
    started, peers_finished = [], []

    class Child:
        def __init__(self, arm, gpu):
            self.arm, self.pid, self.polls = arm, len(started) + 1, 0
            started.append(arm)

        def poll(self):
            self.polls += 1
            if self.arm == 'bad':
                return 1
            if self.polls < 3:
                return None
            peers_finished.append(self.arm)
            return 0

    with pytest.raises(RuntimeError, match='unstarted arms'):
        run_on_gpu_slots(['bad', 'good', 'queued'], [2, 3], Child, lambda **x: None, poll_seconds=0)
    assert started == ['bad', 'good']
    assert peers_finished == ['good']


@pytest.mark.parametrize('arms,gpus', [([], [2]), (['a'], []), (['a', 'a'], [2]), (['a'], [2, 2])])
def test_invalid_queue_is_rejected(arms, gpus):
    with pytest.raises(ValueError, match='nonempty and unique'):
        run_on_gpu_slots(arms, gpus, None, None)
