import json
import pytest

from scripts.run_benchmark import run_on_gpu_slots, unfinished_arms, snapshot_interrupted_outputs


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


def resume_fixture(tmp_path):
    config = dict(arms=['done', 'pending'], gpus=[4, 5])
    for arm in config['arms']:
        directory = tmp_path / 'train' / arm
        directory.mkdir(parents=True)
        (directory / 'config.json').write_text(json.dumps(dict(
            arm=arm, config=config, smoke=False, code_sha256={})))
        (directory / 'last.pt').write_bytes(b'checkpoint')
        (directory / 'best.pt').write_bytes(b'best')
    (tmp_path / 'train/done/complete.json').write_text('{}')
    return config


def test_resume_skips_complete_without_mutating_original_gpu_recipe(tmp_path):
    config = resume_fixture(tmp_path)
    assert unfinished_arms(tmp_path, config) == ['pending']
    assert config['gpus'] == [4, 5]


def test_resume_rejects_changed_recipe_or_code(tmp_path):
    config = resume_fixture(tmp_path)
    with pytest.raises(ValueError, match='identity mismatch'):
        unfinished_arms(tmp_path, dict(config, gpus=[0, 1]))
    path = tmp_path / 'train/pending/config.json'
    identity = json.loads(path.read_text())
    identity['code_sha256'] = {str(tmp_path / 'missing.py'): 'stale'}
    path.write_text(json.dumps(identity))
    with pytest.raises(ValueError, match='code hash mismatch'):
        unfinished_arms(tmp_path, config)


def test_resume_rejects_missing_checkpoint(tmp_path):
    config = resume_fixture(tmp_path)
    (tmp_path / 'train/pending/last.pt').unlink()
    with pytest.raises(ValueError, match='requires last.pt and best.pt'):
        unfinished_arms(tmp_path, config)


def test_recovery_preserves_partial_outputs_without_copying_checkpoints(tmp_path):
    config = resume_fixture(tmp_path)
    (tmp_path / 'pipeline.json').write_text('{"status":"interrupted"}')
    directory = tmp_path / 'train/pending'
    (directory / 'input_audit.json').write_text('{"training_samples":32640}')
    (directory / 'valid_task_predictions.npz').write_bytes(b'predictions')
    recovery = tmp_path / 'recovery'
    recovery.mkdir()
    snapshot_interrupted_outputs(tmp_path, recovery, unfinished_arms(tmp_path, config))
    target = recovery / 'before_resume/pending'
    assert (target / 'input_audit.json').read_bytes() == (directory / 'input_audit.json').read_bytes()
    assert (target / 'valid_task_predictions.npz').read_bytes() == b'predictions'
    assert not (target / 'last.pt').exists()
    assert (directory / 'last.pt').read_bytes() == b'checkpoint'
    assert not (recovery / 'before_resume/done').exists()
    assert json.loads((recovery / 'previous_pipeline.json').read_text())['status'] == 'interrupted'
