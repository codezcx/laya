import pytest
from eeg_tasks.emotion_trial_v29 import trial_groups


def test_distinct_recordings_and_invalid_trials():
    row = dict(dataset='SEED', subject='1', path='a', trial='1', class_id=0, split='train', start_sample=0)
    rows = [row, dict(row, start_sample=600), dict(row, path='b')]
    assert trial_groups(rows) == [[0, 1], [2]]
    with pytest.raises(ValueError):trial_groups([row, dict(row)])
    with pytest.raises(ValueError):trial_groups([row, dict(row, start_sample=600, class_id=1)])
