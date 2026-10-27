import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from eeg_align.official_data import check_indices, epoch_batches, require_final_test, build_plan, OfficialFold
from eeg_align.task_options import make_variant


def test_same_subject_different_trials_still_rejected():
    with pytest.raises(ValueError, match='overlap'):
        check_indices(np.array([0]), np.array([1]), ['same', 'same'])


def test_duplicate_rows_usage_and_out_of_range_rejected():
    with pytest.raises(ValueError):
        check_indices(np.array([0, 0]), np.array([1]), ['a', 'b'])
    with pytest.raises(ValueError):
        check_indices(np.array([0]), np.array([2]), ['a', 'b'])
    with pytest.raises(ValueError, match='usage'):
        check_indices(np.array([0]), np.array([1]), ['a', 'b'], ['test', 'train'])


def test_every_training_trial_once_no_drop_last_no_test_rows():
    train = np.arange(1, 104, 2)
    batches = epoch_batches(train, 2, 0, 16)
    assert len(batches[-1]) == 4
    np.testing.assert_array_equal(np.sort(np.concatenate(batches)), train)
    np.testing.assert_array_equal(np.concatenate(batches), np.concatenate(epoch_batches(train, 2, 0, 16)))
    assert not np.array_equal(np.concatenate(batches), np.concatenate(epoch_batches(train, 2, 1, 16)))


def test_test_access_only_after_budget_never_smoke():
    for step, smoke in [(0, False), (19, False), (20, True), (21, False)]:
        with pytest.raises(RuntimeError):
            require_final_test(step, 20, smoke)
    require_final_test(20, 20)


def test_access_control_checks_source_indices():
    fold = OfficialFold.__new__(OfficialFold)
    fold.train_members = {0, 2}
    fold.test_members = {1}
    fold.dataset = 'synthetic'
    fold.task = 0
    fold.labels = np.array([0, 1, 1])
    fold.subjects = np.array(['a', 'b', 'c'])
    assert fold.rows([0, 2], 'train')[1]['source_index'] == 2
    with pytest.raises(ValueError):
        fold.rows([1], 'train')
    with pytest.raises(ValueError):
        fold.rows([0], 'test')


def test_question_text_does_not_reveal_true_label():
    task = dict(task_id='synthetic', question='Which class?', classes=[dict(class_id=str(i), option=f'Class {i}') for i in range(4)])
    for variant in (None, 0, 1):
        outputs = [make_variant(task, label, 'sample', 17, 0, variant) for label in range(4)]
        assert len({out['suffix'] for out in outputs}) == 1
        assert len({out['label'] for out in outputs}) == 4


def test_locked_plan_counts_and_budget():
    config = json.loads(Path('configs/benchmark_train.json').read_text())
    if not Path(config['split_root']).exists():
        pytest.skip('Local official manifests not installed')
    plan = build_plan(config)
    assert plan['planned_runs'] == 426
    assert set(plan['pending']) == {'SEED_VIG', 'ThingsEEG2', 'Nakanishi2015'}
    first = next(j for j in plan['jobs'] if j['dataset'] == 'BNCI2014001')
    assert first['train_n'] == 2304 and first['test_n'] == 288
    assert first['updates'] == 2880 and first['exposures'] == 46080
    assert not set(first['train_subjects']) & set(first['test_subjects'])
