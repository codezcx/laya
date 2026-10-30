from collections import Counter
from types import SimpleNamespace
import numpy as np
import pytest

from eeg_align.multitask_data import TrialStore, window_positions, locked_records
from eeg_align.benchmark_data import catalog
from eeg_align.evaluation import require_split, fixed_training_rows, validation_rows
from scripts.prepare_multitask_splits import extract_methods


def test_frozen_splits_and_train_only_extras():
    tasks = catalog()
    assert len(tasks) == 17
    assert sum(t['role'] == 'benchmark' for t in tasks) == 9
    assert not any('2014004' in t['task_id'] for t in tasks)
    for task in tasks:
        manifest, frame = locked_records(task['task_id'])
        assert frame.groupby('subject')['split'].nunique().max() == 1
        assert manifest['role'] == task['role']
        if task['role'] == 'train_only':
            assert set(frame['split']) == {'train'}


@pytest.mark.parametrize('drop_last', [True, False])
def test_window_positions_match_pinned_upstream(drop_last):
    cls = extract_methods('data/processor/builder.py', 'EEGDatasetBuilder', {'_generate_window_sample', '_milli_sec_to_pts'})
    obj = cls()
    obj.config = SimpleNamespace(fs=200, wnd_len=1000, is_finetune=True,
        category=['a', 'b'], category_query_dict={'a': 0, 'b': 1}, dataset_name='test', task_type=SimpleNamespace(value='test'))
    raw = np.arange(6000, dtype=np.float32)[None]
    events = [('a', 0, -1), ('b', 3100, 4100), ('a', 0, 1000), ('b', 99999, 100000), ('unknown', 0, 5000)]
    expected = obj._generate_window_sample(raw, 'test', np.array([0]), events, drop_last)
    actual, skipped = window_positions(raw.shape[-1], events, 5, ['a', 'b'], drop_last)
    assert [(x['label'], int(x['data'][0]), len(x['data'])) for x in expected] == [(x['label'], x['start'], x['stop'] - x['start']) for x in actual]
    assert len(skipped) == 2


def test_cycle_balances_datasets_without_eval_sampling():
    store = TrialStore.__new__(TrialStore)
    store.tasks = [{'task_id': str(i)} for i in range(17)]
    store.rows = {t: [dict(task=t, label=c, subject='a', split='train', id=f'{t}:{c}') for c in range(2)] for t in range(17)}
    store.pools = {t: {c: {'a': [c]} for c in range(2)} for t in range(17)}
    seen = []
    for step in range(1, 205):
        tid, rows = store.sample(step, 16, 42)
        seen.append(tid)
        assert set(r['split'] for r in rows) == {'train'}
        assert Counter(r['label'] for r in rows) == {0: 8, 1: 8}
        assert (tid, rows) == store.sample(step, 16, 42)
    assert set(Counter(seen).values()) == {12}


def test_test_access_gate():
    with pytest.raises(RuntimeError):
        require_split('test')
    require_split('valid')
    require_split('train')


def test_training_monitor_is_fixed_label_independent_and_train_only():
    rows=[dict(id=str(i),split='train' if i<30 else 'test',label=i%2) for i in range(40)]
    store=SimpleNamespace(rows={0:rows},excluded={0:{0,1}})
    selected=fixed_training_rows(store,0,12)
    assert len(selected)==12 and all(r['split']=='train' and r['id'] not in {'0','1'} for r in selected)
    ids=[r['id'] for r in selected]
    for row in rows:row['label']=9
    assert ids==[r['id'] for r in fixed_training_rows(store,0,12)]


def test_selection_validation_contains_legacy_subset_and_all_classes():
    store = TrialStore.__new__(TrialStore)
    store.rows = {0:[dict(id=str(i), label=i%6, subject=str(i%11), split='valid') for i in range(2000)]}
    selected = validation_rows(store,0,512)
    assert len(selected)==512
    assert set(r['id'] for r in store.evaluation(0,'valid',128)) <= set(r['id'] for r in selected)
    assert set(r['label'] for r in selected)==set(range(6))
    assert selected==validation_rows(store,0,512)


def test_hmc_sleepedf_label_order_is_not_conflated():
    tasks = {t['task_id']: t for t in catalog()}
    assert tasks['hmc']['classes'][1]['option'] == 'REM sleep'
    assert tasks['SleepEDF']['classes'][1]['option'] == 'N1 sleep'
    assert tasks['motor_mv_img']['classes'][2]['option'] == 'Both fists motor imagery'
