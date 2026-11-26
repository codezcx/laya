import hashlib
import json

import h5py
import numpy as np
import pytest
import torch

from jev_train.data import TrialStore, PreparedAdapter
from jev_train.runtime import CONFIG
from jev_train.task_options import make_variant
from jev_train.lr_schedule import rates_at
from jev_train.distributed import rank_rows
from jev_train.evaluation_protocol import require_split, EvaluationLedger, selection_rows


def tiny_store(tmp_path, overlap=False):
    path = tmp_path / 'data.h5'
    x = np.arange(12 * 65 * 800, dtype=np.float32).reshape(12, 65, 800) / 10000
    with h5py.File(path, 'w') as f:
        f['all/X'] = x
        f['all/Y'] = np.arange(12) % 2
    splits = {'train': list(range(8)), 'valid': [8, 9], 'test': [10, 11]}
    if overlap:
        splits['test'][0] = 0
    m = {'datasets': {'demo': {'splits': {s: [dict(path=str(path), group='all',
        subject='s1', indices=ix)] for s, ix in splits.items()},
        'counts': {s: {'windows': len(ix)} for s, ix in splits.items()}}}}
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(m))
    cfg = {'manifest': str(manifest), 'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
           'datasets': ['demo']}
    tasks = [dict(task_id='demo', instruction='Classify', question='Which?',
                  classes=[dict(class_id='0', option='Left'), dict(class_id='1', option='Right')])]
    return TrialStore(cfg, tasks), x


def test_prepared_reader_no_second_transform(tmp_path):
    store, x = tiny_store(tmp_path)
    rows = store.rows[0][:4]
    adapted = PreparedAdapter(store).batch(rows)
    np.testing.assert_array_equal(adapted[0][2][0], x[:4])
    assert {r['split'] for r in store.sample(1, 16, 42)[1]} == {'train'}


def test_exact_membership_and_no_overlap(tmp_path):
    with pytest.raises(ValueError, match='duplicated'):
        tiny_store(tmp_path, overlap=True)


def test_manifest_mutation_rejected(tmp_path):
    store, _ = tiny_store(tmp_path)
    cfg = dict(store.config, manifest_sha256='wrong')
    with pytest.raises(ValueError, match='identity'):
        TrialStore(cfg, store.tasks)


def test_original_test_is_development_valid_deferred(tmp_path):
    store, _ = tiny_store(tmp_path)
    rows = selection_rows(store, 0, 512)
    assert [r['index'] for r in rows] == [10, 11]
    assert all(r['split'] == 'test' for r in rows)
    with pytest.raises(RuntimeError):
        require_split('valid')
    with pytest.raises(RuntimeError):
        require_split('test', training=True)
    ledger = EvaluationLedger()
    ledger.record('test', 2)
    assert ledger.state_dict()['valid_evaluations'] == 0


def test_four_rank_partition_and_option_relabelling(tmp_path):
    store, _ = tiny_store(tmp_path)
    _, rows = store.sample(1, 16, 42)
    assert sum([rank_rows(rows, r, 4) for r in range(4)], []) == rows
    task = store.tasks[0]
    for label in (0, 1):
        q = make_variant(task, label, 'same-sample', 123, 42)
        assert q['option_class_ids'][q['label']] == str(label)
    q0 = make_variant(task, 0, 'same-sample', 123, 42)
    q1 = make_variant(task, 1, 'same-sample', 123, 42)
    assert q0['suffix'] == q1['suffix']


def test_historical_hyperparameters_unchanged():
    assert CONFIG['steps'] == 2040 and CONFIG['bridge_only_steps'] == 340
    assert CONFIG['global_batch'] == 16 and CONFIG['microbatch'] == 4
    assert CONFIG['eval_microbatch'] == 8 and CONFIG['validation_every'] == 340
    assert CONFIG['learning_rates'] == dict(connector=1e-4, jev_lora=5e-6,
                                           decision_head=1e-5, eeg_encoder=1e-6)
    r = rates_at(340, 2040, 340, CONFIG['learning_rates'])
    assert r['connector'] > 0 and r['eeg_encoder'] == r['jev_lora'] == r['decision_head'] == 0
    assert rates_at(341, 2040, 340, CONFIG['learning_rates'])['eeg_encoder'] > 0


def test_medjev_architecture_is_explicit_and_no_meta_learning():
    assert CONFIG['retained_layers'] == 23
    assert CONFIG['lora_rank'] == 16 and CONFIG['lora_alpha'] == 32
    assert CONFIG['head_hidden_width'] == 256
    assert 'No medical checkpoint and no meta-learning' in CONFIG['transfer_policy']
