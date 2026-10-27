import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_official', ROOT / 'scripts/prepare_official_splits.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_mask_summary_preserves_source_indices_and_usage():
    data = SimpleNamespace(subject_ids=np.array(['01','01','02','02']),
        labels=np.array([0,1,0,1]), usage=np.array(['train','test','train','test']))
    ids, summary = audit.mask_summary(data, np.array([False, True, False, False]))
    assert ids.tolist() == [1]
    assert summary == dict(n=1, subjects=['01'], labels={'1':1}, usage={'test':1})


def test_original_splitter_honors_usage_in_cross_mode():
    path = ROOT / 'third_party/EEG-FM-Benchmark-official/utils/dataset_split.py'
    if not path.exists():
        pytest.skip('Pinned third-party repository not installed')
    splitter = audit.import_file('test_original_splitter', path)
    data = SimpleNamespace(subject_ids=np.array([1,1,2,2,3,3]),
        usage=np.array(['train','test']*3))
    splits = splitter.split_dataset_cross(data, 'LOSO')
    assert len(splits) == 3
    assert np.flatnonzero(splits[0]['train_mask']).tolist() == [2,4]
    assert np.flatnonzero(splits[0]['test_mask']).tolist() == [1]
    assert set(data.subject_ids[splits[0]['train_mask']]) == {2,3}


def test_original_loso_uses_all_other_subjects_without_validation_holdout():
    path = ROOT / 'third_party/EEG-FM-Benchmark-official/utils/dataset_split.py'
    if not path.exists():
        pytest.skip('Pinned third-party repository not installed')
    splitter = audit.import_file('test_original_splitter_plain', path)
    data = SimpleNamespace(subject_ids=np.repeat(np.arange(1,10),2), usage=None)
    splits = splitter.split_dataset_cross(data, 'LOSO')
    assert len(splits) == 9
    for i, split in enumerate(splits):
        assert split['test_subject'] == i+1
        assert split['train_mask'].sum() == 16
        assert split['test_mask'].sum() == 2
        assert not np.any(split['train_mask'] & split['test_mask'])
