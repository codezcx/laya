import copy

import numpy as np
import pandas as pd
import pytest

from eeg_align.signal_quality import invalid_epochs, excluded_training_indices
from scripts.audit_multitask_signals import scan_array


def test_qc_detects_degenerate_epochs_but_keeps_individual_missing_channels():
    x = np.ones((6, 3, 100), dtype=np.float32)
    x[0] = 0
    x[1, 1] = 2  # Different DC offsets still provide no temporal EEG signal.
    x[2, 0, 0] = np.nan
    x[3, 1, 2] = np.inf
    x[4] = 0
    x[4, 0] = np.linspace(-1, 1, 100)
    x[5] = np.random.default_rng(42).normal(size=(3, 100)) * 1e-9
    assert list(invalid_epochs(x)) == ['all_zero', 'flatline', 'nonfinite', 'nonfinite', '', '']


def example_frame():
    return pd.DataFrame([dict(array='epochs.npy', row=i, subject='01', split=split,
                              label=i % 2, record_id=f'record:{i}')
                         for i, split in enumerate(['train', 'train', 'valid', 'test'])])


def test_qc_preserves_original_indices_and_metadata():
    frame = example_frame()
    original = frame.copy(deep=True)
    item = dict(frame.iloc[0], index=0, reason='all_zero')
    assert excluded_training_indices(frame, [item]) == {0}
    pd.testing.assert_frame_equal(frame, original)
    assert frame.iloc[1]['record_id'] == 'record:1'


@pytest.mark.parametrize('index', [2, 3])
def test_heldout_exclusion_is_forbidden(index):
    frame = example_frame()
    with pytest.raises(ValueError, match='validation/test'):
        excluded_training_indices(frame, [dict(frame.iloc[index], index=index, reason='all_zero')])


@pytest.mark.parametrize('key,value', [('row', 1), ('subject', '02'), ('label', 1), ('reason', 'low_amplitude')])
def test_changed_qc_identity_or_rule_is_rejected(key, value):
    frame = example_frame()
    item = dict(frame.iloc[0], index=0, reason='all_zero')
    item[key] = value
    with pytest.raises(ValueError):
        excluded_training_indices(frame, [item])


def test_duplicate_exclusion_is_rejected():
    frame = example_frame()
    item = dict(frame.iloc[0], index=0, reason='all_zero')
    with pytest.raises(ValueError, match='duplicate'):
        excluded_training_indices(frame, [item, copy.deepcopy(item)])


def test_scan_preserves_signal_file_and_records_hash(tmp_path):
    x = np.random.default_rng(42).normal(size=(5, 3, 100)).astype('float32')
    x[2] = 0
    path = tmp_path / 'test.npy'
    np.save(path, x)
    before = path.read_bytes()
    result = scan_array(path)
    assert result['invalid'] == {2: 'all_zero'}
    assert len(result['sha256']) == 64
    assert path.read_bytes() == before
