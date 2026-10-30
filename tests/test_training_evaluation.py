from types import SimpleNamespace

import numpy as np
import pytest

from scripts.evaluate_training_split import training_rows, checked_predictions, save_predictions


def test_training_selection_excludes_heldout_and_qc_rows():
    rows = [dict(id=str(i), split=split) for i, split in enumerate(['train', 'valid', 'train', 'test'])]
    store = SimpleNamespace(tasks=[dict(task_id='sample', role='benchmark')], rows={0: rows},
        excluded={0: {2}}, quality={'sample': {'effective_counts': {'train': 1}}})
    assert training_rows(store, 0) == [rows[0]]
    store.quality['sample']['effective_counts']['train'] = 2
    with pytest.raises(ValueError):
        training_rows(store, 0)


def test_train_only_extras_cannot_be_reported_as_benchmark_training_eval():
    store = SimpleNamespace(tasks=[dict(task_id='extra', role='train_only')],
        rows={0: [dict(id='x', split='train')]}, excluded={0: set()},
        quality={'extra': {'effective_counts': {'train': 1}}})
    with pytest.raises(ValueError):
        training_rows(store, 0)


def test_chunks_preserve_ids_labels_and_order(tmp_path):
    rows = [dict(id=f'task:{i}', label=i % 2) for i in range(5)]
    indices = [0, 2, 4]
    probabilities = np.array([[.7, .3], [.6, .4], [.8, .2]], dtype=np.float32)
    path = tmp_path / 'chunk.npz'
    save_predictions(path, indices, rows, probabilities)
    np.testing.assert_array_equal(checked_predictions(path, indices, rows, 2), probabilities)
    with pytest.raises(ValueError):
        checked_predictions(path, [2, 0, 4], rows, 2)
    changed = [dict(r) for r in rows]
    changed[0]['label'] = 1
    with pytest.raises(ValueError):
        checked_predictions(path, indices, changed, 2)


@pytest.mark.parametrize('probabilities', [np.array([[np.nan, .5]]), np.array([[.2, .2]]), np.array([[-.1, 1.1]])])
def test_invalid_probabilities_are_rejected(tmp_path, probabilities):
    rows = [dict(id='task:0', label=0)]
    path = tmp_path / 'bad.npz'
    save_predictions(path, [0], rows, probabilities)
    with pytest.raises(ValueError):
        checked_predictions(path, [0], rows, 2)
