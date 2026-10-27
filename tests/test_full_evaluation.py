import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('full_eval', Path(__file__).resolve().parents[1] / 'scripts/evaluate_benchmark.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_dataset_shards_are_complete_disjoint_and_deterministic():
    rows = [dict(id=f'{tid}-{i}', task=tid) for tid, n in enumerate([1, 7, 8, 101]) for i in range(n)]
    shards = [evaluation.shard_rows(rows, rank, 2) for rank in range(2)]
    ids = [[r['id'] for rr in shard.values() for r in rr] for shard in shards]
    assert not set(ids[0]) & set(ids[1])
    assert set(ids[0] + ids[1]) == {r['id'] for r in rows}
    for tid in range(4):
        assert abs(len(shards[0][tid])-len(shards[1][tid])) <= 1


def test_class_prototypes_preserve_catalog_order():
    root = Path(__file__).resolve().parents[1]
    tasks = json.loads((root / 'configs/benchmark_tasks.json').read_text())['tasks']
    for task in tasks:
        assert len(evaluation.prototype_texts(task)) == len(task['classes'])
    assert evaluation.prototype_texts(tasks[0]) == ['Left', 'Right', 'Foot', 'Tongue']
    assert evaluation.prototype_texts(tasks[2]) == ['Right', 'Foot']
    assert evaluation.prototype_texts(tasks[3]) == ['Target stimulus', 'Non-target stimulus']


def test_json_rejects_nonfinite_values(tmp_path):
    with pytest.raises(ValueError):
        evaluation.write_json(tmp_path / 'result.json', {'accuracy': float('nan')})
    assert not (tmp_path / 'result.json').exists()


def test_json_atomic_publication(tmp_path):
    evaluation.write_json(tmp_path / 'result.json', {'n': 17})
    assert json.loads((tmp_path / 'result.json').read_text()) == {'n': 17}
    assert not (tmp_path / 'result.tmp').exists()
