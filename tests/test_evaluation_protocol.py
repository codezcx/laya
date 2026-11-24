"""Synthetic split-role tests; no model, EEG data, or network access."""
import copy
import json

import pytest

from eeg_align.evaluation_protocol import (POLICY, effective_config, require_split,
    selection_rows, EvaluationLedger, verify_evaluation_contract)
from eeg_align.evaluation import validation_rows
from eeg_align.runtime import CONFIG
from scripts.plot_training_curves import read_records
from scripts.run_benchmark import training_command


class Pool:
    def __init__(self, split='test', n=900):
        self.split = split
        self.calls = []
        self.rows = [dict(id=f'row-{i}', label=i % 3, subject=str(i % 7), split=split) for i in range(n)]

    def evaluation(self, tid, split, cap=None):
        self.calls.append(split)
        assert split == self.split
        return self.rows if cap is None else self.rows[:cap]


def test_original_test_sampling_preserves_algorithm_ids_and_split_labels():
    test, legacy = Pool(), Pool('valid')
    selected = selection_rows(test, 0, 512)
    assert [r['id'] for r in selected] == [r['id'] for r in validation_rows(legacy, 0, 512)]
    assert selected == selection_rows(test, 0, 512)
    assert {r['split'] for r in selected} == {'test'}
    assert len(selection_rows(test, 0, None)) == 900
    assert set(test.calls) == {'test'}
    assert selection_rows(Pool(n=20), 0, 512) == Pool(n=20).rows
    bad = Pool()
    bad.rows[0]['split'] = 'valid'
    with pytest.raises(RuntimeError, match='incorrect split'):
        selection_rows(bad, 0, 512)


def test_no_valid_forward_or_test_gradient_and_no_recipe_mutation():
    before = copy.deepcopy(CONFIG)
    for split in ('valid', 'unknown'):
        with pytest.raises(RuntimeError, match='cannot be forwarded'):
            require_split(split)
    require_split('train', training=True)
    require_split('test')
    with pytest.raises(RuntimeError, match='Only train'):
        require_split('test', training=True)
    current = effective_config(CONFIG)
    assert current['selection_split'] == 'test'
    changed = {key for key in current if current[key] != before.get(key)}
    assert changed == {'selection', 'test_policy', 'selection_split', 'evaluation_policy'}
    assert CONFIG == before
    cmd = training_command('python', [0, 1, 2, 3], 'leaf_infonce_mlp', 'out', backbone='qwen3vl8b')
    assert cmd[cmd.index('--backbone')+1] == 'qwen3vl8b'


def test_evaluation_ledger_survives_resume_and_requires_honest_completion():
    ledger = EvaluationLedger()
    ledger.record('train', 16)
    ledger.record('test', 512)
    resumed = EvaluationLedger(json.loads(json.dumps(ledger.state_dict())))
    resumed.record('test', 1000)
    assert resumed.counts['test_evaluation_samples'] == 1512
    assert resumed.counts['test_evaluations'] == 2
    with pytest.raises(RuntimeError):
        resumed.record('valid', 1)
    counts = resumed.state_dict()
    complete = dict(smoke=False, evaluation_counts=counts, evaluation_policy=POLICY,
                    test_evaluations=2, valid_evaluations=0,
                    development_evaluation=dict(split='test', role='development', scope='full',
                                                independent_test_estimate=False))
    verify_evaluation_contract(complete)
    for key, value in [('split', 'valid'), ('role', 'independent_test'),
                       ('scope', 'subset'), ('independent_test_estimate', True)]:
        bad = copy.deepcopy(complete)
        bad['development_evaluation'][key] = value
        with pytest.raises(RuntimeError):
            verify_evaluation_contract(bad)
    with pytest.raises(RuntimeError):
        verify_evaluation_contract(dict(complete, smoke=True))
    bad_counts = dict(counts, valid_evaluations=1)
    with pytest.raises(ValueError):
        EvaluationLedger(bad_counts)
    clean = EvaluationLedger()
    clean.record('train', 16)
    verify_evaluation_contract(dict(complete, smoke=True, evaluation_counts=clean.state_dict(),
                                   test_evaluations=0, development_evaluation=None))


def test_plot_reader_uses_original_test_not_valid_and_checks_same_ids(tmp_path):
    identity = dict(config=effective_config(CONFIG), language_backbone='qwen3vl8b',
                    language_spec={'run_name': 'fixture'})
    (tmp_path/'config.json').write_text(json.dumps(identity))
    (tmp_path/'metrics.jsonl').write_text(json.dumps(dict(status='training', step=1, loss=1.))+'\n')
    report = dict(step=0, split='test', role='development', scope='subset',
                  independent_test_estimate=False, macro_balanced_accuracy=.5,
                  datasets={'fixture': dict(n=512, accuracy=.5, balanced_accuracy=.5, sample_ids_sha256='same')})
    (tmp_path/'test_step000000_subset.json').write_text(json.dumps(report))
    # Invalid legacy artifact must never be opened by the current-policy reader.
    (tmp_path/'valid_step000000_subset.json').write_text('not JSON')
    config, _, reports, paths = read_records(tmp_path)
    assert config['language_backbone'] == 'qwen3vl8b'
    assert len(reports) == 1 and not any(p.name.startswith('valid_') for p in paths)
    second = copy.deepcopy(report)
    second['step'] = 340
    second['datasets']['fixture']['sample_ids_sha256'] = 'changed'
    (tmp_path/'test_step000340_subset.json').write_text(json.dumps(second))
    with pytest.raises(ValueError, match='IDs changed'):
        read_records(tmp_path)


def test_development_plot_outputs_are_not_named_validation(tmp_path, monkeypatch):
    import sys
    from scripts.plot_training_curves import main
    folder = tmp_path/'train'/'fixture'
    folder.mkdir(parents=True)
    identity = dict(config=effective_config(CONFIG), language_backbone='qwen3vl8b')
    (folder/'config.json').write_text(json.dumps(identity))
    training = [dict(status='training', step=i+1, dataset=name, loss=1., accuracy=.5)
                for i, name in enumerate(CONFIG['datasets'])]
    (folder/'metrics.jsonl').write_text('\n'.join(map(json.dumps, training))+'\n')
    for step in (0, 2040):
        report = dict(step=step, split='test', role='development', scope='subset',
            independent_test_estimate=False, macro_balanced_accuracy=.5,
            datasets={name: dict(n=512, accuracy=.5, balanced_accuracy=.5, sample_ids_sha256=name)
                      for name in CONFIG['datasets'][:9]})
        (folder/f'test_step{step:06d}_subset.json').write_text(json.dumps(report))
    monkeypatch.setattr(sys, 'argv', ['plot_training_curves.py', '--run-dir', str(tmp_path), '--arm', 'fixture'])
    main()
    output = tmp_path/'figures'/'fixture'
    manifest = json.loads((output/'figure_manifest.json').read_text())
    assert manifest['evaluation_split'] == 'test' and manifest['evaluation_role'] == 'development'
    assert manifest['independent_test_estimate'] is False
    assert not any('validation' in p.name for p in output.iterdir())
    assert (output/'development_test.csv').exists()
    svg = (output/'development_test_by_dataset.svg').read_text()
    assert 'Development (original test)' in svg
