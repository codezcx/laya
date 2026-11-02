import numpy as np
import json
import hashlib
import pytest

from scripts.plot_training_curves import padded_limits, plot_selected_losses, read_training_records, trailing_mean, validation_series


def test_trailing_mean_requires_full_window_and_preserves_gaps():
    result = trailing_mean([1, 2, 3, np.nan, 5, 6, 7], 3)
    np.testing.assert_allclose(result, [np.nan, np.nan, 2, np.nan, np.nan, np.nan, 6], equal_nan=True)


def test_axis_range_keeps_all_observations_visible():
    lo, hi = padded_limits([30, 40, 50], lower=0, upper=100)
    assert 0 <= lo < 30 and 50 < hi <= 100
    lo, hi = padded_limits([0, 100], lower=0, upper=100)
    assert (lo, hi) == (0, 100)


def test_validation_keeps_step_zero_and_missing_evaluations_as_gaps():
    report = dict(step=0, macro_balanced_accuracy=.5,
                  datasets={'a': dict(accuracy=.2, balanced_accuracy=.3),
                            'b': dict(accuracy=.6, balanced_accuracy=.7)})
    acc, balanced = validation_series([report], [0, 340])
    np.testing.assert_allclose(acc, [40, np.nan], equal_nan=True)
    np.testing.assert_allclose(balanced, [50, np.nan], equal_nan=True)


def test_selected_losses_preserve_sources_and_filter_only_the_figure(tmp_path):
    folder = tmp_path / 'train'
    folder.mkdir()
    (folder / 'config.json').write_text(json.dumps({'config': {
        'run_name': 'fixture', 'steps': 20, 'datasets': ['bcic_2a', 'tuab']}}))
    rows = [dict(status='training', step=i, dataset='bcic_2a' if i % 2 else 'tuab', loss=1 / i)
            for i in range(1, 21)]
    log = folder / 'metrics.jsonl'
    log.write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    original = log.read_bytes()
    out = tmp_path / 'figures'
    report = plot_selected_losses(folder, out, ['tuab'])
    assert report['displayed_datasets'] == ['tuab']
    assert report['omitted_datasets'] == ['bcic_2a']
    assert report['plotted_updates'] == 10
    assert report['total_training_updates'] == 20
    assert log.read_bytes() == original
    assert 'bcic_2a' not in (out / 'training_loss_selected.csv').read_text()
    assert (out / 'training_loss_selected.png').exists()
    with pytest.raises(FileExistsError):
        plot_selected_losses(folder, out, ['tuab'])
    with pytest.raises(ValueError, match='not configured'):
        plot_selected_losses(folder, out, ['unknown'])
    with pytest.raises(ValueError, match='distinct'):
        plot_selected_losses(folder, out, ['tuab', 'tuab'])
    log.write_text('\n'.join(json.dumps(r) for r in rows[1:]) + '\n')
    with pytest.raises(ValueError, match='contiguous'):
        plot_selected_losses(folder, out, ['tuab'])


def test_continuation_uses_recorded_global_steps_and_separate_planned_endpoint(tmp_path):
    identity = {'config': {'run_name': 'parent', 'steps': 20, 'datasets': ['tuab']},
                'continuation': {'start_step': 20, 'total_steps': 40,
                                 'plan': {'run_name': 'continuation'}}}
    (tmp_path / 'config.json').write_text(json.dumps(identity))
    rows = [dict(status='training', step=i, dataset='tuab', loss=1 / i) for i in range(21, 31)]
    (tmp_path / 'metrics.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    config, train, _ = read_training_records(tmp_path)
    assert config['steps'] == 40 and config['start_step'] == 20
    assert config['run_name'] == 'continuation'
    assert [r['step'] for r in train] == list(range(21, 31))
    report = plot_selected_losses(tmp_path, tmp_path / 'figures', ['tuab'])
    assert report['recorded_step_range'] == [21, 30]
    assert report['planned_endpoint'] == 40
    assert report['total_training_updates'] == 10
    assert report['continuation_start_step'] == 20


def test_joined_loss_history_is_verified_and_has_no_visible_boundary(tmp_path):
    parent, child = tmp_path / 'parent', tmp_path / 'child'
    parent.mkdir()
    child.mkdir()
    base = {'run_name': 'parent', 'steps': 20, 'datasets': ['tuab']}
    (parent / 'config.json').write_text(json.dumps({'config': base}))
    identity = {'config': base, 'continuation': {'start_step': 20, 'total_steps': 40,
        'plan': {'run_name': 'continuation'},
        'parent_config_sha256': hashlib.sha256((parent / 'config.json').read_bytes()).hexdigest()}}
    (child / 'config.json').write_text(json.dumps(identity))
    for folder, first, last in [(parent, 1, 20), (child, 21, 30)]:
        rows = [dict(status='training', step=i, dataset='tuab', loss=1 / i) for i in range(first, last + 1)]
        (folder / 'metrics.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    output = tmp_path / 'joined'
    report = plot_selected_losses(child, output, ['tuab'], parent_folder=parent)
    assert report['recorded_step_range'] == [1, 30]
    assert report['total_training_updates'] == 30
    assert report['continuation_start_step'] == 20
    assert len(report['source_segments']) == 2
    assert report['source_boundary_shown'] is False
    svg = (output / 'training_loss_selected.svg').read_text()
    assert 'Continuation' not in svg and 'continuation' not in svg
    assert 'Recorded steps 1–30' in svg
    (parent / 'config.json').write_text(json.dumps({'config': dict(base, run_name='wrong_parent')}))
    with pytest.raises(ValueError, match='provenance'):
        plot_selected_losses(child, output, ['tuab'], parent_folder=parent)
