import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from eeg_align.split_policy import SEED_PROTOCOL, seed_split, validate_split_frame
from eeg_align.runtime import CONFIG, arm_settings, digest
from eeg_align.lr_schedule import rates_at
from eeg_align.online_leaf import OnlineLEAFTower
from scripts.prepare_multitask_splits import extract_methods


def test_seed_rule_exactly_matches_pinned_upstream():
    cls = extract_methods('data/dataset/seeds/seed.py', 'SeedBuilder', {'_divide_by_uniform_label'})
    official = cls._divide_by_uniform_label(pd.DataFrame([{'label': json.dumps(list(range(15)))}]))
    expected = {i:r['split'] for r in official.to_dict('records') for i in json.loads(r['label'])}
    assert list(seed_split(np.arange(15))) == [expected[i] for i in range(15)]
    for bad in [[-1], [15], [.5], [np.nan]]:
        with pytest.raises(ValueError): seed_split(bad)


def test_trial_split_exception_is_narrow_and_fail_closed():
    frame = pd.DataFrame(dict(subject=['a']*15, record_id=['a_1']*15, event=range(15), split=seed_split(np.arange(15))))
    manifest = {'split_protocol': SEED_PROTOCOL}
    validate_split_frame('seed', manifest, frame)
    with pytest.raises(ValueError): validate_split_frame('tuab', manifest, frame)
    with pytest.raises(ValueError): validate_split_frame('seed', {}, frame)
    wrong = frame.copy(); wrong.loc[0, 'split'] = 'test'
    with pytest.raises(ValueError): validate_split_frame('seed', manifest, wrong)
    with pytest.raises(ValueError): validate_split_frame('seed', manifest, frame.iloc[:-1])


def test_current_seed_reindex_did_not_change_signals_or_other_datasets():
    if 'seedpaper' not in CONFIG['split_root']:
        pytest.skip('Current run is not the SEED-paper adaptation control')
    previous = Path('artifacts/benchmark/multitask_splits_hmc151_20261031')
    current = Path(CONFIG['split_root'])
    for folder in previous.iterdir():
        if folder.is_dir() and folder.name != 'seed':
            for name in ['manifest.json', 'records.csv']:
                assert digest(folder/name) == digest(current/folder.name/name)
    old = pd.read_csv('datasets/multitask_hmc151_20261031/seed/index.csv')
    new = pd.read_csv(Path(CONFIG['cache_root'])/'seed/index.csv')
    pd.testing.assert_frame_equal(old.drop(columns='split'), new.drop(columns='split'))
    assert new.groupby('split').size().to_dict() == {'train':22455, 'valid':7875, 'test':7560}
    assert new.groupby('split')['subject'].nunique().eq(15).all()


def test_adaptation_configuration_and_schedule():
    assert CONFIG['arms'] == ['leaf_tower_mlp', 'leaf_infonce_mlp', 'leaf_sigmoid_mlp']
    setting = arm_settings(CONFIG['arms'][0])
    assert setting == dict(encoder='leaf_tower', encoder_trainable=True, readout='identity')
    assert CONFIG['world_size'] == 4 and CONFIG['global_batch'] == 16
    assert CONFIG['microbatch'] == 4
    peaks = CONFIG['learning_rates']
    assert peaks == dict(connector=1e-4, jev_lora=5e-6, decision_head=1e-5, eeg_encoder=1e-6)
    assert rates_at(340,2040,340,peaks)['eeg_encoder'] == 0
    assert rates_at(341,2040,340,peaks)['eeg_encoder'] > 0


def test_tower_phase_controls_gradients_not_batchnorm_statistics():
    model = OnlineLEAFTower.__new__(OnlineLEAFTower)
    torch.nn.Module.__init__(model)
    model.tower = torch.nn.Sequential(torch.nn.BatchNorm1d(2), torch.nn.Linear(2,2))
    model.set_adaptation_enabled(False)
    model.train()
    assert not model.tower.training
    assert not model(torch.ones(3,2)).requires_grad
    model.set_adaptation_enabled(True)
    model.train()
    model(torch.ones(3,2)).sum().backward()
    assert model.tower[1].weight.grad is not None
    assert model.tower[0].weight.grad is not None
    assert model.tower[0].num_batches_tracked.item() == 0
