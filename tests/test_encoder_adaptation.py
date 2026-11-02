import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from eeg_align.split_policy import SEED_PROTOCOL, seed_split, validate_split_frame
from eeg_align.runtime import CONFIG, arm_settings, digest
from eeg_align.lr_schedule import rates_at
from eeg_align.online_bsr import OnlineBSR
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
    assert CONFIG['arms'] == ['infonce_qformer_lora64']
    setting = arm_settings(CONFIG['arms'][0])
    assert setting['encoder_trainable'] is True and setting['readout'] == 'qformer'
    assert setting['videomae_lora']['rank'] == setting['videomae_lora']['alpha'] == 64
    assert setting['videomae_lora']['dropout'] == 0
    peaks = CONFIG['learning_rates']
    assert peaks['eeg_encoder'] == 1e-6
    assert rates_at(340,2040,340,peaks)['eeg_encoder'] == 0
    assert rates_at(341,2040,340,peaks)['eeg_encoder'] > 0


def test_bsr_phase_controls_parameters_training_and_grad_mode():
    model = OnlineBSR.__new__(OnlineBSR)
    torch.nn.Module.__init__(model)
    model.finetune = model.adaptation_enabled = True
    model.renderer = torch.nn.Module()
    model.renderer.eeg2pix = torch.nn.Linear(2,2)
    model.renderer.pix2eeg = torch.nn.Linear(2,2).requires_grad_(False)
    model.adapter, model.encoder = torch.nn.Linear(2,2), torch.nn.Linear(2,2)
    model._encode = lambda x,ch: model.encoder(model.adapter(model.renderer.eeg2pix(x)))
    model.set_adaptation_enabled(False); model.train()
    assert not model.training
    assert not model(torch.ones(1,2), []).requires_grad
    model.set_adaptation_enabled(True); model.train()
    assert model.training
    model(torch.ones(1,2), []).sum().backward()
    assert model.encoder.weight.grad is not None
    assert model.renderer.eeg2pix.weight.grad is not None
    assert model.renderer.pix2eeg.weight.grad is None
    model.eval()
    assert not model.training and model.encoder.weight.requires_grad
