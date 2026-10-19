import pickle
from pathlib import Path
import numpy as np
import pytest
from eeg_align.mi_data import read_trial,iter_paths,subject_partitions,MITrials


def example(tmp_path,dataset='BNCI2014_001',label='left_hand',raw_name='A01T.mat'):
    path=tmp_path/f'{dataset}_trial-0_label-{label}.pkl'
    r=dict(schema_version='eeg-epoch-v3',paradigm='imagery',dataset=dataset,label=label,freq=200,unit='uV',
           eeg={c:np.arange(800,dtype=np.float32)*(i+1) for i,c in enumerate(['C3','CZ','C4'])},n_samples=800,
           raw_name=raw_name,subject=1,session='0train',run='0',trial=0)
    path.write_bytes(pickle.dumps(r));return path


def test_2a_filters_other_classes_and_maps_hands(tmp_path):
    assert read_trial(example(tmp_path))[1]==0
    assert read_trial(example(tmp_path,label='right_hand'))[1]==1
    for label in ['feet','tongue']:
        with pytest.raises(ValueError,match='Not left/right'):read_trial(example(tmp_path,label=label))


def test_channel_order_units_and_crop(tmp_path):
    p=example(tmp_path);x,y,m=read_trial(p,channels=['C4','C3'],crop_seconds=(.5,2.5))
    assert x.shape==(2,400) and x.dtype==np.float32
    np.testing.assert_array_equal(x[0],3*x[1]);assert x[1,0]==100
    assert m['unit']=='uV' and not m['baseline_available']
    with pytest.raises(ValueError,match='Crop outside'):read_trial(p,crop_seconds=(-1,2))
    with pytest.raises(ValueError,match='Missing channels'):read_trial(p,channels=['FC3'])


def test_physionet_execution_rejected(tmp_path):
    for run in [4,8,12]:assert read_trial(example(tmp_path,dataset='PhysionetMI',raw_name=f'S001R{run:02}.edf'))[1]==0
    with pytest.raises(ValueError,match='imagined'):read_trial(example(tmp_path,dataset='PhysionetMI',raw_name='S001R03.edf'))


def test_heldout_and_alias_protection():
    for name in ['BNCI2014_004','Dreyer2023A','Schirrmeister2017','GuttmannFlury2025_ME']:
        with pytest.raises(ValueError):list(iter_paths(name,role='source'))


def test_subject_split_stable_no_cross_session_leakage():
    a=subject_partitions([1,1,2,3,4,5], 'example')
    assert a==subject_partitions([5,4,3,2,1], 'example')
    assert set(a.values())=={'train','validation'}


def test_conflicting_dreyer_runs_quarantined(tmp_path):
    p=example(tmp_path,dataset='Dreyer2023')
    r=pickle.loads(p.read_bytes());r.update(subject=13,session='0',run='5R6online');p.write_bytes(pickle.dumps(r))
    with pytest.raises(ValueError,match='quarantined'):read_trial(p)


def test_loader_rejects_incomplete_or_changed_index(tmp_path):
    import hashlib
    import json
    from eeg_align.mi_data import CONFIG
    p=tmp_path/'train.jsonl';p.write_text('{}\n')
    with pytest.raises(FileNotFoundError):MITrials(p)
    done=tmp_path/'complete.json'
    done.write_text(json.dumps(dict(status='passed',config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),manifest_sha256={'train.jsonl':'wrong'})))
    with pytest.raises(ValueError,match='hash mismatch'):MITrials(p)
