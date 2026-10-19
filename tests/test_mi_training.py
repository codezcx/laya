import json
import pickle
from pathlib import Path
import numpy as np
import pytest
from eeg_align.mi_training_data import preprocess,epoch_batches,metrics

def trial(tmp_path,n=600):
    p=tmp_path/'sample_label-left_hand.pkl'
    x=np.stack([np.sin(np.arange(n)/10+i) for i in range(4)]).astype(np.float32)
    r=dict(schema_version='eeg-epoch-v3',paradigm='imagery',dataset='BNCI2014_001',label='left_hand',freq=200,unit='uV',eeg=dict(zip(['C3','CZ','C4','FZ'],x)),n_samples=n,subject=1,session='0train',run='0',trial=0)
    p.write_bytes(pickle.dumps(r));return dict(path=str(p),dataset='BNCI2014_001',label=0),r

def test_full_channels_and_tail_retained(tmp_path):
    row,r=trial(tmp_path);x,names,m=preprocess(row)
    assert x.shape==(4,740) and m['original_samples']==600 and m['token_length']==16
    expected=np.stack(list(r['eeg'].values()));expected=(expected-expected.mean(-1,keepdims=True))/100
    np.testing.assert_allclose(x[:,:600],expected,atol=1e-8)
    np.testing.assert_allclose(x[:,600:],np.repeat(x[:,599:600],140,axis=1))

def test_unknown_position_not_silently_dropped(tmp_path):
    row,r=trial(tmp_path);r['eeg']['UNKNOWN']=np.arange(600,dtype=np.float32);Path(row['path']).write_bytes(pickle.dumps(r))
    with pytest.raises(ValueError,match='Unknown electrode'):preprocess(row)

def test_buckets_cover_all_trials_once_and_reshuffle():
    rows=[dict(token_length=(i%7+1)*20) for i in range(49193)]
    a=epoch_batches(rows,42,0,64);b=epoch_batches(rows,42,1,64)
    assert len(a)==769 and min(map(len,a))>=4
    assert sorted(np.concatenate(a))==list(range(len(rows)))
    assert not np.array_equal(np.concatenate(a),np.concatenate(b))
    assert np.array_equal(np.concatenate(a),np.concatenate(epoch_batches(rows,42,0,64)))

def test_metrics_macro_not_dominated_by_larger_source():
    rows=[dict(dataset='a',subject='1',label=i%2,probability_right=float(i%2)) for i in range(100)]
    rows += [dict(dataset='b',subject='2',label=i,probability_right=float(1-i)) for i in range(2)]
    m=metrics(rows);assert m['macro_dataset_accuracy']==.5 and m['macro_dataset_balanced_accuracy']==.5
    assert m['accuracy']==100/102
