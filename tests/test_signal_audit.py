import numpy as np
from scripts.audit_benchmark import frequency_scores


def test_frequency_control_recovers_all_candidates_despite_phase_gain_offset():
    f=np.arange(9.25,15,.5);t=np.arange(831)/200
    phase=np.arange(8)[None,:,None]*.7
    x=np.sin(2*np.pi*f[:,None,None]*t[None,None,:]+phase)
    expected=np.arange(len(f))
    np.testing.assert_array_equal(frequency_scores(x,200,f).argmax(-1),expected)
    altered=x*np.arange(1,9)[None,:,None]+np.arange(8)[None,:,None]*100
    np.testing.assert_array_equal(frequency_scores(altered,200,f).argmax(-1),expected)


def test_leaf_2a_keeps_labeled_runs_after_one_calibration_run():
    from eeg_align.leaf_preprocessing import raw_2a_epochs
    runs = np.empty((1,4), dtype=object)
    original = []
    for i,label in enumerate([None,1,4,2]):
        record = np.empty((1,1), dtype=[('X',object),('trial',object),('y',object)])
        x = np.arange(1700, dtype=float)[:,None] + np.arange(25)[None,:]**2 + i
        record[0,0] = (x, np.array([] if label is None else [7]).reshape(-1,1),
                       np.array([] if label is None else [label]).reshape(-1,1))
        runs[0,i] = record
        if label is not None:
            v=x[7:1507,:22].T.copy();original.append(v-v.mean(0,keepdims=True))
    x,y,selected=raw_2a_epochs({'data':runs})
    np.testing.assert_array_equal(y,[0,3,1])
    np.testing.assert_array_equal(x,np.stack(original))
    np.testing.assert_allclose(x.mean(1),0,atol=1e-12)
    assert [r['run_index'] for r in selected]==[1,2,3]


def test_leaf_2a_rejects_incomplete_raw_epoch():
    import pytest
    from eeg_align.leaf_preprocessing import raw_2a_epochs
    run=np.empty((1,1),dtype=[('X',object),('trial',object),('y',object)])
    run[0,0]=(np.zeros((1000,25)),np.array([[0]]),np.array([[1]]))
    data=np.empty((1,1),dtype=object);data[0,0]=run
    with pytest.raises(AssertionError):raw_2a_epochs({'data':data})
