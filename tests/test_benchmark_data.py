import numpy as np
from eeg_align.benchmark_data import bipolar_matrix, to_electrodes, segment, subject_split, classification_metrics, catalog
from eeg_align.task_options import make_question


def test_bipolar_differences_preserved():
    names=['Fp1-F3','F3-C3','Fp2-F4','F4-C4']
    _, _, a = bipolar_matrix(names)
    potential=np.random.default_rng(0).normal(size=(2,6,120))
    observed=np.einsum('ce,bet->bct',a,potential)
    recovered, _, report=to_electrodes(observed,names)
    np.testing.assert_allclose(np.einsum('ce,bet->bct',a,recovered),observed,atol=1e-6)
    assert report['relative_residual'] < 1e-6


def test_full_epoch_preserved():
    x=np.arange(6010)[None,None]
    chunks=segment(x)
    np.testing.assert_array_equal(np.concatenate(chunks,axis=-1),x)
    assert max(v.shape[-1] for v in chunks)<=2000


def test_subject_isolation_and_options():
    subjects=['a','b','c','a','c']
    splits=subject_split(subjects)
    assert set(splits.values())=={'train','validation','test'}
    assert splits==subject_split(subjects)
    for task in catalog():
        for cls in task['classes']:
            q=make_question(task,cls['class_id'],'same-signal',0)
            assert q['option_class_ids'][q['label']]==cls['class_id']
    tasks={t['task_id']:t for t in catalog()}
    assert tasks['BNCI2015001']['classes'][0]['option']=='Right hand motor imagery'
    assert tasks['BNCI2014008']['classes'][0]['option']=='Target stimulus'


def test_unbalanced_metrics():
    m=classification_metrics([0,0,0,1],[[1,0]]*4,2)
    assert m['accuracy']==.75 and m['balanced_accuracy']==.5


def test_missing_bipolar_measurement_is_not_zero_constraint():
    channels=['Fp1-F3','F3-C3','Fp1-C3']
    observed=np.zeros((1,3,100),dtype=np.float32)
    observed[:,0]=2;observed[:,1]=3
    x,names,report=to_electrodes(observed,channels)
    assert report['missing_lead_instances']==1
    np.testing.assert_allclose(x[:,names.index('Fp1')]-x[:,names.index('C3')],5,atol=1e-6)


def test_sampler_balances_tasks():
    from eeg_align.benchmark_data import BalancedSampler
    rows=[dict(task=t,label=c,subject='one') for t in range(3) for c in range(t+2)]
    sampler=BalancedSampler(rows);sample=sampler(rows,1,batch=12)
    assert [sum(rows[i]['task']==t for i in sample) for t in range(3)]==[4,4,4]


def test_cross_subject_folds_have_no_subject_overlap():
    from eeg_align.signal_data import cross_subject_groups
    subjects=list(map(str,range(1,10)))
    seen=[]
    for fold in range(9):
        mapping,protocol=cross_subject_groups(subjects,'BNCI2014001',fold)
        assert list(mapping.values()).count('test')==1
        assert list(mapping.values()).count('validation')==1
        assert list(mapping.values()).count('train')==7
        seen.extend(s for s,v in mapping.items() if v=='test')
    assert sorted(seen)==subjects
    subjects=list(map(str,range(78)));seen=[]
    for fold in range(10):
        mapping,_=cross_subject_groups(subjects,'SleepEDF',fold)
        seen.extend(s for s,v in mapping.items() if v=='test')
        assert set(mapping.values())=={'train','validation','test'}
    assert len(seen)==78 and len(set(seen))==78
