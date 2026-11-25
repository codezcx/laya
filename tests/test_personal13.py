import json
import numpy as np
import pytest
import torch
from personal.common import CONFIG, ROOT, check_rows, check_source_rows
from personal.data import TrialStore, PreparedAdapter, normalize_trials, segment, source_group, to_electrodes
from personal.modules import ConnectorDelta, UserParameters, canonical_logits
from personal.plan import build, select_support, split_person
from personal.meta import clone_initialization, first_order_gradients, episode_schedule


def row(i,session='A08T',label=None):
    return dict(id=f'{session}:{i}',dataset='bcic_2a',subject='8',task=0,split='test',
                record=session,row=i,event=i%48,start=i*1000,samples=800,label=i%4 if label is None else label)


def test_authorized_rows_and_split_identity():
    s,q=row(0),row(1,'A08E')
    case=dict(dataset='bcic_2a',subject='8',task=0,support_ids=[s['id']],query_ids=[q['id']],
              row_splits={s['id']:'test',q['id']:'test'})
    check_rows([s],case,gradient=True)
    check_rows([q],case)
    with pytest.raises(RuntimeError):
        check_rows([q],case,gradient=True)
    with pytest.raises(RuntimeError):
        check_rows([dict(s,subject='5')],case,gradient=True)
    with pytest.raises(RuntimeError):
        check_rows([dict(s,split='train')],case,gradient=True)
    with pytest.raises(RuntimeError,match='overlap'):
        check_rows([s],dict(case,query_ids=case['support_ids']))


def test_complete_bcic_sessions_and_same_seeded_support():
    rows=[row(i) for i in range(288)]+[row(i,'A08E') for i in range(288)]
    support,query,details=split_person(rows,'bcic_2a',4)
    assert len(support)==100 and len(query)==288
    assert len({r['id'] for r in support})==100
    assert [sum(r['label']==c for r in support) for c in range(4)]==[25]*4
    assert not {source_group(r) for r in support}&{source_group(r) for r in query}
    with pytest.raises(ValueError,match='288'):
        split_person(rows[:-1],'bcic_2a',4)
    small=select_support(rows[:288],4,20,42)
    assert {r['id'] for r in small}<={r['id'] for r in support}


def test_balanced_remainder_does_not_duplicate_trials():
    rows=[dict(row(i),dataset='seed',label=i%3) for i in range(300)]
    selected=select_support(rows,3,100,42)
    assert len(selected)==len({r['id'] for r in selected})==100
    assert sorted(sum(r['label']==c for r in selected) for c in range(3))==[33,33,34]
    with pytest.raises(ValueError,match='needs'):
        select_support(rows[:9],3,100,42)


def test_connector_lora_identity_and_gradient():
    module=UserParameters(CONFIG)
    assert sum(p.numel() for p in module.parameters())==40960
    x=torch.randn(2,3,5120)
    assert torch.count_nonzero(module.delta(x))==0
    module.delta(x).sum().backward()
    assert module.delta.b.grad.abs().sum()>0
    assert all(p.grad is not None and p.grad.isfinite().all() for p in module.parameters())


def test_semantic_option_reordering():
    x=torch.tensor([[3.,1.,2.]],requires_grad=True)
    ordered=canonical_logits(x,[[2,0,1]])
    torch.testing.assert_close(ordered,torch.tensor([[1.,2.,3.]]))
    ordered.sum().backward()
    assert x.grad.isfinite().all()


def test_optimization_matches_completed_four_user_run():
    previous=json.loads((ROOT/CONFIG['paired_reference']/'plan.json').read_text())['config']
    for name in ('parent','parent_sha256','seed','steps','batch_size','microbatch','eval_batch',
                 'encode_batch','learning_rate','weight_decay','clip_grad_norm','connector_lora_rank','connector_lora_scale'):
        assert CONFIG[name]==previous[name]
    assert CONFIG['arms']==['source_erm','fomaml']


def test_trial_normalization_independence_and_montage():
    x=np.random.default_rng(42).normal(size=(4,22,800)).astype(np.float32)
    one=normalize_trials(x[:1]); x[1:]*=10000
    np.testing.assert_array_equal(one,normalize_trials(x)[:1])
    with pytest.raises(ValueError,match='constant'):
        normalize_trials(np.zeros_like(x))
    original=np.ones((2,2,200),dtype=np.float32)
    recovered,names,_=to_electrodes(original,['C3-M1','C4-M1'])
    np.testing.assert_allclose(recovered[:,names.index('C3')]-recovered[:,names.index('M1')],1.,atol=1e-6)


def test_chunking_preserves_long_sleep_epochs():
    x=np.arange(6000)[None,None]
    pieces=segment(x)
    assert [v.shape[-1] for v in pieces]==[2000]*3
    np.testing.assert_array_equal(np.concatenate(pieces,axis=-1),x)


def test_meta_plan_has_unseen_targets_and_train_only_sources():
    store = TrialStore()
    p = build(store)
    assert p['counts'] == dict(bcic_2a=2, hmc=2)
    assert p['source_counts'] == dict(bcic_2a=5, hmc=8)
    for case in p['cases']:
        assert len(case['support_ids']) == 100
        assert not set(case['support_ids']) & set(case['query_ids'])
        assert case['original_split'] == 'test'
        assert not case['shared_train_seen_subject']
        assert case['query_in_shared_train_pool'] == 0
        assert len(case['query_ids']) == (288 if case['dataset']=='bcic_2a' else 340 if case['subject']=='2' else 344)
    for case in p['sources']:
        assert len(case['support_ids']) == len(case['query_ids']) == 100
        assert not set(case['support_ids']) & set(case['query_ids'])
        index = {r['id']: r for r in store.rows[case['task']]}
        for part in ['support','query']:
            rows = [index[i] for i in case[part+'_ids']]
            check_source_rows(rows, case, part)
            assert {r['split'] for r in rows} == {'train'}
            with pytest.raises(RuntimeError):
                check_source_rows([dict(rows[0], split='test')], case, part)
        with pytest.raises(RuntimeError):
            check_source_rows([index[case['query_ids'][0]]], case, 'support')
    assert not set(i for c in p['sources'] for part in ['support','query'] for i in c[part+'_ids']) & set(
        i for c in p['cases'] for part in ['support','query'] for i in c[part+'_ids'])
    for name, episodes in p['schedules'].items():
        assert len(episodes) == 60
        assert sum(len(e['batches']) for e in episodes) == 360
        assert sum(len(b['ids']) for e in episodes for b in e['batches']) == 5760
        assert episodes == episode_schedule([c for c in p['sources'] if c['dataset']==name], CONFIG)
        assert all([b['part'] for b in e['batches']] == ['support']*5+['query'] for e in episodes)


def test_first_order_meta_gradient_is_query_gradient_at_adapted_weights():
    initial = torch.nn.Linear(1,1,bias=False)
    with torch.no_grad():
        initial.weight.fill_(1.)
    adapted = clone_initialization(initial)
    inner = torch.optim.SGD(adapted.parameters(), lr=.1)
    (.5*(adapted.weight-2).square().sum()).backward()
    inner.step()
    assert initial.weight.item() == 1.
    adapted.zero_grad()
    (.5*(adapted.weight-4).square().sum()).backward()
    first_order_gradients(initial, adapted)
    torch.testing.assert_close(initial.weight.grad, torch.tensor([[-2.9]]))
    torch.optim.SGD(initial.parameters(), lr=.1).step()
    torch.testing.assert_close(initial.weight, torch.tensor([[1.29]]))
    with pytest.raises(RuntimeError):
        first_order_gradients(initial, initial)


def test_bcic_preprocessing_matches_historical_saved_signal():
    old=torch.load(ROOT/'artifacts/benchmark/leaf_personalization_20261123/subjects/bcic_2a_s8.pt',
                   weights_only=False,map_location='cpu')
    store=TrialStore(); byid={r['id']:r for r in store.rows[0]}
    rows=[byid[r['id']] for r in old['rows'][:4]]
    actual=PreparedAdapter(store).batch(rows)[0][2][0]
    np.testing.assert_array_equal(actual,old['x'][:4].numpy())
