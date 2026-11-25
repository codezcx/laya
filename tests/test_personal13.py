import json
import numpy as np
import pytest
import torch
from personal.common import CONFIG, ROOT, check_rows
from personal.data import TrialStore, PreparedAdapter, normalize_trials, segment, source_group, to_electrodes
from personal.modules import ConnectorDelta, UserParameters, canonical_logits
from personal.plan import build, select_support, split_person, personal_batches
from personal.initialization import common_state, audit_initializations, load_initialization
from personal.curves import aggregate, plot
from jev_train.checkpoint import tensor_fingerprint, capture_rng, restore_rng


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
    assert CONFIG['arms']==['ordinary_lora','source_erm','fomaml']
    assert CONFIG['initialization_seed']==42
    assert CONFIG['eval_steps']==[0,5,10,20,50,100]


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


def test_curve_plan_exact_users_samples_batches():
    store = TrialStore()
    plan = build(store)
    previous = json.loads((ROOT/CONFIG['initialization_reference']/'plan.json').read_text())
    assert plan['cases'] == previous['cases']
    assert plan['counts'] == dict(bcic_2a=2, hmc=2)
    assert not plan['offline_retraining']
    for case in plan['cases']:
        assert len(case['support_ids']) == 100
        assert not set(case['support_ids']) & set(case['query_ids'])
        assert case['original_split'] == 'test'
        assert not case['shared_train_seen_subject']
        assert case['query_in_shared_train_pool'] == 0
        assert len(case['query_ids']) == (288 if case['dataset']=='bcic_2a' else 340 if case['subject']=='2' else 344)
        schedule = plan['schedules'][case['key']]
        assert schedule == personal_batches(case)
        assert {i for b in schedule for i in b['indices']} == set(range(100))
        for batch in schedule:
            # Reproduce the previous personal trainer's exact sampling rule.
            rng = np.random.default_rng(np.random.SeedSequence([42, case['order'], batch['step']]))
            assert batch['indices'] == rng.integers(100, size=16).tolist()
    assert not any(r['split'] == 'valid' for rows in store.rows.values() for r in rows)


def test_all_initializations_share_seed42_origin_and_reuse_unchanged_weights():
    records = audit_initializations()
    ordinary = common_state()
    assert tensor_fingerprint(ordinary) == '88da1153c2e270122ff7e9af1bb9db58e5824a9fc415426022e66af5f8112542'
    assert not ordinary['delta.b'].count_nonzero()
    for name in CONFIG['datasets']:
        for arm in CONFIG['arms']:
            r = records[name][arm]
            state = load_initialization(r)
            assert tensor_fingerprint(state) == r['final_sha']
            assert r['starting_sha'] == tensor_fingerprint(ordinary)
            if arm != 'ordinary_lora':
                assert state['delta.b'].count_nonzero()
                assert not torch.equal(state['delta.a'], ordinary['delta.a'])
                assert r['source_episodes'] == 60


def test_seed_constructor_does_not_consume_callers_cpu_rng():
    before = torch.get_rng_state()
    first = common_state()
    assert torch.equal(before, torch.get_rng_state())
    second = common_state()
    assert tensor_fingerprint(first) == tensor_fingerprint(second)


def test_paired_curves_wait_for_all_users_and_methods(tmp_path):
    cases = [dict(key=f'{d}{i}', dataset=d) for d in ['bcic_2a','hmc'] for i in [1,2]]
    arms = CONFIG['arms']
    points = []
    for case in cases:
        for arm in arms:
            value = .5 if arm == 'ordinary_lora' else .6 if arm == 'source_erm' else .7
            metric = dict(accuracy=value, balanced_accuracy=value, ce=1-value)
            points.append(dict(case=case, arm=arm, step=0, support=metric, query=metric,
                               baseline=dict(balanced_accuracy=.4)))
    incomplete = aggregate(points[:-1], cases, arms, [0,5])
    assert '0' in incomplete['matched']['bcic_2a']
    assert not incomplete['matched']['hmc']
    assert not incomplete['equal_task']
    complete = aggregate(points, cases, arms, [0,5])
    assert complete['equal_task']['0']['fomaml']['delta_vs_ordinary_pp'] == pytest.approx(20.)
    assert complete['matched']['hmc']['0']['ordinary_lora']['users'] == 2
    assert '5' not in complete['equal_task']
    with pytest.raises(RuntimeError, match='Duplicate'):
        aggregate(points + points[:1], cases, arms, [0,5])
    plot(complete, tmp_path, arms, [0,5])
    assert (tmp_path/'adaptation_curves.png').stat().st_size > 1000
    assert (tmp_path/'support_query_curves.pdf').exists()


def test_bcic_preprocessing_matches_historical_saved_signal():
    old=torch.load(ROOT/'artifacts/benchmark/leaf_personalization_20261123/subjects/bcic_2a_s8.pt',
                   weights_only=False,map_location='cpu')
    store=TrialStore(); byid={r['id']:r for r in store.rows[0]}
    rows=[byid[r['id']] for r in old['rows'][:4]]
    actual=PreparedAdapter(store).batch(rows)[0][2][0]
    np.testing.assert_array_equal(actual,old['x'][:4].numpy())
