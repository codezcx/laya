import json
import hashlib
from collections import Counter
import numpy as np
import pytest
import torch
from personal.common import CONFIG, ROOT, check_rows
from personal.data import TrialStore, PreparedAdapter, SignalBatch, normalize_trials, segment, source_group
from personal.modules import UserParameters, canonical_logits, install_tower_lora, tower_lora_parameters
from personal.plan import build, select_support, split_person, personal_batches
from personal.initialization import common_state, audit_initializations, load_initialization
from personal.curves import aggregate, plot
from jev_train.checkpoint import tensor_fingerprint


@pytest.fixture(scope='module')
def store(): return TrialStore()


@pytest.fixture(scope='module')
def plan(store): return build(store)


def row(i,session='A08T'):
    return dict(id=f'{session}:{i}',dataset='bcic_2a',subject='8',task=0,split='test',
                record=session,row=i,event=i%48,start=i*1000,samples=800,label=i%4)


def test_gradient_authorization_and_query_exclusion():
    s,q=row(0),row(1,'A08E')
    case=dict(dataset='bcic_2a',subject='8',task=0,support_ids=[s['id']],query_ids=[q['id']],row_splits={s['id']:'test',q['id']:'test'})
    check_rows([s],case,gradient=True); check_rows([q],case)
    for bad in [q,dict(s,subject='3'),dict(s,split='train')]:
        with pytest.raises(RuntimeError): check_rows([bad],case,gradient=True)
    with pytest.raises(RuntimeError,match='overlap'): check_rows([s],dict(case,query_ids=case['support_ids']))


def test_bcic_original_complete_sessions():
    rows=[row(i) for i in range(288)]+[row(i,'A08E') for i in range(288)]
    support,query,_=split_person(rows,'bcic_2a',4)
    assert len(support)==100 and len(query)==288
    assert not {source_group(r) for r in support}&{source_group(r) for r in query}
    with pytest.raises(ValueError,match='288'): split_person(rows[:-1],'bcic_2a',4)


def test_balanced_support_has_no_duplicates():
    rows=[dict(row(i),dataset='seed',label=i%3) for i in range(300)]
    selected=select_support(rows,3,100,42)
    assert len(selected)==len({r['id'] for r in selected})==100
    assert sorted(Counter(r['label'] for r in selected).values())==[33,33,34]


def test_exact_21_users_nested_support_and_schedules(plan,store):
    previous=json.loads((ROOT/CONFIG['paired_reference']/'plan.json').read_text())
    assert plan['cases']==previous['cases']
    assert plan['counts']==dict(bcic_2a=9,seed=4,hmc=4,Nakanishi2015=4)
    assert len(plan['episodes'])==168
    assert Counter(e['support_total'] for e in plan['episodes'])=={100:84,48:42,24:42}
    assert len({e['key'] for e in plan['episodes']})==168
    assert sorted(k for j in plan['jobs'] for k in j['episodes'])==sorted(e['key'] for e in plan['episodes'])
    assert [len(j['episodes']) for j in plan['jobs']]==[42]*4
    for job in plan['jobs']:
        phases=[next(e['phase'] for e in plan['episodes'] if e['key']==key) for key in job['episodes']]
        assert phases==sorted(phases)
    byid={r['id']:r for rows in store.rows.values() for r in rows}
    for case in plan['cases']:
        sets=plan['support_sets'][case['key']]
        assert set(sets['24'])<set(sets['48'])<set(sets['100'])
        assert sets['100']==case['support_ids']
        for n in CONFIG['support_sizes']:
            ids=sets[str(n)]; counts=Counter(byid[i]['label'] for i in ids)
            assert len(ids)==len(set(ids))==n and len(counts)==case['classes']
            assert max(counts.values())-min(counts.values())<=1
            assert not set(ids)&set(case['query_ids'])
            active=dict(case,support_ids=ids)
            schedule=plan['schedules'][f'{case["key"]}_n{n}']
            assert schedule==personal_batches(active)
            assert {i for b in schedule for i in b['indices']}==set(range(n))
            check_rows([byid[i] for i in ids],active,gradient=True)
        assert case['query_ids']==next(c['query_ids'] for c in previous['cases'] if c['key']==case['key'])


def test_protocol_and_initialization_are_matched():
    previous=json.loads((ROOT/CONFIG['matched_reference']/'plan.json').read_text())['config']
    for name in ('parent','parent_sha256','seed','steps','batch_size','microbatch','learning_rate','weight_decay','clip_grad_norm','connector_lora_rank','connector_lora_scale','eval_steps'):
        assert CONFIG[name]==previous[name]
    assert CONFIG['tower_lora_rank']==4 and CONFIG['tower_learning_rate']==CONFIG['learning_rate']
    records=audit_initializations()
    last=common_state('last_only'); first=common_state('first_only')
    assert tensor_fingerprint(last)=='88da1153c2e270122ff7e9af1bb9db58e5824a9fc415426022e66af5f8112542'
    for arm in CONFIG['arms']:
        state=load_initialization(records[arm])
        assert tensor_fingerprint(state)==records[arm]['final_sha']
        if arm in ('both','tower_both'):
            for name,value in {**last,**first}.items(): torch.testing.assert_close(state[name],value,rtol=0,atol=0)
    before=torch.get_rng_state(); common_state('both'); assert torch.equal(before,torch.get_rng_state())


@pytest.mark.parametrize('arm,count',[('first_only',21504),('last_only',40960),('both',62464),('tower_both',62464)])
def test_connector_lora_shapes_identity_gradients(arm,count):
    user=UserParameters(CONFIG,arm)
    assert sum(p.numel() for p in user.parameters())==count
    losses=[]
    for name,dim in [('delta',5120),('first',256)]:
        if hasattr(user,name):
            delta=getattr(user,name); value=delta(torch.randn(2,3,dim))
            assert value.shape==(2,3,5120) and not value.count_nonzero()
            losses.append(value.sum())
    sum(losses).backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in user.parameters())


def test_weight_lora_reaches_mha_direct_weight_and_preserves_base():
    torch.manual_seed(42); torch.backends.mha.set_fastpath_enabled(False)
    class Mini(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.attn=torch.nn.MultiheadAttention(8,2,batch_first=True); self.ff=torch.nn.Linear(8,8)
        def forward(self,x): return self.ff(self.attn(x,x,x,need_weights=False)[0])
    m=Mini().eval().requires_grad_(False); x=torch.randn(2,3,8)
    baseline=m(x).detach(); original={n:p.detach().clone() for n,p in m.named_parameters()}
    records=install_tower_lora(m,2,1.,42)
    assert {(r['module'],r['parameter']) for r in records}=={('attn','in_proj_weight'),('attn.out_proj','weight'),('ff','weight')}
    with torch.no_grad(): torch.testing.assert_close(m(x),baseline,rtol=0,atol=0)
    parameters=tower_lora_parameters(m); m(x).square().mean().backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in parameters.values())
    for name,p in parameters.items():
        if name.endswith('.b'): assert p.grad.abs().sum()>0
    optimizer=torch.optim.AdamW(parameters.values(),lr=.01); optimizer.step()
    assert not torch.equal(m(x),baseline)
    for name,p in m.named_parameters():
        if name in parameters: continue
        assert p.grad is None and not p.requires_grad
        old_name=name.replace('.parametrizations.weight.original','.weight').replace('.parametrizations.in_proj_weight.original','.in_proj_weight')
        torch.testing.assert_close(p,original[old_name],rtol=0,atol=0)


def test_tower_lora_reset_optimizer_and_freeze_guard():
    from personal.model import PersonalServer
    from personal.worker import optimizer_for,update
    server=object.__new__(PersonalServer); server.model=torch.nn.Module()
    server.model.add_module('eeg_encoder',torch.nn.Sequential(torch.nn.Linear(4,4)))
    server.model.add_module('connector',torch.nn.Linear(4,4)); server.model.requires_grad_(False)
    install_tower_lora(server.model.eeg_encoder,2,1.,42); server.model.requires_grad_(False)
    server.tower_enabled=False; server.initial_tower=server.tower_state()
    server.personal_names={'eeg_encoder.'+n for n in tower_lora_parameters(server.model.eeg_encoder)}
    server.frozen_versions={n:p._version for n,p in server.model.named_parameters() if n not in server.personal_names}
    server.frozen_buffers={}; user=torch.nn.Linear(4,4)
    server.configure_arm('tower_both'); optimizer=optimizer_for(user,server)
    user(server.model.connector(server.model.eeg_encoder(torch.ones(2,4)))).sum().backward(); update(user,optimizer)
    server.assert_frozen(); assert tensor_fingerprint(server.tower_state())!=tensor_fingerprint(server.initial_tower)
    server.configure_arm('last_only')
    assert tensor_fingerprint(server.tower_state())==tensor_fingerprint(server.initial_tower)
    with torch.no_grad(): server.model.connector.weight.add_(1)
    with pytest.raises(RuntimeError,match='modified'): server.assert_frozen()


def test_project_first_and_second_locations():
    from personal.model import PersonalServer
    server=object.__new__(PersonalServer); server.model=torch.nn.Module()
    server.model.connector=torch.nn.Sequential(torch.nn.LayerNorm(256),torch.nn.Linear(256,5120),torch.nn.GELU(),torch.nn.Linear(5120,5120)).requires_grad_(False)
    x=torch.randn(2,2,256); baseline=server.project(x,None)
    for arm in CONFIG['arms']:
        user=UserParameters(CONFIG,arm); y=server.project(x,user)
        torch.testing.assert_close(y,baseline,rtol=0,atol=0)
        y.square().mean().backward()
        assert all(p.grad is not None and p.grad.isfinite().all() for p in user.parameters())


def test_paired_summary_waits_for_all_members_and_stratifies(plan,tmp_path):
    points=[]
    for ep in plan['episodes']:
        if ep['support_total']!=100: continue
        case=dict(next(c for c in plan['cases'] if c['key']==ep['case_key']))
        value=.5 if ep['arm']=='last_only' else .6
        metric=dict(accuracy=value,balanced_accuracy=value,ce=1-value)
        points.append(dict(episode=ep,case=case,arm=ep['arm'],support_n=100,step=100,query=metric,support=metric,
                           baseline=dict(accuracy=.4,balanced_accuracy=.4,ce=.6)))
    report=aggregate(points,plan)
    assert report['equal_task_all_exposures']['100']['100']['both']['delta_vs_reference_pp']==pytest.approx(10.)
    assert report['unseen_datasets']==['bcic_2a','hmc']
    incomplete=aggregate(points[:-1],plan)
    assert not incomplete['equal_task_all_exposures']['100']
    assert incomplete['equal_task_unseen_subjects']['100']
    assert not report['matched']['bcic_2a']['24']
    with pytest.raises(RuntimeError,match='Duplicate'): aggregate(points+points[:1],plan)
    plot(report,tmp_path)
    assert (tmp_path/'location_curves.png').stat().st_size>1000


def test_signal_preprocessing_and_chunking(store):
    old=torch.load(ROOT/'artifacts/benchmark/leaf_personalization_20261123/subjects/bcic_2a_s8.pt',weights_only=False,map_location='cpu')
    byid={r['id']:r for r in store.rows[0]}; rows=[byid[r['id']] for r in old['rows'][:4]]
    np.testing.assert_array_equal(PreparedAdapter(store).batch(rows)[0][2][0],old['x'][:4].numpy())
    inputs=SignalBatch.prepare(store,rows)
    assert len(inputs)==4 and len(inputs[[3,0,0]])==3
    np.testing.assert_array_equal(inputs.pieces[0].numpy(),old['x'][:4].numpy())
    x=np.arange(6000)[None,None]; pieces=segment(x)
    assert [v.shape[-1] for v in pieces]==[2000]*3
    np.testing.assert_array_equal(np.concatenate(pieces,axis=-1),x)


def test_trial_normalization_does_not_leak_other_trials():
    x=np.random.default_rng(42).normal(size=(4,22,800)).astype(np.float32)
    one=normalize_trials(x[:1]); x[1:]*=10000
    np.testing.assert_array_equal(one,normalize_trials(x)[:1])
    with pytest.raises(ValueError,match='constant'): normalize_trials(np.zeros_like(x))


@pytest.mark.parametrize('name,expected',[
    ('seed','56d2fcd7086c66d783fb241675b4f2baf81c50d8d719674843c4b27ca76095fa'),
    ('hmc','b885b94a1c69db263bde165c53edae83003e7c6386f6810a742589429642798c'),
    ('Nakanishi2015','7cc6dcb070dae97b5ac83048f41a1704af611e61af0c5183971f84ccd893ef13')])
def test_other_tasks_match_historical_preprocessing_digest(name,expected,store,plan):
    # Audited against 6f3cde9's PreparedAdapter in an isolated worktree.
    case=next(c for c in plan['cases'] if c['dataset']==name)
    byid={r['id']:r for r in store.rows[case['task']]}
    rows=[byid[i] for i in case['support_ids'][:2]+case['query_ids'][:2]]
    digest=hashlib.sha256()
    for piece in PreparedAdapter(store).batch(rows)[0][2]: digest.update(piece.tobytes())
    assert digest.hexdigest()==expected


def test_canonical_option_order():
    x=torch.tensor([[3.,1.,2.]],requires_grad=True); y=canonical_logits(x,[[2,0,1]])
    torch.testing.assert_close(y,torch.tensor([[1.,2.,3.]])); y.sum().backward(); assert x.grad.isfinite().all()
