"""Paired source episodes and forward-only support-conditioned adaptation."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from eeg_align.personalization import (CONFIG, AmortizedFiLM, assert_episode,
    make_manifest, module_hash, select_support, separate_session_rows, trial_key)


def rows_2a(subject='1', split='train'):
    return [dict(id=f'2a:{subject}:{session}:{i}',subject=subject,split=split,
                 record_id=f'A{int(subject):02d}{session}',row=i,event=i%48,samples=800,label=i%4)
            for session in ('T','E') for i in range(288)]


def rows_physio(subject='1',split='train'):
    return [dict(id=f'p:{subject}:{run}:{i}',subject=subject,split=split,
                 record_id=f'S{int(subject):03d}R{run:02d}',row=i,event=i,samples=800,label=i%4)
            for run in (4,6,8,10,12,14) for i in range(16)]


def fake_store():
    return SimpleNamespace(tasks=[dict(task_id=n,classes=list('abcd')) for n in CONFIG['datasets']],
        rows={0:rows_2a()+rows_2a('6','valid')+rows_2a('8','test'),
              1:rows_physio()+rows_physio('70','valid')+rows_physio('90','test')},
        excluded={0:set(),1:set()},meta={0:dict(index_sha256='a'),1:dict(index_sha256='b')})


def test_session_trial_identity():
    rows=rows_2a()
    a,b=separate_session_rows(rows,'bcic_2a',4)
    assert len(a)==len(b)==288
    assert len({trial_key(r) for r in rows})==576
    assert all(r['record_id'].endswith('T') for r in a)
    assert all(r['record_id'].endswith('E') for r in b)


@pytest.mark.parametrize('bad',['duplicate','incomplete','mixed_subject','mixed_split'])
def test_invalid_session_fails(bad):
    rows=rows_2a()
    if bad=='duplicate':
        rows.append(rows[0].copy())
    elif bad=='incomplete':
        rows.pop()
    elif bad=='mixed_subject':
        rows[0]['subject']='2'
    else:
        rows[0]['split']='test'
    with pytest.raises(ValueError):
        separate_session_rows(rows,'bcic_2a',4)


def test_physio_different_recordings():
    a,b=separate_session_rows(rows_physio(),'motor_mv_img',4)
    assert len(a)==64 and len(b)==32
    assert {r['record_id'] for r in a}.isdisjoint({r['record_id'] for r in b})


def test_support_balanced_unique_deterministic():
    a,_=separate_session_rows(rows_2a(),'bcic_2a',4)
    sampled=select_support(a,4,5,42)
    assert sampled==select_support(list(reversed(a)),4,5,42)
    assert len({r['id'] for r in sampled})==20
    assert [sum(r['label']==i for r in sampled) for i in range(4)]==[5]*4
    assert sampled!=select_support(a,4,5,43)


def test_manifest_only_source_gradients_and_matched_stream():
    cfg=dict(CONFIG,steps=12)
    first=make_manifest(fake_store(),cfg)
    assert first==make_manifest(fake_store(),cfg)
    cohorts={c['key']:c for c in first['cohorts']}
    assert len(cohorts)==5
    assert {c['subject'] for c in cohorts.values() if c['split']=='test'}=={'8'}
    counts={n:0 for n in CONFIG['datasets']}
    for ep in first['episodes']:
        c=cohorts[ep['key']]
        assert c['split']=='train'
        assert len(ep['support_ids'])==20 and len(ep['query_ids'])==16
        assert not set(ep['support_ids'])&set(ep['query_ids'])
        assert_episode(c,ep,True)
        counts[c['dataset']]+=1
    assert set(counts.values())=={6}
    assert not first['test_used_for_selection']


def test_forbid_query_leakage_and_heldout_training():
    m=make_manifest(fake_store(),dict(CONFIG,steps=2))
    c=next(c for c in m['cohorts'] if c['split']=='train')
    ep=dict(support_ids=[c['support_pool'][0]['id']],query_ids=[c['query'][0]['id']])
    with pytest.raises(ValueError,match='Only original train'):
        assert_episode(dict(c,split='test'),ep,True)
    with pytest.raises(ValueError,match='Duplicate trials'):
        assert_episode(c,dict(ep,query_ids=ep['support_ids']),True)
    with pytest.raises(ValueError,match='boundary'):
        assert_episode(c,dict(ep,query_ids=['another_subject']),True)


def test_manifest_does_not_change_original_rows():
    s=fake_store()
    before=copy.deepcopy(s.rows)
    make_manifest(s,dict(CONFIG,steps=2))
    assert s.rows==before


def small_adapter():
    return AmortizedFiLM(width=8,text_width=6,label_width=3,context_width=10,bottleneck=4)


def test_identity_initialization_and_permutation_invariance():
    torch.manual_seed(42)
    m=small_adapter()
    support=torch.randn(20,4,8)
    labels=torch.randn(20,6)
    query=torch.randn(5,4,8)
    u=m.summarize(support,labels)
    ix=torch.randperm(20)
    torch.testing.assert_close(u,m.summarize(support[ix],labels[ix]))
    assert torch.equal(query,m(query,u))
    assert m.summarize(support[:3],labels[:3]).shape==(10,)


def test_empty_or_unmatched_support_rejected():
    m=small_adapter()
    for n in (0,3):
        with pytest.raises(ValueError):
            m.summarize(torch.randn(n,4,8),torch.randn(1,6))


@pytest.mark.parametrize('train_tower',[False,True])
def test_frozen_lm_passes_gradients_and_support_branch(train_tower):
    torch.manual_seed(7)
    tower=nn.Linear(7,8).requires_grad_(train_tower)
    adapter=small_adapter()
    lm=nn.Sequential(nn.Linear(8,12),nn.GELU(),nn.Linear(12,4)).requires_grad_(False)
    before=module_hash(lm)
    optimizer=torch.optim.AdamW(list(adapter.parameters())+(list(tower.parameters()) if train_tower else []),lr=.01)
    nonzero=set()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        support=tower(torch.randn(20,4,7))
        if train_tower:
            support.retain_grad()
        u=adapter.summarize(support,torch.randn(20,6))
        query=tower(torch.randn(6,4,7))
        loss=nn.functional.cross_entropy(lm(adapter(query,u).mean(1)),torch.tensor([0,1,2,3,0,1]))
        loss.backward()
        for name,module in adapter.named_children():
            if any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters()):
                nonzero.add(name)
        if train_tower and _==2:
            assert support.grad.abs().sum()>0
        optimizer.step()
    assert nonzero=={'label_projection','element','context','condition','down','up'}
    assert module_hash(lm)==before and all(p.grad is None for p in lm.parameters())
    assert all((p.grad is not None)==train_tower for p in tower.parameters())


@pytest.mark.parametrize('train_tower',[False,True])
def test_replicated_support_distributed_gradient_equivalence(train_tower):
    torch.manual_seed(8)
    tower=nn.Linear(7,8).requires_grad_(train_tower)
    adapter=small_adapter()
    nn.init.normal_(adapter.up.weight,std=.03)
    lm=nn.Linear(8,4).requires_grad_(False)
    support=torch.randn(20,4,7)
    text=torch.randn(20,6)
    query=torch.randn(16,4,7)
    labels=torch.arange(16)%4
    params=list(adapter.parameters())+(list(tower.parameters()) if train_tower else [])
    def grads(indices):
        for p in params:
            p.grad=None
        u=adapter.summarize(tower(support),text)
        loss=nn.functional.cross_entropy(lm(adapter(tower(query[indices]),u).mean(1)),labels[indices])
        loss.backward()
        return [p.grad.clone() for p in params]
    whole=grads(slice(None))
    left,right=grads(slice(0,8)),grads(slice(8,16))
    for a,b,c in zip(whole,left,right):
        torch.testing.assert_close(a,(b+c)/2,atol=1e-7,rtol=1e-4)


def test_forward_only_context_and_reload(tmp_path):
    m=small_adapter()
    nn.init.normal_(m.up.weight,std=.03)
    support=torch.randn(20,4,8)
    labels=torch.randn(20,6)
    query=torch.randn(5,4,8)
    before=module_hash(m)
    with torch.no_grad():
        a=m(query,m.summarize(support,labels))
        b=m(query,m.summarize(support+5,labels))
    assert module_hash(m)==before and not torch.equal(a,b)
    path=tmp_path/'adapter.pt'
    torch.save(m.state_dict(),path)
    m2=small_adapter()
    m2.load_state_dict(torch.load(path,weights_only=True))
    torch.testing.assert_close(a,m2(query,m2.summarize(support,labels)),rtol=0,atol=0)


def test_locked_budget_and_no_language_updates():
    assert CONFIG['arms']==['tower_frozen','tower_full']
    assert CONFIG['steps']==340 and CONFIG['seed']==42
    assert CONFIG['adapter_learning_rate']==1e-4 and CONFIG['tower_learning_rate']==1e-6
    assert CONFIG['shots_per_class']==5 and CONFIG['global_query_batch']==16
    assert CONFIG['test_subjects']=={'bcic_2a':['8']}
    assert sum(p.numel() for p in AmortizedFiLM().parameters())<250000
