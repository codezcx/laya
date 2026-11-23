import pytest
import torch
from torch import nn

from eeg_align.personalization import (CONFIG, ResidualAdapter, check_support_rows,
    copy_trainable, frozen_hash, select_support, separate_session_rows,
    trainable_state, fixed_server_versions, make_manifest)
from eeg_align.task_options import make_variant


def rows_2a():
    return [dict(id=f'{session}:{i}', record_id=f'A08{session}', event=i%48,row=i,samples=800,
                 subject='8', split='test', label=i%4)
            for session in ('T','E') for i in range(288)]


def test_sessions_and_nested_shots():
    support, query = separate_session_rows(rows_2a(), 'bcic_2a', 4)
    assert len(support) == len(query) == 288
    previous = set()
    for k in (5,10,20):
        chosen = select_support(support, 4, k, 42)
        ids = {r['id'] for r in chosen}
        assert len(ids) == 4*k and previous <= ids
        assert not ids & {r['id'] for r in query}
        previous = ids
    assert select_support(support,4,5,42) != select_support(support,4,5,43)


def test_duplicate_windows_must_not_be_independent_trials():
    rows = rows_2a()
    rows.append(dict(rows[0], id='another_window'))
    with pytest.raises(ValueError, match='Multiple windows'):
        separate_session_rows(rows, 'bcic_2a', 4)


def test_physio_crossrun_and_short_support_rejected():
    rows = [dict(id=f'{run}:{i}', record_id=f'S089R{run:02d}', event=i,
                 subject='89', split='test', label=(i%2 if run in (4,8,12) else i%2+2))
            for run in (4,6,8,10,12,14) for i in range(14)]
    support, query = separate_session_rows(rows, 'motor_mv_img', 4)
    assert len(support) == 56 and len(query) == 28
    assert len(select_support(support,4,10,42)) == 40
    with pytest.raises(ValueError, match='independent trials'):
        select_support(support,4,20,42)


def test_only_exact_support_can_receive_gradients():
    support, query = separate_session_rows(rows_2a(), 'bcic_2a', 4)
    support = select_support(support,4,5,42)
    case = dict(subject='8', original_split='test', support_ids=[r['id'] for r in support],
                query_ids=[r['id'] for r in query])
    check_support_rows(support, case)
    check_support_rows([support[0]]*16, case)
    with pytest.raises(ValueError, match='authorized support'):
        check_support_rows(query[:1], case)
    with pytest.raises(ValueError, match='authorized support'):
        check_support_rows([dict(support[0],subject='9')], case)
    with pytest.raises(ValueError, match='leakage'):
        check_support_rows(support, dict(case,query_ids=[support[0]['id']]))


def test_residual_identity_update_and_reload():
    torch.manual_seed(42)
    model = ResidualAdapter()
    assert sum(p.numel() for p in model.parameters()) == 33088
    x = torch.randn(8,16,256)
    assert torch.equal(model(x),x)
    initial = trainable_state(model)
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4)
    model(x).square().mean().backward()
    optimizer.step()
    adapted = model(x).detach()
    assert not torch.equal(adapted,x)
    state = trainable_state(model)
    copy_trainable(model,initial)
    assert torch.equal(model(x),x)
    copy_trainable(model,state)
    assert torch.equal(model(x),adapted)


def test_query_prompt_is_label_independent():
    task = dict(instruction='Imagine moving a body part.',question='Which body part?',
                classes=[dict(class_id=str(i),option=str(i)) for i in range(4)])
    prompts = [make_variant(task,i,'sample',0,42,0) for i in range(4)]
    assert len({v['suffix'] for v in prompts}) == 1
    assert all(v['option_class_ids'] == ['0','1','2','3'] for v in prompts)


def test_budget_and_objective_are_matched():
    assert CONFIG['arms'] == ['residual_frozen']
    assert CONFIG['steps'] == 100 and CONFIG['batch_size'] == 16
    assert CONFIG['seeds'] == [42]
    assert CONFIG['datasets'] == {'bcic_2a': [5], 'motor_mv_img': [5]}
    assert CONFIG['objective'] == 'jev_ce' and CONFIG['microbatch'] == 4
    assert CONFIG['learning_rate'] == 1e-4 and CONFIG['weight_decay'] == .01
    assert CONFIG['arm_settings']['residual_frozen']['train_lm_lora'] is False


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.base=nn.Linear(4,4)
        self.lora_A=nn.Parameter(torch.randn(2,4)*.1)
        self.lora_B=nn.Parameter(torch.randn(4,2)*.1)

    def forward(self,x):
        return self.base(x)+x@self.lora_A.T@self.lora_B.T


def test_frozen_lm_still_passes_input_gradient():
    server=nn.Module()
    server.text=TinyLM()
    server.requires_grad_(False)
    x=torch.randn(5,4,requires_grad=True)
    server.text(x).square().mean().backward()
    assert x.grad.abs().sum()>0
    assert all(p.grad is None for p in server.parameters())


def test_accumulation_matches_full_ce_gradient():
    torch.manual_seed(11)
    a=nn.Linear(4,4)
    b=nn.Linear(4,4)
    b.load_state_dict(a.state_dict())
    x=torch.randn(16,4)
    y=torch.arange(16)%4
    nn.functional.cross_entropy(a(x),y).backward()
    for start in range(0,16,4):
        (nn.functional.cross_entropy(b(x[start:start+4]),y[start:start+4])/4).backward()
    assert torch.allclose(a.weight.grad,b.weight.grad,atol=1e-7)
    assert torch.allclose(a.bias.grad,b.bias.grad,atol=1e-7)

def test_independent_adapter_optimizer_and_frozen_lm():
    torch.manual_seed(42)
    personal = ResidualAdapter(width=4, bottleneck=2)
    frozen_lm = TinyLM().requires_grad_(False).eval()
    initial = trainable_state(personal)
    fixed = frozen_hash(frozen_lm)
    versions = fixed_server_versions(frozen_lm)
    x = torch.randn(16, 4)
    y = torch.arange(16) % 4
    outputs = []
    for user in range(2):
        copy_trainable(personal, initial)
        optimizer = torch.optim.AdamW(personal.parameters(), lr=1e-4)
        assert not optimizer.state
        for step in range(3):
            optimizer.zero_grad()
            nn.functional.cross_entropy(frozen_lm(personal(x)), y).backward()
            assert all(p.grad is None for p in frozen_lm.parameters())
            assert personal.up.weight.grad.abs().sum() > 0
            optimizer.step()
        outputs.append(personal(x).detach().clone())
    assert torch.equal(*outputs)
    assert frozen_hash(frozen_lm) == fixed
    assert fixed_server_versions(frozen_lm) == versions


def fake_store():
    from types import SimpleNamespace
    tasks = [dict(task_id=name, classes=list(range(4))) for name in CONFIG['datasets']]
    rows = {0: [], 1: []}
    for split, subjects in [('test', [8, 9]), ('valid', [6, 7])]:
        for subject in subjects:
            rows[0].extend(dict(r, id=f'{subject}_{r["id"]}', subject=str(subject),
                record_id=f'A{subject:02d}{r["record_id"][-1]}', split=split) for r in rows_2a())
    for split, subjects in [('test', range(89, 110)), ('valid', [70, 71])]:
        for subject in subjects:
            rows[1].extend(dict(id=f'{subject}_{run}_{i}', subject=str(subject), split=split,
                record_id=f'S{subject:03d}R{run:02d}', event=i,
                label=(i % 2 if run in (4, 8, 12) else i % 2+2))
                for run in (4, 6, 8, 10, 12, 14) for i in range(14))
    return SimpleNamespace(tasks=tasks, rows=rows, meta={i:dict(index_sha256=str(i)) for i in rows})


def test_manifest_all_test_users_and_no_extra_sweep():
    manifest = make_manifest(fake_store())
    cases = manifest['cases']
    assert len(cases) == len(manifest['cohorts']) == 23
    assert [sum(c['worker'] == i for c in cases) for i in range(4)] == [6, 6, 6, 5]
    assert {(c['shots_per_class'], c['seed'], c['arm']) for c in cases} == {(5,42,'residual_frozen')}
    assert {c['original_split'] for c in cases} == {'test'}
    assert all(len(set(c['support_ids'])) == 20 for c in cases)
    assert all(not set(c['support_ids']) & set(c['query_ids']) for c in cases)


def test_smoke_is_validation_only():
    manifest = make_manifest(fake_store(), smoke=True)
    assert len(manifest['cases']) == 2
    assert {c['subject'] for c in manifest['cases']} == {'6', '70'}
    assert {c['original_split'] for c in manifest['cases']} == {'valid'}
    assert {c['worker'] for c in manifest['cases']} == {0, 1}


def test_missing_original_test_user_rejected():
    store = fake_store()
    store.rows[1] = [r for r in store.rows[1] if r['subject'] != '109']
    with pytest.raises(ValueError, match='Unexpected held-out'):
        make_manifest(store)


def test_aggregate_users_not_trials_and_dataset_macro():
    from eeg_align.run_personalization import aggregate_results
    def result(dataset, subject, before, after, n):
        return dict(case=dict(dataset=dataset,subject=subject),
            baseline=dict(jev=dict(balanced_accuracy=before)),
            adapted=dict(jev=dict(balanced_accuracy=after,n=n)),
            support=dict(jev=dict(accuracy=.7,n=20)),train_seconds=10)
    rs = [result('bcic_2a','8',.4,.6,288),
          result('motor_mv_img','89',.5,.4,28), result('motor_mv_img','90',.3,.3,32)]
    report = aggregate_results(rs)
    assert report['dataset_equal_mean']['adapted_bacc'] == pytest.approx(.475)
    assert report['subject_equal_mean']['adapted_bacc'] == pytest.approx(1.3/3)
    assert report['rows'][0]['improved'] == 1
    assert report['rows'][1]['worse'] == report['rows'][1]['tied'] == 1
    with pytest.raises(ValueError, match='Duplicate user'):
        aggregate_results(rs+[rs[0]])


def test_batch_stream_preserves_reference_sampling():
    import numpy as np
    from eeg_align.personalization import ids_hash
    pool, _ = separate_session_rows(rows_2a(), 'bcic_2a', 4)
    support = select_support(pool, 4, 5, 42)
    labels = np.array([r['label'] for r in support])
    def stream():
        rng = np.random.default_rng(42)
        return [support[int(rng.choice(np.flatnonzero(labels == j % 4)))]
                for _ in range(100) for j in range(16)]
    assert ids_hash(stream()) == ids_hash(stream())
    assert set(r['id'] for r in stream()) <= set(r['id'] for r in support)
