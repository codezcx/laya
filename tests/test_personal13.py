import copy

import pytest
import torch

from personal.common import CONFIG, check_rows
from personal.modules import ConnectorDelta, KVPrefix, PrefixAttention, ResidualAdapter, UserParameters, canonical_logits, support_loss
from personal.plan import exact_total_support, paired_reference, partitions, select_support


def test_personal_valid_boundary_and_no_query_gradients():
    case = dict(subject='A01', task=0, support_ids=['s'], query_ids=['q'])
    support = dict(id='s', split='valid', subject='A01', task=0)
    check_rows([support], case, gradient=True)
    check_rows([dict(support, id='q')], case)
    for row in (dict(support, id='q'), dict(support, split='test'), dict(support, subject='A02')):
        with pytest.raises(RuntimeError):
            check_rows([row], case, gradient=True)
    with pytest.raises(RuntimeError, match='overlap'):
        check_rows([support], dict(case, query_ids=['s']))


def test_session_split_uses_author_e_before_t():
    rows = [dict(id=str(i), index=i, label=i % 2) for i in range(4)]
    p = [dict(record=x, trial=str(i)) for i,x in enumerate(['E','E','T','T'])]
    support, query = partitions(rows, 'MI_BCIC_IV2a', p)
    assert [r['id'] for r in support] == ['2','3']
    assert [r['id'] for r in query] == ['0','1']


def test_paired_support_selection_and_no_silent_shot_reduction():
    rows = [dict(id=str(i), label=i % 3) for i in range(30)]
    selected = select_support(rows, 3, 5, 42)
    assert selected == select_support(rows, 3, 5, 42)
    assert len({r['id'] for r in selected}) == 15
    with pytest.raises(ValueError, match='needs'):
        select_support(rows[:6], 3, 5, 42)


def test_adapter_lora_identity_and_gradients():
    x = torch.randn(2, 4, 256)
    adapter = ResidualAdapter()
    torch.testing.assert_close(adapter(x), x, rtol=0, atol=0)
    delta = ConnectorDelta(dim=256, rank=4)
    assert torch.count_nonzero(delta(x)) == 0
    (adapter(x).sum() + delta(x).sum()).backward()
    assert adapter.up.weight.grad.abs().sum() > 0
    assert delta.b.grad.abs().sum() > 0


def test_three_40960_parameter_controls():
    p = torch.zeros(8, 5120)
    for arm in ('soft_prompt', 'connector_lora', 'kv_prefix'):
        user = UserParameters(arm, p, CONFIG)
        assert sum(v.numel() for v in user.parameters()) == 40960
    assert sum(v.numel() for v in UserParameters('residual', p, CONFIG).parameters()) == 33088


def test_semantic_reordering_and_teacher_gate():
    x = torch.tensor([[3., 1., 2.]], requires_grad=True)
    ordered = canonical_logits(x, [[2, 0, 1]])
    torch.testing.assert_close(ordered, torch.tensor([[1., 2., 3.]]))
    teacher = torch.tensor([[.2, .3, .5]])
    loss, ce, kd = support_loss(ordered, torch.tensor([1]), teacher, torch.tensor([False]))
    torch.testing.assert_close(loss, ce)
    assert kd == 0
    loss.backward()
    assert x.grad.isfinite().all()
    with pytest.raises(ValueError, match='detached'):
        support_loss(ordered, torch.tensor([1]), teacher.requires_grad_(), torch.tensor([True]))


def test_prefix_noop_when_disabled_and_frozen_base():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention
    config = Qwen3_5TextConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, layer_types=['full_attention'])
    config._attn_implementation = 'sdpa'
    base = Qwen3_5Attention(config, 0).eval().requires_grad_(False)
    wrap = PrefixAttention(base, 0)
    x = torch.randn(2, 4, 32)
    pos = (torch.ones(2, 4, 8), torch.zeros(2, 4, 8))
    mask = torch.ones(2, 1, 4, 4, dtype=torch.bool).tril()
    torch.testing.assert_close(wrap(x, pos, mask)[0], base(x, pos, mask)[0], atol=0, rtol=0)
    prefix = KVPrefix([0], tokens=3, heads=1, dim=8)
    object.__setattr__(wrap, 'active_prefix', prefix)
    output = wrap(x, pos, mask)[0]
    output.sum().backward()
    assert prefix.values['0'].grad.abs().sum() > 0
    assert all(p.grad is None for p in base.parameters())
    assert not any('keys' in n or 'values' in n for n, _ in wrap.named_parameters())
    # Later input tokens cannot affect earlier outputs (with prefix fixed).
    xx = x.clone()
    xx[:, -1] += 100
    torch.testing.assert_close(wrap(xx, pos, mask)[0][:, :-1], output[:, :-1], atol=1e-6, rtol=1e-6)
    with pytest.raises(ValueError, match='uncached'):
        wrap(x, pos, mask, past_key_values=object())


def test_personal_recipe_keeps_historical_optimization():
    assert CONFIG['steps'] == 100 and CONFIG['batch_size'] == 16 and CONFIG['microbatch'] == 4
    assert CONFIG['learning_rate'] == 1e-4 and CONFIG['seed'] == 42
    assert CONFIG['weight_decay'] == .01 and CONFIG['original_split'] == 'valid'


def test_58_support_is_total_unique_nested_and_query_unchanged():
    pool = [dict(id=f's{i}', label=i % 4) for i in range(58)]
    query = [dict(id=f'q{i}', label=i % 4) for i in range(8)]
    reference = dict(support_ids=[r['id'] for r in pool[:20]], query_ids=[r['id'] for r in query])
    selected = exact_total_support(pool, query, reference, 58)
    assert len(selected) == len({r['id'] for r in selected}) == 58
    assert selected[0]['label'] == 0 and selected[-1]['label'] == 3
    assert set(reference['support_ids']) <= {r['id'] for r in selected}
    with pytest.raises(ValueError, match='57 trials'):
        exact_total_support(pool[:-1], query, reference, 58)
    with pytest.raises(RuntimeError, match='query IDs'):
        exact_total_support(pool, query[::-1], reference, 58)
    with pytest.raises(RuntimeError, match='every previous'):
        exact_total_support(pool, query, dict(reference, support_ids=['not_in_pool']), 58)
    with pytest.raises(RuntimeError, match='exactly once'):
        exact_total_support(pool[:-1] + [pool[0]], query, reference, 58)


def test_paired_reference_rejects_optimization_drift(tmp_path, monkeypatch):
    import json
    import personal.plan as module
    root = tmp_path / 'reference'
    root.mkdir()
    config = copy.deepcopy(CONFIG)
    config['paired_reference'] = 'reference'
    old = copy.deepcopy(config)
    old['shots_per_class'] = 5
    case = dict(key='MI_BCIC_IV2a_A07', order=21)
    (root / 'queue.json').write_text(json.dumps(dict(state='stopped_by_user')))
    (root / 'plan.json').write_text(json.dumps(dict(config=old, cases=[case])))
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'CONFIG', config)
    assert paired_reference()[case['key']]['order'] == 21
    config['learning_rate'] *= 2
    with pytest.raises(RuntimeError, match='learning_rate'):
        paired_reference()
