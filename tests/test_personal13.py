import pytest
import torch

from personal.common import CONFIG, check_rows
from personal.modules import ConnectorDelta, KVPrefix, PrefixAttention, ResidualAdapter, UserParameters, canonical_logits, support_loss
from personal.plan import build, partitions, select_support
from personal.data import TrialStore, normalize_trials


def test_personal_split_boundary_and_no_query_gradients():
    case = dict(subject='6', original_split='valid', task=0, support_ids=['s'], query_ids=['q'])
    support = dict(id='s', split='valid', subject='6', task=0, record='A06T')
    check_rows([support], case, gradient=True)
    check_rows([dict(support, id='q', record='A06E')], case)
    for row in (dict(support, id='q'), dict(support, split='test'), dict(support, subject='7'),
                dict(support, record='A06E')):
        with pytest.raises(RuntimeError):
            check_rows([row], case, gradient=True)
    with pytest.raises(RuntimeError, match='overlap'):
        check_rows([support], dict(case, query_ids=['s']))


def synthetic_pool(session='A06T'):
    return [dict(id=f'{session}:{i}', label=i%4, record=session, row=i, event=i%48) for i in range(288)]


def test_full_sessions_and_nested_balanced_support():
    pool, query = partitions(synthetic_pool() + synthetic_pool('A06E'))
    assert len(query) == 288
    small = select_support(pool, 4, 5, 42)
    big = select_support(pool, 4, 25, 42)
    assert len(big) == len({r['id'] for r in big}) == 100
    assert {r['id'] for r in small} <= {r['id'] for r in big}
    assert [sum(r['label']==c for r in big) for c in range(4)] == [25]*4
    assert not {r['id'] for r in big} & {r['id'] for r in query}
    with pytest.raises(ValueError, match='288'):
        partitions(pool[:-1] + query)
    with pytest.raises(ValueError, match='needs'):
        select_support(pool, 4, 73, 42)


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
    assert CONFIG['weight_decay'] == .01 and CONFIG['support_total'] == 100
    # Historical bf16 JEV logits depend on inference batch shape; restore 8.
    assert CONFIG['eval_batch'] == 8 and CONFIG['encode_batch'] == 32



def test_trial_normalization_does_not_use_other_trials():
    import numpy as np
    x = np.random.default_rng(42).normal(size=(4,22,800)).astype(np.float32)
    one = normalize_trials(x[:1])
    x[1:] *= 10000
    np.testing.assert_array_equal(one, normalize_trials(x)[:1])
    with pytest.raises(ValueError, match='constant'):
        normalize_trials(np.zeros_like(x))


def test_actual_historical_cohort_is_complete_and_subject_disjoint():
    p = build(TrialStore())
    assert [(c['subject'], c['original_split']) for c in p['cases']] == [('6','valid'),('7','valid'),('8','test'),('9','test')]
    for c in p['cases']:
        assert not c['shared_train_seen_subject']
        assert len(c['support_ids']) == 100 and len(c['query_ids']) == 288
        assert c['support_class_counts'] == dict.fromkeys(['0','1','2','3'],25)
