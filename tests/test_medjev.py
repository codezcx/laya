from types import SimpleNamespace
import copy

import pytest
import torch
from torch import nn

from jev_train.medjev import BatchedResidualReadout, LoRALinear, inject_lora
from vendor.MedJev.medjev.model import MarkerEmbedding
from vendor.MedJev.medjev.readout import DecisionReadout, ResidualReadout
from jev_train.task_options import make_variant, OPTION_END, DECIDE, QUESTION_END


def readouts():
    torch.manual_seed(42)
    config = SimpleNamespace(num_attention_heads=4, num_key_value_heads=2, head_dim=8, hidden_size=32)
    attn = SimpleNamespace(q_proj=nn.Linear(32, 64, bias=False), k_proj=nn.Linear(32, 16, bias=False),
                           q_norm=nn.LayerNorm(8), k_norm=nn.LayerNorm(8))
    layer = SimpleNamespace(self_attn=attn, input_layernorm=nn.LayerNorm(32))
    original = ResidualReadout(DecisionReadout(layer, config), 8, 42)
    batched = BatchedResidualReadout(DecisionReadout(layer, config), 8, 42)
    batched.load_state_dict(original.state_dict())
    return original, batched


def test_batched_readout_matches_author_per_request():
    original, batched = readouts()
    hidden = torch.randn(3, 12, 32)
    angles = torch.randn(3, 12, 8)
    rope = angles.cos(), angles.sin()
    options = torch.tensor([[2, 5, -1], [3, 6, 9], [1, 4, -1]])
    decisions = torch.tensor([8, 11, 7])
    got = batched(hidden, rope, options, decisions, options >= 0)
    for i in range(3):
        ix = options[i][options[i] >= 0].tolist()
        original.set_context(ix, int(decisions[i]))
        expected = original(hidden[i:i+1], tuple(r[i:i+1] for r in rope))[0, :, 0]
        torch.testing.assert_close(got[i, :len(ix)], expected, rtol=1e-6, atol=1e-6)
        assert torch.isneginf(got[i, len(ix):]).all()


def test_readout_backward_matches_author():
    original, batched = readouts()
    x = torch.randn(1, 12, 32, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    angles = torch.randn(1, 12, 8)
    rope = angles.cos(), angles.sin()
    original.set_context([3, 6, 9], 11)
    loss_a = original(x, rope)[0, :, 0].square().sum()
    loss_b = batched(y, rope, torch.tensor([[3, 6, 9]]), torch.tensor([11])).square().sum()
    ga = torch.autograd.grad(loss_a, x)[0]
    gb = torch.autograd.grad(loss_b, y)[0]
    torch.testing.assert_close(ga, gb, rtol=1e-5, atol=1e-6)


def test_zero_initialized_residual_and_untrained_qk():
    original, batched = readouts()
    assert torch.count_nonzero(batched.head_mlp[-1].weight) == 0
    assert torch.count_nonzero(batched.head_mlp[-1].bias) == 0
    hidden = torch.randn(1, 10, 32)
    rope = torch.ones(1, 10, 8), torch.zeros(1, 10, 8)
    original.set_context([2, 5], 9)
    torch.testing.assert_close(batched(hidden, rope, torch.tensor([[2, 5]]), torch.tensor([9]))[0],
                               original.qk_logits(hidden, rope))


def test_lora_initial_identity_and_rank16_scaling():
    base = nn.Linear(32, 48)
    x = torch.randn(3, 5, 32, requires_grad=True)
    expected = base(x).detach()
    layer = LoRALinear(base, 16, 32)
    torch.testing.assert_close(layer(x), expected)
    assert layer.scale == 2 and not layer.base.weight.requires_grad
    layer(x).square().mean().backward()
    assert x.grad is not None and layer.lora_B.grad.norm() > 0
    assert layer.base.weight.grad is None


def test_lora_matches_peft():
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(123)
    base = nn.Linear(32, 48, bias=False)
    ours = LoRALinear(copy.deepcopy(base), 16, 32)
    peft = get_peft_model(nn.Sequential(nn.Linear(32, 48, bias=False)),
                         LoraConfig(r=16, lora_alpha=32, target_modules=['0'], bias='none'))
    theirs = peft.base_model.model[0]
    with torch.no_grad():
        ours.lora_B.normal_(0, .01)
        theirs.base_layer.weight.copy_(base.weight)
        theirs.lora_A.default.weight.copy_(ours.lora_A)
        theirs.lora_B.default.weight.copy_(ours.lora_B)
    x = torch.randn(2, 5, 32)
    torch.testing.assert_close(ours(x), peft(x), rtol=1e-6, atol=1e-6)


def test_lora_targets_exclude_connector_and_mlp_head():
    root = nn.Module()
    root.q_proj = nn.Linear(32, 32)
    root.head_mlp = nn.Sequential(nn.Linear(32, 8), nn.Linear(8, 1))
    assert inject_lora(root, 16, 32) == ['q_proj']
    assert not isinstance(root.head_mlp[0], LoRALinear)


def test_markers_do_not_update_vocabulary():
    embedding = nn.Embedding(12, 32).requires_grad_(False)
    markers = MarkerEmbedding(embedding, [12, 13, 14], 2)
    ids = torch.tensor([[0, 12, 4, 13, 14]])
    out = markers(ids)
    torch.testing.assert_close(out[0, 1], embedding.weight[2])
    out.square().sum().backward()
    assert embedding.weight.grad is None and markers.markers.grad is not None


def test_prompt_markers_and_target_independence():
    task = dict(task_id='demo', instruction='Motor imagery', question='Which movement?',
                classes=[dict(class_id=str(i), option=f'Class{i}') for i in range(4)])
    questions = [make_variant(task, i, 'trial1', 4, 42) for i in range(4)]
    assert len({q['suffix'] for q in questions}) == 1
    for i, q in enumerate(questions):
        assert q['option_class_ids'][q['label']] == str(i)
        assert q['suffix'].count(OPTION_END) == 4
        assert q['suffix'].count(QUESTION_END) == q['suffix'].count(DECIDE) == 1
        assert q['suffix'].endswith(DECIDE)


def test_reject_option_after_decision():
    _, head = readouts()
    with pytest.raises(ValueError, match='precede'):
        head(torch.randn(1, 8, 32), (torch.ones(1, 8, 8), torch.zeros(1, 8, 8)),
             torch.tensor([[1, 6]]), torch.tensor([5]))


@pytest.mark.parametrize('device,dtype', [('cpu', torch.float32),
    pytest.param('cuda', torch.bfloat16, marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_real_transformer_bridge_forward_backward_and_padding(device, dtype):
    import re
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    from jev_train.medjev import MedJevBridge
    markers = (OPTION_END, DECIDE, QUESTION_END)
    class Tokenizer:
        def encode(self, text, **kwargs):
            pieces = re.split('(' + '|'.join(re.escape(m) for m in markers) + ')', text)
            return [idx for part in pieces for idx in
                    ([100 + markers.index(part)] if part in markers else [ord(c) % 90 for c in part])]
        def apply_chat_template(self, messages, **kwargs):
            return '<user>' + messages[0]['content'] + '</end>'
    config = Qwen3_5TextConfig(vocab_size=104, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        layer_types=['full_attention', 'full_attention'],
        rope_parameters={'rope_type':'default', 'rope_theta':10000., 'partial_rotary_factor':1.,
                         'mrope_section':[1, 1, 2], 'mrope_interleaved':True}, use_cache=False)
    config._attn_implementation = 'sdpa'
    text = Qwen3_5TextModel(config).to(device=device, dtype=dtype)
    head = BatchedResidualReadout(DecisionReadout(text.layers[-1], config), 8, 42)
    text.layers = nn.ModuleList(list(text.layers)[:1])
    text.config.num_hidden_layers = 1
    text.config.layer_types = ['full_attention']
    text.norm = nn.Identity()
    text.embed_tokens = MarkerEmbedding(text.embed_tokens, [100, 101, 102], 2)
    model = MedJevBridge(text, Tokenizer(), head, [100, 101, 102], {})
    task = dict(task_id='toy', question='Left or right?', instruction='MI',
                classes=[dict(class_id='0', option='left'), dict(class_id='1', option='right')])
    prompts = [make_variant(task, i, str(i), 0, 42, 0) for i in (0, 1)]
    x = torch.randn(2, 4, 256, device=device, requires_grad=True)
    valid = torch.tensor([[True, True, False, False], [True, True, True, True]], device=device)
    model.eval()
    together = model(x, valid, prompts)
    separate = torch.cat([model(x[i:i+1], valid[i:i+1], prompts[i:i+1]) for i in range(2)])
    tolerance = .03 if dtype == torch.bfloat16 else 1e-4
    torch.testing.assert_close(together, separate, rtol=tolerance, atol=tolerance/10)
    torch.nn.functional.cross_entropy(together, torch.tensor([0, 1], device=device)).backward()
    assert x.grad[0, :2].norm() > 0 and (x.grad[0, 2:] == 0).all()
    assert model.text.embed_tokens.markers.grad is not None
    assert model.text.embed_tokens.original.weight.grad is None
    groups = model.parameter_groups()
    params = [p for g in groups for p in g['params']]
    assert len(params) == len(set(map(id, params)))
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
