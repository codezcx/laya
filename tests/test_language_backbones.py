"""Small CPU fixtures for the native-Qwen control; no network or test EEG."""
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from eeg_align.jev import LoadedLoRA
from eeg_align.language_backbones import (add_fresh_lora, backbone_spec, choice_token_ids,
                                         verify_native_checkpoint, build_decision_bridge)
from eeg_align.runtime import CONFIG
from scripts.run_benchmark import training_command, validate_resume


def test_native_profile_preserves_shared_recipe_and_jev_default():
    before = copy.deepcopy(CONFIG)
    assert backbone_spec()['kind'] == 'jev'
    profile = backbone_spec('qwen3vl4b')
    assert profile['upstream_repo'] == 'Qwen/Qwen3-VL-4B-Instruct'
    assert profile['lora_rank'] == 32 and profile['lora_alpha'] == 64
    assert CONFIG == before
    with pytest.raises(ValueError, match='Unknown language'):
        backbone_spec('made-up')


def test_native_lora_initially_identical_and_updates_only_low_rank():
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(4, 6), nn.GELU(), nn.Linear(6, 3))
    x = torch.randn(5, 4)
    expected = model(x).detach()
    original = {n: p.detach().clone() for n, p in model.named_parameters()}
    modules = add_fresh_lora(model, 2, 4, ['0', '2'])
    assert modules == ['0', '2']
    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
    loss = model(x).square().mean()
    loss.backward()
    for module in (model[0], model[2]):
        assert isinstance(module, LoadedLoRA)
        assert module.base.weight.grad is None
        assert module.lora_B.grad.abs().sum() > 0
    torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.01).step()
    assert not torch.equal(model(x), expected)
    for name, value in original.items():
        layer, key = name.split('.')
        torch.testing.assert_close(getattr(model[int(layer)].base, key), value, rtol=0, atol=0)
    with pytest.raises(ValueError, match='do not exist'):
        add_fresh_lora(nn.Linear(2, 2), 2, 4, ['q_proj'])


def test_official_content_check_rejects_tampering(tmp_path):
    data = json.dumps({'weight_map': {'a': 'model-1.safetensors'}}).encode()
    (tmp_path/'model.safetensors.index.json').write_bytes(data)
    (tmp_path/'model-1.safetensors').write_bytes(b'fixture')
    spec = dict(kind='qwen3_vl_native', path=str(tmp_path), sha256={
        'model-1.safetensors': hashlib.sha256(b'fixture').hexdigest()}, git_blob_sha1={
        'model.safetensors.index.json': hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()})
    assert len(verify_native_checkpoint(spec)) == 2
    (tmp_path/'model-1.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError, match='weight hash mismatch'):
        verify_native_checkpoint(spec)
    (tmp_path/'model.safetensors.index.json').write_bytes(b'{}')
    with pytest.raises(ValueError, match='metadata hash mismatch'):
        verify_native_checkpoint(spec, weights=False)


class TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(text)-65] if len(text) == 1 and 'A' <= text <= 'P' else [1, 2]


class TinyText(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=12)
        self.embedding = nn.Embedding(32, 12)

    def get_input_embeddings(self):
        return self.embedding

    def gradient_checkpointing_enable(self, **kwargs):
        pass

    def forward(self, inputs_embeds, **kwargs):
        return SimpleNamespace(last_hidden_state=inputs_embeds.cumsum(1))


def test_dynamic_connector_and_frozen_body_input_gradients():
    torch.manual_seed(4)
    text = TinyText()
    model = build_decision_bridge(text, torch.randn(16, 12), torch.zeros(16), TinyTokenizer(),
                      train_head=True, head_choices=16, input_dim=4)
    assert model.connector[1].weight.shape == (12, 4)
    assert model.connector[3].weight.shape == (12, 12)
    tokens = torch.randn(2, 3, 4, requires_grad=True)
    result = model(tokens, torch.ones(2, 3, dtype=torch.bool),
                   labels=torch.tensor([0, 2]), num_choices=[2, 4])
    assert result['logits'].shape == (2, 4)
    assert torch.isneginf(result['logits'][0, 2:]).all()
    result['loss'].backward()
    assert tokens.grad.abs().sum() > 0
    assert model.connector[1].weight.grad.abs().sum() > 0
    assert text.embedding.weight.grad is None
    assert model.head_weight.grad is not None


def test_verbalizers_and_launcher_forward_backbone(tmp_path):
    assert choice_token_ids(TinyTokenizer()) == list(range(16))
    bad = SimpleNamespace(encode=lambda *args, **kwargs: [0, 1])
    with pytest.raises(ValueError, match='single tokens'):
        choice_token_ids(bad)
    command = training_command('python', [0, 1, 2, 3], 'leaf_infonce_mlp', 'out',
                               backbone='qwen3vl4b')
    assert command[command.index('--backbone')+1] == 'qwen3vl4b'
    directory = tmp_path/'train'/'leaf_infonce_mlp'
    directory.mkdir(parents=True)
    (directory/'config.json').write_text(json.dumps(dict(config=CONFIG, arm='leaf_infonce_mlp',
        smoke=False, language_backbone='qwen3vl4b', code_sha256={})))
    for name in ('last.pt', 'best.pt'):
        (directory/name).write_bytes(b'fixture')
    validate_resume(tmp_path, 'leaf_infonce_mlp', 'qwen3vl4b')
    with pytest.raises(ValueError, match='backbone mismatch'):
        validate_resume(tmp_path, 'leaf_infonce_mlp', 'jev')
