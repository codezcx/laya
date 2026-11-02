import copy
import json
from pathlib import Path

import pytest
import torch
from transformers import VideoMAEConfig, VideoMAEModel

from eeg_align.checkpoint import capture_model, restore_model
from eeg_align.encoder_lora import (LoRALinear, TARGETS, is_lora, base_fingerprint,
                                   inject_videomae_lora, verify_matched_recipe,
                                   verify_control_initialization)
from eeg_align.online_bsr import OnlineBSR
from eeg_align.runtime import CONFIG


def tiny_video():
    config = VideoMAEConfig(image_size=8, patch_size=4, num_frames=4, tubelet_size=2,
                           hidden_size=8, num_hidden_layers=2, num_attention_heads=2,
                           intermediate_size=16, use_mean_pooling=False)
    config._attn_implementation = 'sdpa'
    return VideoMAEModel(config)


@pytest.mark.parametrize('autocast', [False, True])
def test_zero_initial_delta_preserves_dtype_output_and_input_gradient(autocast):
    torch.manual_seed(1)
    base = torch.nn.Linear(8, 8)
    wrapped = LoRALinear(copy.deepcopy(base), 4, 4, 0)
    x = torch.randn(2, 3, 8, requires_grad=True)
    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=autocast):
        expected, actual = base(x), wrapped(x)
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().mean().backward()
    assert x.grad.abs().sum() > 0
    assert wrapped.lora_B.grad.abs().sum() > 0
    assert not wrapped.lora_A.grad.any()
    assert all(p.grad is None and not p.requires_grad for p in wrapped.base.parameters())


def test_real_videomae_freeze_backward_and_checkpoint_roundtrip(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(9)
    model = tiny_video().eval()
    x = torch.randn(2, 4, 3, 8, 8)
    before = model(x).last_hidden_state.detach()
    spec = dict(rank=4, alpha=4, dropout=0., targets=list(TARGETS))
    audit = inject_videomae_lora(model, spec)
    assert audit['module_count'] == 12
    torch.testing.assert_close(model(x).last_hidden_state, before, rtol=0, atol=0)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    model.train()
    target = torch.randn_like(before)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        (model(x).last_hidden_state * target).mean().backward()
        assert all(p.grad is None for n,p in model.named_parameters() if not is_lora(n))
        optimizer.step()
    assert base_fingerprint(model) == audit['base_sha256']
    assert any(p.abs().sum()>0 for n,p in model.named_parameters() if n.endswith('lora_B'))
    model.eval()
    expected = model(x).last_hidden_state.detach()
    path = tmp_path / 'lora.pt'
    torch.save(capture_model(model), path)
    saved = torch.load(path, weights_only=False)
    assert all(is_lora(n) for n in saved['trainable'])
    with torch.no_grad():
        for n,p in model.named_parameters():
            if is_lora(n): p.add_(.1)
    restore_model(model, saved)
    torch.testing.assert_close(model(x).last_hidden_state, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match='layout'):
        inject_videomae_lora(model, spec)


def test_lora_phase_preserves_renderer_and_sharedscale_training():
    model = OnlineBSR.__new__(OnlineBSR)
    torch.nn.Module.__init__(model)
    model.finetune = model.adaptation_enabled = True
    model.lora_audit = dict(rank=2)
    model.renderer = torch.nn.Module()
    model.renderer.eeg2pix = torch.nn.Linear(4,4)
    model.renderer.pix2eeg = torch.nn.Linear(4,4).requires_grad_(False)
    model.adapter = torch.nn.Linear(4,4)
    model.encoder = torch.nn.Sequential(LoRALinear(torch.nn.Linear(4,4),2,2,0))
    model._encode = lambda x,ch: model.encoder(model.adapter(model.renderer.eeg2pix(x)))
    for _ in range(2):
        model.set_adaptation_enabled(False)
        model.train()
        assert not model(torch.ones(2,4), []).requires_grad
        model.set_adaptation_enabled(True)
        model.train()
        model(torch.ones(2,4), []).square().sum().backward()
        assert model.renderer.eeg2pix.weight.grad.abs().sum() > 0
        assert model.adapter.weight.grad.abs().sum() > 0
        assert model.encoder[0].lora_B.grad.abs().sum() > 0
        assert model.encoder[0].base.weight.grad is None
        assert not model.encoder[0].base.weight.requires_grad
        assert model.renderer.pix2eeg.weight.grad is None
        model.zero_grad(set_to_none=True)


def test_recipe_is_exactly_locked_to_full_finetuning_control():
    control = verify_matched_recipe(CONFIG)
    assert control['recipe_matched']
    for field, value in [('seed', 5), ('steps', 100), ('weight_decay', .1)]:
        changed = copy.deepcopy(CONFIG)
        changed[field] = value
        with pytest.raises(ValueError, match='recipe changed'):
            verify_matched_recipe(changed)
    changed = copy.deepcopy(CONFIG)
    changed['learning_rates']['eeg_encoder'] = 1e-4
    with pytest.raises(ValueError, match='recipe changed'):
        verify_matched_recipe(changed)
    changed = copy.deepcopy(CONFIG)
    changed['arm_settings'][changed['arms'][0]]['encoder_trainable'] = False
    with pytest.raises(ValueError, match='Only VideoMAE'):
        verify_matched_recipe(changed)
    proof = copy.deepcopy(control['initial_state'])
    verify_control_initialization(proof, control)
    proof['readout_sha256'] = 'different'
    with pytest.raises(RuntimeError, match='initialization changed'):
        verify_control_initialization(proof, control)
