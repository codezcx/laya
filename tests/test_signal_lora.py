import copy

import pytest
import torch
from torch import nn

from jev_train.lora import LoRALinear, LoRAWeight, inject_lora, inject_tower_lora, is_lora
from jev_train.online_leaf import OnlineLEAFTower
from jev_train.checkpoint import capture_model, restore_model, tensor_fingerprint
from jev_train.runtime import CONFIG


class TinyTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(32, 32)
        self.attn = nn.MultiheadAttention(32, 4, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32))
        self.norm = nn.LayerNorm(32)

    def forward(self, x):
        x = self.projection(x)
        x = self.attn(x, x, x, need_weights=False)[0]
        return self.norm(x + self.ffn(x))


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda',
    marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_tower_fused_attention_and_linear_gradients_and_freeze(device):
    torch.manual_seed(4)
    tower = TinyTower().to(device).eval()
    original = copy.deepcopy(tower)
    targets = inject_tower_lora(tower, CONFIG['lora_rank'], CONFIG['lora_alpha'])
    assert 'attn.in_proj_weight' in targets and 'attn.out_proj.weight' in targets
    assert len(targets) == 5
    x = torch.randn(3, 7, 32, device=device)
    torch.testing.assert_close(tower(x), original(x))
    frozen = {n:p for n,p in tower.named_parameters() if not is_lora(n)}
    before = tensor_fingerprint(frozen)
    optimizer = torch.optim.AdamW([p for p in tower.parameters() if p.requires_grad], lr=1e-3)
    target = torch.randn_like(x)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == 'cuda'):
            loss = (tower(x).float() - target).square().mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for n,p in tower.named_parameters() if is_lora(n))
        assert all(p.grad.norm() > 0 for n,p in tower.named_parameters() if n.endswith('lora_B'))
        optimizer.step()
    assert tensor_fingerprint(frozen) == before
    assert all(p.grad is None for p in frozen.values())


def test_weight_and_forward_lora_are_equivalent():
    linear = nn.Linear(32, 48)
    layer = LoRALinear(copy.deepcopy(linear), 16, 32)
    weight = LoRAWeight(linear.weight, 16, 32)
    with torch.no_grad():
        weight.lora_A.copy_(layer.lora_A)
        layer.lora_B.normal_(0, .01)
        weight.lora_B.copy_(layer.lora_B)
    x = torch.randn(4, 5, 32)
    torch.testing.assert_close(layer(x), torch.nn.functional.linear(x, weight(linear.weight), linear.bias))


def test_tower_stage_switch_never_unfreezes_base():
    holder = OnlineLEAFTower.__new__(OnlineLEAFTower)
    nn.Module.__init__(holder)
    holder.tower = TinyTower()
    inject_tower_lora(holder.tower, 16, 32)
    for enabled in (False, True, False, True):
        holder.set_adaptation_enabled(enabled)
        holder.train()
        assert not holder.tower.training
        assert all(p.requires_grad == (enabled and is_lora(n)) for n,p in holder.tower.named_parameters())


def test_connector_checkpoint_restores_random_frozen_base_and_lora():
    model = nn.Module()
    model.connector = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 64))
    model.connector.requires_grad_(False)
    assert inject_lora(model.connector, 16, 32, {'1', '3'}) == ['1', '3']
    assert all(is_lora(n) for n,p in model.connector.named_parameters() if p.requires_grad)
    before = tensor_fingerprint(model.state_dict())
    state = capture_model(model)
    assert state['format_version'] == 3 and len(state['initialized_frozen']) == 6
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1)
    restore_model(model, state)
    assert tensor_fingerprint(model.state_dict()) == before


def test_shared_rank_is_single_source_of_truth():
    assert CONFIG['lora_rank'] == 16 and CONFIG['lora_alpha'] == 32
    setting = CONFIG['arm_settings']['leaf_tower_mlp']
    assert setting['encoder_adaptation'] == setting['connector_adaptation'] == 'lora'
    assert not any(k in CONFIG for k in ('tower_rank', 'connector_rank'))
