"""Shared LoRA primitives; one rank/alpha applies to Tower, connector and LM."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize


def is_lora(name):
    return name.endswith(('lora_A', 'lora_B'))


class LoRALinear(nn.Module):
    """Ordinary float32 LoRA with a frozen Linear base (PEFT-compatible)."""
    def __init__(self, base, rank, alpha):
        super().__init__()
        if rank <= 0:
            raise ValueError('LoRA rank must be positive')
        self.base = base.requires_grad_(False)
        self.rank, self.scale = rank, alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (F.linear(F.linear(x.float(), self.lora_A), self.lora_B)
                               * self.scale).to(x.dtype)


class LoRAWeight(nn.Module):
    """Weight parametrization also handles MHA's functional in/out projections."""
    def __init__(self, weight, rank, alpha):
        super().__init__()
        if weight.ndim != 2 or rank <= 0:
            raise ValueError('LoRA requires a matrix and positive rank')
        self.rank, self.scale = rank, alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, weight.shape[1], device=weight.device))
        self.lora_B = nn.Parameter(torch.zeros(weight.shape[0], rank, device=weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, original):
        # Disable autocast for the float32 low-rank product, as in LoRALinear.
        with torch.autocast(original.device.type, enabled=False):
            delta = self.lora_B @ self.lora_A * self.scale
        return original + delta.to(original.dtype)


def inject_lora(root, rank, alpha, targets=None):
    if targets is None:
        targets = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}
    selected = [(n, m) for n, m in root.named_modules()
                if isinstance(m, nn.Linear) and n.split('.')[-1] in targets]
    for name, module in selected:
        parent, _, key = name.rpartition('.')
        setattr(root.get_submodule(parent), key, LoRALinear(module, rank, alpha))
    return [n for n, _ in selected]


def inject_tower_lora(root, rank, alpha):
    """All Tower Linear weights and each fused MHA QKV weight; no conv/bias/norm."""
    root.requires_grad_(False)
    selected = []
    for name, module in list(root.named_modules()):
        keys = ['weight'] if isinstance(module, nn.Linear) else []
        if isinstance(module, nn.MultiheadAttention):
            if module.in_proj_weight is None:
                raise ValueError('Expected LEAF fused QKV attention')
            keys.append('in_proj_weight')
        for key in keys:
            weight = getattr(module, key)
            parametrize.register_parametrization(module, key, LoRAWeight(weight, rank, alpha))
            selected.append(f'{name}.{key}')
    if not selected:
        raise ValueError('No Tower LoRA targets found')
    return selected
