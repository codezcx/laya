"""VideoMAE-only low-rank updates and a locked full-finetuning control audit."""
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .checkpoint import tensor_fingerprint
from .runtime import digest


TARGETS = ('attention.attention.query', 'attention.attention.key',
           'attention.attention.value', 'attention.output.dense',
           'intermediate.dense', 'output.dense')
INITIAL_KEYS = ('common_connector_sha256', 'jev_adapter_head_sha256',
                'initial_encoder_sha256', 'readout_sha256')


class LoRALinear(nn.Module):
    """Unmerged alpha/r * BA; zero B preserves initial outputs and their dtype."""
    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        if not isinstance(base, nn.Linear) or not 0 < rank <= min(base.in_features, base.out_features):
            raise ValueError('LoRA requires a valid rank and a linear base')
        if not math.isfinite(alpha) or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError('Invalid LoRA scale/dropout')
        self.base = base.requires_grad_(False)
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        result = self.base(x)
        delta = F.linear(F.linear(self.dropout(x).to(self.lora_A.dtype), self.lora_A), self.lora_B)
        return result + (delta * self.scale).to(result.dtype)


def is_lora(name):
    return name.rsplit('.', 1)[-1] in {'lora_A', 'lora_B'}


def base_fingerprint(model):
    """Hash original VideoMAE tensors, independent of inserted wrapper names."""
    values = {'parameter:' + n.replace('.base.', '.'): p
              for n, p in model.named_parameters() if not is_lora(n)}
    values.update({'buffer:' + n: b for n, b in model.named_buffers()})
    return tensor_fingerprint(values)


def lora_fingerprint(model):
    return tensor_fingerprint({n: p for n, p in model.named_parameters() if is_lora(n)})


def inject_videomae_lora(model, spec):
    if spec['targets'] != list(TARGETS):
        raise ValueError('VideoMAE LoRA targets must be explicit and complete')
    names = [f'encoder.layer.{i}.{target}' for i in range(model.config.num_hidden_layers)
             for target in TARGETS]
    modules = dict(model.named_modules())
    if set(names) != {n for n, m in modules.items() if isinstance(m, nn.Linear)}:
        raise ValueError('Unexpected VideoMAE linear layout or already injected model')
    for name in names:
        layer = modules[name]
        if not 0 < spec['rank'] <= min(layer.in_features, layer.out_features):
            raise ValueError(f'Invalid LoRA rank for {name}')
    before = base_fingerprint(model)
    model.requires_grad_(False)
    for name in names:
        parent, _, child = name.rpartition('.')
        setattr(model.get_submodule(parent), child,
                LoRALinear(modules[name], spec['rank'], spec['alpha'], spec['dropout']))
    if base_fingerprint(model) != before:
        raise RuntimeError('LoRA injection changed base weights')
    return dict(**spec, modules=names, module_count=len(names),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                base_sha256=before, initialization='Kaiming-uniform A; zero B; no base-weight merge')


def verify_matched_recipe(config):
    reference = config.get('matched_control')
    if reference is None:
        return None
    root = Path(reference['directory'])
    for name, sha in reference['sha256'].items():
        if digest(root / name) != sha:
            raise ValueError(f'Matched control file changed: {name}')
    identity = json.loads((root / 'config.json').read_text())
    old = identity['config']
    ignored = {'run_name', 'arms', 'arm_settings', 'gpus', 'matched_control'}
    if {k:v for k,v in config.items() if k not in ignored} != {k:v for k,v in old.items() if k not in ignored}:
        raise ValueError('Matched control recipe changed beyond adaptation/launch fields')
    for arm in config['arms']:
        setting = config['arm_settings'][arm]
        if {k:v for k,v in setting.items() if k != 'videomae_lora'} != old['arm_settings'][reference['arm']]:
            raise ValueError('Only VideoMAE LoRA may differ from the full-tuning control')
    complete = json.loads((root / 'complete.json').read_text())
    if identity['smoke'] or complete['updates'] != config['steps'] or complete['test_evaluations'] != 0:
        raise ValueError('Matched control is incomplete or used test data')
    return dict(directory=str(root), recipe_matched=True,
                control_validation_bacc=complete['validation']['macro_balanced_accuracy'],
                control_runtime=identity['runtime'],
                initial_state=json.loads((root / 'initial_state.json').read_text()))


def verify_control_initialization(proof, control):
    if control is not None:
        for key in INITIAL_KEYS:
            if proof[key] != control['initial_state'][key]:
                raise RuntimeError(f'Matched control initialization changed: {key}')
