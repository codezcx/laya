"""Matched connector deltas and weight-level Tower LoRA, including fused MHA."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize


class ConnectorDelta(nn.Module):
    def __init__(self, dim=5120, rank=4, scale=1., out_dim=None):
        super().__init__()
        self.a = nn.Parameter(torch.empty(rank, dim))
        self.b = nn.Parameter(torch.zeros(dim if out_dim is None else out_dim, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = scale

    def forward(self, x):
        return F.linear(F.linear(x.float(), self.a), self.b) * self.scale


class UserParameters(nn.Module):
    def __init__(self, config, arm='last_only', dim=5120):
        super().__init__()
        if arm not in config['arms']:
            raise ValueError(f'Unknown personal arm: {arm}')
        # Independent seeded construction preserves the identical second-layer
        # initial A in last-only, both, and Tower+both arms.
        for name, width in [('delta',dim),('first',256)]:
            enabled = (name=='delta' and arm!='first_only') or (name=='first' and arm!='last_only')
            if enabled:
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(config['initialization_seed'])
                    self.add_module(name,ConnectorDelta(width,config['connector_lora_rank'],
                                                        config['connector_lora_scale'],out_dim=dim))


class WeightLoRA(nn.Module):
    """Parametrizing weight also covers MHA's direct weight access (not forward hooks)."""
    def __init__(self, weight, rank, scale):
        super().__init__()
        self.a=nn.Parameter(torch.empty(rank,weight.shape[1],device=weight.device,dtype=torch.float32))
        self.b=nn.Parameter(torch.zeros(weight.shape[0],rank,device=weight.device,dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a,a=math.sqrt(5))
        self.scale=scale

    def forward(self, weight):
        with torch.autocast(weight.device.type,enabled=False):
            delta=self.b@self.a
        return weight+(delta*self.scale).to(weight.dtype)


def install_tower_lora(tower, rank, scale, seed):
    targets=[]
    for name,module in list(tower.named_modules()):
        if isinstance(module,nn.Linear):
            targets.append((name,module,'weight'))
        if isinstance(module,nn.MultiheadAttention):
            if module.in_proj_weight is None:
                raise ValueError('Expected fused QKV in the historical LEAF Tower')
            targets.append((name,module,'in_proj_weight'))
    # Initialize on CPU to avoid dependent CUDA RNG/device indexing.
    records=[]
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        for name,module,parameter in targets:
            weight=getattr(module,parameter)
            low=WeightLoRA(weight.detach().cpu(),rank,scale).to(weight.device)
            parametrize.register_parametrization(module,parameter,low)
            module.parametrizations[parameter].original.requires_grad_(False)
            records.append(dict(module=name,parameter=parameter,shape=list(weight.shape),
                                trainable_parameters=sum(p.numel() for p in low.parameters())))
    return records


def tower_lora_parameters(tower):
    return {name:p for name,p in tower.named_parameters()
            if '.parametrizations.' in '.'+name and (name.endswith('.0.a') or name.endswith('.0.b'))}


def canonical_logits(logits, option_orders):
    out = torch.empty_like(logits)
    order = torch.tensor(option_orders, dtype=torch.long, device=logits.device)
    return out.scatter(1, order, logits)
