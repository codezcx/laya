"""Per-user rank-4 connector delta; no other method implementations retained."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class ConnectorDelta(nn.Module):
    def __init__(self, dim=5120, rank=4, scale=1.):
        super().__init__()
        self.a = nn.Parameter(torch.empty(rank, dim))
        self.b = nn.Parameter(torch.zeros(dim, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = scale

    def forward(self, x):
        return F.linear(F.linear(x.float(), self.a), self.b) * self.scale


class UserParameters(nn.Module):
    def __init__(self, config, dim=5120):
        super().__init__()
        self.delta = ConnectorDelta(dim, config['connector_lora_rank'], config['connector_lora_scale'])


def canonical_logits(logits, option_orders):
    out = torch.empty_like(logits)
    order = torch.tensor(option_orders, dtype=torch.long, device=logits.device)
    return out.scatter(1, order, logits)
