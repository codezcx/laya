"""Order-preserving, parameter-matched control on the same REVE tokens."""
import torch
from torch import nn
from torch.nn import functional as F


class SignalControl(nn.Module):
    def __init__(self):
        super().__init__()
        # ~18.86M parameters, close to the Qwen connector's ~18.89M.
        # All 120 channel/time positions remain distinct; no mean pooling.
        self.connector = nn.Sequential(nn.LayerNorm(512), nn.Flatten(1),
            nn.Linear(120*512,307), nn.GELU(), nn.Linear(307,5))

    def forward(self,tokens,valid,labels=None):
        if tokens.ndim!=3 or tokens.shape[-1]!=512 or valid.shape!=tokens.shape[:2] or valid.dtype!=torch.bool:
            raise ValueError('Expected B,K,512 tokens and boolean B,K mask')
        if tokens.shape[1]>120 or not valid.any(1).all() or not torch.isfinite(tokens[valid]).all():
            raise ValueError('Invalid valid tokens')
        # Normalize before zeroing padding so learned LayerNorm bias cannot leak
        # through invalid positions; absent channels remain explicitly zero.
        x = self.connector[0](tokens.masked_fill(~valid[...,None],0).float())
        x = x.masked_fill(~valid[...,None],0)
        x = F.pad(x,(0,0,0,120-x.shape[1]))
        logits = self.connector[1:](x)
        return {'logits':logits,'loss':None if labels is None else F.cross_entropy(logits,labels)}
