"""Shared query pooling of variable-count REVE channel tokens."""
import torch
from torch import nn
from torch.nn import functional as F

class ChannelPrototypeClassifier(nn.Module):
    def __init__(self, prototypes, arm, n_queries=4):
        super().__init__();self.arm=arm
        self.norm=nn.LayerNorm(512)
        self.queries=nn.Parameter(torch.randn(n_queries,512)*.02)
        self.pool=nn.MultiheadAttention(512,4,dropout=.1,batch_first=True)
        self.readout=nn.Sequential(nn.LayerNorm(512*n_queries),nn.Linear(512*n_queries,512),nn.GELU(),nn.Dropout(.1))
        self.output=nn.Linear(512,7 if arm=='learned_ce' else prototypes.shape[1])
        self.register_buffer('prototypes',F.normalize(prototypes.float(),dim=-1))
        self.log_scale=nn.Parameter(torch.tensor(2.3));self.bias=nn.Parameter(torch.tensor(-2.))
    def forward(self,x,valid):
        if valid.dtype!=torch.bool or not valid.any(-1).all():raise ValueError('Each sample needs a valid channel')
        x=self.norm(x);q=self.queries[None].expand(len(x),-1,-1)
        pooled,_=self.pool(q,x,x,key_padding_mask=~valid,need_weights=False)
        z=self.output(self.readout((pooled+q).flatten(1)))
        if self.arm=='learned_ce':return z
        logits=self.log_scale.clamp(0,4.6).exp()*(F.normalize(z,dim=-1)@self.prototypes.T)
        return logits+self.bias if self.arm=='text_sigmoid' else logits
