"""Shared signal representation with fixed text class prototypes and masked supervision."""
import torch
from torch import nn
from torch.nn import functional as F

CLASS_TASK=torch.tensor([0,0,0,1,1,2,2])

def task_mask(tasks): return tasks[:,None]==CLASS_TASK.to(tasks.device)[None]
def masked_loss(logits,labels,tasks,kind='ce'):
    valid=task_mask(tasks)
    assert valid.gather(1,labels[:,None]).all()
    if kind=='sigmoid':
        target=F.one_hot(labels,logits.shape[-1]).float()
        loss=F.binary_cross_entropy_with_logits(logits,target,reduction='none')
        # Unknown task labels contribute exactly zero gradients.
        return ((loss*valid).sum(-1)/valid.sum(-1)).mean()
    return F.cross_entropy(logits.masked_fill(~valid,-1e4),labels)

class PrototypeClassifier(nn.Module):
    def __init__(self,prototypes,arm):
        super().__init__();self.arm=arm
        self.encoder=nn.Sequential(nn.LayerNorm(1536),nn.Linear(1536,512),nn.GELU(),nn.Dropout(.1))
        self.projection=nn.Linear(512,prototypes.shape[-1])
        self.register_buffer('prototypes',F.normalize(prototypes.float(),dim=-1))
        self.log_scale=nn.Parameter(torch.tensor(2.3));self.bias=nn.Parameter(torch.tensor(-2.))
        self.classifier=nn.Linear(512,7) if arm=='learned_ce' else None
    def forward(self,x):
        h=self.encoder(x)
        if self.classifier is not None:return self.classifier(h)
        z=F.normalize(self.projection(h),dim=-1)
        logits=self.log_scale.clamp(0,4.6).exp()*(z@self.prototypes.T)
        if self.arm=='text_sigmoid':logits=logits+self.bias
        return logits
