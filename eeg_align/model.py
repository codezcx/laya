"""Single EEG representation aligned with complete task/class descriptions."""
import torch
from torch import nn
from torch.nn import functional as F

def contrastive_pair_loss(signal, text, signal_labels, text_labels, temperature=.1, bias=0.):
    """Balanced sigmoid pair supervision; all same-class descriptions are positive.

    Repeated class
    descriptions must not become off-diagonal negatives as in one-to-one CLIP.
    Every row and column needs both positive and negative pairs.
    """
    if temperature<=0 or signal.ndim!=2 or text.ndim!=2 or signal.shape[1]!=text.shape[1]:
        raise ValueError('Invalid embeddings/temperature')
    if signal_labels.shape!=(len(signal),) or text_labels.shape!=(len(text),):
        raise ValueError('Invalid label shape')
    positive=signal_labels[:,None]==text_labels[None,:]
    negative=~positive
    if not all(mask.any(axis).all() for mask in [positive,negative] for axis in [0,1]):
        raise ValueError('Both pair types required for every signal and text anchor')
    scores=F.normalize(signal.float(),dim=-1)@F.normalize(text.float(),dim=-1).T/temperature + bias
    pos=F.softplus(-scores)*positive;neg=F.softplus(scores)*negative
    row=.5*(pos.sum(1)/positive.sum(1)+neg.sum(1)/negative.sum(1))
    col=.5*(pos.sum(0)/positive.sum(0)+neg.sum(0)/negative.sum(0))
    return .5*(row.mean()+col.mean())


class EEGTextAligner(nn.Module):
    def __init__(self, text_vectors, text_labels, control='contrastive'):
        super().__init__()
        self.control=control
        dim=text_vectors.shape[1]
        self.norm=nn.LayerNorm(512)
        self.projector=nn.Sequential(nn.Linear(120*512,307),nn.GELU(),nn.Dropout(.1),nn.Linear(307,dim))
        self.register_buffer('text_vectors',F.normalize(text_vectors.float(),dim=-1))
        self.register_buffer('text_labels',text_labels.long())
        self.log_scale=nn.Parameter(torch.tensor(2.302585))
        self.bias=nn.Parameter(torch.tensor(-2.))
        self.classifier=nn.Linear(dim,5,bias=False) if control=='ce' else None

    def encode(self,tokens,valid):
        if tokens.ndim!=3 or tokens.shape[-1]!=512 or valid.shape!=tokens.shape[:2] or valid.dtype!=torch.bool:
            raise ValueError('Expected B,K,512 tokens and B,K boolean mask')
        if tokens.shape[1]>120 or not valid.any(1).all() or not torch.isfinite(tokens[valid]).all():
            raise ValueError('Invalid token content')
        x=self.norm(tokens.masked_fill(~valid[...,None],0).float()).masked_fill(~valid[...,None],0)
        x=F.pad(x,(0,0,0,120-x.shape[1])).flatten(1)
        return F.normalize(self.projector(x),dim=-1)

    def forward(self,tokens,valid,labels=None):
        z=self.encode(tokens,valid)
        scale=self.log_scale.clamp(0,4.60517).exp()
        if self.classifier is not None:
            logits=scale*F.linear(z,F.normalize(self.classifier.weight,dim=-1))
            loss=None if labels is None else F.cross_entropy(logits,labels)
        else:
            similarities=z@self.text_vectors.T
            logits=torch.stack([similarities[:,self.text_labels==c].mean(1) for c in range(5)],1)*scale
            loss=None if labels is None else contrastive_pair_loss(z,self.text_vectors,labels,self.text_labels,1/scale,self.bias)
        return {'embedding':z,'logits':logits,'loss':loss}
