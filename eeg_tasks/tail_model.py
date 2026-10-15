"""Trainable last two REVE blocks plus shared coordinate pooling and head."""
import torch
from torch import nn
from .spatial_model import spatial_pool,SpatialPrototypeClassifier

class TailClassifier(nn.Module):
 def __init__(self,reve,prototypes,arm,anchors,positions,tune):
  super().__init__();self.layers=reve.transformer.layers[-2:];self.layers.requires_grad_(tune);self.head=SpatialPrototypeClassifier(prototypes,arm,len(anchors));self.register_buffer('anchors',anchors)
  self.register_buffer('positions_emotion',positions[0]);self.register_buffer('positions_mi',positions[1])
 def encode_tail(self,x):
  for attn,ff in self.layers:x=attn(x)+x;x=ff(x)+x
  return x
 def forward(self,x,task):
  channels=60 if task==0 else 3
  if x.shape[1]!=channels*4:raise ValueError('Incorrect prefix shape')
  z=self.encode_tail(x.float()).reshape(len(x),channels,4,512).mean(2);positions=self.positions_emotion if task==0 else self.positions_mi;z=spatial_pool(z,positions,self.anchors)
  return self.head(z,torch.ones(z.shape[:2],dtype=torch.bool,device=z.device))
