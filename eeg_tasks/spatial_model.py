"""Fixed coordinate-weighted anchors retain ordered spatial information."""
import torch
from torch import nn
from torch.nn import functional as F


def spatial_pool(tokens,positions,anchors,sigma=.02):
    if sigma<=0:raise ValueError('sigma must be positive')
    if tokens.shape[-2]!=positions.shape[0]:raise ValueError('Channel/coordinate count mismatch')
    weights=torch.softmax(-torch.cdist(anchors.float(),positions.float()).square()/(2*sigma*sigma),dim=-1)
    return torch.einsum('ac,bcd->bad',weights,tokens.float())

class SpatialPrototypeClassifier(nn.Module):
    def __init__(self,prototypes,arm,n_anchors):
        super().__init__();self.arm=arm
        self.encoder=nn.Sequential(nn.LayerNorm(n_anchors*512),nn.Linear(n_anchors*512,512),nn.GELU(),nn.Dropout(.1))
        self.output=nn.Linear(512,7 if arm=='learned_ce' else prototypes.shape[1])
        self.register_buffer('prototypes',F.normalize(prototypes.float(),dim=-1))
        self.log_scale=nn.Parameter(torch.tensor(2.3));self.bias=nn.Parameter(torch.tensor(-2.))
    def forward(self,x,valid):
        if not valid.all():raise ValueError('Spatial anchors must all be populated')
        z=self.output(self.encoder(x.flatten(1)))
        if self.arm=='learned_ce':return z
        logits=self.log_scale.clamp(0,4.6).exp()*(F.normalize(z,dim=-1)@self.prototypes.T)
        return logits+self.bias if self.arm=='text_sigmoid' else logits

@torch.no_grad()
def prepare_spatial(rows,x,valid,cfg):
    import json
    from pathlib import Path
    from transformers import AutoModel
    base=json.loads(Path('configs/reve_qwen_mi_multi.json').read_text())
    bank=AutoModel.from_pretrained(base['positions_checkpoint'],local_files_only=True,trust_remote_code=True)
    names=json.loads(Path('artifacts/emotion_reve60_v5/0.done.json').read_text())['channels']
    anchors=bank(cfg['anchors']).to(x.device);assert len(anchors)==len(cfg['anchors'])
    output=torch.empty(len(x),len(anchors),512,device=x.device,dtype=x.dtype)
    for task,channels in [(0,names),(1,['C3','Cz','C4'])]:
        ids=[i for i,r in enumerate(rows) if r['task']==task]
        positions=bank(channels).to(x.device);assert len(positions)==len(channels)
        for start in range(0,len(ids),192):
            ii=ids[start:start+192];assert valid[ii,:len(channels)].all() and not valid[ii,len(channels):].any()
            output[ii]=spatial_pool(x[ii,:len(channels)],positions,anchors,cfg['sigma']).to(x.dtype)
    audit={'anchors':cfg['anchors'],'anchor_coordinates':anchors.cpu().tolist(),'sigma':cfg['sigma'],'emotion_channels':names,'mi_channels':['C3','Cz','C4'],'positions_checkpoint':base['positions_checkpoint']}
    return output,torch.ones(output.shape[:2],dtype=torch.bool,device=x.device),audit
