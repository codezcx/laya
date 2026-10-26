"""Trainable signal-to-query path, conditioned on task text without target labels."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .leaf_runtime import CONFIG,UP
from .benchmark_data import catalog
from .normalization import apply_bn_policy
from LEAF import LEAF
from load_config import build_model_config,load_yaml


class OnlineLEAF(torch.nn.Module):
    def __init__(self, bn_policy='updating', eval_mode=False):
        super().__init__()
        self.leaf=LEAF(build_model_config(load_yaml(UP/'configs/LEAF_mpnet.yaml')))
        self.leaf.load_state_dict(torch.load(CONFIG['checkpoint'],map_location='cpu',weights_only=True),strict=True)
        # Reconstruction heads / text-output head do not lie on the query path.
        for name,p in self.leaf.named_parameters():
            if name=='tower.mask_token' or name.startswith(('tower.mask_outProj.','tower.next_outProj.','qformer.output_embed.')):
                p.requires_grad_(False)
        info=json.loads(Path('artifacts/benchmark/instructions.json').read_text());vectors=np.load('artifacts/benchmark/instructions.npy')
        self.register_buffer('instructions',torch.from_numpy(np.stack([vectors[info['texts'].index(t['instruction'])] for t in catalog()])))
        self.unfrozen=False
        self.bn_policy=bn_policy
        self.eval_mode=eval_mode
        if self.eval_mode:self.leaf.eval()
        apply_bn_policy(self.leaf,self.bn_policy)

    def forward(self,batch,count,unfrozen):
        self.unfrozen=bool(unfrozen);self.leaf.train(self.training and self.unfrozen and not self.eval_mode)
        apply_bn_policy(self.leaf,self.bn_policy)
        outputs=[None]*count;lengths=[None]*count
        for tid,indices,pieces in batch:
            inst=self.instructions[tid:tid+1].expand(len(indices),-1);qq=[]
            with torch.set_grad_enabled(torch.is_grad_enabled() and self.unfrozen):
                for piece in pieces:
                    x=torch.from_numpy(piece.copy()).to(device=self.instructions.device)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        _,_,history=self.leaf.forward_with_q(x,inst)
                        q=self.leaf.qformer.final_norm(history[-1])
                    qq.append(q.float())
                q=torch.cat(qq,dim=1)
            for j,index in enumerate(indices):
                lengths[index]=q.shape[1];outputs[index]=F.pad(q[j],(0,0,0,24-q.shape[1]))
        assert all(v is not None for v in outputs)
        tokens=torch.stack(outputs);valid=torch.arange(24,device=tokens.device)[None]<torch.tensor(lengths,device=tokens.device)[:,None]
        return tokens,valid
