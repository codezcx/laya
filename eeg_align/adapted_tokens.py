"""Frozen adapted-REVE tokens, lazily cached in host RAM for JEV training."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import numpy as np
import torch
from transformers import AutoModel
from .mi_training_data import Tokens, preprocess, digest


class AdaptedEncoder:
    def __init__(self,checkpoint):
        ec=json.loads(Path('configs/encoder.json').read_text())
        self.checkpoint=str(checkpoint);self.sha256=digest(checkpoint)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        for split,h in saved['config']['source_manifest_sha256'].items():
            assert digest(Path('datasets/mi_tokens')/f'{split}.jsonl')==h
        self.model,info=AutoModel.from_pretrained(ec['reve_checkpoint'],trust_remote_code=True,local_files_only=True,output_loading_info=True)
        assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
        self.model.load_state_dict(saved['encoder'],strict=True);del saved
        self.model=self.model.cuda().eval().requires_grad_(False)
        self.bank=AutoModel.from_pretrained(ec['positions_checkpoint'],trust_remote_code=True,local_files_only=True).eval().requires_grad_(False)
        self.pool=ThreadPoolExecutor(max_workers=4);self.positions={};self.cache={}

    @torch.no_grad()
    def get(self,records):
        missing=[r for r in records if r['id'] not in self.cache]
        for x,names,r in self.pool.map(preprocess,missing):
            key=tuple(names)
            if key not in self.positions:self.positions[key]=self.bank(names).detach().cuda()[None]
            # Match the original token-cache encoding precision; no train graph.
            z=self.model(torch.from_numpy(x).cuda()[None],self.positions[key]).flatten(1,2)[0]
            assert z.shape==(r['token_length'],512) and torch.isfinite(z).all()
            self.cache[r['id']]=z.half().cpu()
        return [self.cache[r['id']] for r in records]


class AdaptedTokens:
    def __init__(self,split,encoder):
        self.rows=Tokens(split).rows;self.encoder=encoder

    def batch(self,ids,device='cuda'):
        records=[self.rows[int(i)] for i in ids];z=self.encoder.get(records)
        x=torch.nn.utils.rnn.pad_sequence(z,batch_first=True).to(device)
        lengths=torch.tensor([len(v) for v in z],device=device)
        valid=torch.arange(x.shape[1],device=device)[None]<lengths[:,None]
        return x,valid,torch.tensor([r['label'] for r in records],device=device)
