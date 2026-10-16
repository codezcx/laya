"""Source raw-window cache for encoder finetuning, preserving token-cache preprocessing."""
import json
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from transformers import AutoModel
from .cache import ROOT,digest,signals_for_file
from eeg_tasks.channels import CHANNELS,readrows

RAW=Path('artifacts/jev_eeg/source_signals')

def prepare():
    RAW.mkdir(parents=True,exist_ok=False)
    rows=readrows(ROOT/'rows.jsonl');groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r['path']].append(i)
    cfg=json.loads(Path('configs/data.json').read_text())
    signals=np.lib.format.open_memmap(RAW/'signals.npy',mode='w+',dtype=np.float32,shape=(len(rows),30,600))
    def work(ids):return ids,signals_for_file([rows[i] for i in ids],cfg)[0]
    done=0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for ids,x in pool.map(work,groups.values()):
            assert x.shape==(len(ids),3 if rows[ids[0]]['class_id']>=3 else 30,600)
            assert np.isfinite(x).all()
            signals[ids]=0;signals[np.array(ids),:x.shape[1]]=x
            done+=len(ids)
            if done%1000<len(ids):print('Prepared signals',done,len(rows),flush=True)
    assert done==len(rows);signals.flush();del signals
    (RAW/'complete.json').write_text(json.dumps(dict(n=done,manifest_sha256=digest(ROOT/'rows.jsonl'),
        signals_sha256=digest(RAW/'signals.npy'),preprocessing='Identical signals_for_file() as frozen token extraction'),indent=2))

class SignalCache:
    def __init__(self):
        audit=json.loads((RAW/'complete.json').read_text());assert audit['manifest_sha256']==digest(ROOT/'rows.jsonl')
        self.x=np.load(RAW/'signals.npy',mmap_mode='r');self.rows=readrows(ROOT/'rows.jsonl')
        self.channels=np.zeros(len(self.rows),dtype=np.int64)
        for shard in range(4):
            folder=ROOT/str(shard)
            self.channels[np.load(folder/'indices.npy')]=np.load(folder/'lengths.npy')//4
    def batch(self,ids,device):
        y=torch.tensor([self.rows[int(i)]['class_id'] for i in ids],device=device)
        return torch.tensor(np.array(self.x[ids]),device=device),torch.tensor(self.channels[ids],device=device),y

class SignalEncoder(torch.nn.Module):
    def __init__(self,device='cuda',trainable=True):
        super().__init__();cfg=json.loads(Path('configs/encoder.json').read_text())
        self.encoder,audit=AutoModel.from_pretrained(cfg['reve_checkpoint'],trust_remote_code=True,local_files_only=True,output_loading_info=True)
        assert not any(audit.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
        self.encoder.to(device).requires_grad_(trainable)
        bank=AutoModel.from_pretrained(cfg['positions_checkpoint'],trust_remote_code=True,local_files_only=True)
        self.register_buffer('mi_positions',bank(['C3','Cz','C4']).detach().to(device))
        self.register_buffer('emotion_positions',bank(CHANNELS).detach().to(device))
    def forward(self,x,channel_counts):
        # Observed channel counts select positions; no labels or task IDs are passed here.
        outputs=[None]*len(x);masks=[None]*len(x)
        for mi,channels,positions in [(True,3,self.mi_positions),(False,30,self.emotion_positions)]:
            ii=torch.where(channel_counts==channels)[0]
            if not len(ii):continue
            xx=x[ii,:channels]
            tail=(-(600-self.encoder.config.patch_size))%(self.encoder.config.patch_size-self.encoder.config.patch_overlap)
            xx=torch.nn.functional.pad(xx,(0,tail),mode='replicate')
            z=self.encoder(xx,positions[None].expand(len(ii),-1,-1)).flatten(1,2)
            valid=torch.arange(120,device=x.device)<z.shape[1]
            z=torch.nn.functional.pad(z,(0,0,0,120-z.shape[1]))
            for k,i in enumerate(ii.tolist()):outputs[i]=z[k];masks[i]=valid
        return torch.stack(outputs),torch.stack(masks)

if __name__=='__main__':prepare()
