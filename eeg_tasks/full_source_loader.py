"""Join audited full MI sources with the fixed emotion development cache."""
import json
from pathlib import Path
import numpy as np
import torch

def load_full_sources(cfg):
    from .channel_train import load
    root=Path(cfg['mi_full_cache']);audit=json.loads((root/'audit.json').read_text())
    if audit['status']!='passed':raise ValueError('Full MI source audit did not pass')
    oldrows,oldx,oldmask=load();emotion=[i for i,r in enumerate(oldrows) if r['task']==0]
    rows=[oldrows[i] for i in emotion];ex=oldx[emotion];em=oldmask[emotion];del oldx,oldmask
    manifest=[json.loads(s) for s in (root/'manifest.jsonl').read_text().splitlines()];keep=np.load(root/'keep_indices.npy');assert len(keep)==audit['kept_n'] and len(set(keep.tolist()))==len(keep)
    features=np.zeros((len(manifest),3,512),np.float16);seen=[]
    for shard in range(4):
        data=np.load(root/f'{shard}.npz');features[data['indices']]=data['features'];seen.extend(data['indices'].tolist())
    assert sorted(seen)==list(range(len(manifest)))
    mirows=[manifest[int(i)] for i in keep];assert all(r['task']==1 and r['split'] in ('train','validation') for r in mirows)
    mx=torch.zeros(len(keep),62,512,device=ex.device,dtype=ex.dtype);mx[:,:3]=torch.from_numpy(features[keep]).to(ex.device);mask=torch.zeros(mx.shape[:2],device=ex.device,dtype=torch.bool);mask[:,:3]=True
    return rows+mirows,torch.cat([ex,mx]),torch.cat([em,mask])

def cap_training_mi(rows,train,limit):
    from collections import defaultdict
    groups=defaultdict(list);keep=[]
    for i in train:
        r=rows[i]
        if r['task']==0:keep.append(i)
        else:groups[(r['dataset'],r['subject'],r['class_id'])].append(i)
    rng=np.random.default_rng(44)
    for key,ids in sorted(groups.items()):keep.extend(rng.choice(ids,min(limit,len(ids)),replace=False).tolist())
    return sorted(keep)
