"""Append audited FACED spatial features without changing existing source rows."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch


def append_faced(rows,x,valid,cfg):
 root=Path(cfg['faced_source']);plan=json.loads((root/'cohort.json').read_text());expected=set(map(str,plan['train']+plan['validation']));status=json.loads((root/'cohort_status.json').read_text());assert status['state']=='complete' and not status['failed'];digest=hashlib.sha256((root/'cohort.json').read_bytes()).hexdigest();newrows=[];arrays=[]
 for shard in range(4):
  out=root/'features';done=json.loads((out/f'{shard}.done.json').read_text());assert done['cohort_sha256']==digest and done['anchors']==cfg['anchors'] and done['sigma']==cfg['sigma'];rr=[json.loads(s) for s in (out/f'rows_{shard}.jsonl').read_text().splitlines()];z=np.load(out/f'{shard}.npy');assert z.shape==(len(rr),len(cfg['anchors']),512) and np.isfinite(z).all();newrows+=rr;arrays.append(z)
 assert len(newrows)==6720 and {r['subject'] for r in newrows}==expected
 assert len({r['signal_sha256'] for r in newrows})==len(newrows),'Duplicate source windows require audit'
 tr={r['subject'] for r in newrows if r['split']=='train'};va={r['subject'] for r in newrows if r['split']=='validation'};assert not tr&va and tr==set(map(str,plan['train'])) and va==set(map(str,plan['validation']))
 assert all(r['dataset']=='FACED-NEMAR' and r['task']==0 and r['class_id'] in [0,1,2] for r in newrows)
 newx=torch.from_numpy(np.concatenate(arrays)).to(x.device,dtype=x.dtype);newvalid=torch.ones(newx.shape[:2],dtype=torch.bool,device=x.device)
 rows,newrows=select_emotion_sources(rows,newrows,cfg['emotion_sources'])
 return rows+newrows,torch.cat([x,newx]),torch.cat([valid,newvalid])


def select_emotion_sources(rows,newrows,mode):
 # Validation cohorts are identical across arms; only training inclusion changes.
 if mode=='seed_only':newrows=[dict(r,split='unused_source_train') if r['split']=='train' else r for r in newrows]
 elif mode=='faced_only':rows=[dict(r,split='unused_source_train') if r['dataset']=='SEED' and r['split']=='train' else r for r in rows]
 elif mode!='seed_faced':raise ValueError(mode)
 return rows,newrows
