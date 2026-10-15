"""Common C3/Cz/C4 frozen REVE features for task-free source diagnostics."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel

ROOT=Path('artifacts/taskfree_v15_data')
def readrows(p):return [json.loads(s) for s in p.read_text().splitlines()]

@torch.no_grad()
def main():
 ROOT.mkdir(exist_ok=False);torch.set_num_threads(4);rows=[];blocks=[];provenance={}
 old=Path('artifacts/multitask_prototypes_v1');rr=readrows(old/'manifest.jsonl');xx=np.empty((len(rr),1536),np.float32);seen=[]
 for i in range(4):
  assert (old/'features'/f'{i}.done.json').exists();d=np.load(old/'features'/f'{i}.npz');xx[d['indices']]=d['features'];seen+=d['indices'].tolist()
 assert sorted(seen)==list(range(len(rr)))
 ids=[i for i,r in enumerate(rr) if r['dataset']=='SEED' and r['split'] in ('train','validation')];rows.extend(rr[i] for i in ids);blocks.append(xx[ids]);provenance['seed_manifest']=hashlib.sha256((old/'manifest.jsonl').read_bytes()).hexdigest();del xx
 root=Path('artifacts/mi_full_sources_v12');assert json.loads((root/'audit.json').read_text())['status']=='passed';rr=readrows(root/'manifest.jsonl');xx=np.empty((len(rr),1536),np.float32);seen=[]
 for i in range(4):
  d=np.load(root/f'{i}.npz');xx[d['indices']]=d['features'].reshape(-1,1536);seen+=d['indices'].tolist()
 assert sorted(seen)==list(range(len(rr)));ids=np.load(root/'keep_indices.npy');rows.extend(rr[int(i)] for i in ids);blocks.append(xx[ids]);provenance['mi_manifest']=hashlib.sha256((root/'manifest.jsonl').read_bytes()).hexdigest();del xx
 root=Path('artifacts/faced_source_v14');assert json.loads((root/'prepared_integrity.json').read_text())['status']=='complete';plan=json.loads((root/'cohort.json').read_text());base=json.loads(Path('configs/reve_qwen_mi_multi.json').read_text());model,info=AutoModel.from_pretrained(base['reve_checkpoint'],trust_remote_code=True,local_files_only=True,output_loading_info=True);assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']);model=model.cuda().eval().requires_grad_(False);bank=AutoModel.from_pretrained(base['positions_checkpoint'],trust_remote_code=True,local_files_only=True);pos=bank(['C3','Cz','C4']).cuda();assert pos.shape==(3,3)
 for subject in sorted(plan['train']+plan['validation']):
  p=root/'prepared'/f'sub-{subject:03d}';a=json.loads((p/'audit.json').read_text());x=np.load(p/'signals.npy')[:,[a['channels'].index(c) for c in ['C3','Cz','C4']]];rr=readrows(p/'rows.jsonl');assert len(rr)==len(x)==280
  z=[]
  for start in range(0,len(x),64):
   batch=torch.from_numpy(x[start:start+64]).cuda();tail=(-(600-model.config.patch_size))%(model.config.patch_size-model.config.patch_overlap);batch=torch.nn.functional.pad(batch,(0,tail),mode='replicate');v=model(batch,pos[None].expand(len(batch),-1,-1));assert v.shape[1:]==(3,4,512);z.append(v.mean(2).flatten(1).cpu().numpy())
  rows.extend(rr);blocks.append(np.concatenate(z));print('FACED',subject,flush=True)
 xx=np.concatenate(blocks).astype(np.float32);assert xx.shape==(len(rows),1536) and np.isfinite(xx).all();assert all(r['task'] in (0,1) and r['split'] in ('train','validation') for r in rows)
 keys=[(r['dataset'],r['signal_sha256']) for r in rows];assert len(keys)==len(set(keys))
 np.save(ROOT/'features.npy',xx);(ROOT/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));(ROOT/'complete.json').write_text(json.dumps({'n':len(rows),'channels':['C3','Cz','C4'],'provenance':provenance,'faced_cohort_sha256':hashlib.sha256((root/'cohort.json').read_bytes()).hexdigest(),'features_sha256':hashlib.sha256((ROOT/'features.npy').read_bytes()).hexdigest()},indent=2))
if __name__=='__main__':main()
