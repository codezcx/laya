"""Evaluate every predeclared seed on immutable external caches; no seed selection."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from .spatial_model import make_spatial_model,prepare_spatial
from .spatial_external import balanced
from .channel_train import predict

@torch.no_grad()
def main(jobs_path='artifacts/spatial_repeats_v7/jobs.json',output='artifacts/spatial_repeats_v7'):
 root=Path(output);root.mkdir(exist_ok=True);out=root/'external_results.json'
 if out.exists():raise FileExistsError(out)
 payload=json.loads(Path(jobs_path).read_text());jobs=payload['jobs'] if isinstance(payload,dict) else payload
 jobs=[dict(j,arm=json.loads(Path(j['config']).read_text())['arms'][0]) for j in jobs];frozen={}
 for job in jobs:
  cfg=json.loads(Path(job['config']).read_text());p=Path(cfg['output'])/job['arm'];digest=hashlib.sha256((p/'best.pt').read_bytes()).hexdigest();assert digest==json.loads((p/'complete.json').read_text())['sha256'];frozen[job['name']]=digest
 (root/'all_frozen.json').write_text(json.dumps(frozen,indent=2));torch.set_num_threads(4);results={}
 for name,folder,channels in [('seed_vii',Path('artifacts/spatial_prototypes_v6/external_seed_vii'),60),('2b',Path('artifacts/multitask_prototypes_v1/external_2b'),3)]:
  rows=[];arrays=[]
  for i in range(4):
   assert (folder/f'{i}.done.json').exists();arrays.append(np.load(folder/f'features_{i}.npy').reshape(-1,channels,512));rows.extend(json.loads(s) for s in (folder/f'rows_{i}.jsonl').read_text().splitlines())
  x=torch.from_numpy(np.concatenate(arrays)).cuda();valid=torch.ones(x.shape[:2],dtype=torch.bool,device='cuda');base=json.loads(Path(jobs[0]['config']).read_text());x,valid,_=prepare_spatial(rows,x,valid,base);results[name]={}
  for job in jobs:
   cfg=json.loads(Path(job['config']).read_text());assert cfg['anchors']==base['anchors'] and cfg['sigma']==base['sigma'];p=Path(cfg['output'])/job['arm'];state=torch.load(p/'best.pt',weights_only=True,map_location='cuda');model=make_spatial_model(state['prototypes'],job['arm'],cfg).cuda();model.load_state_dict(state);logits=predict(model,x,valid);np.save(root/f"{name}_{job['name']}_logits.npy",logits.numpy());results[name][job['name']]=balanced(rows,logits)
  print(name,{k:v['subject_macro_balanced_accuracy'] for k,v in results[name].items()},flush=True)
 out.write_text(json.dumps({'frozen':frozen,'results':results,'jobs':jobs,'note':'All predeclared configurations reported; exploratory repeated-target evaluation, no target model selection. Encoder pretraining includes2b.'},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--jobs',default='artifacts/spatial_repeats_v7/jobs.json');p.add_argument('--output',default='artifacts/spatial_repeats_v7');a=p.parse_args();main(a.jobs,a.output)
