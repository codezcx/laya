"""Frozen v6 exploratory target evaluation; no target fitting or selection."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from scipy.io import loadmat
from transformers import AutoModel
from .external_emotion import labels,SOURCE
from .data import preprocess,normalized_window
from .spatial_model import SpatialPrototypeClassifier,prepare_spatial
from .channel_train import predict
from .train import metrics

ROOT=Path('artifacts/spatial_prototypes_v6')
def frozen():
 cfg=json.loads(Path('configs/spatial_prototypes_v6.json').read_text());hashes={}
 for arm in cfg['arms']:
  p=ROOT/arm;h=hashlib.sha256((p/'best.pt').read_bytes()).hexdigest();assert h==json.loads((p/'complete.json').read_text())['sha256'];hashes[arm]=h
 return cfg,hashes

@torch.no_grad()
def extract(shard):
 cfg,hashes=frozen();out=ROOT/'external_seed_vii';out.mkdir(exist_ok=True)
 if (out/f'{shard}.done.json').exists():raise FileExistsError(out)
 torch.set_num_threads(4);base=json.loads(Path('configs/reve_qwen_mi_multi.json').read_text());prep=json.loads(Path('configs/multitask_prototypes_v1.json').read_text())
 model=AutoModel.from_pretrained(base['reve_checkpoint'],local_files_only=True,trust_remote_code=True).eval().requires_grad_(False).cuda();bank=AutoModel.from_pretrained(base['positions_checkpoint'],local_files_only=True,trust_remote_code=True)
 channels=json.loads(Path('artifacts/emotion_reve60_v5/0.done.json').read_text())['channels'];names=[s.split()[-1] for s in (SOURCE/'channel_62_pos.locs').read_text().splitlines() if s.strip()];picks=[names.index(c) for c in channels];pos=bank(channels).cuda();assert pos.shape==(60,3)
 lab=labels();mapping={'Sad':0,'Neutral':1,'Happy':2};selected={str(k):mapping[v] for k,v in lab.items() if v in mapping};assert len(selected)==32
 rows=[];features=[];start=time.time()
 for subject in range(shard+1,21,4):
  path=SOURCE/'EEG_preprocessed'/f'{subject}.mat';data=loadmat(path,variable_names=list(selected));signals=[];meta=[]
  for trial,y in selected.items():
   x=preprocess(data[trial][picks],200,prep);n=x.shape[-1]//600
   for s in np.linspace(0,n-1,min(n,10),dtype=int)*600:
    signals.append(normalized_window(x[:,s:s+600]));meta.append({'dataset':'SEED-VII-HNS','subject':str(subject),'task':0,'class_id':y,'path':str(path),'trial':trial,'start_sample':int(s)})
  del data
  for begin in range(0,len(signals),24):
   x=torch.from_numpy(np.stack(signals[begin:begin+24])).cuda();stride=model.config.patch_size-model.config.patch_overlap;tail=(-(600-model.config.patch_size))%stride;x=torch.nn.functional.pad(x,(0,tail),mode='replicate');features.append(model(x,pos[None].expand(len(x),-1,-1)).mean(2).cpu().numpy().astype(np.float16))
  rows.extend(meta);print(shard,subject,len(rows),flush=True)
 np.save(out/f'features_{shard}.npy',np.concatenate(features));(out/f'rows_{shard}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));(out/f'{shard}.done.json').write_text(json.dumps({'n':len(rows),'seconds':time.time()-start,'frozen':hashes,'channels':channels},indent=2))

def balanced(rows,logits):
 result=metrics(rows,logits);by={}
 y=np.array([r['class_id'] for r in rows]);task=rows[0]['task'];pred=logits[:,0:3].argmax(-1).numpy() if task==0 else logits[:,3:5].argmax(-1).numpy()+3
 for subject in sorted({r['subject'] for r in rows}):
  ids=np.array([r['subject']==subject for r in rows]);by[subject]=float(np.mean([np.mean(pred[ids&(y==c)]==c) for c in np.unique(y[ids])]))
 vals=np.array(list(by.values()));boot=np.random.default_rng(2026).choice(vals,(3000,len(vals)),replace=True).mean(1);result.update(subject_macro_balanced_accuracy=float(vals.mean()),subject_balanced_95ci=np.quantile(boot,[.025,.975]).tolist(),subject_balanced=by);return result

@torch.no_grad()
def evaluate():
 cfg,hashes=frozen();target=ROOT/'external_results.json'
 if target.exists():raise FileExistsError(target)
 torch.set_num_threads(4);results={}
 for name,folder,channels in [('seed_vii',ROOT/'external_seed_vii',60),('2b',Path('artifacts/multitask_prototypes_v1/external_2b'),3)]:
  rows=[];arrays=[]
  for i in range(4):
   done=json.loads((folder/f'{i}.done.json').read_text())
   if name=='seed_vii':assert done['frozen']==hashes
   arrays.append(np.load(folder/f'features_{i}.npy').reshape(-1,channels,512));rows.extend(json.loads(s) for s in (folder/f'rows_{i}.jsonl').read_text().splitlines())
  x=torch.from_numpy(np.concatenate(arrays)).cuda();mask=torch.ones(x.shape[:2],dtype=torch.bool,device='cuda');x,mask,audit=prepare_spatial(rows,x,mask,cfg);results[name]={}
  for arm in cfg['arms']:
   state=torch.load(ROOT/arm/'best.pt',weights_only=True,map_location='cuda');model=SpatialPrototypeClassifier(state['prototypes'],arm,len(cfg['anchors'])).cuda();model.load_state_dict(state);logits=predict(model,x,mask);np.save(ROOT/f'external_{name}_{arm}_logits.npy',logits.numpy());results[name][arm]=balanced(rows,logits)
  print(name,{a:m['subject_macro_balanced_accuracy'] for a,m in results[name].items()},flush=True)
 target.write_text(json.dumps({'frozen':hashes,'results':results,'note':'Exploratory after prior target exposure. No target fitting/selection. REVE pretraining includes2b; SEED-VII restricted to Happy/Neutral/Sad. Task-conditioned, not automatic task identification.'},indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--shard',type=int,choices=range(4));p.add_argument('--evaluate',action='store_true');a=p.parse_args();evaluate() if a.evaluate else extract(a.shard)
