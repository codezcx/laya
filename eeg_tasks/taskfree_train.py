"""Five-class task-free training; dataset holdouts never enter selection."""
import argparse,hashlib,json,time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .taskfree_data import ROOT,readrows

class JointClassifier(nn.Module):
 def __init__(self,proto,arm):
  super().__init__();self.arm=arm;self.encoder=nn.Sequential(nn.LayerNorm(1536),nn.Linear(1536,512),nn.GELU(),nn.Dropout(.1));self.output=nn.Linear(512,5 if arm=='ordinary' else proto.shape[-1]);self.register_buffer('prototypes',proto);self.scale=nn.Parameter(torch.tensor(2.3))
 def forward(self,x):
  z=self.output(self.encoder(x))
  return z if self.arm=='ordinary' else self.scale.clamp(0,4.6).exp()*(F.normalize(z,dim=-1)@self.prototypes.T)

def split_indices(rows,held_emotion):
 held={held_emotion,'Lee2019'}
 train=[i for i,r in enumerate(rows) if r['dataset'] not in held and r['split']=='train']
 val=[i for i,r in enumerate(rows) if r['dataset'] not in held and r['split']=='validation']
 test=[i for i,r in enumerate(rows) if r['dataset'] in held]
 assert train and val and test
 assert {rows[i]['dataset'] for i in test}==held
 assert not ({(rows[i]['dataset'],rows[i]['subject']) for i in train}&{(rows[i]['dataset'],rows[i]['subject']) for i in val})
 return train,val,test

def joint_metrics(rows,logits):
 logits=torch.as_tensor(logits);assert logits.shape==(len(rows),5) and torch.isfinite(logits).all()
 y=np.array([r['class_id'] for r in rows]);task=np.array([r['task'] for r in rows]);pred=logits.argmax(1).numpy();prob=logits.softmax(1);tp=torch.stack([prob[:,:3].sum(1),prob[:,3:].sum(1)],1).argmax(1).numpy();oracle=logits.clone();oracle[task==0,3:]=-float('inf');oracle[task==1,:3]=-float('inf');op=oracle.argmax(1).numpy();datasets={}
 for ds in sorted({r['dataset'] for r in rows}):
  subjects={}
  for sub in sorted({r['subject'] for r in rows if r['dataset']==ds}):
   ix=np.array([i for i,r in enumerate(rows) if r['dataset']==ds and r['subject']==sub]);yy=y[ix];classes=np.unique(yy)
   subjects[sub]={'joint_balanced':float(np.mean([np.mean(pred[ix][yy==c]==c) for c in classes])),'oracle_balanced':float(np.mean([np.mean(op[ix][yy==c]==c) for c in classes])),'task_accuracy':float(np.mean(tp[ix]==task[ix])),'joint_accuracy':float(np.mean(pred[ix]==yy))}
  datasets[ds]={k:float(np.mean([v[k] for v in subjects.values()])) for k in next(iter(subjects.values()))};datasets[ds]['subjects']=subjects
 macro={}
 for k in ('joint_balanced','oracle_balanced','task_accuracy','joint_accuracy'):
  macro[k]=float(np.mean([np.mean([v[k] for ds,v in datasets.items() if next(r['task'] for r in rows if r['dataset']==ds)==t]) for t in sorted(set(task))]))
 return {'n':len(rows),'macro_task':macro,'datasets':datasets}

@torch.no_grad()
def predict(model,x):
 model.eval();return torch.cat([model(b).cpu() for b in x.split(512)])

def main(config):
 cfg=json.loads(Path(config).read_text());out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(cfg['seed']);rng=np.random.default_rng(cfg['seed']);assert (ROOT/'complete.json').exists();rows=readrows(ROOT/'rows.jsonl');x=torch.from_numpy(np.load(ROOT/'features.npy')).cuda();train,val,test=split_indices(rows,cfg['heldout_emotion']);y=torch.tensor([r['class_id'] for r in rows],device='cuda')
 # Global centering preserves differences between task descriptions.
 p=torch.from_numpy(np.load('artifacts/multitask_prototypes_v1/text_prototypes.npy')[:5]).cuda();p=F.normalize(p-p.mean(0,keepdim=True),dim=-1);model=JointClassifier(p,cfg['arm']).cuda();opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=.01)
 counts=Counter((rows[i]['task'],rows[i]['dataset'],rows[i]['class_id']) for i in train);ds={t:len({rows[i]['dataset'] for i in train if rows[i]['task']==t}) for t in (0,1)};w=np.array([1/(counts[(rows[i]['task'],rows[i]['dataset'],rows[i]['class_id'])]*ds[rows[i]['task']]*{0:3,1:2}[rows[i]['task']]) for i in train]);w/=w.sum();assert all(np.isclose(sum(ww for i,ww in zip(train,w) if rows[i]['task']==t),.5) for t in (0,1))
 (out/'run.json').write_text(json.dumps({'config':cfg,'train_n':len(train),'val_n':len(val),'heldout_n':len(test),'training_datasets':sorted({rows[i]['dataset'] for i in train}),'heldout_datasets':sorted({rows[i]['dataset'] for i in test}),'data_manifest_sha256':hashlib.sha256((ROOT/'rows.jsonl').read_bytes()).hexdigest(),'selection':'source validation macro-task subject balanced joint accuracy; no task ID in forward or loss'},indent=2));best=-1;start=time.monotonic()
 for step in range(1,cfg['steps']+1):
  model.train();ids=rng.choice(train,cfg['batch_size'],replace=True,p=w);opt.zero_grad(set_to_none=True);loss=F.cross_entropy(model(x[ids]),y[ids]);assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
  if step==1 or step%50==0:
   rec={'step':step,'loss':loss.item(),'seconds':time.monotonic()-start}
   with (out/'train.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
   print(rec,flush=True)
  if step%250==0 or step==cfg['steps']:
   m=joint_metrics([rows[i] for i in val],predict(model,x[val]));m['step']=step
   with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(m)+'\n')
   if m['macro_task']['joint_balanced']>best:
    best=m['macro_task']['joint_balanced'];torch.save(model.state_dict(),out/'best.pt');(out/'best_metrics.json').write_text(json.dumps(m,indent=2))
 (out/'complete.json').write_text(json.dumps({'sha256':hashlib.sha256((out/'best.pt').read_bytes()).hexdigest(),'seconds':time.monotonic()-start},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);main(p.parse_args().config)
