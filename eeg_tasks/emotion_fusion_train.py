"""Matched-montage frequency/REVE fusion with frozen source-only standardization."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .emotion_transfer_probe import readrows,metrics
from .emotion_fusion_cache import ROOT

class Classifier(nn.Module):
 def __init__(self,dim,proto,text=False):
  super().__init__();self.text=text;self.net=nn.Sequential(nn.Linear(dim,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,proto.shape[-1] if text else 3));self.register_buffer('prototypes',proto);self.scale=nn.Parameter(torch.tensor(2.3))
 def forward(self,x):
  z=self.net(x)
  return self.scale.clamp(0,4.6).exp()*(F.normalize(z,dim=-1)@self.prototypes.T) if self.text else z

def load_features(arm):
 manifest=Path('artifacts/emotion_transfer_v16/rows.jsonl');rows=readrows(manifest);spectrum=np.load('artifacts/emotion_transfer_v16/features.npy').reshape(len(rows),-1);reve=np.empty((len(rows),30*512),np.float32);seen=[]
 for i in range(4):
  d=json.loads((ROOT/'features'/f'{i}.done.json').read_text());assert d['manifest_sha256']==hashlib.sha256(manifest.read_bytes()).hexdigest();v=np.load(ROOT/'features'/f'{i}.npz');reve[v['indices']]=v['features'].reshape(-1,30*512);seen.extend(v['indices'].tolist())
 assert sorted(seen)==list(range(len(rows))) and np.isfinite(reve).all()
 x=spectrum if arm=='spectral' else reve if arm=='reve' else np.concatenate([reve,spectrum],1)
 return rows,x

def standardize(x,indices):
 mean=x[indices].mean(0,keepdim=True);std=x[indices].std(0,unbiased=False,keepdim=True).clamp_min(.001)
 return (x-mean)/std,mean,std

@torch.no_grad()
def predict(model,x):
 model.eval();return torch.cat([model(b).cpu() for b in x.split(256)])

def main(arm):
 torch.set_num_threads(4);rows,raw=load_features(arm);raw=torch.from_numpy(raw).cuda();y=torch.tensor([r['class_id'] for r in rows],device='cuda');proto=torch.from_numpy(np.load('artifacts/multitask_prototypes_v1/text_prototypes.npy')[:3]).cuda();proto=F.normalize(proto-proto.mean(0,keepdim=True),dim=-1)
 for source in ['SEED','FACED-NEMAR']:
  out=ROOT/f'{source}_{arm}';out.mkdir(exist_ok=False);torch.manual_seed(42);rng=np.random.default_rng(42);ti=[i for i,r in enumerate(rows) if r['dataset']==source and r['split']=='train'];vi=[i for i,r in enumerate(rows) if r['dataset']==source and r['split']=='validation'];assert not {rows[i]['subject'] for i in ti}&{rows[i]['subject'] for i in vi};x,mean,std=standardize(raw,ti);labels=y[ti].cpu().numpy();w=np.array([1/np.sum(labels==label) for label in labels]);w/=w.sum();model=Classifier(x.shape[-1],proto,arm=='fusion_text').cuda();opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01);best=-1;start=time.monotonic()
  (out/'run.json').write_text(json.dumps({'arm':arm,'source':source,'train_n':len(ti),'validation_n':len(vi),'steps':3000,'seed':42,'learning_rate':1e-4,'batch_size':96,'dim':x.shape[-1],'trainable_parameters':sum(p.numel() for p in model.parameters()),'selection':'source validation subject balanced accuracy'},indent=2))
  for step in range(1,3001):
   model.train();ids=rng.choice(ti,96,replace=True,p=w);opt.zero_grad(set_to_none=True);loss=F.cross_entropy(model(x[ids]),y[ids]);assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
   if step==1 or step%100==0:
    rec={'step':step,'loss':loss.item(),'seconds':time.monotonic()-start}
    with (out/'train.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
    print(source,arm,rec,flush=True)
   if step%250==0:
    m=metrics([rows[i] for i in vi],predict(model,x[vi]).argmax(1).numpy());m['step']=step
    with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(m)+'\n')
    if m['subject_balanced_accuracy']>best:
     best=m['subject_balanced_accuracy'];torch.save({'model':model.state_dict(),'mean':mean.cpu(),'std':std.cpu()},out/'best.pt');(out/'best_metrics.json').write_text(json.dumps(m,indent=2))
  (out/'complete.json').write_text(json.dumps({'sha256':hashlib.sha256((out/'best.pt').read_bytes()).hexdigest(),'seconds':time.monotonic()-start},indent=2))

@torch.no_grad()
def evaluate():
 torch.set_num_threads(4);out=ROOT/'evaluation';out.mkdir(exist_ok=False);arms=['spectral','reve','fusion','fusion_text'];frozen={}
 for source in ['SEED','FACED-NEMAR']:
  for arm in arms:
   p=ROOT/f'{source}_{arm}';h=hashlib.sha256((p/'best.pt').read_bytes()).hexdigest();assert h==json.loads((p/'complete.json').read_text())['sha256'];frozen[p.name]=h
 (out/'frozen.json').write_text(json.dumps(frozen,indent=2));results={}
 for arm in arms:
  rows,raw=load_features(arm)
  for source in ['SEED','FACED-NEMAR']:
   name=f'{source}_{arm}';state=torch.load(ROOT/name/'best.pt',weights_only=True);ids=[i for i,r in enumerate(rows) if r['dataset']!=source];x=(torch.from_numpy(raw[ids])-state['mean'])/state['std'];model=Classifier(x.shape[-1],state['model']['prototypes'],arm=='fusion_text').cuda();model.load_state_dict(state['model']);z=predict(model,x.cuda());rr=[rows[i] for i in ids];m=metrics(rr,z.argmax(1).numpy());m['shuffle']=metrics(rr,z[np.random.default_rng(2026).permutation(len(z))].argmax(1).numpy());results[name]=m;np.save(out/f'{name}_logits.npy',z.numpy());print(name,m['subject_balanced_accuracy'],flush=True)
 (out/'results.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['spectral','reve','fusion','fusion_text']);p.add_argument('--evaluate',action='store_true');a=p.parse_args()
 if a.evaluate:evaluate()
 elif a.arm:main(a.arm)
 else:p.error('Choose arm or evaluate')
