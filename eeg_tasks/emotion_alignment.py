"""Class-conditional cross-source adaptation of the final two REVE blocks."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel
from .emotion_transfer_probe import readrows,metrics
ROOT=Path('artifacts/emotion_alignment_v21')
ARMS={'frozen_ce':(False,0.,0.),'tail_ce':(True,0.,0.),'tail_align':(True,.1,0.),'tail_align_spectral':(True,.1,.1)}

def cross_source_loss(z,labels,domains):
 candidates=domains[:,None]!=domains[None,:];positive=candidates&(labels[:,None]==labels[None,:])
 if not positive.any(1).all():raise ValueError('Every anchor needs a same-class other-source positive')
 z=F.normalize(z,dim=1);scores=(z@z.T)/.1;scores=scores.masked_fill(~candidates,-1e4);logp=scores-torch.logsumexp(scores,dim=1,keepdim=True)
 return -(logp*positive).sum(1).div(positive.sum(1)).mean()

class AlignmentModel(nn.Module):
 def __init__(self,reve,tune):
  super().__init__();self.tune=tune;self.layers=reve.transformer.layers[-2:];self.layers.requires_grad_(tune);self.readout=nn.Sequential(nn.LayerNorm(15360),nn.Linear(15360,256),nn.GELU(),nn.Dropout(.1));self.classifier=nn.Linear(256,3);self.projection=nn.Linear(256,128);self.spectral=nn.Linear(256,150)
 def forward(self,x):
  if not self.tune:self.layers.eval()
  x=x.float()
  for attn,ff in self.layers:x=attn(x)+x;x=ff(x)+x
  h=self.readout(x.reshape(len(x),30,4,512).mean(2).flatten(1))
  return self.classifier(h),self.projection(h),self.spectral(h)

def load():
 manifest=Path('artifacts/emotion_transfer_v16/rows.jsonl');rows=readrows(manifest);digest=hashlib.sha256(manifest.read_bytes()).hexdigest();x=np.empty((len(rows),120,512),np.float16);seen=[]
 for shard in range(4):
  d=json.loads((ROOT/'features'/f'{shard}.done.json').read_text());assert d['manifest_sha256']==digest and d['prefix'] and d['fp32_replay_max_abs']<=1e-5;v=np.load(ROOT/'features'/f'{shard}.npz');x[v['indices']]=v['features'];seen.extend(v['indices'].tolist())
 assert sorted(seen)==list(range(len(rows))) and np.isfinite(x).all()
 return rows,torch.from_numpy(x).cuda()

@torch.no_grad()
def evaluate(model,x,rows):
 model.eval();z=torch.cat([model(b)[0].cpu() for b in x.split(48)]);scores={}
 for ds in sorted({r['dataset'] for r in rows}):
  ids=[i for i,r in enumerate(rows) if r['dataset']==ds];scores[ds]=metrics([rows[i] for i in ids],z[ids].argmax(1).numpy())
 return {'datasets':scores,'macro_balanced':float(np.mean([v['subject_balanced_accuracy'] for v in scores.values()]))}

def main(arm):
 out=ROOT/arm;out.mkdir(exist_ok=False);torch.set_num_threads(4);torch.manual_seed(42);rng=np.random.default_rng(42);rows,x=load();base=json.loads(Path('configs/reve_qwen_mi_multi.json').read_text());reve=AutoModel.from_pretrained(base['reve_checkpoint'],trust_remote_code=True,local_files_only=True).eval().requires_grad_(False).cuda();tune,alignment_weight,spectral_weight=ARMS[arm];model=AlignmentModel(reve,tune).cuda();del reve
 labels=torch.tensor([r['class_id'] for r in rows],device='cuda');domains=torch.tensor([int(r['dataset']=='FACED-NEMAR') for r in rows],device='cuda');ti=[i for i,r in enumerate(rows) if r['split']=='train'];vi=[i for i,r in enumerate(rows) if r['split']=='validation'];assert not {(rows[i]['dataset'],rows[i]['subject']) for i in ti}&{(rows[i]['dataset'],rows[i]['subject']) for i in vi}
 spectrum=torch.from_numpy(np.load('artifacts/emotion_transfer_v16/features.npy').reshape(len(rows),150)).cuda();mean=spectrum[ti].mean(0,keepdim=True);std=spectrum[ti].std(0,unbiased=False,keepdim=True).clamp_min(.001);spectrum=(spectrum-mean)/std
 (out/'spectral_standardizer.json').write_text(json.dumps({'mean':mean.cpu().tolist(),'std':std.cpu().tolist(),'fit_rows':len(ti)},indent=2))
 groups={(d,c):[i for i in ti if int(domains[i])==d and int(labels[i])==c] for d in [0,1] for c in [0,1,2]};assert all(groups.values())
 params=[{'params':[p for n,p in model.named_parameters() if not n.startswith('layers.')],'lr':1e-4}]
 if tune:params.append({'params':model.layers.parameters(),'lr':1e-5})
 opt=torch.optim.AdamW(params,weight_decay=.01);best=-1;start=time.monotonic();audit={'arm':arm,'seed':42,'steps':1000,'warmup_ce_steps':100,'batch_size':48,'sampling':'eight per source/class group; equal class/source mass','alignment_weight':alignment_weight,'spectral_weight':spectral_weight,'trainable_encoder_parameters':sum(p.numel() for p in model.layers.parameters() if p.requires_grad),'train_n':len(ti),'val_n':len(vi),'source_only':True,'auxiliary':'standardized log-band fractions, fit train rows only'};(out/'run.json').write_text(json.dumps(audit,indent=2))
 for step in range(1,1001):
  model.train();ids=np.concatenate([rng.choice(g,8,replace=True) for g in groups.values()]);opt.zero_grad(set_to_none=True);logits,z,s=model(x[ids]);ce=F.cross_entropy(logits,labels[ids]);al=cross_source_loss(z,labels[ids],domains[ids]) if alignment_weight else ce.new_zeros(());sp=F.mse_loss(s,spectrum[ids]) if spectral_weight else ce.new_zeros(());factor=min(1.,max(0.,(step-100)/100));loss=ce+factor*(alignment_weight*al+spectral_weight*sp);assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
  if step==1 or step%25==0:
   rec={'step':step,'loss':loss.item(),'ce':ce.item(),'alignment':al.item(),'spectral':sp.item(),'seconds':time.monotonic()-start}
   with (out/'train.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
   print(rec,flush=True)
  if step%100==0:
   result=evaluate(model,x[vi],[rows[i] for i in vi]);result['step']=step
   with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
   print('validation',result['macro_balanced'],flush=True)
   if result['macro_balanced']>best:
    best=result['macro_balanced'];torch.save(model.state_dict(),out/'best.pt');(out/'best_metrics.json').write_text(json.dumps(result,indent=2))
 (out/'complete.json').write_text(json.dumps({'sha256':hashlib.sha256((out/'best.pt').read_bytes()).hexdigest(),'seconds':time.monotonic()-start},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=list(ARMS),required=True);main(p.parse_args().arm)
