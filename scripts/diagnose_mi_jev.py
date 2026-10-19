"""Matched small-set MI fitting and runtime padding diagnostic; no live-job edits."""
import argparse, json, os, time, hashlib
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from eeg_align.jev import load_jev, JEVBridge
from eeg_align.mi_training_data import Tokens, digest, write_json
from eeg_align.distributed import average_gradients

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--recipe',choices=['mi','tuab'],required=True);p.add_argument('--steps',type=int,default=60);a=p.parse_args()
rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);torch.cuda.set_device(int(os.environ['LOCAL_RANK']));torch.set_num_threads(2)
dist.init_process_group('nccl');torch.manual_seed(42)
if rank==0:a.out.mkdir(parents=True,exist_ok=False)
dist.barrier();data=Tokens('train');rng=np.random.default_rng(42);ids=[]
for ds in sorted({r['dataset'] for r in data.rows}):
 for label in [0,1]:
  candidates=[i for i,r in enumerate(data.rows) if r['dataset']==ds and r['label']==label]
  ids.extend(rng.choice(candidates,4,replace=False).tolist())
if rank==0:
 write_json(a.out/'manifest.json',[data.rows[i] for i in ids])
 write_json(a.out/'config.json',dict(recipe=a.recipe,steps=a.steps,batch=16,seed=42,train_trials=len(ids),code={f:digest(f) for f in ['scripts/diagnose_mi_jev.py','eeg_align/jev.py','eeg_align/mi_training_data.py']},note='Train-only fitting diagnostic; no validation/test selection. MI schedule reproduces first 60 steps of the 2307-step schedule.'))
text,w,b,tok=load_jev();model=JEVBridge(text,w,b,tok,train_lora=True,train_head=True)
groups=[list(model.connector.parameters()),[p for p in model.text.parameters() if p.requires_grad],[model.head_weight,model.head_bias]]
lrs=[3e-5,2e-6,5e-6] if a.recipe=='mi' else [1e-4,5e-6,1e-5]
opt=torch.optim.AdamW([dict(params=g,lr=lr) for g,lr in zip(groups,lrs)],weight_decay=.01)
@torch.no_grad()
def assess(step):
 model.eval();rows=[]
 for i in ids[rank::world]:
  x,m,y=data.batch([i]);prob=model(x,m)['logits'].softmax(-1)[0,1].item();rows.append(dict(id=data.rows[i]['id'],label=int(y.item()),p=prob))
 allrows=[None]*world;dist.all_gather_object(allrows,rows);rows=sum(allrows,[])
 if rank==0:
  q=np.array([r['p'] for r in rows]);y=np.array([r['label'] for r in rows]);write_json(a.out/f'fit_{step:04d}.json',dict(step=step,accuracy=float(((q>.5)==y).mean()),class_mean_p=[float(q[y==c].mean()) for c in [0,1]],loss=float(-(y*np.log(q.clip(1e-8,1))+(1-y)*np.log((1-q).clip(1e-8,1))).mean()),predictions=rows))
@torch.no_grad()
def padding_check():
 model.eval();ordered=sorted(ids,key=lambda i:data.rows[i]['token_length']);pair=[ordered[0],ordered[-1]]
 singles=[]
 for i in pair:
  x,m,y=data.batch([i]);singles.append(model(x,m)['logits'].float())
 x,m,y=data.batch(pair);batch=model(x,m)['logits'].float();single=torch.cat(singles)
 # Invalid padding is removed before assembly; poison it to verify masking.
 poison=x.clone();poison[~m]=100;other=model(poison,m)['logits'].float()
 result=dict(lengths=[data.rows[i]['token_length'] for i in pair],single=single.cpu().tolist(),batch=batch.cpu().tolist(),max_probability_difference=float((single.softmax(-1)-batch.softmax(-1)).abs().max()),max_logit_difference=float((single-batch).abs().max()),poisoned_padding_logit_difference=float((batch-other).abs().max()))
 if rank==0:write_json(a.out/'padding.json',result);print('PADDING',result,flush=True)
padding_check();assess(0);began=time.monotonic()
for step in range(1,a.steps+1):
 # Identical sample order across arms; each step balanced by drawing eight per class.
 rr=np.random.default_rng(1000+step);batch=[]
 for label in [0,1]:batch+=rr.choice([i for i in ids if data.rows[i]['label']==label],8,replace=False).tolist()
 rr.shuffle(batch);model.train();opt.zero_grad(set_to_none=True)
 ratio=min(step/115,1.) if a.recipe=='mi' else 1.
 for g,lr in zip(opt.param_groups,lrs):g['lr']=lr*ratio
 loss_sum=0.
 for i in batch[rank::world]:
  x,m,y=data.batch([i]);z=model(x,m)['logits'];loss=torch.nn.functional.cross_entropy(z,y);(loss*world/len(batch)).backward();loss_sum+=float(loss.detach())
 active=sum(groups,[]);average_gradients(active,world)
 norms=[float(torch.sqrt(sum(p.grad.float().square().sum() for p in g))) for g in groups]
 grad=torch.nn.utils.clip_grad_norm_(active,1.);opt.step()
 total=torch.tensor(loss_sum,device='cuda');dist.all_reduce(total)
 if rank==0:
  row=dict(step=step,loss=total.item()/len(batch),gradient_norm=float(grad),group_gradient_norms=norms,lrs=[g['lr'] for g in opt.param_groups],elapsed=time.monotonic()-began)
  with (a.out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  print('STEP',row,flush=True)
 if step%10==0:assess(step)
if rank==0:write_json(a.out/'complete.json',dict(steps=a.steps))
dist.destroy_process_group()
