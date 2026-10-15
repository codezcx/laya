"""Joint source-only training with 60-channel emotion and 3-channel MI."""
import argparse,json,time,hashlib
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .channel_model import ChannelPrototypeClassifier
from .model import masked_loss
from .train import metrics

@torch.no_grad()
def predict(model,x,mask):
 model.eval();return torch.cat([model(x[i:i+192].float(),mask[i:i+192]).float().cpu() for i in range(0,len(x),192)])

def load():
 root=Path('artifacts/multitask_prototypes_v1');allrows=[json.loads(s) for s in (root/'manifest.jsonl').read_text().splitlines()]
 ids=[i for i,r in enumerate(allrows) if r['task'] in (0,1) and r['split'] in ('train','validation')];rows=[allrows[i] for i in ids];index={r['signal_sha256']:i for i,r in enumerate(rows)}
 x=np.zeros((len(rows),62,512),np.float16);valid=np.zeros((len(rows),62),bool);seen=set()
 for shard in range(4):
  d=np.load(root/'features'/f'{shard}.npz')
  for i,feature in zip(d['indices'],d['features']):
   r=allrows[int(i)]
   if r['task']==1 and r['signal_sha256'] in index:
    j=index[r['signal_sha256']];x[j,:3]=feature.reshape(3,512);valid[j,:3]=True;seen.add(j)
 p=Path('artifacts/emotion_source_probe_v4/rows.jsonl');erows=[json.loads(s) for s in p.read_text().splitlines()];digest=hashlib.sha256(p.read_bytes()).hexdigest()
 for shard in range(4):
  r=Path('artifacts/emotion_reve60_v5');done=json.loads((r/f'{shard}.done.json').read_text());assert done['rows_sha256']==digest
  d=np.load(r/f'{shard}.npz')
  for i,feature in zip(d['indices'],d['features']):
   j=index[erows[int(i)]['signal_sha256']];assert j not in seen;x[j,:len(feature)]=feature;valid[j,:len(feature)]=True;seen.add(j)
 assert len(seen)==len(rows) and np.isfinite(x).all()
 return rows,torch.from_numpy(x).cuda(),torch.from_numpy(valid).cuda()

def main(arm,config_path):
 cfg=json.loads(Path(config_path).read_text());out=Path(cfg['output'])/arm;out.mkdir(parents=True,exist_ok=False)
 torch.set_num_threads(4);torch.manual_seed(cfg['seed']);rng=np.random.default_rng(cfg['seed']);
 if cfg.get('mi_full_cache'):
  from .full_source_loader import load_full_sources
  rows,x,valid=load_full_sources(cfg)
 else:rows,x,valid=load()
 if cfg.get('pool')=='spatial':
  from .spatial_model import prepare_spatial
  x,valid,audit=prepare_spatial(rows,x,valid,cfg);(out/'spatial_audit.json').write_text(json.dumps(audit,indent=2))
 if cfg.get('spectral_mode') in ('concat','only'):
  from .spectral_fusion import prepare_spectral
  spectrum,spectral_audit=prepare_spectral(rows,x.device,cfg);(out/'spectral_audit.json').write_text(json.dumps(spectral_audit,indent=2))
  x=torch.cat([x.float(),spectrum],dim=-1) if cfg['spectral_mode']=='concat' else spectrum
 proto=torch.from_numpy(np.load('artifacts/multitask_prototypes_v1/text_prototypes.npy')).cuda()
 # Center every task to remove shared prompt direction, fixed before this run.
 if cfg.get('center_prototypes',True):
  for lo,hi in [(0,3),(3,5),(5,7)]:proto[lo:hi]=F.normalize(proto[lo:hi]-proto[lo:hi].mean(0,keepdim=True),dim=-1)
 if arm=='random_ce':proto=F.normalize(torch.randn_like(proto),dim=-1)
 torch.manual_seed(cfg['seed'])
 if cfg.get('pool')=='spatial':
  from .spatial_model import SpatialPrototypeClassifier
  model=SpatialPrototypeClassifier(proto,arm,len(cfg['anchors']),feature_dim=x.shape[-1]).cuda()
 else:model=ChannelPrototypeClassifier(proto,arm).cuda()
 opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=.01)
 within=[]
 if cfg.get('heldout_mi_dataset'):
  assert not cfg.get('spectral_mode'),'Domain-specific scaler fitting must be implemented before combining these options'
  from .domain_split import domain_indices
  train,val,within=domain_indices(rows,cfg['heldout_mi_dataset'])
 else:
  train=[i for i,r in enumerate(rows) if r['split']=='train'];val=[i for i,r in enumerate(rows) if r['split']=='validation']
 if cfg.get('mi_train_cap_per_subject_class'):
  from .full_source_loader import cap_training_mi
  train=cap_training_mi(rows,train,cfg['mi_train_cap_per_subject_class'])
 vr=[rows[i] for i in val];labels=torch.tensor([r['class_id'] for r in rows],device='cuda');tasks=torch.tensor([r['task'] for r in rows],device='cuda')
 counts=Counter((rows[i]['task'],rows[i]['dataset'],rows[i]['class_id']) for i in train);ds={t:len({rows[i]['dataset'] for i in train if rows[i]['task']==t}) for t in (0,1)}
 w=np.array([1/(counts[(rows[i]['task'],rows[i]['dataset'],rows[i]['class_id'])]*ds[rows[i]['task']]*{0:3,1:2}[rows[i]['task']]) for i in train]);w/=w.sum()
 (out/'run.json').write_text(json.dumps({'config':cfg,'arm':arm,'train_n':len(train),'val_n':len(val),'within_val_n':len(within),'training_datasets':sorted({rows[i]['dataset'] for i in train}),'validation_datasets':sorted({rows[i]['dataset'] for i in val}),'trainable_parameters':sum(p.numel() for p in model.parameters()),'source_only':True},indent=2))
 start=time.time();best=-1;within_best=-1
 for step in range(1,cfg['steps']+1):
  model.train();ids=rng.choice(train,cfg['batch_size'],replace=True,p=w);opt.zero_grad(set_to_none=True)
  logits=model(x[ids].float(),valid[ids]);loss=masked_loss(logits,labels[ids],tasks[ids],'sigmoid' if arm=='text_sigmoid' else 'ce');assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
  if step==1 or step%50==0:
   record={'step':step,'loss':loss.item(),'seconds':time.time()-start}
   with (out/'train.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
   print(record,flush=True)
  if step%250==0 or step==cfg['steps']:
   scores=metrics(vr,predict(model,x[val],valid[val]));scores['step']=step
   with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(scores)+'\n')
   print('validation',step,scores['macro_task_accuracy'],flush=True)
   if within:
    same=metrics([rows[i] for i in within],predict(model,x[within],valid[within]));same['step']=step
    with (out/'within_validation.jsonl').open('a') as f:f.write(json.dumps(same)+'\n')
    if same['macro_task_accuracy']>within_best:
     within_best=same['macro_task_accuracy'];torch.save(model.state_dict(),out/'within_best.pt');(out/'within_best_metrics.json').write_text(json.dumps({'within':same,'cross_at_same_step':scores},indent=2))
   if scores['macro_task_accuracy']>best:
    best=scores['macro_task_accuracy'];torch.save(model.state_dict(),out/'best.pt');(out/'best_metrics.json').write_text(json.dumps(scores,indent=2))
 (out/'complete.json').write_text(json.dumps({'seconds':time.time()-start,'sha256':hashlib.sha256((out/'best.pt').read_bytes()).hexdigest(),'note':'Source validation only; no external or source test evaluated.'},indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',required=True,choices=['text_ce','text_sigmoid','random_ce','learned_ce']);p.add_argument('--config',default='configs/channel_prototypes_v5.json');a=p.parse_args();main(a.arm,a.config)
