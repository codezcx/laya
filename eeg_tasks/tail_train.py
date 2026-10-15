"""Source-only controlled adaptation of the last two REVE blocks."""
import argparse,hashlib,json,time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModel
from .tail_model import TailClassifier
from .model import masked_loss
from .train import metrics


def load():
 source=Path('artifacts/multitask_prototypes_v1');allrows=[json.loads(s) for s in (source/'manifest.jsonl').read_text().splitlines()];erows=[json.loads(s) for s in Path('artifacts/emotion_source_probe_v4/rows.jsonl').read_text().splitlines()];result={}
 for task,name,rows in [(0,'emotion',erows),(1,'mi',allrows)]:
  arrays=[];indices=[]
  for shard in range(4):
   root=Path('artifacts/reve_prefix_v8')/name;assert (root/f'{shard}.done.json').exists();d=np.load(root/f'{shard}.npz');arrays.append(d['features']);indices.extend(d['indices'].tolist())
  assert len(indices)==len(set(indices));selected=[rows[i] for i in indices];assert all(r['task']==task and r['split'] in ('train','validation') for r in selected)
  result[task]=(selected,torch.from_numpy(np.concatenate(arrays)).cuda())
 return result

@torch.no_grad()
def evaluate(model,data):
 model.eval();rows=[];logits=[]
 for task,(rr,x) in data.items():
  ids=[i for i,r in enumerate(rr) if r['split']=='validation'];rows.extend(rr[i] for i in ids)
  for start in range(0,len(ids),48):logits.append(model(x[ids[start:start+48]],task).float().cpu())
 return metrics(rows,torch.cat(logits))

def main(name):
 cfg=json.loads(Path('configs/reve_tail_v8.json').read_text());armcfg=cfg['runs'][name];out=Path(cfg['output'])/name;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(cfg['seed']);rng=np.random.default_rng(cfg['seed']);data=load();base=json.loads(Path('configs/reve_qwen_mi_multi.json').read_text());reve=AutoModel.from_pretrained(base['reve_checkpoint'],trust_remote_code=True,local_files_only=True).eval().requires_grad_(False).cuda();bank=AutoModel.from_pretrained(base['positions_checkpoint'],trust_remote_code=True,local_files_only=True)
 channels=json.loads(Path('artifacts/emotion_reve60_v5/0.done.json').read_text())['channels'];positions=[bank(channels).cuda(),bank(['C3','Cz','C4']).cuda()];anchors=bank(cfg['anchors']).cuda();checkpoint=Path('artifacts/spatial_prototypes_v6')/armcfg['head']/'best.pt';state=torch.load(checkpoint,weights_only=True,map_location='cuda');model=TailClassifier(reve,state['prototypes'],armcfg['head'],anchors,positions,armcfg['encoder_lr']>0).cuda();model.head.load_state_dict(state);del reve
 params=[{'params':model.head.parameters(),'lr':cfg['head_lr']}]
 if armcfg['encoder_lr']>0:params.append({'params':model.layers.parameters(),'lr':armcfg['encoder_lr']})
 opt=torch.optim.AdamW(params,weight_decay=.01);samplers={}
 for task,(rows,x) in data.items():
  ids=[i for i,r in enumerate(rows) if r['split']=='train'];counts=Counter((rows[i]['dataset'],rows[i]['class_id']) for i in ids);weights=np.array([1/counts[(rows[i]['dataset'],rows[i]['class_id'])] for i in ids]);weights/=weights.sum();samplers[task]=(ids,weights,torch.tensor([r['class_id'] for r in rows],device='cuda'))
 audit={'config':cfg,'run':name,'initial_head_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'base_checkpoint':base['reve_checkpoint'],'trainable_tail_parameters':sum(p.numel() for p in model.layers.parameters() if p.requires_grad),'source_only':True};(out/'run.json').write_text(json.dumps(audit,indent=2));start=time.time();initial=evaluate(model,data);initial['step']=0;(out/'initial_metrics.json').write_text(json.dumps(initial,indent=2));best=initial['macro_task_accuracy'];torch.save(model.state_dict(),out/'best.pt');(out/'best_metrics.json').write_text(json.dumps(initial,indent=2))
 for step in range(1,cfg['steps']+1):
  model.train();opt.zero_grad(set_to_none=True);lossvalue=0
  for task in (0,1):
   ids,w,y=samplers[task];batch=rng.choice(ids,cfg['batch_per_task'],replace=True,p=w);logits=model(data[task][1][batch],task);loss=.5*masked_loss(logits,y[batch],torch.full((len(batch),),task,device='cuda'),'sigmoid' if armcfg['head']=='text_sigmoid' else 'ce');assert torch.isfinite(loss);loss.backward();lossvalue+=loss.item()
  torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
  if step==1 or step%25==0:
   record={'step':step,'loss':lossvalue,'seconds':time.time()-start}
   with (out/'train.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
   print(record,flush=True)
  if step%cfg['validate_every']==0:
   result=evaluate(model,data);result['step']=step
   with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
   print('validation',step,result['macro_task_accuracy'],flush=True)
   if result['macro_task_accuracy']>best:best=result['macro_task_accuracy'];torch.save(model.state_dict(),out/'best.pt');(out/'best_metrics.json').write_text(json.dumps(result,indent=2))
 (out/'complete.json').write_text(json.dumps({'seconds':time.time()-start,'best_checkpoint_sha256':hashlib.sha256((out/'best.pt').read_bytes()).hexdigest()}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);main(p.parse_args().run)
