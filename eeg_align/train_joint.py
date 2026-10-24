"""JEV adaptation of frozen language-conditioned LEAF queries on 2a and SEED."""
import argparse,json,os,time,subprocess
from pathlib import Path
from collections import defaultdict
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from .leaf_data import load,metrics,TASKS,ROOT,CONFIG,digest
from .task_options import make_question
from .jev import load_jev,JEVBridge
from .distributed import average_gradients,atomic_save


def sample_indices(rows,step,seed=42,batch=48):
 rng=np.random.default_rng(np.random.SeedSequence([seed,step]));indices=[]
 # Equal task weighting; class, subject, parent, then window balancing.
 for task in range(2):
  candidates=[i for i,r in enumerate(rows) if r['task']==task];k=len(TASKS[task]['classes']);order=rng.permutation(k)
  for j in range(batch//2):
   label=int(order[(j+step)%k]);ix=[i for i in candidates if rows[i]['label']==label]
   subjects=sorted({rows[i]['subject'] for i in ix});subject=subjects[int(rng.integers(len(subjects)))];ix=[i for i in ix if rows[i]['subject']==subject]
   parents=sorted({rows[i]['parent_id'] for i in ix});parent=parents[int(rng.integers(len(parents)))];ix=[i for i in ix if rows[i]['parent_id']==parent]
   indices.append(ix[int(rng.integers(len(ix)))])
 return indices


def make_variant(task,label,sample,step,seed,variant=None):
 if variant is None:return make_question(task,task['classes'][label]['class_id'],sample,step,seed)
 order=list(range(len(task['classes'])))
 if variant:order.reverse()
 opts='\n'.join(f"{chr(65+i)}) {task['classes'][j]['option']}" for i,j in enumerate(order))
 return dict(suffix=f"\nEEG recording ends.\n[question] {task['question']}\n[options]\n{opts}\n[decision]:",label=order.index(label),num_choices=len(order),option_class_ids=[task['classes'][j]['class_id'] for j in order])


def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 rank,world,local=[int(os.environ[k]) for k in ['RANK','WORLD_SIZE','LOCAL_RANK']];assert world==2
 torch.cuda.set_device(local);torch.set_num_threads(3);dist.init_process_group('nccl',timeout=timedelta(minutes=60));torch.manual_seed(CONFIG['seed'])
 if rank==0:a.out.mkdir(parents=True,exist_ok=False)
 dist.barrier();train,xt=load('train');val,xv=load('validation')
 text,w,b,tok=load_jev();model=JEVBridge(text,w,b,tok,True,True,head_choices=16,input_dim=256)
 groups=[dict(params=model.connector.parameters(),lr=CONFIG['learning_rates'][0]),dict(params=[p for p in model.text.parameters() if p.requires_grad],lr=CONFIG['learning_rates'][1]),dict(params=[model.head_weight,model.head_bias],lr=CONFIG['learning_rates'][2])]
 opt=torch.optim.AdamW(groups,weight_decay=.01);active=[p for g in opt.param_groups for p in g['params']]
 config=dict(CONFIG,smoke=a.smoke,data_manifest_sha256=digest(ROOT/'complete.json'),task_sha256=digest('configs/mi_emotion_tasks.json'),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),code_sha256={str(p):digest(p) for p in Path('eeg_align').glob('*.py')},encoder='frozen LEAF final-normalized instruction-conditioned queries',trainable_parameters=sum(p.numel() for p in active))
 if rank==0:(a.out/'config.json').write_text(json.dumps(config,indent=2))
 prompts={};start=time.monotonic();best=-1
 def forward(rr,x,step,variant=None):
  qs=[make_variant(TASKS[r['task']],r['label'],r['id'],step,CONFIG['seed'],variant) for r in rr]
  for q in qs:
   if q['suffix'] not in prompts:prompts[q['suffix']]=torch.tensor(tok.encode(q['suffix'],add_special_tokens=False),device='cuda')
  inputs=torch.from_numpy(np.array(x,copy=True)).cuda();valid=torch.ones(inputs.shape[:2],device='cuda',dtype=torch.bool)
  z=model(inputs,valid,suffix_ids=[prompts[q['suffix']] for q in qs],num_choices=[q['num_choices'] for q in qs])['logits']
  return z,torch.tensor([q['label'] for q in qs],device='cuda'),qs
 def semantic(z,qs,rr):
  values=[]
  for prob,q,r in zip(z.softmax(-1).detach().cpu().numpy(),qs,rr):
   ids=[c['class_id'] for c in TASKS[r['task']]['classes']];v=[float(prob[q['option_class_ids'].index(i)]) for i in ids];values.append(v+[0.]*(4-len(v)))
  return values
 @torch.no_grad()
 def evaluate(step):
  model.eval();indices=list(range(len(val)))
  if a.smoke:indices=[next(i for i,r in enumerate(val) if r['task']==t and r['label']==c) for t in range(2) for c in range(len(TASKS[t]['classes']))]
  indices=sorted(indices)
  mine=indices[rank::world];result=[]
  for offset in range(0,len(mine),CONFIG['eval_microbatch']):
   ix=mine[offset:offset+CONFIG['eval_microbatch']];rr=[val[i] for i in ix];zz=[]
   for variant in (0,1):
    z,_,qs=forward(rr,xv[ix],step,variant);zz.append(semantic(z,qs,rr))
   result.extend((i,p,q) for i,p,q in zip(ix,*zz))
  gathered=[None]*world;dist.all_gather_object(gathered,result);pred=sorted(sum(gathered,[]));assert [r[0] for r in pred]==indices
  rr=[val[i] for i,_,_ in pred];probs=np.array([p for _,p,_ in pred]);alt=np.array([q for _,_,q in pred]);report=dict(normal=metrics(rr,probs),reversed=metrics(rr,alt))
  report['consistency']={task['datasets'][0]:float(np.mean(probs[[r['task']==t for r in rr]].argmax(1)==alt[[r['task']==t for r in rr]].argmax(1))) for t,task in enumerate(TASKS)}
  report['selection_score']=float(np.mean([m['trial']['balanced_accuracy'] for m in report['normal'].values()]))
  if rank==0:
   (a.out/f'validation_{step:04d}.json').write_text(json.dumps(dict(step=step,metrics=report,predictions=[dict(r,probabilities=p.tolist(),alternate_probabilities=q.tolist()) for r,p,q in zip(rr,probs,alt)])))
  return report
 def save(step,improved=False):
  state=dict(step=step,config=config,optimizer=opt.state_dict(),best_score=best,trainable={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad})
  atomic_save(state,a.out/'last.pt')
  if improved:atomic_save(state,a.out/'best.pt')
 log=(a.out/'metrics.jsonl').open('w',buffering=1) if rank==0 else None
 total=1 if a.smoke else CONFIG['steps']
 for step in range(1,total+1):
  model.train();opt.zero_grad(set_to_none=True);ix=sample_indices(train,step,CONFIG['seed'],CONFIG['global_batch'])[rank::world];stats=torch.zeros(3,device='cuda')
  for offset in range(0,len(ix),CONFIG['microbatch']):
   ids=ix[offset:offset+CONFIG['microbatch']];rr=[train[i] for i in ids];z,y,_=forward(rr,xt[ids],step);loss=F.cross_entropy(z,y,reduction='sum');(loss*world/CONFIG['global_batch']).backward();stats[0]+=loss.detach();stats[1]+=(z.argmax(-1)==y).sum();stats[2]+=len(ids)
  average_gradients(active,world);grad=torch.nn.utils.clip_grad_norm_(active,1.);assert torch.isfinite(grad);opt.step();dist.all_reduce(stats)
  if rank==0:
   item=dict(step=step,steps=total,loss=float(stats[0]/stats[2]),train_accuracy=float(stats[1]/stats[2]),gradient_norm=float(grad),elapsed_seconds=time.monotonic()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
   log.write(json.dumps(item)+'\n');(a.out/'progress.json').write_text(json.dumps(item,indent=2))
   if step==1 or step%10==0:print('TRAIN',item,flush=True)
  if step==1 or step%CONFIG['checkpoint_every']==0 or step==total:
   if rank==0:save(step)
   dist.barrier()
  if step%CONFIG['validation_every']==0 or step==total:
   report=evaluate(step);improved=report['selection_score']>best;best=max(best,report['selection_score'])
   if rank==0:
    log.write(json.dumps(dict(step=step,validation=report))+'\n');save(step,improved)
    if improved:(a.out/'best.json').write_text(json.dumps(dict(step=step,metrics=report),indent=2))
    print('VALIDATION',step,report,flush=True)
   dist.barrier()
 if rank==0:(a.out/'complete.json').write_text(json.dumps(dict(steps=total,smoke=a.smoke,seconds=time.monotonic()-start)))
 dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
