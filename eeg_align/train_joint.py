"""Joint task+class text alignment followed by shuffled-position JEV supervision."""
import argparse,json,os,subprocess,time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoModel
from .joint_data import TASKS,OFFSETS,ROOT,read_rows,raw,Sampler
from .semantic_alignment import text_prototypes,task_contrastive_loss,TEXT_MODEL
from .distributed import average_gradients,atomic_save
from .mi_training_data import digest,write_json
from .jev import load_jev,JEVBridge
from .task_options import make_question


def summarise(pred):
    def metric(rr):
        labels=sorted(set(r['label'] for r in rr));y=np.array([r['label'] for r in rr]);z=np.array([np.argmax(r['probabilities']) for r in rr]);k=len(rr[0]['probabilities']);cm=np.zeros((k,k),int)
        np.add.at(cm,(y,z),1)
        return dict(n=len(rr),accuracy=float(np.mean(y==z)),balanced_accuracy=float(np.mean([np.mean(z[y==v]==v) for v in labels])),confusion_matrix=cm.tolist())
    result={}
    for ds in sorted({r['dataset'] for r in pred}):
        rr=[r for r in pred if r['dataset']==ds];parents=defaultdict(list)
        for r in rr:parents[r['parent_id']].append(r)
        trials=[dict(v[0],probabilities=np.mean([r['probabilities'] for r in v],axis=0).tolist()) for v in parents.values()]
        result[ds]=dict(window=metric(rr),trial=metric(trials))
        if 'alternate_probabilities' in rr[0]:
            alt=[dict(r,probabilities=r['alternate_probabilities']) for r in rr]
            result[ds]['alternate_window']=metric(alt)
            result[ds]['permutation_consistency']=float(np.mean([np.argmax(r['probabilities'])==np.argmax(r['alternate_probabilities']) for r in rr]))
    task_scores=[]
    for t in range(3):
        ds=sorted({r['dataset'] for r in pred if r['task']==t})
        task_scores.append(float(np.mean([result[d]['trial']['balanced_accuracy'] for d in ds])))
    return dict(datasets=result,task_balanced_accuracy=dict(zip([t['task_id'] for t in TASKS],task_scores)),selection_score=float(np.mean(task_scores)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['contrast','jev'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--encoder-checkpoint',type=Path);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    rank,world,local=[int(os.environ[k]) for k in ['RANK','WORLD_SIZE','LOCAL_RANK']];assert world==2
    torch.cuda.set_device(local);torch.set_num_threads(3);dist.init_process_group('nccl',timeout=timedelta(minutes=60))
    c=json.loads(Path('configs/mi_emotion_train.json').read_text());torch.manual_seed(c['seed']);np.random.seed(c['seed'])
    if rank==0:a.out.mkdir(parents=True,exist_ok=False)
    dist.barrier();train=read_rows('train');val=read_rows('validation');sampler=Sampler(train);ec=json.loads(Path('configs/encoder.json').read_text())
    enc,info=AutoModel.from_pretrained(ec['reve_checkpoint'],trust_remote_code=True,local_files_only=True,output_loading_info=True)
    assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    if a.encoder_checkpoint:
        saved=torch.load(a.encoder_checkpoint,map_location='cpu',weights_only=False)
        assert saved['config']['manifest_sha256']==digest(ROOT/'complete.json')
        enc.load_state_dict(saved['encoder'],strict=True);del saved
    contrast=a.stage=='contrast';enc=enc.cuda().requires_grad_(contrast);enc.cls_query_token.requires_grad_(False)
    texts=[k['contrast_text'] for t in TASKS for k in t['classes']]
    model=None;head=None;prototypes=None
    if contrast:
        prototypes=text_prototypes(texts=texts);torch.manual_seed(c['seed']);head=torch.nn.Linear(512,prototypes.shape[1]).cuda()
        groups=[dict(params=[p for p in enc.parameters() if p.requires_grad],lr=c['encoder_lr']),dict(params=head.parameters(),lr=c['contrast_head_lr'])]
    else:
        assert a.encoder_checkpoint;enc.eval();torch.manual_seed(c['seed']);text,w,b,tok=load_jev();model=JEVBridge(text,w,b,tok,True,True,head_choices=16)
        groups=[dict(params=model.connector.parameters(),lr=c['jev_learning_rates'][0]),dict(params=[p for p in model.text.parameters() if p.requires_grad],lr=c['jev_learning_rates'][1]),dict(params=[model.head_weight,model.head_bias],lr=c['jev_learning_rates'][2])]
    opt=torch.optim.AdamW(groups,weight_decay=.01);active=[p for g in opt.param_groups for p in g['params']];base_lr=[g['lr'] for g in opt.param_groups]
    config=dict(c,stage=a.stage,smoke=a.smoke,manifest_sha256=digest(ROOT/'complete.json'),encoder_checkpoint=str(a.encoder_checkpoint),encoder_sha256=digest(a.encoder_checkpoint) if a.encoder_checkpoint else None,texts=texts,text_model=TEXT_MODEL,encoder=ec,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),code_sha256={str(p):digest(p) for p in sorted(Path('eeg_align').glob('*.py'))},task_catalog_sha256=digest('configs/mi_emotion_tasks.json'))
    if rank==0:
        write_json(a.out/'config.json',config)
        if contrast:atomic_save(dict(texts=texts,vectors=prototypes.cpu()),a.out/'text_prototypes.pt')
    pool=ThreadPoolExecutor(max_workers=4);cache={};prompt_cache={};began=time.monotonic();best=-1.;log=(a.out/'metrics.jsonl').open('w',buffering=1) if rank==0 else None
    def batches(rr,micro):
        grouped=defaultdict(list)
        for r,(x,p) in zip(rr,pool.map(raw,rr)):
            grouped[(x.shape,p.tobytes())].append((r,x,p))
        for values in grouped.values():
            for i in range(0,len(values),micro):
                v=values[i:i+micro];yield torch.from_numpy(np.stack([x for r,x,p in v])).cuda(),torch.from_numpy(np.stack([p for r,x,p in v])).cuda(),[r for r,x,p in v]
    def embedding(rr):
        missing=[r for r in rr if r['id'] not in cache]
        with torch.no_grad():
            for x,p,records in batches(missing,4):
                z=enc(x,p).flatten(1,2)
                for r,h in zip(records,z):cache[r['id']]=h.half().cpu()
        z=[cache[r['id']] for r in rr];lengths=torch.tensor([len(v) for v in z],device='cuda');x=torch.nn.utils.rnn.pad_sequence(z,batch_first=True).cuda();mask=torch.arange(x.shape[1],device='cuda')[None]<lengths[:,None]
        return x,mask
    def question(r,step,variant=None):
        task=TASKS[r['task']];ids=[k['class_id'] for k in task['classes']]
        if variant is None:q=make_question(task,ids[r['label']],r['id'],step,c['seed'])
        else:
            order=list(range(len(ids))) if variant==0 else list(reversed(range(len(ids))))
            opts='\n'.join(f"{chr(65+i)}) {task['classes'][j]['option']}" for i,j in enumerate(order))
            q=dict(suffix=f"\nEEG recording ends.\n[question] {task['question']}\n[options]\n{opts}\n[decision]:",label=order.index(r['label']),num_choices=len(ids),option_class_ids=[ids[i] for i in order])
        s=q['suffix']
        if s not in prompt_cache:prompt_cache[s]=torch.tensor(tok.encode(s,add_special_tokens=False),device='cuda')
        return q,prompt_cache[s]
    def jev_logits(rr,step,variant=None):
        x,mask=embedding(rr);qq=[question(r,step,variant) for r in rr];counts=[q['num_choices'] for q,s in qq];labels=torch.tensor([q['label'] for q,s in qq],device='cuda')
        z=model(x,mask,suffix_ids=[s for q,s in qq],num_choices=counts)['logits']
        return z,labels,[q for q,s in qq]
    def semantic_probs(z,qq,rr):
        answer=[]
        for prob,q,r in zip(z.softmax(-1).detach().cpu().numpy(),qq,rr):
            ids=[k['class_id'] for k in TASKS[r['task']]['classes']];answer.append([float(prob[q['option_class_ids'].index(v)]) for v in ids])
        return answer
    @torch.no_grad()
    def evaluate(step):
        enc.eval()
        if contrast:head.eval()
        else:model.eval()
        selected=val
        if a.smoke:selected=sum([[r for r in val if r['task']==t][:4] for t in range(3)],[])
        mine=[];rr=selected[rank::world]
        if contrast:
            for start in range(0,len(rr),64):
                for x,p,records in batches(rr[start:start+64],c['contrast_microbatch']):
                    with torch.autocast('cuda',dtype=torch.bfloat16):z=head(enc(x,p).mean((1,2)))
                    for h,r in zip(z,records):
                        t=r['task'];lo=OFFSETS[t];k=len(TASKS[t]['classes']);prob=(F.normalize(h.float(),dim=-1)@prototypes[lo:lo+k].T/c['temperature']).softmax(-1).tolist()
                        mine.append({**{key:r[key] for key in ['id','parent_id','task','dataset','subject','label']},'probabilities':prob})
        else:
            for start in range(0,len(rr),c['jev_microbatch']):
                records=rr[start:start+c['jev_microbatch']];z,y,qq=jev_logits(records,step,0);prob=semantic_probs(z,qq,records);z,y,qq=jev_logits(records,step,1);alt=semantic_probs(z,qq,records)
                for r,v,w in zip(records,prob,alt):mine.append({**{key:r[key] for key in ['id','parent_id','task','dataset','subject','label']},'probabilities':v,'alternate_probabilities':w})
        parts=[None]*world;dist.all_gather_object(parts,mine);pred=sum(parts,[]);assert len(pred)==len(selected) and len({r['id'] for r in pred})==len(pred)
        return summarise(pred),pred
    def save(step,improved):
        saved=dict(step=step,config=config,optimizer=opt.state_dict(),best_score=best)
        if contrast:saved.update(encoder=enc.state_dict(),head=head.state_dict())
        else:saved['trainable']={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
        atomic_save(saved,a.out/'last.pt')
        if improved:atomic_save(saved,a.out/'best.pt')
    total=1 if a.smoke else c[a.stage+'_steps']
    for step in range(1,total+1):
        enc.train(contrast)
        if contrast:head.train()
        else:model.train()
        opt.zero_grad(set_to_none=True)
        for g,lr in zip(opt.param_groups,base_lr):g['lr']=lr*(min(1.,step/c['warmup_steps']) if contrast else 1.)
        indices=sampler.batch(step);rr=[train[i] for i in indices[rank::world]];stats=torch.zeros(3,device='cuda');taskloss={}
        if contrast:
            representations=[];labels=[];tasks=[]
            for x,p,records in batches(rr,c['contrast_microbatch']):
                with torch.autocast('cuda',dtype=torch.bfloat16):z=head(enc(x,p).mean((1,2))).float()
                representations.append(z);labels.extend([r['label']+OFFSETS[r['task']] for r in records]);tasks.extend([r['task'] for r in records])
                for h,r in zip(z.detach(),records):
                    t=r['task'];lo=OFFSETS[t];k=len(TASKS[t]['classes']);pred=(F.normalize(h,dim=-1)@prototypes[lo:lo+k].T).argmax();stats[1]+=int(pred)==r['label']
            loss,parts=task_contrastive_loss(torch.cat(representations),torch.tensor(labels,device='cuda'),torch.tensor(tasks,device='cuda'),prototypes,dict(enumerate(c['task_weights'])),c['temperature']);loss.backward();stats[0]=loss.detach()*len(rr);stats[2]=len(rr);taskloss={TASKS[t]['task_id']:float(v.detach()) for t,v in parts.items()};del representations,parts,loss
        else:
            for start in range(0,len(rr),c['jev_microbatch']):
                records=rr[start:start+c['jev_microbatch']];z,y,qq=jev_logits(records,step);loss=F.cross_entropy(z,y,reduction='sum');(loss*world/64).backward();stats[0]+=loss.detach();stats[1]+=(z.argmax(-1)==y).sum();stats[2]+=len(records)
        average_gradients(active,world);grad=torch.nn.utils.clip_grad_norm_(active,1.);assert torch.isfinite(grad);opt.step();dist.all_reduce(stats)
        if rank==0:
            item=dict(step=step,steps=total,loss=float(stats[0]/stats[2]),train_accuracy=float(stats[1]/stats[2]),task_contrast_losses=taskloss,gradient_norm=float(grad),elapsed_seconds=time.monotonic()-began)
            log.write(json.dumps(item)+'\n');write_json(a.out/'progress.json',item)
            if step==1 or step%10==0:print('TRAIN',item,flush=True)
        if step%c['validation_every']==0 or step==total:
            result,pred=evaluate(step);improved=result['selection_score']>best;best=max(best,result['selection_score'])
            if rank==0:
                log.write(json.dumps(dict(step=step,validation=result))+'\n');write_json(a.out/f'validation_{step:04d}.json',dict(step=step,metrics=result,predictions=pred));save(step,improved)
                if improved:write_json(a.out/'best.json',dict(step=step,metrics=result))
                print('VALIDATION',step,result['task_balanced_accuracy'],flush=True)
            dist.barrier()
    if rank==0:write_json(a.out/'complete.json',dict(steps=total,smoke=a.smoke,seconds=time.monotonic()-began))
    pool.shutdown();dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
