"""Online text-conditioned EEG → LEAF → JEV joint supervised fine-tuning."""
import argparse,json,os,time,subprocess,math
from pathlib import Path
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from .leaf_runtime import CONFIG,digest
from .benchmark_data import catalog,metrics,BalancedSampler
from .signal_data import ROOT,load_rows,SignalStore
from .online_leaf import OnlineLEAF
from .task_options import make_question
from .jev import load_jev,JEVBridge
from .distributed import average_gradients,atomic_save
from .lr_schedule import GroupLRSchedule
TASKS=catalog()


def make_variant(task,label,sample,step,seed,variant=None):
    if variant is None:return make_question(task,task['classes'][label]['class_id'],sample,step,seed)
    order=list(range(len(task['classes'])))
    if variant:order.reverse()
    opts='\n'.join(f"{chr(65+i)}) {task['classes'][j]['option']}" for i,j in enumerate(order))
    return dict(suffix=f"\nEEG recording ends.\n[question] {task['question']}\n[options]\n{opts}\n[decision]:",label=order.index(label),num_choices=len(order),option_class_ids=[task['classes'][j]['class_id'] for j in order])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--smoke',action='store_true');parser.add_argument('--evaluate-checkpoint',type=Path);parser.add_argument('--split',choices=['validation','test'],default='validation');a=parser.parse_args()
    rank,world,local=[int(os.environ[k]) for k in ['RANK','WORLD_SIZE','LOCAL_RANK']]
    assert world==CONFIG['world_size'] and CONFIG['global_batch']%world==0
    torch.cuda.set_device(local);torch.set_num_threads(3);dist.init_process_group('nccl',timeout=timedelta(minutes=60));torch.manual_seed(CONFIG['seed'])
    if rank==0:a.out.mkdir(parents=True,exist_ok=False)
    dist.barrier()
    train=load_rows('train',CONFIG['fold']);val=load_rows(a.split,CONFIG['fold'],None if a.evaluate_checkpoint else CONFIG['validation_cap_per_dataset'])
    sampler=BalancedSampler(train);signals=SignalStore()
    text,w,b,tok=load_jev();model=JEVBridge(text,w,b,tok,True,True,head_choices=16,input_dim=256)
    model.add_module('eeg_encoder',OnlineLEAF().cuda())
    groups=[dict(name='connector',params=list(model.connector.parameters()),lr=CONFIG['learning_rates'][0]),dict(name='jev_lora',params=[p for p in model.text.parameters() if p.requires_grad],lr=CONFIG['learning_rates'][1]),dict(name='decision_head',params=[model.head_weight,model.head_bias],lr=CONFIG['learning_rates'][2]),dict(name='eeg_encoder',params=[p for p in model.eeg_encoder.parameters() if p.requires_grad],lr=CONFIG['encoder_learning_rate'])]
    opt=torch.optim.AdamW(groups,weight_decay=.01);active=[p for g in opt.param_groups for p in g['params']]
    total=2 if a.smoke else CONFIG['steps']
    encoder_freeze_steps=1 if a.smoke else CONFIG['encoder_warmup_steps']
    schedule=GroupLRSchedule(opt,total,1 if a.smoke else math.ceil(total*CONFIG['lr_schedule']['warmup_ratio']),CONFIG['lr_schedule']['min_ratio'],encoder_freeze_steps,CONFIG['lr_schedule']['kind'])
    if a.evaluate_checkpoint:
        restored=torch.load(a.evaluate_checkpoint,map_location='cpu',weights_only=False)['trainable'];parameters=dict(model.named_parameters())
        assert set(restored)=={n for n,p in parameters.items() if p.requires_grad}
        with torch.no_grad():
            for n,v in restored.items():parameters[n].copy_(v)
    config=dict(CONFIG,actual_world_size=world,smoke=a.smoke,data_manifest_sha256=digest(ROOT/f"fold{CONFIG['fold']}"/'complete.json'),task_sha256=digest('configs/benchmark_tasks.json'),instructions_sha256=digest('artifacts/benchmark/instructions.npy'),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),code_sha256={str(p):digest(p) for p in Path('eeg_align').glob('*.py')},encoder='online task-conditioned LEAF; warmup then unfreeze',trainable_parameters=sum(p.numel() for p in active),train_epochs_available=len(train),validation_epochs=len(val))
    if rank==0:(a.out/'config.json').write_text(json.dumps(config,indent=2))
    prompts={};start=time.monotonic();best=-1

    def forward(rows,step,variant=None):
        questions=[make_variant(TASKS[r['task']],r['label'],r['id'],step,CONFIG['seed'],variant) for r in rows]
        for q in questions:
            if q['suffix'] not in prompts:prompts[q['suffix']]=torch.tensor(tok.encode(q['suffix'],add_special_tokens=False),device='cuda')
        unfreeze=step>encoder_freeze_steps or a.evaluate_checkpoint is not None
        inputs,valid=model.eeg_encoder(signals.batch(rows),len(rows),unfrozen=unfreeze)
        logits=model(inputs,valid,suffix_ids=[prompts[q['suffix']] for q in questions],num_choices=[q['num_choices'] for q in questions])['logits']
        return logits,torch.tensor([q['label'] for q in questions],device='cuda'),questions

    def semantic(logits,questions,rows):
        values=[]
        for prob,q,r in zip(logits.softmax(-1).detach().cpu().numpy(),questions,rows):
            ids=[c['class_id'] for c in TASKS[r['task']]['classes']];v=[float(prob[q['option_class_ids'].index(i)]) for i in ids];values.append(v+[0.]*(16-len(v)))
        return values

    @torch.no_grad()
    def evaluate(step):
        model.eval();indices=list(range(len(val)))
        if a.smoke:indices=sorted(next(i for i,r in enumerate(val) if r['task']==t and r['label']==c) for t in range(len(TASKS)) for c in range(len(TASKS[t]['classes'])))
        mine=indices[rank::world];result=[]
        for offset in range(0,len(mine),CONFIG['eval_microbatch']):
            ix=mine[offset:offset+CONFIG['eval_microbatch']];rows=[val[i] for i in ix];zz=[]
            for variant in (0,1):
                logits,_,questions=forward(rows,step,variant);zz.append(semantic(logits,questions,rows))
            result.extend((i,p,q) for i,p,q in zip(ix,*zz))
        gathered=[None]*world;dist.all_gather_object(gathered,result);pred=sorted(sum(gathered,[]));assert [r[0] for r in pred]==indices
        rows=[val[i] for i,_,_ in pred];probs=np.array([p for _,p,_ in pred]);alt=np.array([q for _,_,q in pred])
        report=dict(normal=metrics(rows,probs),reversed=metrics(rows,alt))
        report['consistency']={task['task_id']:float(np.mean(probs[[r['task']==t for r in rows]].argmax(1)==alt[[r['task']==t for r in rows]].argmax(1))) for t,task in enumerate(TASKS)}
        report['selection_score']=float(np.mean([m['epoch']['balanced_accuracy'] for m in report['normal'].values()]))
        if rank==0:(a.out/f'{a.split}_{step:04d}.json').write_text(json.dumps(dict(step=step,metrics=report,predictions=[dict(r,probabilities=p.tolist(),alternate_probabilities=q.tolist()) for r,p,q in zip(rows,probs,alt)])))
        return report

    def save(step,improved=False):
        state=dict(step=step,config=config,optimizer=opt.state_dict(),scheduler=schedule.state_dict(),best_score=best,trainable={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad})
        atomic_save(state,a.out/'last.pt')
        if improved:atomic_save(state,a.out/'best.pt')

    if a.evaluate_checkpoint:
        report=evaluate(0)
        if rank==0:(a.out/'complete.json').write_text(json.dumps(dict(checkpoint=str(a.evaluate_checkpoint),split=a.split,metrics=report),indent=2))
        dist.barrier();dist.destroy_process_group();return
    log=(a.out/'metrics.jsonl').open('w',buffering=1) if rank==0 else None
    probe=next(model.eeg_encoder.leaf.tower.tokenizer.parameters())
    for step in range(1,total+1):
        learning_rates=schedule.set_step(step)
        model.train();opt.zero_grad(set_to_none=True)
        ix=sampler(train,step,CONFIG['seed'],CONFIG['global_batch'])[rank::world];stats=torch.zeros((len(TASKS),3),device='cuda')
        before=probe.detach().clone() if a.smoke else None
        for offset in range(0,len(ix),CONFIG['microbatch']):
            ids=ix[offset:offset+CONFIG['microbatch']];rows=[train[i] for i in ids];logits,y,_=forward(rows,step)
            losses=F.cross_entropy(logits,y,reduction='none');(losses.sum()*world/CONFIG['global_batch']).backward()
            for j,r in enumerate(rows):
                stats[r['task'],0]+=losses[j].detach();stats[r['task'],1]+=(logits[j].argmax()==y[j]);stats[r['task'],2]+=1
        unfrozen=model.eeg_encoder.unfrozen;sync=active if unfrozen else [p for g in opt.param_groups[:3] for p in g['params']]
        average_gradients(sync,world)
        encoder_grad=float(torch.stack([p.grad.float().square().sum() for p in opt.param_groups[3]['params']]).sum().sqrt()) if unfrozen else 0.
        grad=torch.nn.utils.clip_grad_norm_(sync,1.);assert torch.isfinite(grad);opt.step();dist.all_reduce(stats)
        if a.smoke:
            delta=float((probe.detach()-before).abs().max())
            assert (delta>0 and encoder_grad>0) if unfrozen else delta==0,(unfrozen,delta,encoder_grad)
        if rank==0:
            sums=stats.sum(0)
            item=dict(step=step,steps=total,loss=float(sums[0]/sums[2]),train_accuracy=float(sums[1]/sums[2]),learning_rates=learning_rates,encoder_unfrozen=unfrozen,encoder_gradient_norm=encoder_grad,gradient_norm=float(grad),elapsed_seconds=time.monotonic()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,train_by_dataset={task['task_id']:dict(n=int(stats[t,2]),loss=float(stats[t,0]/stats[t,2]),accuracy=float(stats[t,1]/stats[t,2])) for t,task in enumerate(TASKS)})
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
