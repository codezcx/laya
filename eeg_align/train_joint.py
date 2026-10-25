"""Online EEG → LEAF → JEV with matched BN controls and complete checkpoints."""
import argparse,json,os,time,subprocess,math,signal
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
from .checkpoint import (seed_rng,capture_rng,restore_rng,capture_model,restore_model,
                         recover_frozen_warmup,tensor_fingerprint)
from .normalization import bn_summary
TASKS=catalog()


def make_variant(task,label,sample,step,seed,variant=None):
    if variant is None:return make_question(task,task['classes'][label]['class_id'],sample,step,seed)
    order=list(range(len(task['classes'])))
    if variant:order.reverse()
    opts='\n'.join(f"{chr(65+i)}) {task['classes'][j]['option']}" for i,j in enumerate(order))
    return dict(suffix=f"\nEEG recording ends.\n[question] {task['question']}\n[options]\n{opts}\n[decision]:",label=order.index(label),num_choices=len(order),option_class_ids=[task['classes'][j]['class_id'] for j in order])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--bn-policy',choices=['updating','frozen'],required=True)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--recover-warmup',action='store_true')
    parser.add_argument('--evaluate-checkpoint',type=Path)
    parser.add_argument('--split',choices=['validation','test'],default='validation')
    a=parser.parse_args()
    if a.resume and (a.smoke or a.evaluate_checkpoint):parser.error('Resume is only for production training')
    if a.recover_warmup and not a.resume:parser.error('--recover-warmup requires --resume')
    rank,world,local=[int(os.environ[k]) for k in ['RANK','WORLD_SIZE','LOCAL_RANK']]
    assert world==CONFIG['world_size'] and CONFIG['global_batch']%world==0
    torch.cuda.set_device(local);torch.set_num_threads(3)
    dist.init_process_group('nccl',timeout=timedelta(minutes=60));seed_rng(CONFIG['seed'])
    if rank==0:a.out.mkdir(parents=True,exist_ok=False)
    dist.barrier()
    stopping=False
    def request_stop(signum,frame):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGINT,request_stop)
    train=load_rows('train',CONFIG['fold'])
    val=load_rows(a.split,CONFIG['fold'],None if a.evaluate_checkpoint else CONFIG['validation_cap_per_dataset'])
    sampler=BalancedSampler(train);signals=SignalStore()
    text,w,b,tok=load_jev();model=JEVBridge(text,w,b,tok,True,True,head_choices=16,input_dim=256)
    model.add_module('eeg_encoder',OnlineLEAF(a.bn_policy).cuda())
    groups=[dict(name='connector',params=list(model.connector.parameters()),lr=CONFIG['learning_rates'][0]),
            dict(name='jev_lora',params=[p for p in model.text.parameters() if p.requires_grad],lr=CONFIG['learning_rates'][1]),
            dict(name='decision_head',params=[model.head_weight,model.head_bias],lr=CONFIG['learning_rates'][2]),
            dict(name='eeg_encoder',params=[p for p in model.eeg_encoder.parameters() if p.requires_grad],lr=CONFIG['encoder_learning_rate'])]
    opt=torch.optim.AdamW(groups,weight_decay=.01)
    active=[p for g in opt.param_groups for p in g['params']]
    horizon=2 if a.smoke else CONFIG['steps']
    stop_step=2 if a.smoke else CONFIG['stop_step']
    encoder_freeze_steps=1 if a.smoke else CONFIG['encoder_warmup_steps']
    schedule=GroupLRSchedule(opt,horizon,1 if a.smoke else math.ceil(horizon*CONFIG['lr_schedule']['warmup_ratio']),
                             CONFIG['lr_schedule']['min_ratio'],encoder_freeze_steps,CONFIG['lr_schedule']['kind'])
    config=dict(CONFIG,bn_policy=a.bn_policy,actual_world_size=world,smoke=a.smoke,
                data_manifest_sha256=digest(ROOT/f"fold{CONFIG['fold']}"/'complete.json'),
                task_sha256=digest('configs/benchmark_tasks.json'),instructions_sha256=digest('artifacts/benchmark/instructions.npy'),
                git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                code_sha256={str(p):digest(p) for p in Path('eeg_align').glob('*.py')},
                trainable_parameters=sum(p.numel() for p in active),train_epochs_available=len(train),validation_epochs=len(val))
    start_step=0;best=-1;resume_info=None
    if a.resume:
        restored=torch.load(a.resume,map_location='cpu',mmap=True,weights_only=False)
        for key in ['seed','steps','global_batch','learning_rates','encoder_learning_rate','encoder_warmup_steps',
                    'fold','checkpoint','upstream_commit','data_manifest_sha256','task_sha256','instructions_sha256']:
            if restored['config'][key]!=config[key]:raise ValueError(f'Resume mismatch: {key}')
        start_step=int(restored['step']);assert start_step<stop_step<=horizon
        if a.recover_warmup:
            resume_info=recover_frozen_warmup(model,restored,encoder_freeze_steps)
            seed_rng(CONFIG['branch_seed'])
        else:
            if restored['config']['bn_policy']!=a.bn_policy or restored['config']['actual_world_size']!=world:
                raise ValueError('Exact resume requires matching BN policy and world size')
            restore_model(model,restored);restore_rng(restored['rng_per_rank'][rank])
        opt.load_state_dict(restored['optimizer']);schedule.load_state_dict(restored['scheduler'])
        assert schedule.last_step==start_step
        best=float(restored['best_score'])
        config.update(resume_checkpoint=str(a.resume.resolve()),resume_checkpoint_sha256=digest(a.resume),
                      resume_step=start_step,resume_info=resume_info)
        del restored
    elif a.evaluate_checkpoint:
        restored=torch.load(a.evaluate_checkpoint,map_location='cpu',mmap=True,weights_only=False)
        if restored['config']['bn_policy']!=a.bn_policy:raise ValueError('Evaluation BN policy mismatch')
        restore_model(model,restored);start_step=int(restored['step']);del restored
    if rank==0:(a.out/'config.json').write_text(json.dumps(config,indent=2))
    prompts={};start=time.monotonic()
    diagnostic_indices=sorted(next(i for i,r in enumerate(val) if r['task']==t and r['label']==c)
                              for t in range(len(TASKS)) for c in range(len(TASKS[t]['classes'])))

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
            ids=[c['class_id'] for c in TASKS[r['task']]['classes']]
            v=[float(prob[q['option_class_ids'].index(i)]) for i in ids];values.append(v+[0.]*(16-len(v)))
        return values

    @torch.no_grad()
    def evaluate(step):
        model.eval();indices=diagnostic_indices if a.smoke else list(range(len(val)))
        mine=indices[rank::world];result=[]
        for offset in range(0,len(mine),CONFIG['eval_microbatch']):
            ix=mine[offset:offset+CONFIG['eval_microbatch']];rows=[val[i] for i in ix];zz=[]
            for variant in (0,1):
                logits,_,questions=forward(rows,step,variant);zz.append(semantic(logits,questions,rows))
            result.extend((i,p,q) for i,p,q in zip(ix,*zz))
        gathered=[None]*world;dist.all_gather_object(gathered,result)
        pred=sorted(sum(gathered,[]));assert [r[0] for r in pred]==indices
        rows=[val[i] for i,_,_ in pred];probs=np.array([p for _,p,_ in pred]);alt=np.array([q for _,_,q in pred])
        report=dict(normal=metrics(rows,probs),reversed=metrics(rows,alt))
        report['consistency']={task['task_id']:float(np.mean(probs[[r['task']==t for r in rows]].argmax(1)==alt[[r['task']==t for r in rows]].argmax(1))) for t,task in enumerate(TASKS)}
        report['selection_score']=float(np.mean([m['epoch']['balanced_accuracy'] for m in report['normal'].values()]))
        report['bn_state']=bn_summary(model.eeg_encoder)
        if rank==0:(a.out/f'{a.split}_{step:04d}.json').write_text(json.dumps(dict(step=step,metrics=report,predictions=[dict(r,probabilities=p.tolist(),alternate_probabilities=q.tolist()) for r,p,q in zip(rows,probs,alt)])))
        return report

    def save(step,improved=False,initial=False):
        rng_states=[None]*world;dist.all_gather_object(rng_states,capture_rng())
        if rank==0:
            state=dict(**capture_model(model),step=step,config=config,optimizer=opt.state_dict(),
                       scheduler=schedule.state_dict(),rng_per_rank=rng_states,best_score=best)
            atomic_save(state,a.out/('initial.pt' if initial else 'last.pt'))
            if improved:atomic_save(state,a.out/'best.pt')
            if initial:
                proof=dict(step=step,trainable_sha256=tensor_fingerprint(state['trainable']),
                           buffers_sha256=tensor_fingerprint(state['buffers']),bn_state=bn_summary(model.eeg_encoder),
                           source=config.get('resume_checkpoint'),source_sha256=config.get('resume_checkpoint_sha256'),
                           branch_seed=CONFIG['branch_seed'])
                temp=a.out/'initial_state.tmp';temp.write_text(json.dumps(proof,indent=2));temp.replace(a.out/'initial_state.json')
        dist.barrier()

    @torch.no_grad()
    def reload_check(step,path):
        """Prove a saved model reproduces logits after destroying relevant live state."""
        rng=capture_rng();was_training=model.training;model.eval()
        rows=[val[i] for i in diagnostic_indices]
        def predict():
            chunks=[]
            for offset in range(0,len(rows),CONFIG['eval_microbatch']):
                logits,_,_=forward(rows[offset:offset+CONFIG['eval_microbatch']],step,0)
                chunks.append(F.pad(logits.float(),(0,16-logits.shape[-1]),value=float('-inf')).cpu())
            return torch.cat(chunks)
        before=predict()
        state=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        # Loading is tested from changed affine parameters and BN statistics.
        model.head_weight[0,0].add_(1.)
        for name,buffer in model.named_buffers():
            if name.endswith('running_mean'):buffer.add_(1.)
            elif name.endswith('running_var'):buffer.mul_(2.)
            elif name.endswith('num_batches_tracked'):buffer.add_(5)
        restore_model(model,state)
        after=predict();torch.testing.assert_close(before,after,rtol=0,atol=0)
        result=dict(step=step,samples=len(rows),logits_exact_match=True,max_abs_logit_difference=0.,
                    prediction_agreement=float((before.argmax(-1)==after.argmax(-1)).float().mean()),
                    checkpoint_format=state['format_version'],saved_buffers=len(state['buffers']),
                    bn_state=bn_summary(model.eeg_encoder),probabilities=after.softmax(-1).tolist(),
                    sample_ids=[r['id'] for r in rows])
        if rank==0:(a.out/f'reload_check_{step:04d}.json').write_text(json.dumps(result,indent=2))
        del state;model.train(was_training);restore_rng(rng)
        return {k:v for k,v in result.items() if k not in {'probabilities','sample_ids'}}

    if a.evaluate_checkpoint:
        report=evaluate(start_step)
        if rank==0:(a.out/'complete.json').write_text(json.dumps(dict(checkpoint=str(a.evaluate_checkpoint),split=a.split,metrics=report),indent=2))
        dist.barrier();dist.destroy_process_group();return
    log=(a.out/'metrics.jsonl').open('w',buffering=1) if rank==0 else None
    if a.resume:
        save(start_step,initial=True)
        checked=reload_check(start_step,a.out/'initial.pt')
        if rank==0:log.write(json.dumps(dict(step=start_step,checkpoint_check=checked))+'\n')
    probe=next(model.eeg_encoder.leaf.tower.tokenizer.parameters())
    for step in range(start_step+1,stop_step+1):
        learning_rates=schedule.set_step(step);model.train();opt.zero_grad(set_to_none=True)
        ix=sampler(train,step,CONFIG['seed'],CONFIG['global_batch'])[rank::world]
        stats=torch.zeros((len(TASKS),3),device='cuda')
        before=probe.detach().clone() if a.smoke else None
        for offset in range(0,len(ix),CONFIG['microbatch']):
            ids=ix[offset:offset+CONFIG['microbatch']];rows=[train[i] for i in ids];logits,y,_=forward(rows,step)
            losses=F.cross_entropy(logits,y,reduction='none')
            if not torch.isfinite(losses).all():raise FloatingPointError('Nonfinite per-example loss')
            (losses.sum()*world/CONFIG['global_batch']).backward()
            for j,r in enumerate(rows):
                stats[r['task'],0]+=losses[j].detach();stats[r['task'],1]+=(logits[j].argmax()==y[j]);stats[r['task'],2]+=1
        unfrozen=model.eeg_encoder.unfrozen
        sync=active if unfrozen else [p for g in opt.param_groups[:3] for p in g['params']]
        average_gradients(sync,world)
        encoder_grad=float(torch.stack([p.grad.float().square().sum() for p in opt.param_groups[3]['params']]).sum().sqrt()) if unfrozen else 0.
        grad=torch.nn.utils.clip_grad_norm_(sync,1.);assert torch.isfinite(grad);opt.step();dist.all_reduce(stats)
        if a.smoke:
            delta=float((probe.detach()-before).abs().max())
            assert (delta>0 and encoder_grad>0) if unfrozen else delta==0,(unfrozen,delta,encoder_grad)
        if rank==0:
            sums=stats.sum(0)
            item=dict(step=step,steps=horizon,stop_step=stop_step,loss=float(sums[0]/sums[2]),
                      train_accuracy=float(sums[1]/sums[2]),learning_rates=learning_rates,
                      encoder_unfrozen=unfrozen,encoder_gradient_norm=encoder_grad,gradient_norm=float(grad),
                      bn_state=bn_summary(model.eeg_encoder),elapsed_seconds=time.monotonic()-start,
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      train_by_dataset={task['task_id']:dict(n=int(stats[t,2]),loss=float(stats[t,0]/stats[t,2]),accuracy=float(stats[t,1]/stats[t,2])) for t,task in enumerate(TASKS)})
            log.write(json.dumps(item)+'\n')
            (a.out/'progress.json').write_text(json.dumps(item,indent=2))
            if step==start_step+1 or step%10==0:print('TRAIN',item,flush=True)
        flag=torch.tensor(int(stopping),device='cuda');dist.all_reduce(flag,op=dist.ReduceOp.MAX);stopping=bool(flag.item())
        validation_due=step%CONFIG['validation_every']==0 or step==stop_step
        if validation_due and not stopping:
            report=evaluate(step);improved=report['selection_score']>best;best=max(best,report['selection_score'])
            save(step,improved)
            checked=reload_check(step,a.out/'last.pt')
            if rank==0:
                log.write(json.dumps(dict(step=step,validation=report,checkpoint_check=checked))+'\n')
                if improved:(a.out/'best.json').write_text(json.dumps(dict(step=step,metrics=report),indent=2))
                print('VALIDATION',step,report['selection_score'],'RELOAD',checked,flush=True)
        elif step==start_step+1 or step%CONFIG['checkpoint_every']==0 or stopping:
            save(step)
        if stopping:break
    if rank==0:
        (a.out/('stopped.json' if stopping else 'complete.json')).write_text(json.dumps(dict(step=step,steps=horizon,stop_step=stop_step,smoke=a.smoke,bn_policy=a.bn_policy,seconds=time.monotonic()-start,best_score=best),indent=2))
        log.close()
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
