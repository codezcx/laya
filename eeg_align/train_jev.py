"""TUAB-only record-level binary classification with frozen REVE and active JEV LoRA."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from .jev import load_jev,JEVBridge
from .distributed import average_gradients,atomic_save
from .tuab_data import ROOT,digest,readrows,write_json,load_windows,metrics


def record_loss(window_logits,labels):
    return F.cross_entropy(window_logits.reshape(len(labels),8,2).mean(1),labels)


def backward_bags(model,tokens,labels,microbatch=2,scale=1.):
    """Exact two-pass bag gradient, bounded activation memory; requires no dropout."""
    def forward(x):
        return model(x,torch.ones(x.shape[:2],device=x.device,dtype=torch.bool))['logits']
    with torch.no_grad():
        chunks=[forward(tokens[i:i+microbatch]) for i in range(0,len(tokens),microbatch)]
    detached=torch.cat(chunks).detach().requires_grad_(True)
    loss=record_loss(detached,labels)
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite record loss')
    grad,=torch.autograd.grad(loss*scale,detached)
    for i in range(0,len(tokens),microbatch):
        logits=forward(tokens[i:i+microbatch])
        # Check forward determinism before using cached output gradients.
        if not torch.allclose(logits.detach(),detached[i:i+microbatch],atol=1e-5,rtol=1e-5):
            raise RuntimeError('Two-pass forward changed; gradient cache is invalid')
        logits.backward(grad[i:i+microbatch])
    return float(loss.detach()),detached.detach().reshape(len(labels),8,2).mean(1).softmax(-1)[:,1].cpu().tolist()


def state(model):return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}


@torch.no_grad()
def evaluate(model,rows,rank,world,microbatch=4,shuffle=False):
    model.eval();mine=[];sources=rows
    if shuffle:
        sources=[rows[i] for i in np.random.default_rng(1701).permutation(len(rows))]
    for i in range(rank,len(rows),world):
        x=load_windows([sources[i]],0,'cuda');logits=[]
        for j in range(0,len(x),microbatch):
            z=x[j:j+microbatch]
            logits.append(model(z,torch.ones(z.shape[:2],device=z.device,dtype=torch.bool))['logits'])
        mean=torch.cat(logits).mean(0);p=float(mean.softmax(-1)[1]);loss=float(F.cross_entropy(mean[None],torch.tensor([rows[i]['label']],device='cuda')))
        mine.append(dict(id=rows[i]['id'],label=rows[i]['label'],probability_abnormal=p,loss=loss))
    parts=[None]*world;dist.all_gather_object(parts,mine)
    predictions=sorted(sum(parts,[]),key=lambda r:r['id'])
    assert len(predictions)==len(rows) and len({r['id'] for r in predictions})==len(rows)
    result=metrics([r['label'] for r in predictions],[r['probability_abnormal'] for r in predictions])
    result.update(loss=float(np.mean([r['loss'] for r in predictions])),shuffled_signal=shuffle)
    return result,predictions


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=5);p.add_argument('--batch-records',type=int,default=16)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    rank,world,local=[int(os.environ[k]) for k in ['RANK','WORLD_SIZE','LOCAL_RANK']]
    torch.cuda.set_device(local);torch.set_num_threads(4)
    dist.init_process_group('nccl',timeout=timedelta(minutes=60))
    if rank==0:a.out.mkdir(parents=True,exist_ok=False)
    dist.barrier();torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    train=readrows(ROOT/'records_train.jsonl');val=readrows(ROOT/'records_validation.jsonl')
    test=readrows(ROOT/'records_test.jsonl')
    assert not {r['subject'] for r in train}&{r['subject'] for r in val+test}
    assert not {r['subject'] for r in val}&{r['subject'] for r in test}
    if a.smoke:val=val[:8]
    assert a.batch_records%world==0
    text,w,b,tok=load_jev();model=JEVBridge(text,w,b,tok,train_lora=True,train_head=True)
    # No prior MI/emotion connector or fine-tuning checkpoint is loaded.
    assert not any(isinstance(m,torch.nn.Dropout) and m.p for m in model.modules())
    assert getattr(text.config,'attention_dropout',0.)==0.
    opt=torch.optim.AdamW([
        dict(params=model.connector.parameters(),lr=1e-4),
        dict(params=[p for p in model.text.parameters() if p.requires_grad],lr=5e-6),
        dict(params=[model.head_weight,model.head_bias],lr=1e-5)],weight_decay=.01)
    nsteps=2 if a.smoke else a.epochs*math.ceil(len(train)/a.batch_records)
    paths=['eeg_align/jev.py','eeg_align/tuab_data.py','eeg_align/train_jev.py','eeg_align/distributed.py','configs/encoder.json']
    config=dict(task='TUAB normal vs abnormal only',epochs=a.epochs,steps=nsteps,seed=a.seed,world_size=world,
        batch_records=a.batch_records,windows_per_record=8,microbatch_windows=2,smoke=a.smoke,
        init='original local JEV adapter and decision rows; fresh random connector; no MI/emotion weights',
        train_encoder=False,train_lora=True,train_head=True,aggregation='mean window logits; one record-level cross entropy',
        learning_rates=[1e-4,5e-6,1e-5],schedule='constant',selection='validation record balanced accuracy; earliest on ties',
        data_sha256=digest(ROOT/'complete.json'),token_shard_hashes={str(i):digest(ROOT/'tokens'/f'shard_{i}.json') for i in range(4)},
        code_sha256={s:digest(s) for s in paths},git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    if rank==0:write_json(a.out/'config.json',config)
    began=time.monotonic();best=-1.;step=0
    log=(a.out/'metrics.jsonl').open('a',buffering=1) if rank==0 else None
    def validate():
        nonlocal best
        result,predictions=evaluate(model,val,rank,world)
        if rank==0:
            log.write(json.dumps(dict(step=step,validation=result))+'\n')
            write_json(a.out/f'validation_{step:04d}.json',dict(step=step,metrics=result,predictions=predictions))
            print('VALIDATION',step,json.dumps(result),flush=True)
        if result['balanced_accuracy']>best:
            best=result['balanced_accuracy']
            if rank==0:
                atomic_save(dict(trainable=state(model),step=step,config=config),a.out/'best.pt')
                write_json(a.out/'best.json',dict(step=step,metrics=result))
    validate()
    for epoch in range(a.epochs):
        order=np.random.default_rng(a.seed+epoch).permutation(len(train))
        for start in range(0,len(order),a.batch_records):
            ids=order[start:start+a.batch_records];mine=ids[rank::world]
            # The tail contains at least one record per rank in this audited dataset.
            assert len(mine)>0
            rows=[train[i] for i in mine]
            x=load_windows(rows,epoch,'cuda');labels=torch.tensor([r['label'] for r in rows],device='cuda')
            model.train();opt.zero_grad(set_to_none=True)
            loss,prob=backward_bags(model,x,labels,scale=world*len(rows)/len(ids))
            active=[p for group in opt.param_groups for p in group['params']]
            average_gradients(active,world);grad=torch.nn.utils.clip_grad_norm_(active,1.)
            if not torch.isfinite(grad):raise FloatingPointError('Nonfinite gradient')
            opt.step();step+=1
            stats=torch.tensor([loss*len(rows),sum((p>.5)==r['label'] for p,r in zip(prob,rows))],device='cuda');dist.all_reduce(stats)
            if a.smoke:
                h=hashlib.sha256()
                for value in state(model).values():h.update(value.numpy().tobytes())
                hashes=[None]*world;dist.all_gather_object(hashes,h.hexdigest());assert len(set(hashes))==1
                if rank==0:print('Four-rank parameter synchronization verified',step,flush=True)
            if rank==0:
                item=dict(step=step,epoch=epoch+1,loss=float(stats[0])/len(ids),accuracy=float(stats[1])/len(ids),
                          gradient_norm=float(grad),elapsed_seconds=time.monotonic()-began)
                log.write(json.dumps(item)+'\n');write_json(a.out/'progress.json',dict(step=step,steps=nsteps,epoch=epoch+1))
                if step==1 or step%10==0 or a.smoke:print(json.dumps(item),flush=True)
            endpoint=step==nsteps
            if step%50==0 or start+a.batch_records>=len(order) or endpoint:
                validate()
                if rank==0:atomic_save(dict(trainable=state(model),optimizer=opt.state_dict(),step=step,epoch=epoch,config=config),a.out/'last.pt')
                dist.barrier()
            if endpoint:break
        if step==nsteps:break
    if not a.smoke:
        saved=torch.load(a.out/'best.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(saved['trainable'],strict=False);selected=saved['step'];del saved
        for name,rows,shuffle in [('test',test,False),('validation_shuffled',val,True)]:
            result,predictions=evaluate(model,rows,rank,world,shuffle=shuffle)
            if rank==0:write_json(a.out/f'{name}.json',dict(step=selected,metrics=result,predictions=predictions))
    if rank==0:write_json(a.out/'complete.json',dict(steps=step,smoke=a.smoke,best_sha256=digest(a.out/'best.pt'),seconds=time.monotonic()-began))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
