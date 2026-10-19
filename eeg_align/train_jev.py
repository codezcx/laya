"""Full-channel MI trial classification: frozen REVE/Qwen, active trainable JEV LoRA."""
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
from .mi_training_data import CONFIG,config,Tokens,epoch_batches,metrics,digest,write_json

def state(model):return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}

@torch.no_grad()
def evaluate(model,data,rank,world,microbatch=2,limit=None):
    model.eval();mine=[]
    ids=np.arange(len(data.rows)) if limit is None else np.arange(min(limit,len(data.rows)))
    ids=sorted(ids,key=lambda i:data.rows[i]['token_length'])[rank::world]
    for start in range(0,len(ids),microbatch):
        ix=ids[start:start+microbatch];x,mask,y=data.batch(ix)
        logits=model(x,mask)['logits'];p=logits.softmax(-1)[:,1].cpu().tolist();loss=F.cross_entropy(logits,y,reduction='none').cpu().tolist()
        for i,prob,l in zip(ix,p,loss):
            r=data.rows[i];mine.append(dict(id=r['id'],dataset=r['dataset'],subject=r['subject'],label=r['label'],probability_right=prob,loss=l))
    parts=[None]*world;dist.all_gather_object(parts,mine);pred=sorted(sum(parts,[]),key=lambda r:r['id'])
    assert len(pred)==len(set(r['id'] for r in pred))==(len(data.rows) if limit is None else min(limit,len(data.rows)))
    result=metrics(pred);result['loss']=float(np.mean([r['loss'] for r in pred]));return result,pred

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();c=config()
    rank,world,local=[int(os.environ[k]) for k in ['RANK','WORLD_SIZE','LOCAL_RANK']]
    torch.cuda.set_device(local);torch.set_num_threads(4);dist.init_process_group('nccl',timeout=timedelta(minutes=60))
    if rank==0:a.out.mkdir(parents=True,exist_ok=False)
    dist.barrier();torch.manual_seed(c['seed']);np.random.seed(c['seed']);random.seed(c['seed'])
    train=Tokens('train');val=Tokens('validation')
    assert not {(r['dataset'],r['subject']) for r in train.rows}&{(r['dataset'],r['subject']) for r in val.rows}
    text,w,b,tok=load_jev();model=JEVBridge(text,w,b,tok,train_lora=True,train_head=True)
    opt=torch.optim.AdamW([dict(params=model.connector.parameters(),lr=c['learning_rates'][0]),dict(params=[p for p in model.text.parameters() if p.requires_grad],lr=c['learning_rates'][1]),dict(params=[model.head_weight,model.head_bias],lr=c['learning_rates'][2])],weight_decay=.01)
    batches0=epoch_batches(train.rows,c['seed'],0,c['batch_trials']);steps=2 if a.smoke else c['epochs']*len(batches0)
    warmup=max(1,round(steps*c['warmup_fraction']))
    paths=['eeg_align/jev.py','eeg_align/mi_training_data.py','eeg_align/train_jev.py','eeg_align/distributed.py','configs/encoder.json','configs/mi_train.json']
    provenance=dict(c,steps=steps,world_size=world,smoke=a.smoke,train_trials=len(train.rows),validation_trials=len(val.rows),
        init='original JEV adapter and A/B rows; fresh connector; no TUAB/previous MI weights',
        train_encoder=False,train_qwen_base=False,train_lora=True,train_head=True,
        loss='one cross entropy per trial, ordinary sample mean',schedule='linear warmup then cosine decay',
        data_sha256=digest(Path(c['cache'])/'complete.json'),code_sha256={p:digest(p) for p in paths},
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    if rank==0:write_json(a.out/'config.json',provenance)
    began=time.monotonic();best=-1.;step=0;log=(a.out/'metrics.jsonl').open('a',buffering=1) if rank==0 else None
    def validate():
        nonlocal best
        result,pred=evaluate(model,val,rank,world,c['microbatch_trials'],limit=16 if a.smoke else None)
        if rank==0:
            log.write(json.dumps(dict(step=step,validation=result))+'\n');write_json(a.out/f'validation_{step:04d}.json',dict(step=step,metrics=result,predictions=pred))
            print('VALIDATION',step,result['accuracy'],result['macro_dataset_balanced_accuracy'],flush=True)
        if result['macro_dataset_balanced_accuracy']>best:
            best=result['macro_dataset_balanced_accuracy']
            if rank==0:
                atomic_save(dict(trainable=state(model),step=step,config=provenance),a.out/'best.pt');write_json(a.out/'best.json',dict(step=step,metrics=result))
    validate()
    for epoch in range(c['epochs']):
        batches=epoch_batches(train.rows,c['seed'],epoch,c['batch_trials'])
        if a.smoke:
            # Include variable lengths and the longest available trial.
            order=np.argsort([r['token_length'] for r in train.rows]);batches=[np.r_[order[:8],order[-8:]],np.r_[order[8:16],order[-16:-8]]]
        for batch_index,ids in enumerate(batches):
            assert len(ids)>=world
            ratio=(step+1)/warmup if step<warmup else c['minimum_lr_ratio']+(1-c['minimum_lr_ratio'])*.5*(1+math.cos(math.pi*(step-warmup)/max(1,steps-warmup)))
            for g,lr in zip(opt.param_groups,c['learning_rates']):g['lr']=lr*ratio
            mine=ids[rank::world];model.train();opt.zero_grad(set_to_none=True);total_loss=0.;correct=0
            for start in range(0,len(mine),c['microbatch_trials']):
                ix=mine[start:start+c['microbatch_trials']];x,mask,y=train.batch(ix)
                z=model(x,mask)['logits'];loss=F.cross_entropy(z,y,reduction='sum')
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                (loss*world/len(ids)).backward();total_loss+=float(loss.detach());correct+=int((z.argmax(-1)==y).sum())
            active=[p for g in opt.param_groups for p in g['params']];average_gradients(active,world)
            grad=torch.nn.utils.clip_grad_norm_(active,1.)
            if not torch.isfinite(grad):raise FloatingPointError('Nonfinite gradient')
            opt.step();step+=1
            stats=torch.tensor([total_loss,correct],device='cuda');dist.all_reduce(stats)
            if a.smoke:
                h=hashlib.sha256()
                for value in state(model).values():h.update(value.numpy().tobytes())
                hashes=[None]*world;dist.all_gather_object(hashes,h.hexdigest());assert len(set(hashes))==1
                print('SYNCHRONIZED',rank,step,flush=True)
            if rank==0:
                item=dict(step=step,epoch=epoch+1,loss=float(stats[0])/len(ids),accuracy=float(stats[1])/len(ids),gradient_norm=float(grad),learning_rates=[g['lr'] for g in opt.param_groups],elapsed_seconds=time.monotonic()-began)
                log.write(json.dumps(item)+'\n');write_json(a.out/'progress.json',dict(step=step,steps=steps,epoch=epoch+1))
                if step==1 or step%10==0 or a.smoke:print(json.dumps(item),flush=True)
            endpoint=step==steps
            if step%c['validation_every']==0 or batch_index==len(batches)-1 or endpoint:
                validate()
                if rank==0:atomic_save(dict(trainable=state(model),optimizer=opt.state_dict(),step=step,epoch=epoch,config=provenance),a.out/'last.pt')
                dist.barrier()
            if endpoint:break
        if step==steps:break
    if not a.smoke:
        saved=torch.load(a.out/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(saved['trainable'],strict=False);selected=saved['step'];del saved
        # Test objects are instantiated only after validation-based model selection.
        result,pred=evaluate(model,Tokens('heldout'),rank,world,c['microbatch_trials'])
        if rank==0:write_json(a.out/'test_2b.json',dict(step=selected,metrics=result,predictions=pred))
    if rank==0:write_json(a.out/'complete.json',dict(steps=step,smoke=a.smoke,best_sha256=digest(a.out/'best.pt'),seconds=time.monotonic()-began))
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
