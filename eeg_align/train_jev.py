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
from .jev import load_jev,JEVBridge,SUFFIX
from .mi_options import REVERSED_SUFFIX,swapped
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
        reversed_p=None
        if hasattr(model,'option_suffixes'):
            reverse=model(x,mask,suffix_ids=[model.option_suffixes[1]]*len(ix))['logits']
            reversed_p=reverse.softmax(-1)[:,0].cpu().tolist()
        for j,(i,prob,l) in enumerate(zip(ix,p,loss)):
            r=data.rows[i];mine.append(dict(id=r['id'],dataset=r['dataset'],subject=r['subject'],label=r['label'],probability_right=prob,loss=l))
            if reversed_p is not None:mine[-1]['reversed_probability_right']=reversed_p[j]
    parts=[None]*world;dist.all_gather_object(parts,mine);pred=sorted(sum(parts,[]),key=lambda r:r['id'])
    assert len(pred)==len(set(r['id'] for r in pred))==(len(data.rows) if limit is None else min(limit,len(data.rows)))
    result=metrics(pred);result['loss']=float(np.mean([r['loss'] for r in pred]))
    if hasattr(model,'option_suffixes'):
        rev=metrics([dict(r,probability_right=r['reversed_probability_right']) for r in pred])
        result['reversed_options']=rev
        result['permutation_consistency']=float(np.mean([(r['probability_right']>.5)==(r['reversed_probability_right']>.5) for r in pred]))
        result['mean_right_probability_gap']=float(np.mean([abs(r['probability_right']-r['reversed_probability_right']) for r in pred]))
        result['selection_score']=(result['macro_dataset_balanced_accuracy']+rev['macro_dataset_balanced_accuracy'])/2
    else:result['selection_score']=result['macro_dataset_balanced_accuracy']
    return result,pred

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--resume',type=Path);p.add_argument('--recipe',choices=['mi','tuab-lr'],default='mi');p.add_argument('--shuffle-options',action='store_true');p.add_argument('--encoder-checkpoint',type=Path);p.add_argument('--microbatch',type=int);a=p.parse_args();c=config()
    if a.recipe=='tuab-lr':c.update(learning_rates=[1e-4,5e-6,1e-5],warmup_fraction=0.,minimum_lr_ratio=1.)
    if a.microbatch is not None:c['microbatch_trials']=a.microbatch
    if a.resume and a.smoke:raise ValueError('Smoke and resume are mutually exclusive')
    rank,world,local=[int(os.environ[k]) for k in ['RANK','WORLD_SIZE','LOCAL_RANK']]
    torch.cuda.set_device(local);torch.set_num_threads(4);dist.init_process_group('nccl',timeout=timedelta(minutes=60))
    if rank==0:a.out.mkdir(parents=True,exist_ok=False)
    dist.barrier();torch.manual_seed(c['seed']);np.random.seed(c['seed']);random.seed(c['seed'])
    encoder=None
    if a.encoder_checkpoint:
        from .adapted_tokens import AdaptedEncoder,AdaptedTokens
        encoder=AdaptedEncoder(a.encoder_checkpoint)
    def data(split):return AdaptedTokens(split,encoder) if encoder is not None else Tokens(split)
    train=data('train');val=data('validation')
    # Encoder/text initialization must not change the matched connector seed.
    torch.manual_seed(c['seed'])
    assert not {(r['dataset'],r['subject']) for r in train.rows}&{(r['dataset'],r['subject']) for r in val.rows}
    text,w,b,tok=load_jev();model=JEVBridge(text,w,b,tok,train_lora=True,train_head=True)
    if a.shuffle_options:
        model.option_suffixes=[torch.tensor(tok.encode(s,add_special_tokens=False),device=w.device) for s in [SUFFIX,REVERSED_SUFFIX]]
    opt=torch.optim.AdamW([dict(params=model.connector.parameters(),lr=c['learning_rates'][0]),dict(params=[p for p in model.text.parameters() if p.requires_grad],lr=c['learning_rates'][1]),dict(params=[model.head_weight,model.head_bias],lr=c['learning_rates'][2])],weight_decay=.01)
    batches0=epoch_batches(train.rows,c['seed'],0,c['batch_trials']);steps=2 if a.smoke else c['epochs']*len(batches0)
    warmup=max(1,round(steps*c['warmup_fraction']))
    paths=['eeg_align/jev.py','eeg_align/mi_training_data.py','eeg_align/train_jev.py','eeg_align/distributed.py','configs/encoder.json','configs/mi_train.json','eeg_align/mi_options.py','eeg_align/adapted_tokens.py']
    provenance=dict(c,encoder_checkpoint=str(a.encoder_checkpoint) if encoder else None,encoder_sha256=encoder.sha256 if encoder else None,shuffle_options=a.shuffle_options,steps=steps,world_size=world,smoke=a.smoke,train_trials=len(train.rows),validation_trials=len(val.rows),
        init='original JEV adapter and A/B rows; fresh connector; no TUAB/previous MI weights',
        train_encoder=False,train_qwen_base=False,train_lora=True,train_head=True,
        loss='one cross entropy per trial, ordinary sample mean',schedule='constant' if a.recipe=='tuab-lr' else 'linear warmup then cosine decay',recipe=a.recipe,
        data_sha256=digest(Path(c['cache'])/'complete.json'),code_sha256={p:digest(p) for p in paths},
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    resume_step=0
    resume_best=-1.
    if a.resume:
        saved=torch.load(a.resume,map_location='cpu',weights_only=False)
        prior=saved['config']
        assert prior.get('encoder_sha256')==provenance['encoder_sha256']
        assert prior.get('shuffle_options',False)==a.shuffle_options
        assert prior['data_sha256']==provenance['data_sha256']
        for key in c:assert prior[key]==c[key],f'Resume config mismatch: {key}'
        for path,h in prior['code_sha256'].items():
            if path!='eeg_align/train_jev.py':assert digest(path)==h,f'Resume code mismatch: {path}'
        assert set(saved['trainable'])=={n for n,p in model.named_parameters() if p.requires_grad}
        model.load_state_dict(saved['trainable'],strict=False);opt.load_state_dict(saved['optimizer'])
        resume_step=int(saved['step']);assert 0<resume_step<steps
        best_path=a.resume.parent/'best.pt';best_info=json.loads((a.resume.parent/'best.json').read_text())
        assert best_info['step']<=resume_step
        resume_best=best_info['metrics'].get('selection_score',best_info['metrics']['macro_dataset_balanced_accuracy'])
        if rank==0:
            import shutil
            shutil.copy2(best_path,a.out/'best.pt');shutil.copy2(a.resume.parent/'best.json',a.out/'best.json')
        provenance.update(resume_checkpoint=str(a.resume),resume_sha256=digest(a.resume),resume_step=resume_step,prior_run_config=prior)
        del saved
        print('RESUMED',rank,resume_step,flush=True)
    if rank==0:write_json(a.out/'config.json',provenance)
    began=time.monotonic();best=resume_best;step=resume_step;log=(a.out/'metrics.jsonl').open('a',buffering=1) if rank==0 else None
    def validate():
        nonlocal best
        result,pred=evaluate(model,val,rank,world,c['microbatch_trials'],limit=16 if a.smoke else None)
        if rank==0:
            log.write(json.dumps(dict(step=step,validation=result))+'\n');write_json(a.out/f'validation_{step:04d}.json',dict(step=step,metrics=result,predictions=pred))
            print('VALIDATION',step,result['accuracy'],result['macro_dataset_balanced_accuracy'],flush=True)
        if result['selection_score']>best:
            best=result['selection_score']
            if rank==0:
                atomic_save(dict(trainable=state(model),step=step,config=provenance),a.out/'best.pt');write_json(a.out/'best.json',dict(step=step,metrics=result))
    if not a.resume:validate()
    for epoch in range(c['epochs']):
        batches=epoch_batches(train.rows,c['seed'],epoch,c['batch_trials'])
        if a.smoke:
            # Include variable lengths and the longest available trial.
            order=np.argsort([r['token_length'] for r in train.rows]);batches=[np.r_[order[:8],order[-8:]],np.r_[order[8:16],order[-16:-8]]]
        for batch_index,ids in enumerate(batches):
            if epoch*len(batches0)+batch_index<resume_step:continue
            assert len(ids)>=world
            ratio=1. if a.recipe=='tuab-lr' else (step+1)/warmup if step<warmup else c['minimum_lr_ratio']+(1-c['minimum_lr_ratio'])*.5*(1+math.cos(math.pi*(step-warmup)/max(1,steps-warmup)))
            for g,lr in zip(opt.param_groups,c['learning_rates']):g['lr']=lr*ratio
            mine=ids[rank::world];model.train();opt.zero_grad(set_to_none=True);total_loss=0.;correct=0
            for start in range(0,len(mine),c['microbatch_trials']):
                ix=mine[start:start+c['microbatch_trials']];x,mask,y=train.batch(ix)
                if a.shuffle_options:
                    reverse=torch.tensor([swapped(train.rows[i]['id'],epoch,c['seed']) for i in ix],device=y.device)
                    y=torch.where(reverse,1-y,y)
                    suffix=[model.option_suffixes[int(v)] for v in reverse.tolist()]
                    z=model(x,mask,suffix_ids=suffix)['logits']
                else:z=model(x,mask)['logits']
                loss=F.cross_entropy(z,y,reduction='sum')
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
        result,pred=evaluate(model,data('heldout'),rank,world,c['microbatch_trials'])
        if rank==0:write_json(a.out/'test_2b.json',dict(step=selected,metrics=result,predictions=pred))
    if rank==0:write_json(a.out/'complete.json',dict(steps=step,smoke=a.smoke,best_sha256=digest(a.out/'best.pt'),seconds=time.monotonic()-began))
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
