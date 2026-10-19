"""Matched frozen-REVE linear probe; all-channel MI cache, fixed spatial pooling."""
import argparse
import json
import os
from pathlib import Path
import time
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from .mi_training_data import Tokens,config,digest,write_json,metrics
from .distributed import average_gradients,atomic_save


def spatial_pool(tokens,channel_positions,channel_names):
    """Fixed L/mid/R readout; no learned attention, missing groups become zeros."""
    n=len(channel_names);assert tokens.ndim==2 and tokens.shape[0]%n==0
    channels=tokens.reshape(n,-1,512).mean(1)
    # Standard Z electrodes define the midline even when template coordinates
    # have tiny numerical x offsets. All other groups use anatomical x sign.
    mid=torch.tensor([name.upper().endswith('Z') for name in channel_names],device=tokens.device)
    xx=channel_positions[:,0];mid=mid | (xx.abs()<1e-6)
    groups=[(xx<0)&~mid,mid,(xx>0)&~mid]
    assert torch.stack(groups).sum(0).eq(1).all()
    return torch.cat([channels[g].mean(0) if g.any() else channels.new_zeros(512) for g in groups])


def features(split,rank,world,out,bank):
    data=Tokens(split);rows=data.rows[rank::world];values=[];positions={}
    with torch.no_grad():
        for i,r in enumerate(rows):
            names=tuple(r['position_names'])
            if names not in positions:
                p=bank(list(names)).cuda();assert p.shape==(len(names),3);positions[names]=p
            start=r['cache_offset'];n=r['token_length']
            z=torch.from_numpy(np.array(data.arrays[r['cache_shard']][start:start+n*512]).reshape(n,512)).cuda().float()
            values.append(spatial_pool(z,positions[names],names).cpu())
            if i%2000==0:print('FEATURES',split,rank,i,len(rows),flush=True)
    x=torch.stack(values);assert torch.isfinite(x).all()
    # Reproducible snapshots avoid re-pooling for later inspection.
    atomic_save(dict(features=x,ids=[r['id'] for r in rows]),out/f'features_{split}_{rank}.pt')
    return rows,x.cuda(),torch.tensor([r['label'] for r in rows],device='cuda')


@torch.no_grad()
def evaluate(head,rows,x,y,rank,world):
    logits=head(x);p=logits.softmax(-1)[:,1].cpu().tolist();loss=F.cross_entropy(logits,y,reduction='none').cpu().tolist()
    local=[dict(id=r['id'],dataset=r['dataset'],subject=r['subject'],label=r['label'],probability_right=v,loss=l) for r,v,l in zip(rows,p,loss)]
    parts=[None]*world;dist.all_gather_object(parts,local);pred=sorted(sum(parts,[]),key=lambda r:r['id'])
    assert len(pred)==len({r['id'] for r in pred})
    m=metrics(pred);m['loss']=float(np.mean([r['loss'] for r in pred]));return m,pred


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
    rank,world,local=[int(os.environ[k]) for k in ['RANK','WORLD_SIZE','LOCAL_RANK']]
    torch.cuda.set_device(local);torch.set_num_threads(2);dist.init_process_group('nccl',timeout=timedelta(minutes=30))
    if rank==0:a.out.mkdir(parents=True,exist_ok=False)
    dist.barrier();c=json.loads(Path('configs/mi_probe.json').read_text());torch.manual_seed(c['seed'])
    from transformers import AutoModel
    ec=json.loads(Path('configs/encoder.json').read_text())
    bank=AutoModel.from_pretrained(ec['positions_checkpoint'],trust_remote_code=True,local_files_only=True).eval().requires_grad_(False)
    tr,x,y=features('train',rank,world,a.out,bank);vr,vx,vy=features('validation',rank,world,a.out,bank)
    # Train-only standardization across all four disjoint shards.
    count=torch.tensor(len(x),device='cuda',dtype=torch.float64);s=x.double().sum(0);ss=x.double().square().sum(0)
    for value in [count,s,ss]:dist.all_reduce(value)
    mean=(s/count).float();std=((ss/count-(s/count).square()).clamp_min(0).sqrt()).float().clamp_min(1e-6)
    x=(x-mean)/std;vx=(vx-mean)/std
    head=torch.nn.Linear(1536,2).cuda();opt=torch.optim.AdamW(head.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
    provenance=dict(c,cache_sha256=digest(Path(config()['cache'])/'complete.json'),source_manifest_sha256={name:digest(Path(config()['cache'])/f'{name}.jsonl') for name in ['train','validation','heldout']},code_sha256=digest(__file__),trainable_parameters=sum(p.numel() for p in head.parameters()),encoder_frozen=True)
    if rank==0:write_json(a.out/'config.json',provenance)
    best=-1.;bad=0;step=0;began=time.monotonic();log=(a.out/'metrics.jsonl').open('x',buffering=1) if rank==0 else None
    # All ranks hold equal-sized or nearly equal-sized shards. Construct global
    # trial batches then route each original sorted index to its owning rank.
    total=int(count.item());n_batches=int(np.ceil(total/c['batch_size']))
    for epoch in range(c['epochs']):
        order=np.random.default_rng(c['seed']+epoch).permutation(total)
        batches=np.array_split(order,n_batches);train_sum=0.;correct=0
        for ids in batches:
            ix=torch.tensor(ids[ids%world==rank]//world,device='cuda',dtype=torch.long)
            opt.zero_grad(set_to_none=True);z=head(x[ix]);loss=F.cross_entropy(z,y[ix],reduction='sum')
            (loss*world/len(ids)).backward();average_gradients(head.parameters(),world);opt.step();step+=1
            train_sum+=float(loss.detach());correct+=int((z.argmax(-1)==y[ix]).sum())
        stats=torch.tensor([train_sum,correct],device='cuda');dist.all_reduce(stats)
        result,pred=evaluate(head,vr,vx,vy,rank,world)
        score=result['macro_dataset_balanced_accuracy'];improved=score>best
        if improved:
            best=score;bad=0
            if rank==0:
                atomic_save(dict(head=head.state_dict(),mean=mean.cpu(),std=std.cpu(),epoch=epoch+1,config=provenance),a.out/'best.pt')
                write_json(a.out/'best.json',dict(epoch=epoch+1,metrics=result));write_json(a.out/'validation_best.json',dict(epoch=epoch+1,metrics=result,predictions=pred))
        else:bad+=1
        if rank==0:
            item=dict(epoch=epoch+1,step=step,train_loss=float(stats[0])/total,train_accuracy=float(stats[1])/total,validation=result,elapsed_seconds=time.monotonic()-began)
            log.write(json.dumps(item)+'\n');write_json(a.out/'progress.json',dict(epoch=epoch+1,max_epochs=c['epochs'],bad_epochs=bad))
            print('EPOCH',epoch+1,'train',item['train_accuracy'],'val',result['accuracy'],'macro_BA',score,flush=True)
        if bad>=c['patience']:break
    dist.barrier();best_state=torch.load(a.out/'best.pt',map_location='cpu',weights_only=False);head.load_state_dict(best_state['head'])
    # No test evaluation or model selection before source-only early stopping.
    te,tx,ty=features('heldout',rank,world,a.out,bank);tx=(tx-mean)/std
    result,pred=evaluate(head,te,tx,ty,rank,world)
    if rank==0:
        write_json(a.out/'test_2b.json',dict(epoch=best_state['epoch'],metrics=result,predictions=pred))
        write_json(a.out/'complete.json',dict(epochs=epoch+1,best_epoch=best_state['epoch'],seconds=time.monotonic()-began,best_sha256=digest(a.out/'best.pt')))
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
