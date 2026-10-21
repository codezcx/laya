"""End-to-end REVE MI control, identical source manifests and raw preprocessing."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoModel
from .distributed import average_gradients, atomic_save
from .mi_training_data import Tokens, preprocess, epoch_batches, digest, write_json, metrics


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--resume',type=Path)
    a=parser.parse_args();rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']));torch.set_num_threads(3)
    dist.init_process_group('nccl',timeout=timedelta(minutes=60))
    c=json.loads(Path('configs/mi_reve_finetune.json').read_text());torch.manual_seed(c['seed']);np.random.seed(c['seed'])
    if rank==0:a.out.mkdir(parents=True,exist_ok=a.resume is not None)
    dist.barrier();started=time.monotonic()
    rows={s:Tokens(s).rows for s in ['train','validation','heldout']}
    assert len(rows['train'])==49193 and len(rows['validation'])==12508
    ec=json.loads(Path('configs/encoder.json').read_text())
    encoder,info=AutoModel.from_pretrained(ec['reve_checkpoint'],trust_remote_code=True,local_files_only=True,output_loading_info=True)
    assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    encoder=encoder.cuda().train().requires_grad_(True)
    # This parameter belongs only to the unused optional attention-pooling path.
    encoder.cls_query_token.requires_grad_(False)
    head=torch.nn.Linear(512,2).cuda()
    bank=AutoModel.from_pretrained(ec['positions_checkpoint'],trust_remote_code=True,local_files_only=True).eval().requires_grad_(False)
    parameters=[p for p in encoder.parameters() if p.requires_grad]+list(head.parameters())
    optimizer=torch.optim.AdamW([{'params':[p for p in encoder.parameters() if p.requires_grad],'lr':c['encoder_lr']},
        {'params':head.parameters(),'lr':c['head_lr']}],weight_decay=c['weight_decay'])
    provenance=dict(c,encoder=ec,source_manifest_sha256={s:digest(Path('datasets/mi_tokens')/f'{s}.jsonl') for s in rows},
        code_sha256={f:digest(f) for f in ['eeg_align/train_reve.py','eeg_align/mi_training_data.py','eeg_align/mi_data.py','configs/encoder.json','configs/mi_reve_finetune.json']},
        trainable_parameters=sum(p.numel() for p in parameters),note='All active encoder parameters trained; unused cls_query_token excluded. No frozen-feature z-score: encoder outputs change during training.')
    if rank==0:write_json(a.out/'config.json',provenance)
    pool=ThreadPoolExecutor(max_workers=4);positions={}
    def batches(selected):
        grouped=defaultdict(list)
        for x,names,r in pool.map(preprocess,selected):
            key=(tuple(names),x.shape[-1]);grouped[key].append((x,r))
        for key,items in grouped.items():
            if key[0] not in positions:positions[key[0]]=bank(list(key[0])).detach().cuda()
            for start in range(0,len(items),c['microbatch_size']):
                part=items[start:start+c['microbatch_size']]
                yield torch.from_numpy(np.stack([x for x,r in part])).cuda(),positions[key[0]][None].expand(len(part),-1,-1),[r for x,r in part]
    def forward(x,p):
        with torch.autocast('cuda',dtype=torch.bfloat16):return head(encoder(x,p).mean((1,2)))
    @torch.no_grad()
    def evaluate(split):
        encoder.eval();head.eval();local=[];selected=rows[split][rank::world]
        for start in range(0,len(selected),64):
            for x,p,rr in batches(selected[start:start+64]):
                z=forward(x,p).float();y=torch.tensor([r['label'] for r in rr],device='cuda');prob=z.softmax(-1)[:,1].tolist();loss=F.cross_entropy(z,y,reduction='none').tolist()
                local.extend(dict(id=r['id'],dataset=r['dataset'],subject=r['subject'],label=r['label'],probability_right=v,loss=l) for r,v,l in zip(rr,prob,loss))
        parts=[None]*world;dist.all_gather_object(parts,local);pred=sorted(sum(parts,[]),key=lambda r:r['id'])
        assert len(pred)==len(rows[split]) and len({r['id'] for r in pred})==len(pred)
        m=metrics(pred);m['loss']=float(np.mean([r['loss'] for r in pred]));encoder.train();head.train();return m,pred
    step=0;best=-1.;resume_step=0
    if a.resume:
        saved=torch.load(a.resume,map_location='cpu',weights_only=False)
        assert saved['config']==provenance
        encoder.load_state_dict(saved['encoder']);head.load_state_dict(saved['head']);optimizer.load_state_dict(saved['optimizer'])
        resume_step=saved['step'];best=saved['best_score'];del saved
    log=(a.out/'metrics.jsonl').open('a' if a.resume else 'x',buffering=1) if rank==0 else None
    total=c['epochs']*math.ceil(len(rows['train'])/c['batch_size'])
    for epoch in range(c['epochs']):
        for indices in epoch_batches(rows['train'],c['seed'],epoch,c['batch_size']):
            step+=1
            if step<=resume_step:continue
            factor=min(1.,step/c['warmup_steps'])
            for group,lr in zip(optimizer.param_groups,[c['encoder_lr'],c['head_lr']]):group['lr']=lr*factor
            optimizer.zero_grad(set_to_none=True);stats=torch.zeros(3,device='cuda')
            for x,p,rr in batches([rows['train'][i] for i in indices[rank::world]]):
                y=torch.tensor([r['label'] for r in rr],device='cuda');z=forward(x,p).float();loss=F.cross_entropy(z,y,reduction='sum')
                (loss*world/len(indices)).backward()
                stats+=torch.tensor([loss.detach().item(),(z.argmax(-1)==y).sum().item(),len(rr)],device='cuda')
            average_gradients(parameters,world);norm=torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step();dist.all_reduce(stats)
            assert int(stats[2])==len(indices)
            item=dict(step=step,epoch=epoch+1,train_loss=float(stats[0]/stats[2]),train_accuracy=float(stats[1]/stats[2]),gradient_norm=float(norm),elapsed_seconds=time.monotonic()-started)
            if rank==0:
                log.write(json.dumps(item)+'\n')
                if step%10==0 or step==1:print('TRAIN',item,flush=True)
            if a.smoke:
                if rank==0:write_json(a.out/'smoke.json',dict(item,gradient_check='all active encoder/head parameters have finite synchronized gradients'))
                pool.shutdown();dist.destroy_process_group();return
            if step%c['validation_every']==0 or step%math.ceil(len(rows['train'])/c['batch_size'])==0 or step==total:
                m,pred=evaluate('validation');improved=m['macro_dataset_balanced_accuracy']>best
                best=max(best,m['macro_dataset_balanced_accuracy'])
                if rank==0:
                    log.write(json.dumps(dict(step=step,epoch=epoch+1,validation=m,elapsed_seconds=time.monotonic()-started))+'\n')
                    state=dict(encoder=encoder.state_dict(),head=head.state_dict(),optimizer=optimizer.state_dict(),step=step,best_score=best,config=provenance)
                    atomic_save(state,a.out/'last.pt')
                    if improved:
                        atomic_save(state,a.out/'best.pt');write_json(a.out/'best.json',dict(step=step,metrics=m));write_json(a.out/'validation_best.json',pred)
                    print('VALIDATION',step,m['accuracy'],m['macro_dataset_balanced_accuracy'],flush=True)
                dist.barrier()
    saved=torch.load(a.out/'best.pt',map_location='cpu',weights_only=False);encoder.load_state_dict(saved['encoder']);head.load_state_dict(saved['head'])
    m,pred=evaluate('heldout')
    if rank==0:
        write_json(a.out/'test_2b.json',dict(step=saved['step'],metrics=m,predictions=pred));write_json(a.out/'complete.json',dict(step=step,best_step=saved['step'],elapsed_seconds=time.monotonic()-started))
    pool.shutdown();dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
