"""Four-GPU source-only connector training with explicit task-free metrics."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer
from .cache_multitask_v30 import ROOT, digest
from .model import load_text_only
from .multitask_bridge import MultitaskBridge, PREFIX, SUFFIX, CLASSES


class TokenCache:
    def __init__(self):
        verified = json.loads((ROOT/'verified.json').read_text())
        assert verified['manifest_sha256'] == digest(ROOT/'rows.jsonl')
        self.rows = [json.loads(s) for s in (ROOT/'rows.jsonl').read_text().splitlines()]
        self.location = {}; self.arrays = []; self.lengths = []
        for shard in range(4):
            folder = ROOT/str(shard)
            self.arrays.append(np.load(folder/'tokens.npy', mmap_mode='r'))
            self.lengths.append(np.load(folder/'lengths.npy'))
            for j,i in enumerate(np.load(folder/'indices.npy')):
                assert int(i) not in self.location
                self.location[int(i)] = (shard,j)
        assert sorted(self.location) == list(range(len(self.rows)))

    def batch(self, ids, device):
        locations = [self.location[int(i)] for i in ids]
        lengths = [int(self.lengths[s][j]) for s,j in locations]
        x = np.stack([self.arrays[s][j,:max(lengths)] for s,j in locations])
        tokens = torch.from_numpy(x.astype(np.float32)).to(device)
        mask = torch.arange(max(lengths),device=device)[None] < torch.tensor(lengths,device=device)[:,None]
        labels = torch.tensor([self.rows[int(i)]['class_id'] for i in ids], device=device)
        return tokens,mask,labels


def metrics(rows, logits):
    y = np.array([r['class_id'] for r in rows]); pred = logits.argmax(1)
    task = (y>=3); routing = (pred>=3)
    oracle = logits.copy()
    oracle[task,:3] = -np.inf; oracle[~task,3:] = -np.inf
    within = oracle.argmax(1); result = {}
    for ds in sorted({r['dataset'] for r in rows}):
        people = []
        for subject in sorted({r['subject'] for r in rows if r['dataset']==ds}):
            ii = np.array([i for i,r in enumerate(rows) if r['dataset']==ds and r['subject']==subject])
            classes = np.unique(y[ii])
            people.append(dict(joint=float(np.mean([np.mean(pred[ii][y[ii]==c]==c) for c in classes])),
                               oracle=float(np.mean([np.mean(within[ii][y[ii]==c]==c) for c in classes])),
                               routing=float(np.mean(routing[ii]==task[ii]))))
        result[ds] = {k:float(np.mean([v[k] for v in people])) for k in ['joint','oracle','routing']}
    tasks = [['SEED','FACED-NEMAR'], ['Cho2017','Dreyer2023','Lee2019_MI']]
    score = float(np.mean([np.mean([result[d]['joint'] for d in names]) for names in tasks]))
    return dict(datasets=result, selection_score=score, n=len(y))


@torch.no_grad()
def evaluate(model, cache, ids, rank, world, device):
    model.eval(); local = []
    for start in range(0,len(ids[rank::world]),8):
        ii = ids[rank::world][start:start+8]
        tokens,mask,_ = cache.batch(ii,device)
        logits = model(tokens,mask)['logits'].cpu().numpy()
        local.extend(zip(ii,logits))
    gathered = [None]*world
    dist.all_gather_object(gathered,local)
    if rank: return None
    pairs = sorted([p for part in gathered for p in part], key=lambda p:p[0])
    assert [i for i,_ in pairs] == sorted(ids)
    return metrics([cache.rows[i] for i,_ in pairs], np.stack([z for _,z in pairs]))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--steps',type=int,default=500)
    parser.add_argument('--output',required=True)
    parser.add_argument('--arm',choices=['qwen','signal'],default='qwen'); args = parser.parse_args()
    assert args.steps > 0
    rank = int(os.environ['RANK']); local = int(os.environ['LOCAL_RANK']); world = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local); device = torch.device(f'cuda:{local}')
    torch.set_num_threads(4); torch.manual_seed(42)
    dist.init_process_group('nccl',device_id=device)
    out = Path(args.output)
    if rank==0: out.mkdir(parents=True,exist_ok=False)
    dist.barrier()
    cache = TokenCache(); rows = cache.rows
    train = [i for i,r in enumerate(rows) if r['split']=='train']
    val = [i for i,r in enumerate(rows) if r['split']=='validation']
    groups = {}
    for i in train:
        r = rows[i]; groups.setdefault((r['dataset'],r['class_id']),[]).append(i)
    assert len(groups)==12
    # Both tasks have six dataset/class groups; equal groups imply equal tasks.
    keys = sorted(groups); rng = np.random.default_rng(4200+rank)
    cfg = json.loads(Path('configs/reve_qwen_mi_multi.json').read_text())
    if args.arm=='qwen':
        tokenizer = AutoTokenizer.from_pretrained(cfg['qwen_checkpoint'],local_files_only=True)
        ids = [tokenizer.encode(s,add_special_tokens=False) for s in 'ABCDE']
        assert all(len(i)==1 for i in ids)
        text,head,audit = load_text_only(cfg['qwen_checkpoint'],[i[0] for i in ids],str(device),torch.bfloat16)
        enc = lambda s: torch.tensor(tokenizer.encode(s,add_special_tokens=False),device=device)
        model = MultitaskBridge(text,head,enc(PREFIX),enc(SUFFIX)).to(device)
    else:
        from .signal_control_v30 import SignalControl
        model = SignalControl().to(device); audit = {'language_model_loaded':False}
    wrapped = DistributedDataParallel(model,device_ids=[local],broadcast_buffers=False)
    opt = torch.optim.AdamW(model.connector.parameters(),lr=1e-4,weight_decay=.01)
    if rank==0:
        (out/'run.json').write_text(json.dumps(dict(seed=42,rank_seeds=[4200+i for i in range(world)],
            steps=args.steps,world=world,batch_per_rank=12,learning_rate=1e-4,train_n=len(train),val_n=len(val),
            source_manifest_sha256=digest(ROOT/'rows.jsonl'),cache_audit_sha256=digest(ROOT/'verified.json'),
            arm=args.arm,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            trainer_sha256=digest(Path(__file__)),
            qwen_checkpoint=cfg['qwen_checkpoint'] if args.arm=='qwen' else None,loading=audit,
            prefix=PREFIX if args.arm=='qwen' else None,suffix=SUFFIX if args.arm=='qwen' else None,classes=CLASSES,
            selection='task macro of dataset macro of subject balanced joint five-class accuracy',
            source_only=True,trainable='connector only'),indent=2))
    best = -1.; start = time.monotonic()
    for step in range(args.steps+1):
        if step:
            model.train(); opt.zero_grad(set_to_none=True)
            selected = [int(rng.choice(groups[k])) for k in keys]
            # Three microbatches, each rank sees every source/class once per update.
            loss_sum = torch.zeros((),device=device)
            for begin in range(0,12,4):
                tokens,mask,y = cache.batch(selected[begin:begin+4],device)
                # Synchronize every microbatch to avoid hidden DDP accumulation assumptions.
                result = wrapped(tokens,mask,y)
                assert torch.isfinite(result['loss'])
                (result['loss']/3).backward(); loss_sum += result['loss'].detach()/3
            norm = torch.nn.utils.clip_grad_norm_(model.connector.parameters(),1.)
            assert torch.isfinite(norm)
            opt.step(); dist.all_reduce(loss_sum); loss_sum /= world
            if rank==0 and (step==1 or step%10==0):
                record = dict(step=step,loss=float(loss_sum),seconds=time.monotonic()-start)
                with (out/'train.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
                print(record,flush=True)
        if step%100==0 or step==args.steps:
            result = evaluate(model,cache,val,rank,world,device)
            if rank==0:
                result.update(step=step,seconds=time.monotonic()-start)
                with (out/'validation.jsonl').open('a') as f: f.write(json.dumps(result)+'\n')
                if result['selection_score'] > best:
                    best = result['selection_score']; torch.save(model.connector.state_dict(),out/'best.pt')
                    (out/'best_metrics.json').write_text(json.dumps(result,indent=2))
                print('validation',result,flush=True)
            dist.barrier()
    if rank==0:
        torch.save(model.connector.state_dict(),out/'last.pt')
        (out/'complete.json').write_text(json.dumps(dict(best_sha256=digest(out/'best.pt'),
            last_sha256=digest(out/'last.pt'),steps=args.steps,seconds=time.monotonic()-start),indent=2))
    dist.destroy_process_group()


if __name__ == '__main__': main()
