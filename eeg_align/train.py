"""Source-only training of the task/class text aligner and matched CE control."""
import argparse
import json
import random
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from .cache import digest, ROOT
from .data import TokenCache, metrics
from .model import EEGTextAligner

BANK = Path('artifacts/eeg_text_alignment/embedding_bank')

def load_bank():
    audit=json.loads((BANK/'complete.json').read_text())
    for name,sha in audit['hashes'].items():
        assert digest(BANK/name)==sha
    rows=[json.loads(s) for s in (BANK/'rows.jsonl').read_text().splitlines()]
    vectors=np.load(BANK/'last.npy')
    train=np.array([r['split']=='train' for r in rows])
    labels=np.array([r['class_id'] for r in rows])
    return torch.from_numpy(vectors[train]),torch.from_numpy(labels[train])

@torch.no_grad()
def validate(model,cache,ids):
    model.eval();values=[]
    for start in range(0,len(ids),256):
        x,m,_=cache.batch(ids[start:start+256],'cuda')
        values.append(model(x,m)['logits'].cpu().numpy())
    return metrics([cache.rows[i] for i in ids],np.concatenate(values))

def main():
    p=argparse.ArgumentParser();p.add_argument('--control',choices=['contrastive','ce'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    cache=TokenCache();val=[i for i,r in enumerate(cache.rows) if r['split']=='validation']
    groups={}
    for i,r in enumerate(cache.rows):
        if r['split']=='train':groups.setdefault((r['dataset'],r['class_id']),[]).append(i)
    assert len(groups)==12 and val
    for dataset in {r['dataset'] for r in cache.rows}:
        subjects={s:{r['subject'] for r in cache.rows if r['dataset']==dataset and r['split']==s} for s in ['train','validation']}
        assert not subjects['train']&subjects['validation']
    vectors,labels=load_bank();model=EEGTextAligner(vectors,labels,args.control).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    config=dict(control=args.control,seed=args.seed,steps=args.steps,batch_size=120,lr=1e-4,
                groups=[list(k) for k in sorted(groups)],validation_interval=100,
                git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                source_manifest_sha256=digest(ROOT/'rows.jsonl'),text_bank_sha256=digest(BANK/'complete.json'),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                source_hashes={str(p):digest(p) for p in [Path(__file__),Path('eeg_align/model.py'),Path('eeg_align/data.py')]})
    (args.out/'config.json').write_text(json.dumps(config,indent=2))
    rng=np.random.default_rng(args.seed);best=-1.;began=time.monotonic()
    with (args.out/'metrics.jsonl').open('w',buffering=1) as log:
        for step in range(1,args.steps+1):
            model.train();ids=np.concatenate([rng.choice(groups[k],10,replace=True) for k in sorted(groups)])
            x,m,y=cache.batch(ids,'cuda');opt.zero_grad(set_to_none=True)
            loss=model(x,m,y)['loss'];assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            if step==1 or step%20==0:
                item=dict(step=step,loss=loss.item(),elapsed_seconds=time.monotonic()-began)
                log.write(json.dumps(item)+'\n');print(json.dumps(item),flush=True)
            if step%100==0 or step==args.steps:
                result=validate(model,cache,val);item=dict(step=step,validation=result)
                log.write(json.dumps(item)+'\n');print(json.dumps(item),flush=True)
                if result['selection_score']>best:
                    best=result['selection_score']
                    torch.save(dict(model=model.state_dict(),config=config,step=step,validation=result),args.out/'best.pt')
                    (args.out/'best.json').write_text(json.dumps(item,indent=2))
    (args.out/'complete.json').write_text(json.dumps(dict(best_sha256=digest(args.out/'best.pt'),
        config_sha256=digest(args.out/'config.json'),elapsed_seconds=time.monotonic()-began),indent=2))

if __name__=='__main__':main()
