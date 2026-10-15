"""Source-only frozen-backbone interface diagnosis. Never reads target cache."""
import argparse, json, time, hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer
from .dg_pilot import Features, BASE_CONFIG, prompt_bank, qwen_forward, write_json
from .model import load_model

ROOT = Path('artifacts/mi_interface_v2')

def examples(data, split, task, tiny=False):
    rng = np.random.default_rng(42)
    groups = {}
    for i,r in enumerate(data.rows):
        if r['original_split'] != split or r['dataset'] == 'Lee2019_MI': continue
        for t in ([0] if task == 'mi' else [1,2]):
            if t and r['aux_margins'][t-1] < .1: continue
            y = r['label'] if t == 0 else r['aux_labels'][t-1]
            groups.setdefault((r['dataset'],t,y), []).append((i,t,y))
    assert len(groups) == (4 if task == 'mi' else 8)
    # Equal dataset/task/label weight; tiny MI 32 trials, QA <=64 trials.
    n = 8 if tiny else (min(map(len,groups.values())) if split == 'train' else min(64,min(map(len,groups.values()))))
    items = [g[j] for g in groups.values() for j in rng.choice(len(g),n,replace=False)]
    rng.shuffle(items)
    return items

@torch.no_grad()
def evaluate(model,data,items,bank,perturb=None,swap=0):
    model.eval(); correct=[]; losses=[]; predictions=[]
    shuffled = np.random.default_rng(723).permutation(len(items))
    for begin in range(0,len(items),16):
        rows=items[begin:begin+16]
        ids=[items[shuffled[begin+j]][0] if perturb=='shuffle' else r[0] for j,r in enumerate(rows)]
        z,mask,_=data.get(ids)
        if perturb=='zero': z.zero_()
        tasks=[r[1] for r in rows]; y=torch.tensor([r[2]^swap for r in rows],device='cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=qwen_forward(model,z,mask,tasks,[swap]*len(rows),bank)
        losses.extend(F.cross_entropy(logits,y,reduction='none').tolist())
        correct.extend((logits.argmax(-1)==y).tolist())
        predictions.extend((logits.argmax(-1)^swap).tolist())
    per_task={str(t):float(np.mean([c for c,r in zip(correct,items) if r[1]==t])) for t in sorted({r[1] for r in items})}
    return {'n':len(items),'accuracy':float(np.mean(correct)),'loss':float(np.mean(losses)),'per_task':per_task,'predictions':predictions}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--arm',choices=['tiny_mi','tiny_qa','source_mi','source_qa'],required=True); p.add_argument('--steps',type=int,default=300)
    args=p.parse_args(); out=ROOT/args.arm; out.mkdir(parents=True,exist_ok=True)
    if (out/'run.json').exists(): raise FileExistsError(out)
    torch.set_num_threads(4); torch.manual_seed(42); np.random.seed(42)
    task='mi' if args.arm.endswith('mi') else 'qa'; tiny=args.arm.startswith('tiny')
    mode='legacy' if task=='mi' else 'harmonized'; data=Features(mode)
    train=examples(data,'train',task,tiny); val=examples(data,'validation',task)
    assert not {data.rows[r[0]]['path'] for r in train}&{data.rows[r[0]]['path'] for r in val}
    cfg=json.loads(BASE_CONFIG.read_text()); write_json(out/'run.json',{'args':vars(args),'config':cfg,'mode':mode,'train_examples':len(train),'val_examples':len(val),'train':train,'validation':val,'frozen':['REVE','Qwen'],'lr':1e-4,'seed':42,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    model=load_model(cfg,stage='warmup'); model.encoder.cpu()
    assert not any(p.requires_grad for p in model.text.parameters())
    bank=prompt_bank(AutoTokenizer.from_pretrained(cfg['qwen_checkpoint'],local_files_only=True),'cuda:0')
    params=list(model.connector.parameters())+list(model.head.parameters())
    opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01)
    rng=np.random.default_rng(43); started=time.monotonic(); best=-1
    # Tiny arms test capacity with fixed options; full QA randomizes A/B.
    for step in range(1,args.steps+1):
        model.train(); opt.zero_grad(set_to_none=True); losses=[]
        for micro in range(2):
            batch=[train[int(i)] for i in rng.integers(len(train),size=8)]
            z,mask,_=data.get([r[0] for r in batch]); swaps=rng.integers(2,size=8) if task=='qa' and not tiny else np.zeros(8,dtype=int)
            labels=torch.tensor([r[2]^int(s) for r,s in zip(batch,swaps)],device='cuda')
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=qwen_forward(model,z,mask,[r[1] for r in batch],swaps,bank)
                loss=F.cross_entropy(logits,labels)
            (loss/2).backward(); losses.append(loss.item())
        grad=torch.nn.utils.clip_grad_norm_(params,1.); opt.step()
        if step==1 or step%10==0:
            log={'step':step,'loss':float(np.mean(losses)),'gradient_norm':float(grad),'seconds':time.monotonic()-started}
            with (out/'train.jsonl').open('a') as f: f.write(json.dumps(log)+'\n')
            print(json.dumps(log),flush=True)
        if step%100==0 or step==args.steps:
            scores={'step':step,'train':evaluate(model,data,train if tiny else train[:256],bank),'validation':evaluate(model,data,val,bank)}
            with (out/'validation.jsonl').open('a') as f: f.write(json.dumps(scores)+'\n')
            print(json.dumps({k:({a:b for a,b in v.items() if a!='predictions'} if isinstance(v,dict) else v) for k,v in scores.items()}),flush=True)
            score=scores['train' if tiny else 'validation']['accuracy']
            if score>best:
                best=score; model.save(out/'best'); write_json(out/'best_metrics.json',scores)
    # Reload the selected connector/head, without loading a second backbone.
    from safetensors.torch import load_file
    saved=load_file(str(out/'best/bridge.safetensors'),device='cuda:0')
    model.connector.load_state_dict({k[10:]:v for k,v in saved.items() if k.startswith('connector.')})
    model.head.load_state_dict({k[5:]:v for k,v in saved.items() if k.startswith('head.')})
    checks={}
    for name,items in [('train',train if tiny else train[:256]),('validation',val)]:
        checks[name]={}
        for perturb in [None,'shuffle','zero']:
            checks[name][perturb or 'normal']=evaluate(model,data,items,bank,perturb)
        checks[name]['swapped']=evaluate(model,data,items,bank,swap=1)
    write_json(out/'diagnostics.json',checks)
    write_json(out/'complete.json',{'seconds':time.monotonic()-started,'steps':args.steps,'best_accuracy':best,'target_accessed':False})

if __name__=='__main__': main()
