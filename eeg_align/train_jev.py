"""Matched JEV experiments; the pretrained JEV adapter is active in every arm."""
import argparse,json,random,time,subprocess
from pathlib import Path
import numpy as np
import torch
from .cache import ROOT,digest
from .data import TokenCache,metrics
from .signals import SignalCache,SignalEncoder,RAW
from .jev import load_jev,JEVBridge,CHECKPOINT,PREFIX,SUFFIX

ARMS={'projector':(False,False),'lora':(True,False),'lora_encoder':(True,True)}

def validation_panel(rows):
    groups={}
    for i,r in enumerate(rows):
        if r['split']=='validation':groups.setdefault((r['dataset'],r['subject'],r['class_id']),[]).append(i)
    rng=np.random.default_rng(20261016)
    return sorted(int(i) for k in sorted(groups) for i in rng.choice(groups[k],min(2,len(groups[k])),replace=False))

def features(cache,raw,encoder,ids):
    if encoder is None:return cache.batch(ids,'cuda')
    x,channels,y=raw.batch(ids,'cuda');z,m=encoder(x,channels)
    return z,m,y

@torch.no_grad()
def validate(model,cache,raw,encoder,ids):
    model.eval();values=[]
    if encoder is not None:encoder.eval()
    for start in range(0,len(ids),12):
        z,m,y=features(cache,raw,encoder,ids[start:start+12])
        values.append(model(z,m)['logits'].cpu().numpy())
    return metrics([cache.rows[i] for i in ids],np.concatenate(values))

def save(path,model,encoder,config,step,result):
    # Save changed tensors only; original backbone/adapter remain referenced by provenance.
    state={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad}
    torch.save(dict(trainable=state,encoder=None if encoder is None else encoder.state_dict(),
                    config=config,step=step,validation=result),path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--steps',type=int,default=500);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    cache=TokenCache();train=[i for i,r in enumerate(cache.rows) if r['split']=='train'];val=validation_panel(cache.rows)
    for ds in {r['dataset'] for r in cache.rows}:
        a={r['subject'] for r in cache.rows if r['dataset']==ds and r['split']=='train'}
        b={r['subject'] for r in cache.rows if r['dataset']==ds and r['split']=='validation'}
        assert not a&b
    groups={}
    for i in train:
        r=cache.rows[i];groups.setdefault((r['dataset'],r['class_id']),[]).append(i)
    assert len(groups)==12
    train_lora,train_encoder=ARMS[args.arm]
    text,w,b,tok=load_jev();torch.manual_seed(args.seed)
    model=JEVBridge(text,w,b,tok,train_lora)
    encoder=SignalEncoder(trainable=True).eval() if train_encoder else None
    raw=SignalCache() if train_encoder else None
    params=[dict(params=model.connector.parameters(),lr=1e-4)]
    if train_lora:params.append(dict(params=[p for n,p in model.text.named_parameters() if p.requires_grad],lr=1e-5))
    if train_encoder:params.append(dict(params=encoder.parameters(),lr=1e-6))
    opt=torch.optim.AdamW(params,weight_decay=.01)
    config=dict(arm=args.arm,seed=args.seed,steps=args.steps,warmup_steps=50,batch_size=12,microbatch=6,
        learning_rates=dict(projector=1e-4,lora=1e-5,encoder=1e-6),jev_lora_always_active=True,
        checkpoint=str(CHECKPOINT.resolve()),prefix=PREFIX,suffix=SUFFIX,head='pretrained JEV A-E effective rows, frozen',
        train_lora=train_lora,train_encoder=train_encoder,validation_ids=val,validation_panel_n=len(val),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        manifest_sha256=digest(ROOT/'rows.jsonl'),checkpoint_provenance_sha256=digest(Path('artifacts/jev_eeg/checkpoint_audit.json')),
        code_sha256={str(q):digest(q) for q in [Path(__file__),Path('eeg_align/jev.py'),Path('eeg_align/signals.py'),Path('eeg_align/data.py')]},
        trainable_counts=dict(projector=sum(p.numel() for p in model.connector.parameters()),
            lora=sum(p.numel() for p in model.text.parameters() if p.requires_grad),
            encoder=0 if encoder is None else sum(p.numel() for p in encoder.parameters() if p.requires_grad)))
    (args.out/'config.json').write_text(json.dumps(config,indent=2))
    rng=np.random.default_rng(args.seed);best=-1;began=time.monotonic()
    with (args.out/'metrics.jsonl').open('w',buffering=1) as log:
        for step in range(1,args.steps+1):
            model.train();opt.zero_grad(set_to_none=True)
            # First 50 updates warm up the new modality connector. Adapter remains active.
            warm=step<=50
            for n,param in model.text.named_parameters():
                if n.endswith(('lora_A','lora_B')):param.requires_grad_(train_lora and not warm)
            if encoder is not None:encoder.requires_grad_(not warm).eval()
            ids=[int(rng.choice(groups[k])) for k in sorted(groups)];loss_sum=0.
            for start in range(0,len(ids),6):
                ii=ids[start:start+6]
                # Match the exact cached fp16 encoder output in all warmup arms.
                z,m,y=features(cache,raw,None if warm else encoder,ii)
                out=model(z,m,y);loss=out['loss']/2
                assert torch.isfinite(loss);loss.backward();loss_sum+=loss.item()
            all_params=[p for group in opt.param_groups for p in group['params'] if p.grad is not None]
            norm=torch.nn.utils.clip_grad_norm_(all_params,1.);assert torch.isfinite(norm)
            opt.step()
            if step==1 or step%10==0:
                item=dict(step=step,loss=loss_sum,elapsed_seconds=time.monotonic()-began,peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9)
                log.write(json.dumps(item)+'\n');print(json.dumps(item),flush=True)
            if step%100==0 or step==args.steps:
                result=validate(model,cache,raw,encoder,val);item=dict(step=step,validation_panel=result)
                log.write(json.dumps(item)+'\n');print(json.dumps(item),flush=True)
                if result['selection_score']>best:
                    best=result['selection_score'];save(args.out/'best.pt',model,encoder,config,step,result)
                    (args.out/'best.json').write_text(json.dumps(item,indent=2))
    (args.out/'complete.json').write_text(json.dumps(dict(best_sha256=digest(args.out/'best.pt'),
        elapsed_seconds=time.monotonic()-began),indent=2))

if __name__=='__main__':main()
