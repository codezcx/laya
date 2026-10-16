"""Coverage-controlled actual JEV training. Use torchrun; pretrained LoRA always active."""
import argparse,json,random,time,subprocess,os,hashlib
from datetime import timedelta
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from .cache import ROOT,digest
from .data import TokenCache,metrics
from .jev import load_jev,JEVBridge,CHECKPOINT,PREFIX,SUFFIX
from .sampling import BalancedCoverageSampler
from .distributed import average_gradients,atomic_save


def features(cache,raw,encoder,ids):
    if encoder is None:return cache.batch(ids,'cuda')
    x,channels,y=raw.batch(ids,'cuda');z,m=encoder(x,channels)
    return z,m,y

@torch.no_grad()
def validate(model,cache,raw,encoder,ids):
    model.eval();values=[]
    if encoder is not None:encoder.eval()
    for start in range(0,len(ids),12):
        z,m,_=features(cache,raw,encoder,ids[start:start+12]);values.append(model(z,m)['logits'].cpu().numpy())
    return metrics([cache.rows[i] for i in ids],np.concatenate(values))

@torch.no_grad()
def distributed_validate(model,cache,ids,rank,world):
    model.eval();mine=ids[rank::world];values=[]
    for start in range(0,len(mine),12):
        z,m,_=cache.batch(mine[start:start+12],'cuda');values.append(model(z,m)['logits'].cpu().numpy())
        if rank==0 and start%1200==0:print('full source validation',start,'/',len(mine),'per rank',flush=True)
    part=(mine,np.concatenate(values));all_parts=[None]*world
    dist.all_gather_object(all_parts,part)
    logits={i:z for ii,zz in all_parts for i,z in zip(ii,zz)}
    assert len(logits)==len(ids) and set(logits)==set(ids)
    return metrics([cache.rows[i] for i in ids],np.stack([logits[i] for i in ids]))

def cpu_trainable(model,names):
    params=dict(model.named_parameters())
    return {name:params[name].detach().cpu().clone() for name in names}

def restore_trainable(model,state,names):
    if set(state)!=set(names):raise ValueError('Trainable parameter set changed')
    model.load_state_dict(state,strict=False)

def rng_state():
    return dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),python=random.getstate(),numpy=np.random.get_state())

def restore_rng(state):
    torch.set_rng_state(state['cpu']);torch.cuda.set_rng_state(state['cuda']);random.setstate(state['python']);np.random.set_state(state['numpy'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['projector','lora'],required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--rounds',type=int,default=2)
    p.add_argument('--per-group',type=int,default=4);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--smoke',action='store_true')
    p.add_argument('--stop-after',type=int);args=p.parse_args()
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local);torch.set_num_threads(4)
    dist.init_process_group('nccl',timeout=timedelta(minutes=60))
    if args.rounds<1 or args.per_group%world:raise ValueError('Invalid coverage/rank configuration')
    if rank==0:
        if args.resume:
            assert (args.out/'last.pt').exists() and not (args.out/'complete.json').exists()
        else:args.out.mkdir(parents=True,exist_ok=False)
    dist.barrier()
    torch.manual_seed(args.seed);random.seed(args.seed);np.random.seed(args.seed)
    cache=TokenCache();groups={}
    for i,r in enumerate(cache.rows):
        if r['split']=='train':groups.setdefault((r['dataset'],r['class_id']),[]).append(i)
    assert len(groups)==12
    for ds in {r['dataset'] for r in cache.rows}:
        a={r['subject'] for r in cache.rows if r['dataset']==ds and r['split']=='train'}
        b={r['subject'] for r in cache.rows if r['dataset']==ds and r['split']=='validation'}
        assert not a&b
    sampler=BalancedCoverageSampler(groups,args.per_group,args.seed)
    total=2 if args.smoke else args.rounds*sampler.steps_per_round
    warmup=0 if args.smoke else 50
    val=[i for i,r in enumerate(cache.rows) if r['split']=='validation']
    if args.smoke:
        val=[next(i for i in val if (cache.rows[i]['dataset'],cache.rows[i]['class_id'])==key) for key in sorted(groups)]
    train_lora=args.arm=='lora';text,w,b,tok=load_jev();torch.manual_seed(args.seed)
    model=JEVBridge(text,w,b,tok,train_lora)
    names=[n for n,p in model.named_parameters() if p.requires_grad]
    params=[dict(params=list(model.connector.parameters()),lr=1e-4)]
    if train_lora:params.append(dict(params=[p for p in model.text.parameters() if p.requires_grad],lr=1e-5))
    opt=torch.optim.AdamW(params,weight_decay=.01)
    code_paths=['eeg_align/train_jev.py','eeg_align/jev.py','eeg_align/data.py','eeg_align/sampling.py','eeg_align/distributed.py']
    config=dict(arm=args.arm,seed=args.seed,rounds=args.rounds,steps=total,steps_per_round=sampler.steps_per_round,
        per_group=args.per_group,world_size=world,batch_size=12*args.per_group,microbatch=6,warmup_steps=warmup,
        learning_rates=dict(projector=1e-4,lora=1e-5),jev_lora_always_active=True,train_lora=train_lora,train_encoder=False,
        smoke=args.smoke,checkpoint=str(CHECKPOINT.resolve()),prefix=PREFIX,suffix=SUFFIX,
        validation_ids=val,validation_n=len(val),selection='complete source validation at coverage-round endpoints',
        manifest_sha256=digest(ROOT/'rows.jsonl'),checkpoint_provenance_sha256=digest(Path('artifacts/jev_eeg/checkpoint_audit.json')),
        code_sha256={q:digest(q) for q in code_paths},
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),trainable_names=names)
    best=-1.;prior_seconds=0.;attempt=time.time()
    if args.resume:
        saved=torch.load(args.out/'last.pt',map_location='cpu',weights_only=False)
        previous=saved['config']
        for k,v in config.items():
            if k!='git_commit' and previous[k]!=v:raise ValueError(f'Resume config mismatch: {k}')
        config=previous;restore_trainable(model,saved['trainable'],names);opt.load_state_dict(saved['optimizer'])
        sampler.load_state_dict(saved['sampler']);best=saved['best_score'];prior_seconds=saved['elapsed_seconds']
        restore_rng(saved['rng_states'][rank]);assert sampler.step==saved['step'];del saved
    elif rank==0:(args.out/'config.json').write_text(json.dumps(config,indent=2))
    dist.barrier();began=time.monotonic()
    log=(args.out/'metrics.jsonl').open('a',buffering=1) if rank==0 else None
    for step in range(sampler.step+1,total+1):
        model.train();opt.zero_grad(set_to_none=True)
        for name,param in model.text.named_parameters():
            if name.endswith(('lora_A','lora_B')):param.requires_grad_(train_lora and step>warmup)
        global_ids=sampler.next_batch();ids=sampler.rank_batch(global_ids,rank,world);loss_sum=0.
        for start in range(0,len(ids),6):
            ii=ids[start:start+6];z,m,y=cache.batch(ii,'cuda')
            loss=model(z,m,y)['loss']*(len(ii)/len(ids));assert torch.isfinite(loss)
            loss.backward();loss_sum+=loss.item()
        active=[p for group in opt.param_groups for p in group['params'] if p.requires_grad]
        average_gradients(active,world);grad=torch.nn.utils.clip_grad_norm_(active,1.);opt.step()
        loss_tensor=torch.tensor(loss_sum,device='cuda');dist.all_reduce(loss_tensor);loss_tensor/=world
        if args.smoke:
            h=hashlib.sha256()
            for tensor in cpu_trainable(model,names).values():h.update(tensor.numpy().tobytes())
            hashes=[None]*world;dist.all_gather_object(hashes,h.hexdigest());assert len(set(hashes))==1,'Ranks diverged'
            if rank==0:print('parameter sync verified',step,hashes[0],flush=True)
        elapsed=prior_seconds+time.monotonic()-began
        if rank==0 and (step==1 or step%10==0 or args.smoke):
            item=dict(step=step,loss=float(loss_tensor),gradient_norm=float(grad),elapsed_seconds=elapsed,attempt=attempt,
                draws=step*config['batch_size'],coverage_round=sampler.round,peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9)
            log.write(json.dumps(item)+'\n');print(json.dumps(item),flush=True)
        endpoint=step%sampler.steps_per_round==0 or step==total
        if endpoint:
            result=distributed_validate(model,cache,val,rank,world)
            if result['selection_score']>best:
                best=result['selection_score']
                if rank==0:
                    atomic_save(dict(trainable=cpu_trainable(model,names),encoder=None,config=config,step=step,validation=result),args.out/'best.pt')
                    (args.out/'best.json').write_text(json.dumps(dict(step=step,full_source_validation=result),indent=2))
            if rank==0:
                log.write(json.dumps(dict(step=step,full_source_validation=result,attempt=attempt))+'\n')
                print('FULL SOURCE',step,json.dumps(result),flush=True)
        checkpoint=step%100==0 or endpoint or step==args.stop_after or args.smoke
        if checkpoint:
            states=[None]*world;dist.all_gather_object(states,rng_state())
            if rank==0:
                atomic_save(dict(trainable=cpu_trainable(model,names),optimizer=opt.state_dict(),sampler=sampler.state_dict(),
                    config=config,step=step,best_score=best,rng_states=states,elapsed_seconds=prior_seconds+time.monotonic()-began),args.out/'last.pt')
                (args.out/'progress.json').write_text(json.dumps(dict(step=step,steps=total,coverage_round=sampler.round),indent=2))
            dist.barrier()
        if step==args.stop_after and step<total:
            if rank==0:print('Controlled stop after checkpoint',step,flush=True)
            dist.destroy_process_group();return
    if rank==0:
        (args.out/'complete.json').write_text(json.dumps(dict(best_sha256=digest(args.out/'best.pt'),
            elapsed_seconds=prior_seconds+time.monotonic()-began,steps=total,smoke=args.smoke),indent=2))
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
