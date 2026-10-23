"""Frozen joint REVE -> global token mean -> independently fitted linear probes."""
import argparse,json,os,time,warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from .joint_data import read_rows,raw,ROOT
from .mi_training_data import digest,write_json


def metrics(rows,probabilities):
    def calc(rr,pp):
        y=np.array([r['label'] for r in rr]);pred=pp.argmax(-1)
        return dict(n=len(y),accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),confusion_matrix=confusion_matrix(y,pred,labels=np.arange(pp.shape[1])).tolist())
    results={}
    for ds in sorted({r['dataset'] for r in rows}):
        ids=[i for i,r in enumerate(rows) if r['dataset']==ds];rr=[rows[i] for i in ids];pp=probabilities[ids];parents=defaultdict(list)
        for i,r in enumerate(rr):parents[r['parent_id']].append(i)
        trial_rows=[rr[ix[0]] for ix in parents.values()];trial_prob=np.stack([pp[ix].mean(0) for ix in parents.values()])
        results[ds]=dict(window=calc(rr,pp),trial=calc(trial_rows,trial_prob))
    return dict(datasets=results,macro_dataset_balanced_accuracy=float(np.mean([v['trial']['balanced_accuracy'] for v in results.values()])))


def fit_cached(out,world,rows,started):
    arrays={}
    for split,rr in rows.items():
        x=np.empty((len(rr),512),np.float32);seen=[]
        for shard in range(world):
            d=np.load(out/f'{split}_rank{shard}.npz');x[d['indices']]=d['features'];seen.extend(d['indices'].tolist())
        assert sorted(seen)==list(range(len(rr)));arrays[split]=x
    results={}
    with threadpool_limits(limits=1):
        for task,name,k in [(0,'MI',2),(1,'SEED',3)]:
            ix={s:[i for i,r in enumerate(rr) if r['task']==task] for s,rr in rows.items()}
            rr={s:[rows[s][i] for i in ii] for s,ii in ix.items()}
            x={s:arrays[s][ii] for s,ii in ix.items()};y=np.array([r['label'] for r in rr['train']])
            assert set(y)==set(range(k));scaler=StandardScaler();xtrain=scaler.fit_transform(x['train']);xval=scaler.transform(x['validation'])
            classifier=LogisticRegression(C=1.,max_iter=2000,tol=1e-5,solver='lbfgs',random_state=42)
            write_json(out/'progress.json',dict(stage='fit_probe',task=name))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always',ConvergenceWarning);classifier.fit(xtrain,y)
            converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
            result=dict(converged=converged,iterations=classifier.n_iter_.tolist(),train=metrics(rr['train'],classifier.predict_proba(xtrain)),validation=metrics(rr['validation'],classifier.predict_proba(xval)))
            results[name]=result;write_json(out/f'{name}_metrics.json',result)
            np.savez(out/f'{name}_linear.npz',weight=classifier.coef_,bias=classifier.intercept_,classes=classifier.classes_,feature_mean=scaler.mean_,feature_scale=scaler.scale_)
            print('PROBE',name,json.dumps(result),flush=True)
    write_json(out/'results.json',results);write_json(out/'complete.json',dict(seconds=time.monotonic()-started,converged=all(r['converged'] for r in results.values())))


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--fit-only',action='store_true');a=p.parse_args()
    if a.fit_only:
        config=json.loads((a.out/'config.json').read_text());assert config['manifest_sha256']==digest(ROOT/'complete.json')
        assert not (a.out/'complete.json').exists()
        rows={s:[r for r in read_rows(s) if r['task'] in (0,1)] for s in ['train','validation']}
        shards=sorted(a.out.glob('train_rank*.npz'));assert len(shards)==2
        fit_cached(a.out,len(shards),rows,time.monotonic());return
    if a.checkpoint is None:raise ValueError('Checkpoint required for extraction')
    rank,world,local=[int(os.environ[k]) for k in ['RANK','WORLD_SIZE','LOCAL_RANK']]
    torch.cuda.set_device(local);torch.set_num_threads(3);dist.init_process_group('nccl',timeout=timedelta(minutes=60))
    if rank==0:a.out.mkdir(parents=True,exist_ok=False)
    dist.barrier();started=time.monotonic();saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    assert saved['config']['manifest_sha256']==digest(ROOT/'complete.json')
    ec=saved['config']['encoder'];encoder,info=AutoModel.from_pretrained(ec['reve_checkpoint'],trust_remote_code=True,local_files_only=True,output_loading_info=True)
    assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    encoder.load_state_dict(saved['encoder'],strict=True);encoder=encoder.cuda().eval().requires_grad_(False)
    config=dict(checkpoint=str(a.checkpoint),checkpoint_sha256=digest(a.checkpoint),encoder_step=saved['step'],encoder_frozen=True,contrast_head_used=False,readout='mean all real REVE channel/time tokens -> StandardScaler(train only) -> logistic Linear(512,K)',probe_C=1.,max_iter=2000,manifest_sha256=digest(ROOT/'complete.json'),split='unchanged train/validation; no heldout/test access')
    if rank==0:write_json(a.out/'config.json',config)
    del saved
    rows={s:[r for r in read_rows(s) if r['task'] in (0,1)] for s in ['train','validation']}
    with ThreadPoolExecutor(max_workers=4) as pool,torch.no_grad():
        for split,allrows in rows.items():
            mine=list(range(rank,len(allrows),world));features=np.empty((len(mine),512),np.float32)
            for start in range(0,len(mine),64):
                ix=mine[start:start+64];records=[allrows[i] for i in ix];groups=defaultdict(list)
                for j,(x,pos) in enumerate(pool.map(raw,records)):groups[(x.shape,pos.tobytes())].append((j,x,pos))
                for group in groups.values():
                    for offset in range(0,len(group),16):
                        part=group[offset:offset+16];x=torch.from_numpy(np.stack([v[1] for v in part])).cuda();pos=torch.from_numpy(np.stack([v[2] for v in part])).cuda()
                        with torch.autocast('cuda',dtype=torch.bfloat16):z=encoder(x,pos).mean((1,2))
                        z=z.float().cpu().numpy();assert np.isfinite(z).all()
                        for (j,_,_),v in zip(part,z):features[start+j]=v
                if start%2048==0:
                    print('ENCODE',rank,split,min(start+64,len(mine)),len(mine),flush=True)
                    write_json(a.out/f'progress_rank{rank}.json',dict(stage='encode',split=split,done=min(start+64,len(mine)),total=len(mine),seconds=time.monotonic()-started))
            np.savez(a.out/f'{split}_rank{rank}.npz',indices=np.array(mine),features=features)
    del encoder;torch.cuda.empty_cache();dist.barrier()
    if rank==0:fit_cached(a.out,world,rows,started)
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
