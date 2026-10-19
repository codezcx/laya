"""Full-montage, variable-duration MI caches and subject-disjoint trial batches."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from .mi_data import MITrials, read_trial, canonical_channel

CONFIG=Path('configs/mi_train.json')
def config():return json.loads(CONFIG.read_text())
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write_json(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))
def readrows(path):return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]

def source_rows():
    c=config();rows=[]
    for split in ['train','validation','heldout']:
        ds=MITrials(Path(c['index'])/f'{split}.jsonl',split=split)
        allowed=c['heldout_datasets'] if split=='heldout' else c['source_datasets']
        rows.extend(dict(r,id=hashlib.sha256(r['path'].encode()).hexdigest()[:24]) for r in ds.rows if r['dataset'] in allowed)
    assert len({r['id'] for r in rows})==len(rows)
    sets=[{(r['dataset'],r['subject']) for r in rows if r['split']==s} for s in ['train','validation','heldout']]
    assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
    assert len({r['signal_sha256'] for r in rows})==len(rows)
    return rows

def position_names():
    c=json.loads(Path('configs/encoder.json').read_text())
    names=json.loads((Path(c['positions_checkpoint'])/'config.json').read_text())['position_names']
    # Preserve exact bank spelling when possible, otherwise normalize case only.
    return names,{canonical_channel(n):n for n in names}

def preprocess(row):
    # Strict loader checks ALL retained EEG channels, not just C3/Cz/C4.
    x,y,m=read_trial(row['path'],row['dataset'],channels=None)
    names,lookup=position_names();excluded={'HEO','VEO','HEOG','VEOG','EOG','TRIG','TRIGGER','STI 014'}
    keep=[i for i,n in enumerate(m['channels']) if canonical_channel(n) not in excluded]
    labels=[m['channels'][i] for i in keep]
    unknown=[n for n in labels if canonical_channel(n) not in lookup]
    if unknown:raise ValueError(f"Unknown electrode positions in {row['dataset']}: {unknown}")
    mapped=[n if n in names else lookup[canonical_channel(n)] for n in labels]
    if len(set(map(canonical_channel,mapped)))!=len(mapped):raise ValueError('Duplicated position names')
    x=x[keep].copy();n=x.shape[-1]
    if n<200:raise ValueError('Epoch shorter than one REVE patch')
    x-=x.mean(-1,keepdims=True)
    x=np.clip(x,-1000,1000)/100
    # HF REVE uses unfold without padding. Extend only the final partial patch,
    # so original trailing samples are retained rather than silently discarded.
    target=200+math.ceil((n-200)/180)*180
    x=np.pad(x,((0,0),(0,target-n)),mode='edge').astype(np.float32)
    tokens=len(mapped)*(1+(target-200)//180)
    return x,mapped,dict(row,channels=labels,position_names=mapped,original_samples=n,
        padded_samples=target,padded_tail_samples=target-n,token_length=tokens,label=y,
        excluded_channels=[m['channels'][i] for i in range(len(m['channels'])) if i not in keep])

def encode(shard,world=4):
    import torch
    from transformers import AutoModel
    torch.set_num_threads(4);c=config();out=Path(c['cache']);out.mkdir(exist_ok=True,parents=True)
    done=out/f'shard_{shard}.json'
    if done.exists():raise FileExistsError(done)
    rows=source_rows()[shard::world]
    ec=json.loads(Path('configs/encoder.json').read_text())
    model,info=AutoModel.from_pretrained(ec['reve_checkpoint'],trust_remote_code=True,local_files_only=True,output_loading_info=True)
    assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    model=model.cuda().eval().requires_grad_(False)
    bank=AutoModel.from_pretrained(ec['positions_checkpoint'],trust_remote_code=True,local_files_only=True)
    binary=out/f'tokens_{shard}.bin';manifest=out/f'rows_{shard}.jsonl';offset=0;hist=Counter();groups=defaultdict(list)
    def flush(key,handle,records):
        nonlocal offset
        items=groups.pop(key);x=torch.from_numpy(np.stack([i[0] for i in items])).cuda()
        p=bank(list(key[0])).detach().cuda();assert p.shape==(x.shape[1],3)
        with torch.no_grad():z=model(x,p[None].expand(len(x),-1,-1)).flatten(1,2).cpu().numpy().astype(np.float16)
        assert np.isfinite(z).all()
        for token,(_,_,r) in zip(z,items):
            assert token.shape==(r['token_length'],512)
            token.tofile(handle);r.update(cache_shard=shard,cache_offset=offset)
            records.write(json.dumps(r)+'\n');offset+=token.size;hist[r['dataset']]+=1
    with binary.open('xb') as f,manifest.open('x') as records,ThreadPoolExecutor(max_workers=8) as pool:
        for i,item in enumerate(pool.map(preprocess,rows)):
            key=(tuple(item[1]),item[0].shape[-1]);groups[key].append(item)
            if len(groups[key])>=32:flush(key,f,records)
            if i%1000==0:print('ENCODE',shard,i,len(rows),flush=True)
        for key in list(groups):flush(key,f,records)
    write_json(done,dict(rows=len(rows),datasets=dict(hist),files={p.name:digest(p) for p in [binary,manifest]},
        config_sha256=digest(CONFIG),index_sha256=digest(Path(c['index'])/'complete.json'),code_sha256=digest(__file__),encoder=ec))
    print('ENCODE COMPLETE',shard,len(rows),flush=True)

def finalize():
    c=config();root=Path(c['cache']);rows=[];hashes={}
    for shard in range(4):
        p=root/f'shard_{shard}.json';v=json.loads(p.read_text())
        assert v['config_sha256']==digest(CONFIG) and v['index_sha256']==digest(Path(c['index'])/'complete.json')
        assert v['code_sha256']==digest(__file__)
        for name,h in v['files'].items():assert digest(root/name)==h;hashes[name]=h
        rows+=readrows(root/f'rows_{shard}.jsonl')
    expected={r['id'] for r in source_rows()};assert len(rows)==len(expected) and {r['id'] for r in rows}==expected
    rows.sort(key=lambda r:r['id']);counts={}
    for split in ['train','validation','heldout']:
        selected=[r for r in rows if r['split']==split];p=root/f'{split}.jsonl'
        p.write_text(''.join(json.dumps(r)+'\n' for r in selected));hashes[p.name]=digest(p)
        counts[split]={ds:dict(n=sum(r['dataset']==ds for r in selected),subjects=len({r['subject'] for r in selected if r['dataset']==ds}),labels=dict(Counter(r['label'] for r in selected if r['dataset']==ds))) for ds in sorted({r['dataset'] for r in selected})}
    write_json(root/'complete.json',dict(status='passed',files=hashes,counts=counts,config_sha256=digest(CONFIG),index_sha256=digest(Path(c['index'])/'complete.json'),code_sha256=digest(__file__),token_lengths=dict(Counter(r['token_length'] for r in rows))))
    print('CACHE AUDITED',len(rows),flush=True)

class Tokens:
    def __init__(self,split):
        c=config();root=Path(c['cache']);done=json.loads((root/'complete.json').read_text())
        assert done['status']=='passed' and done['config_sha256']==digest(CONFIG)
        assert done['code_sha256']==digest(__file__) and done['index_sha256']==digest(Path(c['index'])/'complete.json')
        path=root/f'{split}.jsonl';assert digest(path)==done['files'][path.name]
        self.rows=readrows(path);self.arrays=[np.memmap(root/f'tokens_{i}.bin',mode='r',dtype=np.float16) for i in range(4)]
    def batch(self,ids,device='cuda'):
        import torch
        records=[self.rows[int(i)] for i in ids];length=max(r['token_length'] for r in records)
        x=np.zeros((len(ids),length,512),dtype=np.float16);valid=np.zeros((len(ids),length),dtype=bool)
        for i,r in enumerate(records):
            n=r['token_length'];start=r['cache_offset'];x[i,:n]=self.arrays[r['cache_shard']][start:start+n*512].reshape(n,512);valid[i,:n]=True
        return torch.from_numpy(x).to(device),torch.from_numpy(valid).to(device),torch.tensor([r['label'] for r in records],device=device)

def epoch_batches(rows,seed,epoch,batch_size):
    # Shuffle every trial, sort only inside large random pools to reduce padding,
    # then shuffle batches. Every trial appears once, including short final pools.
    rng=np.random.default_rng(seed+epoch);order=rng.permutation(len(rows));batches=[]
    for start in range(0,len(order),batch_size*32):
        pool=sorted(order[start:start+batch_size*32],key=lambda i:rows[i]['token_length'])
        batches.extend(np.array_split(pool,max(1,math.ceil(len(pool)/batch_size))))
    rng.shuffle(batches)
    return batches

def metrics(predictions):
    from sklearn.metrics import confusion_matrix,roc_auc_score
    def calc(rows):
        y=np.array([r['label'] for r in rows]);p=np.array([r['probability_right'] for r in rows]);pred=(p>.5).astype(int)
        cm=confusion_matrix(y,pred,labels=[0,1]);rec=np.divide(cm.diagonal(),cm.sum(1),out=np.zeros(2),where=cm.sum(1)>0)
        return dict(n=len(rows),accuracy=float((y==pred).mean()),balanced_accuracy=float(rec.mean()),left_recall=float(rec[0]),right_recall=float(rec[1]),confusion=cm.tolist(),auroc=float(roc_auc_score(y,p)) if len(set(y))==2 else None)
    out=calc(predictions);out['datasets']={}
    for ds in sorted({r['dataset'] for r in predictions}):
        subset=[r for r in predictions if r['dataset']==ds];v=calc(subset)
        v['subjects']={s:calc([r for r in subset if r['subject']==s]) for s in sorted({r['subject'] for r in subset})}
        out['datasets'][ds]=v
    out['macro_dataset_accuracy']=float(np.mean([r['accuracy'] for r in out['datasets'].values()]))
    out['macro_dataset_balanced_accuracy']=float(np.mean([r['balanced_accuracy'] for r in out['datasets'].values()]))
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int);p.add_argument('--finalize',action='store_true');a=p.parse_args()
    if a.finalize:finalize()
    else:
        assert a.shard in range(4);encode(a.shard)
