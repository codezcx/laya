"""Subject-disjoint MI/MAT manifests and task/class/record sampling."""
import hashlib,json,math
from collections import defaultdict,Counter
from functools import lru_cache
from pathlib import Path
import numpy as np
from .mi_training_data import preprocess,digest
from .task_options import load_tasks
ROOT=Path('datasets/mi_mat')
TASKS=list(load_tasks().values())
OFFSETS=np.cumsum([0]+[len(t['classes']) for t in TASKS[:-1]]).tolist()
BATCH_SIZES=json.loads(Path('configs/mi_emotion_train.json').read_text())['task_batch_sizes']


def prepare():
    import mne
    if (ROOT/'complete.json').exists():raise FileExistsError('Manifest already complete')
    ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'signals').mkdir(exist_ok=True)
    source=Path('datasets/mi_emotion');old=json.loads((source/'complete.json').read_text());rows=[]
    for split in ['train','validation','heldout']:
        path=source/f'{split}.jsonl';assert digest(path)==old['manifest_sha256'][split]
        rows.extend(r for line in path.read_text().splitlines() if (r:=json.loads(line))['task']==0)
    folder=Path('/inspire/hdd/global_user/zhangchenxi-253108310322/mne_data/EEGMat')
    paths=sorted(folder.glob('Subject??_?.edf'));assert len(paths)==72
    for path in paths:
        subject=int(path.stem[7:9]);condition=int(path.stem[-1]);assert 0<=subject<36 and condition in (1,2)
        split='train' if subject<28 else 'validation' if subject<32 else 'test'
        record=mne.io.read_raw_edf(path,preload=True,verbose=False)
        picks=[n for n in record.ch_names if n.startswith('EEG ') and n!='EEG A2-A1']
        names=[n.removeprefix('EEG ') for n in picks];assert len(names)==19
        coordinates(tuple(names));record.pick(picks)
        record.filter(.5,45.,verbose=False);record.resample(200,verbose=False)
        x=(record.get_data()*1e6).astype(np.float32)
        assert np.isfinite(x).all() and np.all(x.std(-1)>1e-8)
        parent='MAT-'+path.stem;output=ROOT/'signals'/f'{parent}.npy';np.save(output,x)
        source_hash=digest(path)
        # All nonoverlapping 5-second windows; never split a subject across partitions.
        for w in range(x.shape[1]//1000):
            rows.append(dict(id=f'{parent}-{w:04d}',parent_id=parent,path=str(output),source_path=str(path),source_sha256=source_hash,dataset='MAT',subject=str(subject),session='1',trial=condition,task=1,label=condition-1,split=split,start=w*1000,samples=1000,channels=names,token_length=114))
        print('PREPARED',path.name,split,x.shape[1]//1000,flush=True)
    assert len({r['id'] for r in rows})==len(rows)
    groups={s:{(r['dataset'],r['subject']) for r in rows if r['split']==s} for s in ['train','validation','test','heldout']}
    assert all(not groups[a]&groups[b] for a in groups for b in groups if a<b)
    counts={};hashes={}
    for split in groups:
        rr=[r for r in rows if r['split']==split];p=ROOT/f'{split}.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in rr));hashes[split]=digest(p)
        counts[split]={ds:dict(windows=len(z),records=len({r['parent_id'] for r in z}),subjects=len({r['subject'] for r in z}),labels=dict(Counter(r['label'] for r in z))) for ds in sorted({r['dataset'] for r in rr}) if (z:=[r for r in rr if r['dataset']==ds])}
    (ROOT/'complete.json').write_text(json.dumps(dict(counts=counts,manifest_sha256=hashes,task_catalog_sha256=digest('configs/mi_emotion_tasks.json'),source_mi_manifest_sha256=digest(source/'complete.json'),sample_rate=200,units='uV',mat_window_samples=1000,mat_excluded_channels=['EEG A2-A1','ECG ECG']),indent=2))
    print('COMPLETE',counts,flush=True)


def read_rows(split):
    done=json.loads((ROOT/'complete.json').read_text());p=ROOT/f'{split}.jsonl'
    assert digest(p)==done['manifest_sha256'][split]
    assert digest('configs/mi_emotion_tasks.json')==done['task_catalog_sha256']
    return [json.loads(l) for l in p.read_text().splitlines()]

@lru_cache(maxsize=256)
def signal(path):return np.load(path,mmap_mode='r')
@lru_cache(maxsize=128)
def coordinates(names):
    bank=json.loads((Path(json.loads(Path('configs/encoder.json').read_text())['positions_checkpoint'])/'positions.json').read_text())
    return np.asarray([bank[n] for n in names],dtype=np.float32)

def raw(r):
    if r['task']==0:
        x,names,_=preprocess(r);return x,coordinates(tuple(names))
    n=r['samples'];x=np.array(signal(r['path'])[:,r['start']:r['start']+n],copy=True)
    if x.shape!=(len(r['channels']),n) or not np.isfinite(x).all():raise ValueError('Invalid MAT window')
    x-=x.mean(-1,keepdims=True);x=np.clip(x,-1000,1000)/100
    padded=200+180*math.ceil(max(0,n-200)/180)
    x=np.pad(x,((0,0),(0,padded-n)),mode='edge')
    return x,coordinates(tuple(r['channels']))

class Sampler:
    def __init__(self,rows):
        self.rows=rows;self.groups={}
        self.mi_indices=np.array([i for i,r in enumerate(rows) if r['task']==0],dtype=np.int64)
        self.mi_steps_per_epoch=math.ceil(len(self.mi_indices)/32)
        self._mi_epoch=None;self._mi_order=None
        for t in range(len(TASKS)):
            classes={}
            for label in range(len(TASKS[t]['classes'])):
                ds=defaultdict(lambda:defaultdict(list))
                for i,r in enumerate(rows):
                    if r['task']==t and r['label']==label:ds[r['dataset']][r['parent_id']].append(i)
                classes[label]={d:list(parents.values()) for d,parents in ds.items()}
                assert classes[label]
            self.groups[t]=classes
    def full_mi_batch(self,step,seed=42):
        """One-based steps: every MI trial exactly once per shuffled epoch."""
        if step<1:raise ValueError('Steps are one-based')
        epoch,offset=divmod(step-1,self.mi_steps_per_epoch)
        key=(seed,epoch)
        if self._mi_epoch!=key:
            self._mi_order=np.random.default_rng(np.random.SeedSequence([seed,epoch,104729])).permutation(self.mi_indices)
            self._mi_epoch=key
        mi=self._mi_order[offset*32:(offset+1)*32].tolist()
        # Keep the last short MI batch: no drop_last and no repeated MI padding.
        return mi+self.batch(step,seed)[32:]

    def batch(self,step,seed=42):
        rng=np.random.default_rng(np.random.SeedSequence([seed,step]));result=[]
        for t,n in enumerate(BATCH_SIZES):
            classes=self.groups[t];order=rng.permutation(len(classes))
            for j in range(n):
                label=int(order[(j+step)%len(classes)]);ds=classes[label];d=list(ds)[int(rng.integers(len(ds)))];parents=ds[d];windows=parents[int(rng.integers(len(parents)))];result.append(windows[int(rng.integers(len(windows)))])
        # Task blocks retained so rank-striding includes each task on both GPUs.
        return result

if __name__=='__main__':prepare()
