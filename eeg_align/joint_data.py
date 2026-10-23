"""Subject-disjoint MI/SEED manifests and task/class/parent-trial sampling."""
import hashlib,json,math,re,zipfile
from collections import defaultdict,Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.io import loadmat
from .mi_training_data import Tokens,preprocess,digest
from .seed_positions import load_seed_positions
from .task_options import load_tasks
BASE=Path('/inspire/hdd/global_user/zhangchenxi-253108310322/mne_data')
ROOT=Path('datasets/mi_emotion')
TASKS=list(load_tasks().values())
OFFSETS=[0,2,5]

def workbook_labels(path):
    ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(path) as z:
        strings=[''.join(x.itertext()) for x in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si',ns)]
        table=[]
        for row in ET.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('.//m:row',ns):
            cells=[]
            for c in row:
                v=c.find('m:v',ns)
                if v is not None:cells.append(strings[int(v.text)] if c.get('t')=='s' else v.text)
            table.append(cells)
    result={}
    for a,b in zip(table[::2],table[1::2]):
        assert len(a)==len(b)==21
        result.update({int(i):label.lower() for i,label in zip(a[1:],b[1:])})
    assert set(result)==set(range(1,81))
    return result

def prepare():
    if (ROOT/'complete.json').exists():raise FileExistsError('Manifest already complete')
    ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'signals').mkdir(exist_ok=True)
    rows=[]
    for split in ['train','validation','heldout']:
        for r in Tokens(split).rows:
            rows.append(dict(r,task=0,parent_id=r['id']))
    labels7=workbook_labels(BASE/'SEED-VII/emotion_label_and_stimuli_order.xlsx')
    seedroot=BASE/'SEED/SEED/SEED_EEG/Preprocessed_EEG'
    labels3=loadmat(seedroot/'label.mat')['label'].reshape(-1).astype(int)+1
    assert len(labels3)==15 and set(labels3)=={0,1,2}
    plans=[];splits={}
    for task,ds,n,folder in [(1,'SEED',15,seedroot),(2,'SEED-VII',20,BASE/'SEED-VII/EEG_preprocessed')]:
        subjects=np.random.default_rng(42).permutation(np.arange(1,n+1));nv=n//5
        mapping={int(s):'validation' if i<nv else 'test' if i<2*nv else 'train' for i,s in enumerate(subjects)}
        splits[ds]=mapping
        for path in sorted(folder.glob('*.mat')):
            if path.name=='label.mat':continue
            subject=int(path.stem.split('_')[0]);assert subject in mapping
            plans.append((task,ds,path,subject,mapping[subject]))
    def process(plan):
        task,ds,path,subject,split=plan;mat=loadmat(path);items=[]
        keys=[k for k in mat if not k.startswith('__')]
        keys=sorted([k for k in keys if (k.isdigit() if task==2 else re.search(r'eeg\d+$',k))])
        assert len(keys)==(80 if task==2 else 15),(path,keys)
        file_hash=digest(path);stats=[]
        names=[l.split()[-1] for l in Path(json.loads(Path('configs/seed_positions.json').read_text())['template_files'][ds]).read_text().splitlines() if l.strip()]
        assert len(names)==62;load_seed_positions(names)
        for key in keys:
            trial=int(key) if task==2 else int(re.search(r'eeg(\d+)$',key).group(1))
            label=int(labels3[trial-1]) if task==1 else [c['class_id'] for c in TASKS[task]['classes']].index(labels7[trial])
            x=np.asarray(mat[key],dtype=np.float32)
            if x.ndim!=2 or x.shape[0]!=62 or not np.isfinite(x).all() or np.any(x.std(-1)<1e-8):raise ValueError(f'Invalid raw EEG: {path}:{key}')
            parent=hashlib.sha256(f'{ds}/{path.name}/{key}'.encode()).hexdigest()[:24]
            output=ROOT/'signals'/f'{parent}.npy';np.save(output,x)
            count=x.shape[1]//800
            assert count>0
            # Fixed eight nonoverlapping windows per validation/test parent; all train windows.
            windows=range(count) if split=='train' else np.unique(np.linspace(0,count-1,min(8,count),dtype=int))
            for w in windows:
                items.append(dict(id=f'{parent}-{w:04d}',parent_id=parent,path=str(output),source_path=str(path),source_key=key,source_sha256=file_hash,dataset=ds,subject=str(subject),session=path.stem if task==1 else str((trial-1)//20+1),trial=trial,task=task,label=label,split=split,start=int(w)*800,samples=800,channels=names,token_length=310))
            stats.append(float(np.median(x.std(-1))))
        print('PREPARED',ds,path.name,len(items),'median_channel_std_uV',float(np.median(stats)),flush=True)
        return items
    with ThreadPoolExecutor(max_workers=4) as pool:
        for items in pool.map(process,plans):rows.extend(items)
    assert len({r['id'] for r in rows})==len(rows)
    groups={s:{(r['dataset'],r['subject']) for r in rows if r['split']==s} for s in ['train','validation','test','heldout']}
    assert all(not groups[a]&groups[b] for a in groups for b in groups if a<b)
    counts={};hashes={}
    for split in groups:
        rr=[r for r in rows if r['split']==split];p=ROOT/f'{split}.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in rr));hashes[split]=digest(p)
        counts[split]={ds:dict(windows=len(z),trials=len({r['parent_id'] for r in z}),subjects=len({r['subject'] for r in z}),labels=dict(Counter(r['label'] for r in z))) for ds in sorted({r['dataset'] for r in rr}) if (z:=[r for r in rr if r['dataset']==ds])}
    (ROOT/'complete.json').write_text(json.dumps(dict(counts=counts,manifest_sha256=hashes,emotion_subject_splits=splits,sample_rate=200,units='uV',window_samples=800,validation_windows_per_parent=8,task_catalog_sha256=digest('configs/mi_emotion_tasks.json'),positions_sha256=digest('configs/seed_positions.json')),indent=2))
    print('COMPLETE',counts,flush=True)

def read_rows(split):
    done=json.loads((ROOT/'complete.json').read_text());p=ROOT/f'{split}.jsonl'
    assert digest(p)==done['manifest_sha256'][split]
    assert digest('configs/mi_emotion_tasks.json')==done['task_catalog_sha256']
    assert digest('configs/seed_positions.json')==done['positions_sha256']
    return [json.loads(l) for l in p.read_text().splitlines()]

@lru_cache(maxsize=256)
def signal(path):return np.load(path,mmap_mode='r')
@lru_cache(maxsize=128)
def coordinates(names,emotion):
    if emotion:return load_seed_positions(names)
    bank=json.loads((Path(json.loads(Path('configs/encoder.json').read_text())['positions_checkpoint'])/'positions.json').read_text())
    return np.asarray([bank[n] for n in names],dtype=np.float32)

def raw(r):
    if r['task']==0:
        x,names,_=preprocess(r);return x,coordinates(tuple(names),False)
    x=np.array(signal(r['path'])[:,r['start']:r['start']+800],copy=True)
    if x.shape!=(62,800) or not np.isfinite(x).all():raise ValueError('Invalid window')
    x-=x.mean(-1,keepdims=True);x=np.clip(x,-1000,1000)/100
    x=np.pad(x,((0,0),(0,120)),mode='edge')
    return x,coordinates(tuple(r['channels']),True)

class Sampler:
    def __init__(self,rows):
        self.rows=rows;self.groups={}
        self.mi_indices=np.array([i for i,r in enumerate(rows) if r['task']==0],dtype=np.int64)
        self.mi_steps_per_epoch=math.ceil(len(self.mi_indices)/32)
        self._mi_epoch=None;self._mi_order=None
        for t in range(3):
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
        for t,n in enumerate([32,16,16]):
            classes=self.groups[t];order=rng.permutation(len(classes))
            for j in range(n):
                label=int(order[(j+step)%len(classes)]);ds=classes[label];d=list(ds)[int(rng.integers(len(ds)))];parents=ds[d];windows=parents[int(rng.integers(len(parents)))];result.append(windows[int(rng.integers(len(windows)))])
        # Task blocks retained so rank-striding includes each task on both GPUs.
        return result

if __name__=='__main__':prepare()
