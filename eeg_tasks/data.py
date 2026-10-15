"""Explicit subject splits and uniform three-channel signal preprocessing."""
import hashlib,json,pickle,re,os
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter,sosfiltfilt,resample_poly

CONFIG=Path(os.environ.get('EEG_TASK_CONFIG','configs/multitask_prototypes_v1.json'))
def config(): return json.loads(CONFIG.read_text())
def write_json(p,x): Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def subject_split(subjects,seed):
    ids=np.random.default_rng(seed).permutation(sorted(subjects)).tolist()
    n=len(ids);v=max(1,round(n*.2));return {s:('test' if j<v else 'validation' if j<2*v else 'train') for j,s in enumerate(ids)}
def preprocess(x,fs,cfg):
    x=np.asarray(x,dtype=np.float64)
    if not np.isfinite(x).all(): raise ValueError('Nonfinite raw signal')
    if fs!=200:
        from math import gcd
        g=gcd(int(fs),200);x=resample_poly(x,200//g,int(fs)//g,axis=-1)
    x=sosfiltfilt(butter(4,cfg['bandpass'],btype='bandpass',fs=200,output='sos'),x,axis=-1)
    return x

def normalized_window(x):
    x=x-x.mean(-1,keepdims=True)
    if (x.std(-1)<1e-12).any(): raise ValueError('Flat EEG channel')
    # Same per-channel normalization across all data sources.
    return np.clip(x/x.std(-1,keepdims=True),-15,15).astype('float32')

def prepare():
    cfg=config();root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True)
    if (root/'data.done.json').exists(): raise FileExistsError(root)
    rows=[]; signals=[]; hashes={};quarantine=[]
    def add(x,meta):
        assert x.shape==(3,600)
        x=normalized_window(x);h=hashlib.sha256(x.tobytes()).hexdigest()
        if h in hashes:
            prev=rows[hashes[h]]
            if prev['split']!=meta['split'] or prev['class_id']!=meta['class_id']:raise ValueError(('Duplicate leakage/conflict',meta,prev))
            quarantine.append(meta);return
        hashes[h]=len(rows); rows.append(dict(meta,index=len(rows),signal_sha256=h));signals.append(x)
    # SEED: every recording/session of a person shares a split.
    folder=Path(cfg['seed_root']);files=sorted(folder.glob('[0-9]*.mat'))
    splits=subject_split({int(p.name.split('_')[0]) for p in files},42)
    channels=[s.split()[-1].upper() for s in Path(cfg['seed_channel_file']).read_text().splitlines() if s.strip()]
    picks=[channels.index(c.upper()) for c in cfg['channels']]
    labels=loadmat(folder/'label.mat')['label'].reshape(-1)
    assert len(files)==45 and len(labels)==15 and set(labels)=={-1,0,1}
    for p in files:
        data=loadmat(p);subject=int(p.name.split('_')[0]);keys={int(re.search(r'eeg(\d+)$',k).group(1)):k for k in data if re.search(r'eeg(\d+)$',k)}
        assert set(keys)==set(range(1,16))
        for trial,k in sorted(keys.items()):
            x=preprocess(data[k][picks],200,cfg);n=x.shape[-1]//600
            starts=np.linspace(0,n-1,min(n,cfg['seed_windows_per_trial']),dtype=int)*600
            for start in starts:add(x[:,start:start+600],dict(dataset='SEED',subject=str(subject),split=splits[subject],task=0,class_id=int(labels[trial-1]+1),path=str(p),trial=str(trial),start_sample=int(start),record_id=f'{p.stem}/{trial}'))
        print('SEED',p.name,len(rows),flush=True)
    # EEGMat _1 is background before task; _2 arithmetic. Use matched first60s.
    import mne
    folder=Path(cfg['mat_root']);files=sorted(folder.glob('Subject*_*.edf'))
    splits_mat=subject_split({int(p.stem[7:9]) for p in files},43)
    assert len(files)==72
    for p in files:
        subject=int(p.stem[7:9]);condition=int(p.stem[-1]);assert condition in (1,2)
        raw=mne.io.read_raw_edf(p,preload=False,verbose=False)
        picks=[raw.ch_names.index('EEG '+c) for c in cfg['channels']]
        fs=raw.info['sfreq'];x=preprocess(raw.get_data(picks=picks,stop=min(raw.n_times,int(fs*60))),fs,cfg)
        for start in range(0,min(x.shape[-1]//600,cfg['mat_windows_per_record'])*600,600):
            add(x[:,start:start+600],dict(dataset='EEGMat',subject=str(subject),split=splits_mat[subject],task=2,class_id=4+condition,path=str(p),trial=str(condition),start_sample=start,record_id=p.stem))
    # MI: preserve audited original subject splits; Lee entirely excluded from fitting.
    base=json.loads(Path(cfg['base_config']).read_text());groups=defaultdict(list)
    for split in ('train','validation','test'):
        for line in (Path(base['manifest_dir'])/f'{split}.jsonl').read_text().splitlines():
            r=json.loads(line)
            if r['dataset']=='Lee2019_MI':
                if split!='test':continue
                dest='external_lee'
            else: dest=split
            groups[(r['dataset'],str(r['subject']),r['label'],dest)].append(r)
    rng=np.random.default_rng(44)
    for (dataset,subject,label,split),g in sorted(groups.items()):
        for j in rng.choice(len(g),min(len(g),cfg['mi_max_per_subject_class_split']),replace=False):
            r=g[j]
            with open(r['path'],'rb') as f:item=pickle.load(f)
            assert item['freq']==200 and item['label'] in ('left_hand','right_hand')
            assert base['labels'].index(item['label'])==label
            x=np.stack([item['eeg'][c.upper()] for c in cfg['channels']]);x=preprocess(x,200,cfg)
            add(x[:,:600],dict(dataset=dataset,subject=subject,split=split,task=1,class_id=3+label,path=r['path'],trial=str(r.get('trial','')),start_sample=0,record_id=r['path']))
    split_sets={s:{(r['dataset'],r['subject']) for r in rows if r['split']==s} for s in {r['split'] for r in rows}}
    for a in split_sets:
        for b in split_sets:
            if a!=b:assert not split_sets[a]&split_sets[b]
    np.save(root/'signals.npy',np.stack(signals));(root/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write_json(root/'data.done.json',{'n':len(rows),'counts':dict(Counter(f"{r['dataset']}/{r['split']}/{r['class_id']}" for r in rows)),'seed_subject_splits':splits,'mat_subject_splits':splits_mat,'duplicate_windows_removed':len(quarantine),'config':cfg,'manifest_sha256':hashlib.sha256((root/'manifest.jsonl').read_bytes()).hexdigest()})
    print('DONE',len(rows),flush=True)

if __name__=='__main__':prepare()
