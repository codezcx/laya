"""Record-level TUAB splits, preprocessing, and frozen REVE tokens."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path('datasets/tuab')
RAW=Path('/inspire/dataset/tuab/v0/v3.0.1/edf')
CHANNELS=['Fp1','Fp2','F7','F3','Fz','F4','F8','T7','C3','Cz','C4','T8','P7','P3','Pz','P4','P8','O1','O2']
PROMPT='Is this EEG clinically normal or abnormal?'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_json(path,data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False))


def readrows(path):return [json.loads(s) for s in Path(path).read_text().splitlines()]


def canonical(name):
    name=name.upper().removeprefix('EEG ').removesuffix('-REF').removesuffix('-LE')
    return {'T3':'T7','T4':'T8','T5':'P7','T6':'P8'}.get(name,name)


def window_starts(duration,count):
    # Ignore first minute when enough EEG remains; cap retained record at 20 min.
    end=min(duration,1200.)
    begin=60. if end>=60+count*10 else 0.
    if end-begin<count*10:raise ValueError('Record too short for nonoverlapping windows')
    edges=np.linspace(begin,end,count+1)
    return ((edges[:-1]+edges[1:])/2-5).tolist()


def split_records(rows,seed=42):
    # Stratify subjects by their set of labels, keeping mixed-label subjects intact.
    by_subject={}
    for r in rows:
        if r['official_split']=='train':by_subject.setdefault(r['subject'],set()).add(r['label'])
    strata={}
    for subject,labels in sorted(by_subject.items()):strata.setdefault(tuple(sorted(labels)),[]).append(subject)
    rng=np.random.default_rng(seed);validation=set()
    for subjects in strata.values():
        rng.shuffle(subjects);validation.update(subjects[:max(1,round(.15*len(subjects)))])
    result=[]
    for r in rows:
        r=dict(r);r['split']='test' if r['official_split']=='eval' else 'validation' if r['subject'] in validation else 'train'
        result.append(r)
    sets=[{r['subject'] for r in result if r['split']==s} for s in ['train','validation','test']]
    assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
    return result


def prepare_record(row):
    import mne
    from scipy.signal import butter,sosfiltfilt,resample_poly
    try:
        raw=mne.io.read_raw_edf(row['path'],preload=False,verbose='ERROR')
        names=[canonical(n) for n in raw.ch_names]
        if any(names.count(c.upper())!=1 for c in CHANNELS):raise ValueError('Missing or duplicated required EEG channel')
        picks=[names.index(c.upper()) for c in CHANNELS]
        fs=float(raw.info['sfreq']);duration=raw.n_times/fs
        count=16 if row['split']=='train' else 8
        starts=window_starts(duration,count)
        # Filtering continuous signal avoids independent window-edge transients.
        x=raw.get_data(picks=picks,start=0,stop=min(raw.n_times,int(1202*fs))) * 1e6
        if not np.isfinite(x).all():raise ValueError('Nonfinite EDF signal')
        ratio=Fraction(200/fs).limit_denominator(10000)
        x=resample_poly(x,ratio.numerator,ratio.denominator,axis=-1)
        x=sosfiltfilt(butter(4,[.5,45],btype='bandpass',fs=200,output='sos'),x,axis=-1)
        windows=np.stack([x[:,round(s*200):round(s*200)+2000] for s in starts])
        assert windows.shape==(count,19,2000)
        if np.any(windows.std(-1)<1e-6):raise ValueError('Flat channel in selected windows')
        # Preserve between-channel and between-record amplitude; no per-channel z-score.
        windows-=windows.mean(-1,keepdims=True)
        windows=(np.clip(windows,-1000,1000)/100).astype(np.float32)
        if not np.isfinite(windows).all():raise ValueError('Nonfinite preprocessed signal')
        path=ROOT/'signals'/f"{row['id']}.npy";np.save(path,windows)
        return dict(row,duration=duration,sfreq=fs,window_starts_seconds=starts,windows=count,
                    signal_file=str(path),signal_sha256=digest(path),raw_bytes=Path(row['path']).stat().st_size)
    except Exception as e:return dict(row,excluded=True,reason=f'{type(e).__name__}: {e}')


def prepare(workers=12):
    ROOT.mkdir(parents=True,exist_ok=False);(ROOT/'signals').mkdir()
    rows=[]
    for path in sorted(RAW.rglob('*.edf')):
        split,label,_,name=path.relative_to(RAW).parts
        assert split in ['train','eval'] and label in ['normal','abnormal']
        rows.append(dict(id=name.removesuffix('.edf'),path=str(path),subject=name.split('_')[0],
                         official_split=split,label=int(label=='abnormal')))
    assert len(rows)==2993 and len({r['id'] for r in rows})==len(rows)
    rows=split_records(rows)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        completed=[]
        for i,row in enumerate(pool.map(prepare_record,rows)):
            completed.append(row)
            if i%100==0:print('preprocessed',i,len(rows),flush=True)
    excluded=[r for r in completed if r.get('excluded')];valid=[r for r in completed if not r.get('excluded')]
    files={}
    for split in ['train','validation','test']:
        path=ROOT/f'records_{split}.jsonl'
        path.write_text(''.join(json.dumps(r)+'\n' for r in valid if r['split']==split));files[path.name]=digest(path)
    write_json(ROOT/'questions.json',dict(question=PROMPT,options={'A':'Clinically normal EEG','B':'Clinically abnormal EEG'},label_map={'normal':0,'abnormal':1}))
    write_json(ROOT/'preprocessing.json',dict(channels=CHANNELS,fs=200,seconds=10,bandpass=[.5,45],
        units='uV / 100, per-window channel centering, clipping at +/-1000 uV, no variance normalization',
        reference='original EDF reference retained',train_pool=16,windows_per_record=8,first_minute='skip if enough remains',maximum_seconds=1200,seed=42))
    counts={s:{str(y):sum(r['split']==s and r['label']==y for r in valid) for y in [0,1]} for s in ['train','validation','test']}
    write_json(ROOT/'audit.json',dict(counts=counts,excluded=excluded,raw_records=len(rows),subject_disjoint=True,
        files=files,code_sha256=digest(__file__)))
    assert len(excluded)/len(rows)<.05, 'Over 5% excluded; inspect audit before proceeding'
    write_json(ROOT/'complete.json',dict(audit_sha256=digest(ROOT/'audit.json'),files=files))
    print('PREPARED',counts,'excluded',len(excluded),flush=True)


def extract(shard):
    import torch
    from transformers import AutoModel
    torch.set_num_threads(4)
    rows=sum([readrows(ROOT/f'records_{s}.jsonl') for s in ['train','validation','test']],[])
    rows=rows[shard::4]
    out=ROOT/'tokens';out.mkdir(exist_ok=True)
    config=json.loads(Path('configs/encoder.json').read_text())
    model,info=AutoModel.from_pretrained(config['reve_checkpoint'],trust_remote_code=True,local_files_only=True,output_loading_info=True)
    assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    model=model.cuda().eval().requires_grad_(False)
    bank=AutoModel.from_pretrained(config['positions_checkpoint'],trust_remote_code=True,local_files_only=True)
    positions=bank(CHANNELS).detach().cuda()
    hashes={}
    with torch.no_grad():
        for i,r in enumerate(rows):
            assert digest(r['signal_file'])==r['signal_sha256']
            x=torch.from_numpy(np.load(r['signal_file'])).cuda()
            # 2000 = 200 + 10*180, exactly 11 temporal patches per channel.
            z=model(x,positions[None].expand(len(x),-1,-1))
            assert z.shape==(len(x),19,11,512) and torch.isfinite(z).all(),z.shape
            path=out/f"{r['id']}.npy";np.save(path,z.flatten(1,2).cpu().numpy().astype(np.float16))
            hashes[r['id']]=digest(path)
            if i%50==0:print('encoded',shard,i,len(rows),flush=True)
    write_json(out/f'shard_{shard}.json',dict(files=hashes,encoder=config,manifest_sha256=digest(ROOT/'complete.json'),code_sha256=digest(__file__)))


def selected_windows(row,epoch=0,seed=42):
    if row['split']!='train':return np.arange(8)
    # One of each adjacent temporal pair, deterministically varying by epoch.
    key=int(hashlib.sha256(row['id'].encode()).hexdigest()[:8],16)
    rng=np.random.default_rng(np.random.SeedSequence([seed,epoch,key]))
    return np.arange(8)*2+rng.integers(0,2,8)


def load_windows(rows,epoch,device):
    import torch
    x=np.concatenate([np.load(ROOT/'tokens'/f"{r['id']}.npy")[selected_windows(r,epoch)] for r in rows])
    return torch.from_numpy(x).to(device)


def metrics(labels,probabilities):
    from sklearn.metrics import confusion_matrix,roc_auc_score
    y=np.asarray(labels);p=np.asarray(probabilities);pred=(p>.5).astype(int)
    c=confusion_matrix(y,pred,labels=[0,1]);recall=np.divide(c.diagonal(),c.sum(1),out=np.zeros(2,dtype=float),where=c.sum(1)>0)
    return dict(n=len(y),accuracy=float((y==pred).mean()),balanced_accuracy=float(recall.mean()),
        normal_recall=float(recall[0]),abnormal_recall=float(recall[1]),confusion=c.tolist(),
        auroc=float(roc_auc_score(y,p)) if len(set(y))==2 else None)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--shard',type=int);p.add_argument('--workers',type=int,default=12);a=p.parse_args()
    if a.prepare:prepare(a.workers)
    else:
        assert a.shard in range(4)
        extract(a.shard)
