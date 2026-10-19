"""Read-only mounted MI trial adapter; separate from the live TUAB training path."""
import hashlib
import json
from pathlib import Path
import pickle
import re
import numpy as np

CONFIG=Path(__file__).resolve().parents[1]/'configs/mi_data.json'


def configuration():return json.loads(CONFIG.read_text())


class QuarantinedTrial(ValueError):
    pass


def canonical_channel(name):return str(name).upper().replace('.','').removeprefix('EEG ').removesuffix('-REF')


def read_trial(path,dataset=None,channels=('C3','Cz','C4'),crop_seconds=None):
    """Load trusted mounted pickle, exact semantic labels only; never pad or normalize silently.

    Returns float32 [channels,time] in microvolts, label 0=left/1=right, and provenance.
    crop_seconds refers to the beginning of the exported task epoch, NOT a rest baseline.
    """
    path=Path(path)
    with path.open('rb') as f:r=pickle.load(f)
    if r.get('schema_version')!='eeg-epoch-v3' or r.get('paradigm')!='imagery':raise ValueError('Not an audited imagery epoch')
    cfg=configuration();name=dataset or r.get('dataset')
    if name not in cfg['source_datasets']+cfg['heldout_datasets']:raise ValueError(f'Dataset not approved: {name}')
    if r.get('dataset')!=name:raise ValueError('Dataset provenance mismatch')
    if any(all(str(r.get(k))==v for k,v in rule.items()) for rule in cfg.get('quarantined_runs',[])):
        raise QuarantinedTrial('Run quarantined: identical EEG across subjects with conflicting labels')
    label=r.get('label')
    if label not in cfg['label_map']:raise ValueError(f'Not left/right hand imagery: {label}')
    if path.stem.rsplit('_label-',1)[-1]!=label:raise ValueError('Filename/label mismatch')
    if float(r.get('freq',0))!=200 or r.get('unit')!='uV':raise ValueError('Expected 200Hz microvolts')
    if name=='PhysionetMI':
        match=re.search(r'S\d+R(\d+)\.edf$',str(r.get('raw_name','')),re.I)
        if not match or int(match[1]) not in (4,8,12):raise ValueError('Not authenticated imagined left/right Physionet run')
    if name=='GuttmannFlury2025_MI' and '/MI/' not in str(r.get('raw_name','')):raise ValueError('MI/ME provenance ambiguous')
    if name=='Chang2025' and '_MI' not in Path(r.get('raw_name','')).name:raise ValueError('MI condition not authenticated')
    names={canonical_channel(c):c for c in r['eeg']}
    if len(names)!=len(r['eeg']):raise ValueError('Duplicate normalized channel name')
    selected=list(channels) if channels is not None else list(r['eeg'])
    missing=[c for c in selected if canonical_channel(c) not in names]
    if missing:raise ValueError(f'Missing channels: {missing}; do not substitute nearby electrodes')
    x=np.stack([np.asarray(r['eeg'][names[canonical_channel(c)]],dtype=np.float32) for c in selected])
    if x.ndim!=2 or not np.isfinite(x).all() or x.shape[1]!=r['n_samples']:raise ValueError('Invalid EEG shape or nonfinite values')
    if np.any(x.std(-1)<1e-8):raise ValueError('Flat selected channel')
    if crop_seconds is not None:
        start,end=crop_seconds;i,j=round(start*200),round(end*200)
        if not 0<=i<j<=x.shape[1]:raise ValueError('Crop outside task epoch; no synthetic baseline/padding')
        x=x[:,i:j].copy()
    metadata={k:r.get(k) for k in ['dataset','subject','session','run','trial','event_index','raw_name','start_time','end_time','window_source','annotation_onset','trial_source']}
    metadata.update(path=str(path),channels=selected,freq=200,unit='uV',label_name=label,
                    subject_key=f'{name}:{r["subject"]}',baseline_available=False,crop_seconds=crop_seconds)
    return x,cfg['label_map'][label],metadata


def iter_paths(dataset,role='source',root=None):
    cfg=configuration()
    if role not in ('source','heldout'):raise ValueError('role must be source or heldout')
    if dataset not in cfg[f'{role}_datasets']:raise ValueError(f'{dataset} not allowed for {role}; 2b remains held out')
    folder=Path(root or cfg['root'])/dataset
    if not (folder/'_SUCCESS.json').is_file():raise FileNotFoundError(folder/'_SUCCESS.json')
    for path in sorted(folder.rglob('*.pkl')):
        if path.stem.rsplit('_label-',1)[-1] in cfg['label_map']:yield path


def subject_partitions(subjects,dataset,seed=42,fraction=.2):
    # Separate RNG per dataset so adding another source does not change existing splits.
    salt=int(hashlib.sha256(dataset.encode()).hexdigest()[:8],16)
    ordered=sorted(set(map(str,subjects)))
    if len(ordered)<2:raise ValueError('Need >=2 subjects for a source split')
    rng=np.random.default_rng(np.random.SeedSequence([seed,salt]));rng.shuffle(ordered)
    n=min(len(ordered)-1,max(1,round(len(ordered)*fraction)))
    return {s:('validation' if i<n else 'train') for i,s in enumerate(ordered)}


class MITrials:
    """Lazy record loader. Variable-duration samples; batching/windowing belongs to training."""
    def __init__(self,manifest,split='train',channels=('C3','Cz','C4'),crop_seconds=None):
        if split not in ('train','validation','heldout'):raise ValueError('Invalid split')
        manifest=Path(manifest)
        complete=json.loads((manifest.parent/'complete.json').read_text())
        if complete.get('status')!='passed' or complete.get('config_sha256')!=hashlib.sha256(CONFIG.read_bytes()).hexdigest():
            raise ValueError('Index audit/configuration is incomplete or stale')
        if complete.get('manifest_sha256',{}).get(manifest.name)!=hashlib.sha256(manifest.read_bytes()).hexdigest():
            raise ValueError('Manifest audit hash mismatch')
        self.rows=[json.loads(s) for s in manifest.read_text().splitlines() if s.strip()]
        self.rows=[r for r in self.rows if r['split']==split]
        self.channels=channels;self.crop_seconds=crop_seconds;cfg=configuration()
        allowed=cfg['heldout_datasets'] if split=='heldout' else cfg['source_datasets']
        if not self.rows or any(r['dataset'] not in allowed for r in self.rows):raise ValueError('Empty split or forbidden dataset')
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        r=self.rows[index];x,y,m=read_trial(r['path'],r['dataset'],self.channels,self.crop_seconds)
        if y!=r['label'] or str(m['subject'])!=r['subject']:raise ValueError('Manifest changed')
        return dict(eeg=x,label=y,metadata=m)
