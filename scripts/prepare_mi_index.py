"""Inventory mounted imagery-labelled exports and audit approved left/right loaders. No GPU/download."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from eeg_align.mi_data import CONFIG,QuarantinedTrial,configuration,iter_paths,read_trial,subject_partitions


def raw_2a_labels(root):
    result={}
    for path in sorted(Path(root).glob('A*.mat')):
        runs=loadmat(path,simplify_cells=True)['data'];trial_runs=[r for r in runs if np.asarray(r['y']).size]
        subject=str(int(path.stem[1:3]));session='0train' if path.stem[-1]=='T' else '1test'
        for run,r in enumerate(trial_runs):
            for trial,y in enumerate(np.asarray(r['y']).reshape(-1)):
                result[subject,session,str(run),trial]=int(y)
    assert len(result)==5184
    assert Counter(result.values())=={1:1296,2:1296,3:1296,4:1296}
    return result


def inspect(item):
    name,path=item
    try:
        x,y,m=read_trial(path,name)
        signal_hash=hashlib.sha256(x.astype('<f4').tobytes()).hexdigest()
        return dict(path=str(path),dataset=name,subject=str(m['subject']),session=str(m['session']),run=str(m['run']),
                    trial=m['trial'],event_index=m['event_index'],label=y,label_name=m['label_name'],samples=x.shape[1],
                    signal_sha256=signal_hash,raw_name=m['raw_name'],start_time=m['start_time'],end_time=m['end_time'])
    except QuarantinedTrial as e:return dict(path=str(path),dataset=name,excluded=str(e))
    except Exception as e:return dict(path=str(path),dataset=name,error=f'{type(e).__name__}: {e}')


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('datasets/mi_index'));p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);cfg=configuration();root=Path(cfg['root']);inventory=[]
    for folder in sorted(root.iterdir()):
        success=folder/'_SUCCESS.json'
        if not success.exists():continue
        info=json.loads(success.read_text())
        if info.get('paradigm')!='imagery':continue
        labels=Counter();subjects=set()
        for path in folder.rglob('*.pkl'):
            labels[path.stem.rsplit('_label-',1)[-1]]+=1
            subjects.add(path.relative_to(folder).parts[0])
        inventory.append(dict(dataset=folder.name,exported_trials=sum(labels.values()),subjects=len(subjects),labels=dict(labels)))
    (a.out/'inventory.json').write_text(json.dumps(inventory,indent=2))
    reference=raw_2a_labels(cfg['raw_2a']);summaries={};all_rows=[];errors=[];excluded=[];seen={};duplicates=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for name in cfg['source_datasets']+cfg['heldout_datasets']:
            role='heldout' if name in cfg['heldout_datasets'] else 'source'
            paths=list(iter_paths(name,role));rows=[]
            for r in pool.map(inspect,((name,p) for p in paths)):
                if 'excluded'in r:excluded.append(r);continue
                if 'error'in r:errors.append(r);continue
                if name=='BNCI2014_001':
                    key=(r['subject'],r['session'],r['run'],int(r['event_index']))
                    assert reference[key]==r['label']+1, (key,r['label'],reference.get(key))
                key=r['signal_sha256']
                if key in seen:
                    prior=seen[key];duplicates.append(dict(path=r['path'],duplicate_of=prior['path'],conflicting_label=r['label']!=prior['label']));continue
                seen[key]=r;rows.append(r)
            if role=='source':parts=subject_partitions([r['subject'] for r in rows],name,cfg['seed'],cfg['validation_fraction'])
            for r in rows:r['split']='heldout' if role=='heldout' else parts[r['subject']]
            summaries[name]=dict(candidates=len(paths),accepted=len(rows),subjects=len({r['subject'] for r in rows}),
                labels=dict(Counter(r['label_name'] for r in rows)),samples=dict(Counter(r['samples'] for r in rows)),
                splits=dict(Counter(r['split'] for r in rows)))
            (a.out/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));all_rows+=rows
            print(name,json.dumps(summaries[name]),flush=True)
    for split in ['train','validation','heldout']:
        (a.out/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in all_rows if r['split']==split))
    sets=[{(r['dataset'],r['subject']) for r in all_rows if r['split']==s} for s in ['train','validation','heldout']]
    assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
    report=dict(config=cfg,imagery_tagged_directories=len(inventory),left_right_label_directories=sum({'left_hand','right_hand'}<=set(r['labels']) for r in inventory),
                datasets=summaries,errors=errors,excluded=excluded,duplicates=duplicates,raw_2a_label_crosscheck='all accepted trials match original MAT y',subject_disjoint=True,
                note='imagery metadata includes real movements and nonmotor tasks; candidate count is not a pure MI dataset count')
    (a.out/'audit.json').write_text(json.dumps(report,indent=2));print('errors',len(errors),'duplicates',len(duplicates),flush=True)
    if errors:raise RuntimeError('Invalid trials found; inspect audit before using index')
    if any(r['conflicting_label'] for r in duplicates):raise RuntimeError('Identical selected EEG has conflicting labels; inspect audit')
    (a.out/'complete.json').write_text(json.dumps(dict(status='passed',rows=len(all_rows),config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),manifest_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.out.glob('*.jsonl')},audit_sha256=hashlib.sha256((a.out/'audit.json').read_bytes()).hexdigest()),indent=2))


if __name__=='__main__':main()
