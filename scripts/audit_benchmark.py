"""Read-only audit of trusted local benchmark caches; no training or split mutation."""
import argparse, json, pickle, sys, hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from eeg_align.benchmark_data import catalog,to_electrodes,resample,segment,load_pickle

def frequency_scores(signals,sampling_rate,frequencies):
    x=np.asarray(signals,dtype=np.float64)
    n=x.shape[-1];time=np.arange(n)/sampling_rate;window=np.hanning(n)
    reference=np.exp(-2j*np.pi*np.asarray(frequencies)[:,None]*time[None,:])*window
    x=x-x.mean(axis=-1,keepdims=True)
    energy=np.sum((x*window)**2,axis=-1,keepdims=True)
    power=np.abs(np.einsum('bct,ft->bcf',x,reference))**2/np.maximum(energy,1e-30)
    return power.mean(axis=1)


def frequency_check_train(out,online=False):
    """Training-only signal sanity check, with no fitted weights or test access.

    Score the stated candidate fundamentals by a Hann-windowed Fourier projection
    averaged across channels after per-channel energy normalization. This is a
    fixed positive control for readable frequency information, not a replacement
    model or an estimate of held-out JEV performance.
    """
    from eeg_align.signal_data import load_rows,SignalStore,ROOT as SIGNAL_ROOT
    from eeg_align.leaf_runtime import digest
    task=next(t for t in catalog() if t['task_id']=='Nakanishi2015')
    rows=[r for r in load_rows('train',0) if r['dataset']==task['task_id']]
    folder=SIGNAL_ROOT/task['task_id'];meta=json.loads((folder/'complete.json').read_text())
    signals=np.load(folder/'signals.npy',mmap_mode='r')
    frequencies=np.array([float(c['option'].split()[0]) for c in task['classes']])
    n=signals.shape[-1]
    store=SignalStore() if online else None
    input_shape=None
    predictions=[]
    for offset in range(0,len(rows),64):
        rr=rows[offset:offset+64]
        if online:
            grouped=store.batch(rr)
            assert len(grouped)==1 and len(grouped[0][2])==1
            assert grouped[0][1]==list(range(len(rr)))
            x=grouped[0][2][0];sampling_rate=200.
        else:
            x=np.array(signals[[r['source_index'] for r in rr]],dtype=np.float64);sampling_rate=meta['sampling_rate']
        input_shape=list(x.shape[1:])
        scores=frequency_scores(x,sampling_rate,frequencies);pred=scores.argmax(-1)
        predictions.extend(dict(id=r['id'],subject=r['subject'],label=r['label'],prediction=int(y),scores=s.tolist()) for r,y,s in zip(rr,pred,scores))
    cm=np.zeros((len(frequencies),len(frequencies)),dtype=int)
    for p in predictions:cm[p['label'],p['prediction']]+=1
    report=dict(dataset=task['task_id'],split='train',n=len(rows),frequencies_hz=frequencies.tolist(),
                sampling_rate=sampling_rate,source_channels=meta['channels'],source_samples=n,
                input_mode='exact_online_preprocessing' if online else 'raw_cache',input_shape=input_shape,
                method='Fixed fundamental-only Fourier score, Hann window, channel energy normalization; no fit or parameter selection',
                train_manifest_sha256=digest(SIGNAL_ROOT/'fold0/train.jsonl'),
                accuracy=float(np.trace(cm)/cm.sum()),confusion_matrix=cm.tolist(),
                per_subject={s:dict(n=sum(p['subject']==s for p in predictions),accuracy=float(np.mean([p['label']==p['prediction'] for p in predictions if p['subject']==s]))) for s in sorted({r['subject'] for r in rows})},
                predictions=predictions)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['predictions','confusion_matrix']},indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path);parser.add_argument('--frequency-check-train',action='store_true');parser.add_argument('--online-input',action='store_true');args=parser.parse_args()
    if args.frequency_check_train:
        suffix='online_' if args.online_input else ''
        frequency_check_train(args.out or ROOT/f'artifacts/benchmark/nakanishi_frequency_{suffix}train.json',args.online_input);return
    if args.online_input:parser.error('--online-input requires --frequency-check-train')
    args.out=args.out or ROOT/'artifacts/benchmark/adapter_audit.json'
    sys.path.insert(0,str(ROOT.parent/'EEG-FM-Benchmark'))
    sys.path.insert(0,str(ROOT.parent/'LEAF'))
    from leaf_datasets.electrode_unifier import ElectrodeUnifier
    from leaf_datasets.shared import pipeline
    unifier=ElectrodeUnifier('LEAF-ch65');aliases={k.upper():k for k in unifier.reverse_lookup}
    result={}
    for task in catalog():
        path=Path(task['source'])
        if path.is_dir():
            paths=sorted(path.rglob('*.pkl'));d=pickle.load(paths[0].open('rb'))
            names=list(d['eeg']);x=np.stack([d['eeg'][c] for c in names])[None];fs=d['freq']
            labels=sorted({p.stem.split('_label-')[-1] for p in paths},key=float)
            assert [f'{float(v):.2f} Hz' for v in labels]==[c['option'] for c in task['classes']]
            info=dict(n=len(paths),subjects=len({p.parent.parent.name for p in paths}),labels=labels)
        else:
            d=load_pickle(path)
            names=d.channel_names;fs=d.sampling_rate;x=d.eeg_data[:2]
            values,counts=np.unique(d.labels,return_counts=True)
            assert values.tolist()==list(range(len(task['classes'])))
            info=dict(n=len(d.labels),subjects=len(np.unique(d.subject_ids)),labels=dict(zip(map(str,values),map(int,counts))))
        signal,names,mapping=to_electrodes(x,names)
        keep=[i for i,c in enumerate(names) if c.upper() not in {'A1','A2'}]
        mapping['excluded_reference_electrodes']=[c for c in names if c.upper() in {'A1','A2'}]
        signal=signal[:,keep];names=[aliases[names[i].upper()] for i in keep]
        chunks=segment(resample(signal,fs));mapped=[pipeline(v,names) for v in chunks]
        assert all(np.isfinite(v).all() for v in mapped)
        info.update(source=str(path),input_shape=list(x.shape),source_hz=float(fs),adapter=mapping,
                    segment_shapes=[list(v.shape) for v in mapped],eeg_tokens=8*len(chunks),
                    matched_locations=len({unifier.reverse_lookup[c] for c in names}))
        result[task['task_id']]=info
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2))
        print(task['task_id'],info,flush=True)
        del d,x,signal,mapped

if __name__=='__main__':main()
