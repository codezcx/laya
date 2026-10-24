"""Read-only audit of trusted local benchmark caches; no training or split mutation."""
import argparse, json, pickle, sys, hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from eeg_align.benchmark_data import catalog,to_electrodes,resample,segment,load_pickle

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'artifacts/benchmark/adapter_audit.json');args=parser.parse_args()
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
