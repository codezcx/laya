"""Source-only linear probes to separate feature and interface bottlenecks."""
import argparse, hashlib, json, time, warnings
from collections import Counter
from pathlib import Path
import joblib
import numpy as np
from scipy.signal import welch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def spectral_features(signals):
    f, p = welch(signals, fs=200, nperseg=200, noverlap=100, axis=-1)
    # Explicit half-open bands; no gamma evidence exists in 1–30Hz inputs.
    bands = [(1,4),(4,8),(8,13),(13,20),(20,30)]
    power = np.stack([p[..., (f>=lo)&(f<hi)].sum(-1) for lo,hi in bands], -1)
    logp = np.log(np.maximum(power, 1e-12))
    # Pairwise log-power contrasts retain spatial asymmetry.
    contrasts = np.stack([logp[:,0]-logp[:,1],logp[:,0]-logp[:,2],logp[:,1]-logp[:,2]],1)
    return np.concatenate([logp.reshape(len(p),-1),contrasts.reshape(len(p),-1)],1)


def covariance_features(signals):
    cov = signals @ signals.transpose(0,2,1) / signals.shape[-1]
    return cov[:,np.triu_indices(3)[0],np.triu_indices(3)[1]]


def score(rows, y, pred):
    result = {}
    for dataset in sorted({r['dataset'] for r in rows}):
        by_subject = {}
        for subject in sorted({r['subject'] for r in rows if r['dataset']==dataset}):
            mask=np.array([r['dataset']==dataset and r['subject']==subject for r in rows])
            by_subject[subject]=float(np.mean(y[mask]==pred[mask]))
        result[dataset]={'subject_macro_accuracy':float(np.mean(list(by_subject.values()))),'per_subject':by_subject}
    return {'datasets':result,'macro_dataset_accuracy':float(np.mean([v['subject_macro_accuracy'] for v in result.values()]))}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--feature',required=True,choices=['spectral','covariance','reve','reve_spectral']);a=parser.parse_args()
    root=Path('artifacts/multitask_prototypes_v1');out=Path('artifacts/source_probe_v3')/a.feature
    out.mkdir(parents=True,exist_ok=False)
    allrows=[json.loads(s) for s in (root/'manifest.jsonl').read_text().splitlines()]
    # Exclude target and source test before reading any signal/features into the probe.
    ids=np.array([i for i,r in enumerate(allrows) if r['split'] in ('train','validation') and r['task'] in (0,1)])
    rows=[allrows[i] for i in ids]
    raw=np.load(root/'signals.npy',mmap_mode='r')[ids]
    parts=[]
    if 'reve' in a.feature:
        feat=np.zeros((len(allrows),1536),np.float32)
        for shard in range(4):
            d=np.load(root/'features'/f'{shard}.npz');feat[d['indices']]=d['features']
        parts.append(feat[ids])
    if 'spectral' in a.feature:parts.append(spectral_features(raw))
    if a.feature=='covariance':parts.append(covariance_features(raw))
    x=np.concatenate(parts,1);assert np.isfinite(x).all()
    results={};started=time.time()
    for task in (0,1):
        ti=np.array([i for i,r in enumerate(rows) if r['task']==task and r['split']=='train'])
        vi=np.array([i for i,r in enumerate(rows) if r['task']==task and r['split']=='validation'])
        y=np.array([r['class_id'] for r in rows]);vr=[rows[i] for i in vi]
        assert not {(rows[i]['dataset'],rows[i]['subject']) for i in ti}&{(rows[i]['dataset'],rows[i]['subject']) for i in vi}
        count=Counter((rows[i]['dataset'],int(y[i])) for i in ti)
        weights=np.array([1/count[(rows[i]['dataset'],int(y[i]))] for i in ti]);weights/=weights.mean()
        runs=[];best=-1
        for c in [.001,.01,.1,1.,10.]:
            model=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=2000,random_state=42))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always',ConvergenceWarning)
                model.fit(x[ti],y[ti],logisticregression__sample_weight=weights)
            m=score(vr,y[vi],model.predict(x[vi]));m.update(C=c,convergence_warnings=[str(w.message) for w in caught]);runs.append(m)
            print(a.feature,task,c,m['macro_dataset_accuracy'],flush=True)
            if m['macro_dataset_accuracy']>best:
                best=m['macro_dataset_accuracy'];selected=m;joblib.dump(model,out/f'task{task}.joblib')
        results[str(task)]={'selected':selected,'sweep':runs}
    result={'feature':a.feature,'dimension':x.shape[1],'results':results,'seconds':time.time()-started,'manifest_sha256':hashlib.sha256((root/'manifest.jsonl').read_bytes()).hexdigest(),'note':'Task-specific diagnostic; source validation only. No external/test evaluation. Not universal model or zero-shot result.'}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
