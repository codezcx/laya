"""SEED source-only channel/band/amplitude diagnostics with fixed subject splits."""
import json,re,time,hashlib,warnings
from pathlib import Path
import numpy as np
import joblib
from scipy.io import loadmat
from scipy.signal import welch,butter,sosfiltfilt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from .source_probe import score


def main():
 root=Path('artifacts/multitask_prototypes_v1');out=Path('artifacts/emotion_source_probe_v4');out.mkdir(exist_ok=False)
 cfg=json.loads(Path('configs/multitask_prototypes_v1.json').read_text())
 rows=[json.loads(s) for s in (root/'manifest.jsonl').read_text().splitlines()]
 rows=[r for r in rows if r['dataset']=='SEED' and r['split'] in ('train','validation')]
 groups={}
 for i,r in enumerate(rows):groups.setdefault(r['path'],{}).setdefault(r['trial'],[]).append(i)
 names=[s.split()[-1].upper() for s in Path(cfg['seed_channel_file']).read_text().splitlines() if s.strip()];picks=[names.index(c.upper()) for c in cfg['channels']]
 # Four predeclared comparisons: channel count, amplitude normalization, extra band.
 arrays={k:[] for k in ['3ch_relative_30','62ch_relative_30','62ch_absolute_30','62ch_absolute_45']}
 arrays={k:np.zeros((len(rows),3*5 if k.startswith('3ch') else 62*(6 if k.endswith('45') else 5)),np.float32) for k in arrays}
 start=time.time()
 for path,trials in sorted(groups.items()):
  data=loadmat(path);keys={str(int(re.search(r'eeg(\d+)$',k).group(1))):k for k in data if re.search(r'eeg(\d+)$',k)}
  for trial,ids in trials.items():
   raw=np.asarray(data[keys[trial]],dtype=np.float64);assert raw.shape[0]==62
   filtered={high:sosfiltfilt(butter(4,[1,high],btype='bandpass',fs=200,output='sos'),raw,axis=-1) for high in [30,45]}
   for i in ids:
    s=rows[i]['start_sample']
    for high in [30,45]:
     window=filtered[high][:,s:s+600];assert window.shape==(62,600)
     f,p=welch(window,fs=200,nperseg=200,noverlap=100,axis=-1)
     bands=[(1,4),(4,8),(8,13),(13,20),(20,30)]+([(30,45)] if high==45 else [])
     power=np.stack([p[:,(f>=lo)&(f<hi)].sum(-1) for lo,hi in bands],-1)
     absolute=np.log(np.maximum(power,1e-20))
     arrays[f'62ch_absolute_{high}'][i]=absolute.ravel()
     if high==30:
      relative=absolute-np.log(np.maximum(window.var(-1,keepdims=True),1e-20))
      arrays['62ch_relative_30'][i]=relative.ravel();arrays['3ch_relative_30'][i]=relative[picks].ravel()
  print('extracted',Path(path).name,flush=True)
  del data
 ti=np.array([i for i,r in enumerate(rows) if r['split']=='train']);vi=np.array([i for i,r in enumerate(rows) if r['split']=='validation']);y=np.array([r['class_id'] for r in rows]);vr=[rows[i] for i in vi]
 assert not {rows[i]['subject'] for i in ti}&{rows[i]['subject'] for i in vi}
 results={}
 for arm,x in arrays.items():
  assert np.isfinite(x).all();np.save(out/f'{arm}.npy',x);runs=[];best=-1
  for c in [.001,.01,.1,1.,10.]:
   model=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=2000,random_state=42))
   with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always',ConvergenceWarning);model.fit(x[ti],y[ti])
   m=score(vr,y[vi],model.predict(x[vi]));m.update(C=c,convergence_warnings=[str(w.message) for w in caught]);runs.append(m)
   if m['macro_dataset_accuracy']>best:best=m['macro_dataset_accuracy'];selected=m;joblib.dump(model,out/f'{arm}.joblib')
   print(arm,c,m['macro_dataset_accuracy'],flush=True)
  results[arm]={'selected':selected,'sweep':runs}
 (out/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
 (out/'results.json').write_text(json.dumps({'results':results,'seconds':time.time()-start,'manifest_sha256':hashlib.sha256((root/'manifest.jsonl').read_bytes()).hexdigest(),'note':'Source validation only; raw preprocessed SEED units unchanged. Absolute means no per-window variance normalization, not calibrated physical units.'},indent=2)+'\n')

if __name__=='__main__':
 with threadpool_limits(limits=4):main()
