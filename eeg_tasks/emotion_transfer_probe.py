"""Source-selected common-montage spectral transfer between SEED and FACED."""
import argparse,hashlib,json,warnings
from pathlib import Path
import joblib
import numpy as np
from scipy.signal import welch
from scipy.special import logsumexp
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

ROOT=Path('artifacts/emotion_transfer_v16')
ALIASES={'T3':'T7','T4':'T8','T5':'P7','T6':'P8'}
CHANNELS=['Fp1','Fp2','Fz','F3','F4','F7','F8','FC1','FC2','FC5','FC6','Cz','C3','C4','T7','T8','CP1','CP2','CP5','CP6','Pz','P3','P4','P7','P8','PO3','PO4','Oz','O1','O2']
BANDS=[(1,4),(4,8),(8,13),(13,20),(20,30)]
def canonical(name):return ALIASES.get(name.upper(),name.upper()).upper()
def picks(names):
 names=[canonical(n) for n in names];assert len(names)==len(set(names));return [names.index(canonical(c)) for c in CHANNELS]
def fractions(logpower):return logpower-logsumexp(logpower,axis=-1,keepdims=True)
def readrows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def transform(x,arm):
 if arm=='three_linear':x=x[:,[CHANNELS.index(c) for c in ['C3','Cz','C4']]]
 elif arm in ('spatial_linear','spatial_rbf'):x=x-x.mean(1,keepdims=True)
 elif arm!='full_linear':raise ValueError(arm)
 return x.reshape(len(x),-1)

def metrics(rows,pred):
 y=np.array([r['class_id'] for r in rows]);subjects={}
 for s in sorted({r['subject'] for r in rows}):
  ix=np.array([i for i,r in enumerate(rows) if r['subject']==s]);yy=y[ix];pp=pred[ix];assert set(yy)=={0,1,2};subjects[s]={'accuracy':float(np.mean(pp==yy)),'balanced_accuracy':float(np.mean([np.mean(pp[yy==c]==c) for c in [0,1,2]]))}
 vals=np.array([v['balanced_accuracy'] for v in subjects.values()]);boot=np.random.default_rng(2026).choice(vals,(10000,len(vals)),replace=True).mean(1)
 return {'n':len(rows),'subject_accuracy':float(np.mean([v['accuracy'] for v in subjects.values()])),'subject_balanced_accuracy':float(vals.mean()),'subject_balanced_95ci':np.quantile(boot,[.025,.975]).tolist(),'per_subject':subjects}

def prepare():
 ROOT.mkdir(exist_ok=False);seed=Path('artifacts/emotion_source_probe_v4');cfg=json.loads(Path('configs/multitask_prototypes_v1.json').read_text());names=[s.split()[-1] for s in Path(cfg['seed_channel_file']).read_text().splitlines() if s.strip()];x=np.load(seed/'62ch_relative_30.npy').reshape(-1,62,5)[:,picks(names)];rows=readrows(seed/'rows.jsonl');assert len(x)==len(rows);arrays=[fractions(x)];fr=Path('artifacts/faced_source_v14');assert json.loads((fr/'prepared_integrity.json').read_text())['status']=='complete';plan=json.loads((fr/'cohort.json').read_text())
 for sub in sorted(plan['train']+plan['validation']):
  p=fr/'prepared'/f'sub-{sub:03d}';a=json.loads((p/'audit.json').read_text());signal=np.load(p/'signals.npy')[:,picks(a['channels'])];f,psd=welch(signal,fs=200,nperseg=200,noverlap=100,axis=-1);power=np.stack([psd[...,(f>=lo)&(f<hi)].sum(-1) for lo,hi in BANDS],-1);arrays.append(fractions(np.log(np.maximum(power,1e-20))));rows.extend(readrows(p/'rows.jsonl'))
 x=np.concatenate(arrays).astype(np.float32);assert x.shape==(len(rows),30,5) and np.isfinite(x).all();assert all(r['split'] in ('train','validation') for r in rows);np.save(ROOT/'features.npy',x);(ROOT/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));(ROOT/'data.json').write_text(json.dumps({'n':len(rows),'channels':CHANNELS,'aliases':ALIASES,'bands':BANDS,'feature':'log band fraction among 1-30 Hz; per-window only, no target fitting','seed_source_manifest_sha256':hashlib.sha256((seed/'rows.jsonl').read_bytes()).hexdigest(),'faced_cohort_sha256':hashlib.sha256((fr/'cohort.json').read_bytes()).hexdigest(),'feature_sha256':hashlib.sha256((ROOT/'features.npy').read_bytes()).hexdigest()},indent=2))

def train(arm):
 rows=readrows(ROOT/'rows.jsonl');x=transform(np.load(ROOT/'features.npy'),arm);y=np.array([r['class_id'] for r in rows])
 for source in ['SEED','FACED-NEMAR']:
  out=ROOT/f'{source}_{arm}';out.mkdir(exist_ok=False);ti=np.array([i for i,r in enumerate(rows) if r['dataset']==source and r['split']=='train']);vi=np.array([i for i,r in enumerate(rows) if r['dataset']==source and r['split']=='validation']);assert not {rows[i]['subject'] for i in ti}&{rows[i]['subject'] for i in vi};weights=np.array([len(ti)/(3*np.sum(y[ti]==v)) for v in y[ti]]);best=-1;runs=[]
  for c in [.001,.01,.1,1.,10.]:
   classifier=SVC(C=c,kernel='rbf',gamma='scale',max_iter=20000,random_state=42) if arm=='spatial_rbf' else LogisticRegression(C=c,max_iter=2000,random_state=42);model=make_pipeline(StandardScaler(),classifier)
   with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always');model.fit(x[ti],y[ti],**{('svc' if arm=='spatial_rbf' else 'logisticregression')+'__sample_weight':weights})
   m=metrics([rows[i] for i in vi],model.predict(x[vi]));m.update(C=c,warnings=[str(w.message) for w in caught]);runs.append(m)
   if m['subject_balanced_accuracy']>best:
    best=m['subject_balanced_accuracy'];selected=m;joblib.dump(model,out/'best.joblib')
   print(source,arm,c,m['subject_balanced_accuracy'],flush=True)
  (out/'complete.json').write_text(json.dumps({'source':source,'arm':arm,'selected':selected,'sweep':runs,'sha256':hashlib.sha256((out/'best.joblib').read_bytes()).hexdigest(),'train_n':len(ti),'validation_n':len(vi)},indent=2))

def evaluate():
 out=ROOT/'evaluation';out.mkdir(exist_ok=False);rows=readrows(ROOT/'rows.jsonl');raw=np.load(ROOT/'features.npy');arms=['three_linear','full_linear','spatial_linear','spatial_rbf'];frozen={}
 for source in ['SEED','FACED-NEMAR']:
  for arm in arms:
   p=ROOT/f'{source}_{arm}';d=json.loads((p/'complete.json').read_text());assert d['sha256']==hashlib.sha256((p/'best.joblib').read_bytes()).hexdigest();frozen[p.name]=d['sha256']
 (out/'frozen.json').write_text(json.dumps(frozen,indent=2));results={}
 for source in ['SEED','FACED-NEMAR']:
  ids=np.array([i for i,r in enumerate(rows) if r['dataset']!=source]);rr=[rows[i] for i in ids]
  for arm in arms:
   name=f'{source}_{arm}';model=joblib.load(ROOT/name/'best.joblib');pred=model.predict(transform(raw[ids],arm));m=metrics(rr,pred);m['shuffled']=metrics(rr,pred[np.random.default_rng(2026).permutation(len(pred))]);results[name]=m;np.save(out/f'{name}_predictions.npy',pred);print(name,m['subject_balanced_accuracy'],flush=True)
 (out/'results.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--evaluate',action='store_true');p.add_argument('--arm',choices=['three_linear','full_linear','spatial_linear','spatial_rbf']);a=p.parse_args()
 with threadpool_limits(limits=4):
  if a.prepare:prepare()
  elif a.evaluate:evaluate()
  elif a.arm:train(a.arm)
  else:p.error('Choose a stage')
