"""Matched-validation source-scale controls with fixed spectral representations."""
import argparse,hashlib,json,warnings
from pathlib import Path
import joblib
import numpy as np
from scipy.signal import welch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from .emotion_transfer_probe import CHANNELS,BANDS,picks,fractions,transform,readrows,metrics
ROOT=Path('artifacts/faced_scale_v22');SOURCE=Path('artifacts/faced_full_v22')

def prepare():
 assert json.loads((SOURCE/'cohort_status.json').read_text())['state']=='complete';audit=json.loads((SOURCE/'prepared_integrity.json').read_text());assert audit['status']=='complete' and audit['windows_audited']==27440;ROOT.mkdir(exist_ok=False);old=Path('artifacts/emotion_transfer_v16');rr=readrows(old/'rows.jsonl');ids=[i for i,r in enumerate(rr) if r['dataset']=='SEED'];rows=[rr[i] for i in ids];arrays=[np.load(old/'features.npy')[ids]];plan=json.loads((SOURCE/'cohort.json').read_text());pilot=json.loads(Path('artifacts/faced_source_v14/cohort.json').read_text())
 for subject in sorted(plan['train']+plan['validation']):
  p=SOURCE/'prepared'/f'sub-{subject:03d}';a=json.loads((p/'audit.json').read_text());x=np.load(p/'signals.npy')[:,picks(a['channels'])];new=readrows(p/'rows.jsonl');assert len(new)==280;f,power=welch(x,fs=200,nperseg=200,noverlap=100,axis=-1);bp=np.stack([power[...,(f>=lo)&(f<hi)].sum(-1) for lo,hi in BANDS],-1);arrays.append(fractions(np.log(np.maximum(bp,1e-20))));rows+=new
 x=np.concatenate(arrays).astype(np.float32);assert x.shape==(32840,30,5) and np.isfinite(x).all();assert not {r['subject'] for r in rows if r['dataset']=='FACED-NEMAR'}&set(map(str,plan['reserved_test']));np.save(ROOT/'features.npy',x);(ROOT/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));(ROOT/'data_complete.json').write_text(json.dumps({'n':len(rows),'pilot_train_subjects':pilot['train'],'full_train_subjects':plan['train'],'validation_subjects':plan['validation'],'cohort_sha256':hashlib.sha256((SOURCE/'cohort.json').read_bytes()).hexdigest(),'channels':CHANNELS},indent=2))

def fit(mode,arm):
 out=ROOT/f'{mode}_{arm}';out.mkdir(exist_ok=False);rows=readrows(ROOT/'rows.jsonl');meta=json.loads((ROOT/'data_complete.json').read_text());x=transform(np.load(ROOT/'features.npy'),arm);y=np.array([r['class_id'] for r in rows]);pilot=set(map(str,meta['pilot_train_subjects']));ti=[i for i,r in enumerate(rows) if r['split']=='train' and (mode=='full' or r['dataset']=='SEED' or r['subject'] in pilot)];vi=[i for i,r in enumerate(rows) if r['split']=='validation'];assert len(vi)==8350;assert not {(rows[i]['dataset'],rows[i]['subject']) for i in ti}&{(rows[i]['dataset'],rows[i]['subject']) for i in vi};counts={}
 for i in ti:
  key=(rows[i]['dataset'],int(y[i]));counts[key]=counts.get(key,0)+1
 w=np.array([1/counts[(rows[i]['dataset'],int(y[i]))] for i in ti]);w/=w.mean();best=-1;runs=[]
 for c in [.001,.01,.1,1.,10.]:
  m=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=2000,random_state=42))
  with warnings.catch_warnings(record=True) as caught:
   warnings.simplefilter('always');m.fit(x[ti],y[ti],logisticregression__sample_weight=w)
  scores={}
  for ds in ['SEED','FACED-NEMAR']:
   ids=[i for i in vi if rows[i]['dataset']==ds];scores[ds]=metrics([rows[i] for i in ids],m.predict(x[ids]))
  value=float(np.mean([v['subject_balanced_accuracy'] for v in scores.values()]));record={'C':c,'macro_balanced':value,'datasets':scores,'warnings':[str(v.message) for v in caught]};runs.append(record);print(mode,arm,c,value,flush=True)
  if value>best:best=value;selected=record;joblib.dump(m,out/'best.joblib')
 (out/'complete.json').write_text(json.dumps({'mode':mode,'arm':arm,'train_n':len(ti),'val_n':len(vi),'validation_rows_sha256':hashlib.sha256(json.dumps([rows[i] for i in vi],sort_keys=True).encode()).hexdigest(),'selected':selected,'sweep':runs,'sha256':hashlib.sha256((out/'best.joblib').read_bytes()).hexdigest(),'note':'Source validation only; target evaluation pending.'},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--mode',choices=['pilot','full']);p.add_argument('--arm',choices=['full_linear','spatial_linear']);a=p.parse_args()
 with threadpool_limits(limits=4):
  if a.prepare:prepare()
  elif a.mode and a.arm:fit(a.mode,a.arm)
  else:p.error('Choose a stage')
