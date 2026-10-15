"""Source-validation-only convex probability fusion of frozen spectral/REVE models."""
import hashlib,json
from pathlib import Path
import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .emotion_fusion_train import load_features,Classifier,predict
from .emotion_transfer_probe import metrics
ROOT=Path('artifacts/emotion_late_fusion_v18')

def blend(spectral,reve,alpha):
 if not 0<=alpha<=1:raise ValueError(alpha)
 for x in [spectral,reve]:
  if x.ndim!=2 or x.shape[1]!=3 or not np.isfinite(x).all() or (x<0).any() or not np.allclose(x.sum(1),1,atol=1e-5):raise ValueError('Expected three-class probabilities')
 if spectral.shape!=reve.shape:raise ValueError('Shape mismatch')
 return (1-alpha)*spectral+alpha*reve

@torch.no_grad()
def main():
 ROOT.mkdir(exist_ok=False);torch.set_num_threads(4);rows,raw=load_features('reve');spectral=np.load('artifacts/emotion_transfer_v16/features.npy').reshape(len(rows),-1);choices={};cache={};hashes={}
 for source in ['SEED','FACED-NEMAR']:
  sp=Path('artifacts/emotion_transfer_v16')/f'{source}_full_linear';rp=Path('artifacts/emotion_fusion_v17')/f'{source}_reve'
  for path,filename in [(sp,'best.joblib'),(rp,'best.pt')]:
   h=hashlib.sha256((path/filename).read_bytes()).hexdigest();assert h==json.loads((path/'complete.json').read_text())['sha256'];hashes[str(path)]=h
  sm=joblib.load(sp/'best.joblib');assert list(sm.classes_)==[0,1,2];state=torch.load(rp/'best.pt',weights_only=True);rm=Classifier(raw.shape[1],state['model']['prototypes']).cuda();rm.load_state_dict(state['model']);ids=[i for i,r in enumerate(rows) if r['dataset']==source and r['split']=='validation'];rr=[rows[i] for i in ids];p=sm.predict_proba(spectral[ids]);x=(torch.from_numpy(raw[ids])-state['mean'])/state['std'];q=predict(rm,x.cuda()).softmax(1).numpy();candidates=[]
  for alpha in [0.,.25,.5,.75,1.]:
   m=metrics(rr,blend(p,q,alpha).argmax(1));candidates.append({'alpha':alpha,'validation':m})
  selected=max(candidates,key=lambda c:c['validation']['subject_balanced_accuracy']);choices[source]={'selected_alpha':selected['alpha'],'candidates':candidates};cache[source]=(sm,rm,state)
 # Persist both choices before any held-dataset prediction/scoring.
 (ROOT/'selection.json').write_text(json.dumps({'choices':choices,'frozen_models':hashes,'criterion':'source validation mean subject balanced accuracy; ties choose smaller REVE weight'},indent=2));results={}
 for source,(sm,rm,state) in cache.items():
  ids=[i for i,r in enumerate(rows) if r['dataset']!=source];rr=[rows[i] for i in ids];p=sm.predict_proba(spectral[ids]);x=(torch.from_numpy(raw[ids])-state['mean'])/state['std'];q=predict(rm,x.cuda()).softmax(1).numpy();alpha=choices[source]['selected_alpha'];result={'selected_alpha':alpha}
  for name,a in [('spectral',0.),('reve',1.),('selected_fusion',alpha)]:
   prob=blend(p,q,a);result[name]=metrics(rr,prob.argmax(1));np.save(ROOT/f'{source}_{name}_probabilities.npy',prob)
  # Paired subject bootstrap of fusion minus spectral, independent of window counts.
  s=result['spectral']['per_subject'];f=result['selected_fusion']['per_subject'];d=np.array([f[k]['balanced_accuracy']-s[k]['balanced_accuracy'] for k in sorted(s)]);boot=np.random.default_rng(2026).choice(d,(10000,len(d)),replace=True).mean(1);result['paired_difference']={'mean':float(d.mean()),'subject_95ci':np.quantile(boot,[.025,.975]).tolist()};results[source]=result;print(source,alpha,{k:result[k]['subject_balanced_accuracy'] for k in ['spectral','reve','selected_fusion']},flush=True)
 (ROOT/'results.json').write_text(json.dumps(results,indent=2));(ROOT/'complete.json').write_text(json.dumps({'status':'complete','selection_sha256':hashlib.sha256((ROOT/'selection.json').read_bytes()).hexdigest()},indent=2))
if __name__=='__main__':
 with threadpool_limits(limits=4):main()
