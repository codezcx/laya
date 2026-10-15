"""Frequency-band and continuous-context controls for emotion transfer."""
import argparse,hashlib,json,re
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.signal import welch
from .data import preprocess
from .emotion_transfer_probe import CHANNELS,BANDS,picks,fractions,readrows,metrics
ROOT=Path('artifacts/emotion_context_v20')
MODES={'hz30_s3':(30,600),'hz45_s3':(45,600),'hz30_s9':(30,1800),'hz45_s9':(45,1800)}

def bounds(start,length,total):
 if length<600 or total<length or not 0<=start<=total-600:raise ValueError('Invalid window bounds')
 begin=max(0,min(start-(length-600)//2,total-length))
 assert begin<=start and begin+length>=start+600
 return begin,begin+length

def spectrum(x,high):
 bands=BANDS+([(30,45)] if high==45 else [])
 f,p=welch(x,fs=200,nperseg=200,noverlap=100,axis=-1);power=np.stack([p[...,(f>=lo)&(f<hi)].sum(-1) for lo,hi in bands],-1)
 return fractions(np.log(np.maximum(power,np.finfo(np.float64).tiny)))

def prepare(shard):
 import mne
 from .faced_prepare import read_table,video_events
 out=ROOT/'features';out.mkdir(parents=True,exist_ok=True)
 if (out/f'{shard}.npz').exists():raise FileExistsError(out)
 manifest=Path('artifacts/emotion_transfer_v16/rows.jsonl');rows=readrows(manifest);groups={}
 for i,r in enumerate(rows):groups.setdefault(r['path'],[]).append(i)
 cfg=json.loads(Path('configs/multitask_prototypes_v1.json').read_text());names=[s.split()[-1] for s in Path(cfg['seed_channel_file']).read_text().splitlines() if s.strip()];indices=[];features={k:[] for k in MODES};audit=[];windows=[]
 for path in sorted(groups)[shard::4]:
  ids=groups[path];rr=[rows[i] for i in ids];ds=rr[0]['dataset'];by_trial={}
  for i in ids:by_trial.setdefault(rows[i]['trial'],[]).append(i)
  if ds=='SEED':
   mat=loadmat(path);keys={str(int(re.search(r'eeg(\d+)$',k).group(1))):k for k in mat if re.search(r'eeg(\d+)$',k)};cp=picks(names);fs=200
  else:
   raw=mne.io.read_raw_bdf(path,preload=False,verbose=False);side=json.loads(Path(path.replace('_eeg.bdf','_eeg.json')).read_text());assert raw.info['sfreq']==side['SamplingFrequency'];cp=picks(raw.ch_names);fs=raw.info['sfreq'];events={str(e['video_index']):e for e in video_events(read_table(path.replace('_eeg.bdf','_events.tsv')))}
  for trial,ii in sorted(by_trial.items()):
   if ds=='SEED':signal=np.asarray(mat[keys[trial]][cp],np.float64)
   else:
    e=events[trial];start=round(float(e['onset'])*fs);stop=round((float(e['onset'])+float(e['duration']))*fs);signal=raw.get_data(picks=cp,start=start,stop=stop);assert all(rows[i]['event_onset_seconds']==float(e['onset']) for i in ii)
   filtered={high:preprocess(signal,fs,dict(cfg,bandpass=[1,high])) for high in [30,45]}
   assert filtered[30].shape==filtered[45].shape and filtered[30].shape[0]==30
   for i in ii:
    start=rows[i]['start_sample'];window_audit={'index':i,'windows':{}}
    for mode,(high,length) in MODES.items():
     begin,end=bounds(start,length,filtered[high].shape[-1]);x=filtered[high][:,begin:end];assert x.shape==(30,length) and np.isfinite(x).all();features[mode].append(spectrum(x,high));window_audit['windows'][mode]=[begin,end]
    indices.append(i);windows.append(window_audit)
  audit.append({'path':path,'dataset':ds,'sample_rate':fs,'reference_metadata':'unknown' if ds=='SEED' else side.get('EEGReference'),'units':'unconfirmed MAT amplitude' if ds=='SEED' else 'MNE volts','n':len(ids)})
  if ds=='SEED':del mat
  else:raw.close()
  print(shard,Path(path).name,len(indices),flush=True)
 arrays={k:np.array(v,dtype=np.float32) for k,v in features.items()};assert all(a.shape==(len(indices),30,6 if MODES[k][0]==45 else 5) and np.isfinite(a).all() for k,a in arrays.items());np.savez(out/f'{shard}.npz',indices=np.array(indices),**arrays);(out/f'{shard}.done.json').write_text(json.dumps({'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'channels':CHANNELS,'audit':audit,'windows':windows,'n':len(indices)},indent=2))

def join():
 rows=readrows(Path('artifacts/emotion_transfer_v16/rows.jsonl'));digest=hashlib.sha256(Path('artifacts/emotion_transfer_v16/rows.jsonl').read_bytes()).hexdigest();arrays={k:np.empty((len(rows),30,6 if high==45 else 5),np.float32) for k,(high,_) in MODES.items()};seen=[]
 for i in range(4):
  done=json.loads((ROOT/'features'/f'{i}.done.json').read_text());assert done['manifest_sha256']==digest;d=np.load(ROOT/'features'/f'{i}.npz');seen+=d['indices'].tolist()
  for mode in arrays:arrays[mode][d['indices']]=d[mode]
 assert sorted(seen)==list(range(len(rows)))
 for mode,x in arrays.items():
  p=ROOT/mode;p.mkdir(exist_ok=False);np.save(p/'features.npy',x);(p/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
 (ROOT/'data_complete.json').write_text(json.dumps({'n':len(rows),'manifest_sha256':digest,'modes':list(arrays),'note':'Matched 30 channels; full trial filtering; nested continuous3/9s windows; no target fitting or reference changes.'},indent=2))

def fit(mode,arm):
 from threadpoolctl import threadpool_limits
 from . import emotion_transfer_probe as probe
 probe.ROOT=ROOT/mode
 with threadpool_limits(limits=4):probe.train(arm)

def evaluate():
 import joblib
 frozen={};results={}
 for mode in MODES:
  for source in ['SEED','FACED-NEMAR']:
   for arm in ['full_linear']:
    p=ROOT/mode/f'{source}_{arm}';h=hashlib.sha256((p/'best.joblib').read_bytes()).hexdigest();assert h==json.loads((p/'complete.json').read_text())['sha256'];frozen[str(p)]=h
 out=ROOT/'evaluation';out.mkdir(exist_ok=False);(out/'frozen.json').write_text(json.dumps(frozen,indent=2))
 from .emotion_transfer_probe import transform
 for mode in MODES:
  rows=readrows(ROOT/mode/'rows.jsonl');x=np.load(ROOT/mode/'features.npy')
  for source in ['SEED','FACED-NEMAR']:
   ids=[i for i,r in enumerate(rows) if r['dataset']!=source];rr=[rows[i] for i in ids]
   for arm in ['full_linear']:
    model=joblib.load(ROOT/mode/f'{source}_{arm}'/'best.joblib');pred=model.predict(transform(x[ids],arm));m=metrics(rr,pred);m['shuffle']=metrics(rr,pred[np.random.default_rng(2026).permutation(len(pred))]);name=f'{mode}_{source}_{arm}';results[name]=m;np.save(out/f'{name}_pred.npy',pred);print(name,m['subject_balanced_accuracy'],flush=True)
 (out/'results.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--shard',type=int,choices=range(4));p.add_argument('--join',action='store_true');p.add_argument('--evaluate',action='store_true');p.add_argument('--mode',choices=list(MODES));p.add_argument('--arm',choices=['full_linear']);a=p.parse_args()
 if a.shard is not None:prepare(a.shard)
 elif a.join:join()
 elif a.evaluate:evaluate()
 elif a.mode and a.arm:fit(a.mode,a.arm)
 else:p.error('Choose stage')
