"""All-arm frozen SEED-VII evaluation; includes a jointly trained spectral control."""
import argparse,hashlib,json
from pathlib import Path
import joblib
import numpy as np
import torch
from scipy.io import loadmat
from transformers import AutoModel
from .emotion_alignment import ROOT,ARMS,AlignmentModel
from .emotion_transfer_probe import CHANNELS,picks,readrows,metrics
from .emotion_reference import spectrum
from .external_emotion import SOURCE,labels
from .data import preprocess,normalized_window

def freeze():
 hashes={}
 for arm in ARMS:
  p=ROOT/arm;h=hashlib.sha256((p/'best.pt').read_bytes()).hexdigest();assert h==json.loads((p/'complete.json').read_text())['sha256'];hashes[arm]=h
 return hashes

def spectral_control():
 from sklearn.pipeline import make_pipeline
 from sklearn.preprocessing import StandardScaler
 from sklearn.linear_model import LogisticRegression
 from threadpoolctl import threadpool_limits
 freeze();out=ROOT/'spectral_control';out.mkdir(exist_ok=False);rows=readrows(Path('artifacts/emotion_transfer_v16/rows.jsonl'));x=np.load('artifacts/emotion_transfer_v16/features.npy').reshape(len(rows),150);y=np.array([r['class_id'] for r in rows]);ti=[i for i,r in enumerate(rows) if r['split']=='train'];vi=[i for i,r in enumerate(rows) if r['split']=='validation'];counts={}
 for i in ti:
  key=(rows[i]['dataset'],int(y[i]));counts[key]=counts.get(key,0)+1
 w=np.array([1/counts[(rows[i]['dataset'],int(y[i]))] for i in ti]);w/=w.mean();best=-1;runs=[]
 with threadpool_limits(limits=4):
  for c in [.001,.01,.1,1.,10.]:
   model=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=2000,random_state=42));model.fit(x[ti],y[ti],logisticregression__sample_weight=w);scores={}
   for ds in ['SEED','FACED-NEMAR']:
    ids=[i for i in vi if rows[i]['dataset']==ds];scores[ds]=metrics([rows[i] for i in ids],model.predict(x[ids]))
   score=float(np.mean([v['subject_balanced_accuracy'] for v in scores.values()]));runs.append({'C':c,'macro_balanced':score,'datasets':scores})
   if score>best:best=score;selected=runs[-1];joblib.dump(model,out/'best.joblib')
 (out/'complete.json').write_text(json.dumps({'sha256':hashlib.sha256((out/'best.joblib').read_bytes()).hexdigest(),'selected':selected,'sweep':runs},indent=2))

@torch.no_grad()
def extract(shard):
 hashes=freeze();out=ROOT/'external_seed_vii';out.mkdir(exist_ok=True)
 if (out/f'{shard}.done.json').exists():raise FileExistsError(out)
 torch.set_num_threads(4);base=json.loads(Path('configs/reve_qwen_mi_multi.json').read_text());model=AutoModel.from_pretrained(base['reve_checkpoint'],trust_remote_code=True,local_files_only=True).cuda().eval().requires_grad_(False);bank=AutoModel.from_pretrained(base['positions_checkpoint'],trust_remote_code=True,local_files_only=True);pos=bank(CHANNELS).cuda();assert pos.shape==(30,3);capture={};model.transformer.layers[-2][0].register_forward_pre_hook(lambda module,args:capture.update(x=args[0].detach()));names=[s.split()[-1] for s in (SOURCE/'channel_62_pos.locs').read_text().splitlines() if s.strip()];cp=picks(names);mapping={'Sad':0,'Neutral':1,'Happy':2};lab=labels();selected={str(k):mapping[v] for k,v in lab.items() if v in mapping};assert len(selected)==32;features=[];freq=[];rows=[]
 for subject in range(shard+1,21,4):
  path=SOURCE/'EEG_preprocessed'/f'{subject}.mat';data=loadmat(path,variable_names=list(selected));signals=[]
  for trial,y in selected.items():
   x=preprocess(data[trial][cp],200,{'bandpass':[1,30]});n=x.shape[-1]//600
   for start in np.linspace(0,n-1,10,dtype=int)*600:
    window=x[:,start:start+600];signals.append(normalized_window(window));freq.append(spectrum(window).reshape(150));rows.append({'dataset':'SEED-VII-HNS','subject':str(subject),'task':0,'class_id':y,'path':str(path),'trial':trial,'start_sample':int(start)})
  del data
  for start in range(0,len(signals),32):
   x=torch.from_numpy(np.stack(signals[start:start+32])).cuda();tail=(-(600-model.config.patch_size))%(model.config.patch_size-model.config.patch_overlap);model(torch.nn.functional.pad(x,(0,tail),mode='replicate'),pos[None].expand(len(x),-1,-1));features.append(capture['x'].cpu().numpy().astype(np.float16))
  print(shard,subject,len(rows),flush=True)
 np.save(out/f'{shard}.npy',np.concatenate(features));np.save(out/f'spectral_{shard}.npy',np.array(freq,np.float32));(out/f'rows_{shard}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));(out/f'{shard}.done.json').write_text(json.dumps({'n':len(rows),'frozen':hashes,'channels':CHANNELS},indent=2))

@torch.no_grad()
def evaluate():
 hashes=freeze();out=ROOT/'external_seed_vii';assert not (out/'results.json').exists();torch.set_num_threads(4);rows=[];arrays=[];freq=[]
 for shard in range(4):
  assert json.loads((out/f'{shard}.done.json').read_text())['frozen']==hashes;rows.extend(readrows(out/f'rows_{shard}.jsonl'));arrays.append(np.load(out/f'{shard}.npy'));freq.append(np.load(out/f'spectral_{shard}.npy'))
 assert len(rows)==6400 and len({r['subject'] for r in rows})==20;x=torch.from_numpy(np.concatenate(arrays)).cuda();base=json.loads(Path('configs/reve_qwen_mi_multi.json').read_text());results={}
 for arm in ARMS:
  reve=AutoModel.from_pretrained(base['reve_checkpoint'],trust_remote_code=True,local_files_only=True).eval().requires_grad_(False);model=AlignmentModel(reve,False).cuda();model.load_state_dict(torch.load(ROOT/arm/'best.pt',weights_only=True,map_location='cuda'));model.eval();z=torch.cat([model(b)[0].cpu() for b in x.split(48)]);m=metrics(rows,z.argmax(1).numpy());m['shuffle']=metrics(rows,z[np.random.default_rng(2026).permutation(len(z))].argmax(1).numpy());results[arm]=m;np.save(out/f'{arm}_logits.npy',z.numpy());print(arm,m['subject_balanced_accuracy'],flush=True);del model,reve
 p=ROOT/'spectral_control';digest=hashlib.sha256((p/'best.joblib').read_bytes()).hexdigest();assert digest==json.loads((p/'complete.json').read_text())['sha256'];results['spectral_control']=metrics(rows,joblib.load(p/'best.joblib').predict(np.concatenate(freq)));(out/'results.json').write_text(json.dumps({'frozen':hashes,'spectral_sha256':digest,'results':results,'note':'Repeated-target exploratory evaluation; no target selection. SEED-VII restricted to Happy/Neutral/Sad.'},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--shard',type=int,choices=range(4));p.add_argument('--spectral-control',action='store_true');p.add_argument('--evaluate',action='store_true');a=p.parse_args()
 if a.spectral_control:spectral_control()
 elif a.evaluate:evaluate()
 elif a.shard is not None:extract(a.shard)
 else:p.error('Choose stage')
