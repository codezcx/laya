"""Frozen LEAF baseline and query caches on subject-disjoint 2a/SEED inputs."""
import json,sys,hashlib,subprocess
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from scipy.io import loadmat
import mne
from .task_options import load_tasks
ROOT=Path('datasets/leaf_2a_seed');OUT=Path('artifacts/leaf_2a_seed')
CONFIG=json.loads(Path('configs/leaf_train.json').read_text());UP=Path(CONFIG['leaf_root'])
TASKS=list(load_tasks().values())
sys.path.insert(0,str(UP))
from LEAF import LEAF
from load_config import build_model_config,load_yaml
from leaf_datasets.shared import pipeline
from leaf_datasets.MI_BCIC_IV2a import CH_NAMES as MI_NAMES
from leaf_datasets.EMO_SEED_3 import CH_NAMES as SEED_NAMES
import init_text_embeddings
init_text_embeddings.EMBEDDING_ROOT=UP/'text_embeddings'
from init_text_embeddings import load_embeddings
BASE=Path('/inspire/hdd/global_user/zhangchenxi-253108310322/mne_data')

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()

def metrics(rows,prob):
 result={}
 for task in range(2):
  ix=[i for i,r in enumerate(rows) if r['task']==task];rr=[rows[i] for i in ix];p=prob[ix];k=len(TASKS[task]['classes'])
  def calc(y,p):
   z=np.argmax(p[:,:k],axis=1);cm=np.zeros((k,k),int);np.add.at(cm,(y,z),1)
   return dict(n=len(y),accuracy=float(np.mean(y==z)),balanced_accuracy=float(np.mean(np.diag(cm)/cm.sum(1))),confusion_matrix=cm.tolist())
  parents={r['parent_id']:[] for r in rr}
  for j,r in enumerate(rr):parents[r['parent_id']].append(j)
  result[TASKS[task]['datasets'][0]]=dict(window=calc(np.array([r['label'] for r in rr]),p),trial=calc(np.array([rr[v[0]]['label'] for v in parents.values()]),np.stack([p[v].mean(0) for v in parents.values()])))
 return result

def batches():
 folder=BASE/'MNE-bnci-data/~bci/database/001-2014'
 for subject in range(1,10):
  split='train' if subject<=6 else 'validation' if subject==7 else 'test'
  if split=='test':continue
  for sess in ['T','E']:
   path=folder/f'A{subject:02d}{sess}.mat';runs=loadmat(path,simplify_cells=True)['data'];xx=[];yy=[];ids=[]
   for ri,run in enumerate(runs):
    labels=np.asarray(run['y']).reshape(-1).astype(int);events=np.asarray(run['trial']).reshape(-1).astype(int)
    if not len(labels):continue
    assert len(labels)==len(events) and set(labels)<={1,2,3,4}
    x=np.asarray(run['X'])[:,:22].T; x=x-x.mean(0,keepdims=True)
    for j,(event,y) in enumerate(zip(events,labels)):
     # MATLAB event indices are 1-based, trial onset precedes cue by 2s.
     start=int(event)-1;part=x[:,start:start+1500];assert part.shape==(22,1500)
     xx.append(part);yy.append(int(y)-1);ids.append(f'2a-{subject}-{sess}-{ri}-{j}')
   assert len(xx)==288,(path,len(xx))
   epo=mne.EpochsArray(np.asarray(xx),mne.create_info(MI_NAMES,250,'eeg'),verbose=False)
   epo.filter(.3,40,verbose=False).resample(200,npad='auto',verbose=False)
   x=pipeline(epo.get_data().astype(np.float32)[:,:,400:1200],MI_NAMES)
   rows=[dict(id=i,parent_id=i,task=0,dataset='2a',subject=str(subject),session=sess,label=y,split=split,source=str(path)) for i,y in zip(ids,yy)]
   yield x,rows
 folder=BASE/'SEED/SEED/SEED_EEG/Preprocessed_EEG'
 labels=loadmat(folder/'label.mat')['label'].reshape(-1).astype(int)+1
 old=json.loads(Path('datasets/mi_emotion/complete.json').read_text())['emotion_subject_splits']['SEED']
 for path in sorted(folder.glob('*.mat')):
  if path.name=='label.mat':continue
  subject=path.stem.split('_')[0];split=old[subject]
  if split=='test':continue
  data=loadmat(path)
  for trial in range(1,16):
   keys=[k for k in data if k.endswith('_eeg'+str(trial))];assert len(keys)==1
   x=np.asarray(data[keys[0]],dtype=np.float64)*1e-6
   raw=mne.io.RawArray(x,mne.create_info(SEED_NAMES,200,'eeg'),verbose=False)
   raw.filter(.1,70,verbose=False);raw.notch_filter(50,notch_widths=2,verbose=False)
   x=raw.get_data().astype(np.float32);count=x.shape[1]//800
   windows=np.arange(count) if split=='train' else np.unique(np.linspace(0,count-1,min(8,count),dtype=int))
   x=np.stack([x[:,w*800:(w+1)*800] for w in windows]);x=np.clip(pipeline(x,SEED_NAMES),-10,10)
   parent=f'SEED-{path.stem}-{trial}'
   rows=[dict(id=f'{parent}-{w}',parent_id=parent,task=1,dataset='SEED',subject=subject,session=path.stem,label=int(labels[trial-1]),split=split,source=str(path),window=int(w)) for w in windows]
   yield x,rows

def prepare():
 if (ROOT/'complete.json').exists():raise FileExistsError(ROOT)
 ROOT.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
 assert subprocess.check_output(['git','-C',str(UP),'rev-parse','HEAD'],text=True).strip()==CONFIG['upstream_commit']
 torch.set_num_threads(3);model=LEAF(build_model_config(load_yaml(UP/'configs/LEAF_mpnet.yaml')))
 model.load_state_dict(torch.load(CONFIG['checkpoint'],map_location='cpu',weights_only=True),strict=True);model.cuda().eval().requires_grad_(False)
 texts=load_embeddings('mpnet-base');queries={'train':[],'validation':[]};scores={'train':[],'validation':[]};rows={'train':[],'validation':[]}
 for x,rr in batches():
  assert x.shape[1:]==(65,800) and np.isfinite(x).all()
  t=rr[0]['task'];task=TASKS[t];split=rr[0]['split'];qq=[];ss=[]
  inst=torch.from_numpy(texts[task['instruction']]).cuda()[None]
  proto=torch.from_numpy(np.stack([texts[c['prototype']] for c in task['classes']])).cuda()
  with torch.inference_mode():
   for start in range(0,len(x),64):
    inp=torch.from_numpy(x[start:start+64]).float().cuda()
    emb,_,history=model.forward_with_q(inp,inst.expand(len(inp),-1));q=model.qformer.final_norm(history[-1])
    assert q.shape[1:]==(8,256) and torch.isfinite(q).all()
    qq.append(q.cpu().numpy());p=(emb@proto.T).softmax(-1).cpu().numpy();ss.append(np.pad(p,((0,0),(0,4-p.shape[1]))))
  queries[split].append(np.concatenate(qq));scores[split].append(np.concatenate(ss));rows[split].extend(rr)
  progress={s:len(r) for s,r in rows.items()};(OUT/'prepare_progress.json').write_text(json.dumps(progress));print('ENCODED',rr[0]['id'],len(rr),progress,flush=True)
 result={};hashes={}
 subjects={s:{(r['dataset'],r['subject']) for r in rr} for s,rr in rows.items()};assert not subjects['train']&subjects['validation']
 for split,rr in rows.items():
  assert len({r['id'] for r in rr})==len(rr)
  np.save(ROOT/f'{split}_queries.npy',np.concatenate(queries[split]));np.save(ROOT/f'{split}_probabilities.npy',np.concatenate(scores[split]))
  (ROOT/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rr))
  result[split]=metrics(rr,np.concatenate(scores[split]))
  for name in [f'{split}_queries.npy',f'{split}_probabilities.npy',f'{split}.jsonl']:hashes[name]=digest(ROOT/name)
 source_hashes={path:digest(path) for path in sorted({r['source'] for rr in rows.values() for r in rr})}
 config=dict(configuration=CONFIG,checkpoint_sha256=digest(CONFIG['checkpoint']),task_catalog_sha256=digest('configs/mi_emotion_tasks.json'),files_sha256=hashes,source_sha256=source_hashes,counts={s:dict(Counter(r['dataset'] for r in rr)) for s,rr in rows.items()},test_policy='2a subjects8,9 and existing SEED test subjects untouched; released-model prior exposure not excluded',preprocessing='2a per session; SEED per parent trial normalization/interpolation; no cross-trial session concatenation')
 (ROOT/'complete.json').write_text(json.dumps(config,indent=2));(OUT/'baseline.json').write_text(json.dumps(result,indent=2));print('BASELINE',result,flush=True)

def load(split):
 done=json.loads((ROOT/'complete.json').read_text());assert done['task_catalog_sha256']==digest('configs/mi_emotion_tasks.json')
 for name in [f'{split}.jsonl',f'{split}_queries.npy']:assert digest(ROOT/name)==done['files_sha256'][name]
 rows=[json.loads(x) for x in (ROOT/f'{split}.jsonl').read_text().splitlines()];x=np.load(ROOT/f'{split}_queries.npy',mmap_mode='r');assert x.shape==(len(rows),8,256)
 return rows,x

if __name__=='__main__':prepare()
