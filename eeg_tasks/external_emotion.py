"""Locked SEED-VII happy/neutral/sad external evaluation; no fit or selection."""
import argparse,hashlib,json,time,zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np
import torch
from scipy.io import loadmat
from transformers import AutoModel
from .data import config,preprocess,normalized_window,write_json
from .model import PrototypeClassifier
from .train import metrics,predict
SOURCE=Path('/inspire/hdd/global_user/zhangchenxi-253108310322/mne_data/SEED-VII')

def frozen():
 root=Path(config()['output']);hashes={}
 for arm in config()['arms']:
  expected=json.loads((root/arm/'frozen.json').read_text())['checkpoint_sha256']
  actual=hashlib.sha256((root/arm/'best.pt').read_bytes()).hexdigest();assert actual==expected
  hashes[arm]=actual
 return hashes

def labels():
 ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
 with zipfile.ZipFile(SOURCE/'emotion_label_and_stimuli_order.xlsx') as z:
  strings=[''.join(n.itertext()) for n in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',ns)]
  table=[]
  for row in ET.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('.//s:row',ns):
   vals=[]
   for c in row:
    v=c.find('s:v',ns)
    if v is not None:vals.append(strings[int(v.text)] if c.attrib.get('t')=='s' else v.text)
   table.append(vals)
 mapping={}
 for i in range(0,8,2):
  for index,label in zip(table[i][1:],table[i+1][1:]):mapping[int(index)]=label
 assert len(mapping)==80
 return mapping

@torch.no_grad()
def extract(shard):
 hashes=frozen();cfg=config();out=Path(cfg['output'])/'external_seed_vii';out.mkdir(exist_ok=True)
 if (out/f'{shard}.done.json').exists():raise FileExistsError(out)
 torch.set_num_threads(4);base=json.loads(Path(cfg['base_config']).read_text())
 model=AutoModel.from_pretrained(base['reve_checkpoint'],local_files_only=True,trust_remote_code=True).eval().requires_grad_(False).cuda()
 bank=AutoModel.from_pretrained(base['positions_checkpoint'],local_files_only=True,trust_remote_code=True)
 pos=bank(cfg['channels']).detach().cuda()
 names=[s.split()[-1].upper() for s in (SOURCE/'channel_62_pos.locs').read_text().splitlines() if s.strip()]
 picks=[names.index(c.upper()) for c in cfg['channels']];lab=labels();classmap={'Sad':0,'Neutral':1,'Happy':2}
 selected={str(k):classmap[v] for k,v in lab.items() if v in classmap};assert len(selected)==32
 features=[];rows=[];started=time.monotonic()
 for subject in range(1+shard,21,4):
  path=SOURCE/'EEG_preprocessed'/f'{subject}.mat'
  data=loadmat(path,variable_names=list(selected))
  assert all(k in data for k in selected)
  signals=[];meta=[]
  for key,y in selected.items():
   raw=data[key];assert raw.shape[0]==62
   x=preprocess(raw[picks],200,cfg);n=x.shape[-1]//600
   for start in np.linspace(0,n-1,min(n,10),dtype=int)*600:
    signals.append(normalized_window(x[:,start:start+600]))
    meta.append({'dataset':'SEED-VII-HNS','subject':str(subject),'task':0,'class_id':y,'split':'external','path':str(path),'trial':key,'start_sample':int(start),'record_id':f'{subject}/{key}','original_label':lab[int(key)]})
  del data
  for begin in range(0,len(signals),64):
   x=torch.from_numpy(np.stack(signals[begin:begin+64])).cuda();stride=model.config.patch_size-model.config.patch_overlap;tail=(-(x.shape[-1]-model.config.patch_size))%stride
   x=torch.nn.functional.pad(x,(0,tail),mode='replicate');z=model(x,pos[None].expand(len(x),-1,-1)).mean(2).flatten(1)
   features.append(z.cpu().numpy())
  rows.extend(meta);print('subject',subject,'total',len(rows),flush=True)
 np.save(out/f'features_{shard}.npy',np.concatenate(features));(out/f'rows_{shard}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
 write_json(out/f'{shard}.done.json',{'n':len(rows),'seconds':time.monotonic()-started,'frozen_checkpoints':hashes,'mapping':classmap})

def evaluate():
 hashes=frozen();cfg=config();root=Path(cfg['output']);out=root/'external_seed_vii'
 if (out/'results.json').exists():raise FileExistsError('Do not overwrite target results')
 torch.set_num_threads(4);features=[];rows=[]
 for shard in range(4):
  assert json.loads((out/f'{shard}.done.json').read_text())['frozen_checkpoints']==hashes
  features.append(np.load(out/f'features_{shard}.npy'));rows +=[json.loads(s) for s in (out/f'rows_{shard}.jsonl').read_text().splitlines()]
 x=torch.from_numpy(np.concatenate(features)).cuda();results={}
 for arm in cfg['arms']:
  state=torch.load(root/arm/'best.pt',weights_only=True,map_location='cuda');model=PrototypeClassifier(state['prototypes'],arm).cuda();model.load_state_dict(state)
  logits=predict(model,x);normal=metrics(rows,logits);np.save(out/f'{arm}_logits.npy',logits.numpy())
  # Balanced subject estimates: unequal 12/8/12 trial counts across classes.
  yy=np.array([r['class_id'] for r in rows]);pp=logits[:,:3].argmax(-1).numpy();by_subject={}
  for subject in sorted({r['subject'] for r in rows}):
   ids=np.array([r['subject']==subject for r in rows]);by_subject[subject]=float(np.mean([np.mean(pp[ids&(yy==c)]==c) for c in range(3)]))
  values=np.array(list(by_subject.values()));boot=np.random.default_rng(2026).choice(values,(3000,len(values)),replace=True).mean(1)
  normal['subject_macro_balanced_accuracy']=float(values.mean());normal['subject_balanced_95ci']=np.quantile(boot,[.025,.975]).tolist();normal['per_subject_balanced']=by_subject
  shuffled=np.random.default_rng(723).permutation(len(x));results[arm]={'normal':normal,'shuffled':metrics(rows,predict(model,x[shuffled]))}
 write_json(out/'results.json',{'results':results,'frozen_checkpoints':hashes,'note':'No target fitting. Valence mapping restricted to Happy/Neutral/Sad; not full SEED-VII 7-class recognition. Pretraining exposure not ruled out.'})
 print(json.dumps({k:v['normal']['subject_macro_balanced_accuracy'] for k,v in results.items()}))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--shard',type=int);p.add_argument('--evaluate',action='store_true');a=p.parse_args();evaluate() if a.evaluate else extract(a.shard)
