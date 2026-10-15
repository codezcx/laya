"""One-shot evaluation on all original 2b E trials after checkpoint freeze."""
import argparse,json,pickle,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel
from .data import config,preprocess,normalized_window,write_json
from .external_emotion import frozen
from .model import PrototypeClassifier
from .train import metrics,predict

@torch.no_grad()
def extract(shard):
 hashes=frozen();cfg=config();root=Path(cfg['output']);out=root/'external_2b';out.mkdir(exist_ok=True)
 if (out/f'{shard}.done.json').exists():raise FileExistsError(out)
 base=json.loads(Path(cfg['base_config']).read_text());source=[json.loads(s) for s in (Path(base['manifest_dir'])/'external_2b.jsonl').read_text().splitlines()]
 assert len(source)==2840
 rows=[];signals=[];torch.set_num_threads(4);start=time.monotonic()
 for i in range(shard,len(source),4):
  r=source[i]
  with open(r['path'],'rb') as f:d=pickle.load(f)
  assert d['freq']==200 and base['labels'].index(d['label'])==r['label']
  x=np.stack([d['eeg'][c.upper()] for c in cfg['channels']]);x=preprocess(x,200,cfg);signals.append(normalized_window(x[:,:600]));rows.append(dict(r,task=1,class_id=r['label']+3,split='external_2b'))
 model=AutoModel.from_pretrained(base['reve_checkpoint'],local_files_only=True,trust_remote_code=True).eval().requires_grad_(False).cuda();bank=AutoModel.from_pretrained(base['positions_checkpoint'],local_files_only=True,trust_remote_code=True);pos=bank(cfg['channels']).detach().cuda();features=[]
 for begin in range(0,len(signals),64):
  x=torch.from_numpy(np.stack(signals[begin:begin+64])).cuda();stride=model.config.patch_size-model.config.patch_overlap;tail=(-(600-model.config.patch_size))%stride;x=torch.nn.functional.pad(x,(0,tail),mode='replicate');z=model(x,pos[None].expand(len(x),-1,-1)).mean(2).flatten(1);features.append(z.cpu().numpy())
 np.save(out/f'features_{shard}.npy',np.concatenate(features));(out/f'rows_{shard}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));write_json(out/f'{shard}.done.json',{'n':len(rows),'seconds':time.monotonic()-start,'frozen_checkpoints':hashes})

def evaluate():
 hashes=frozen();cfg=config();root=Path(cfg['output']);out=root/'external_2b';torch.set_num_threads(4)
 if (out/'results.json').exists():raise FileExistsError(out)
 features=[];rows=[]
 for i in range(4):
  assert json.loads((out/f'{i}.done.json').read_text())['frozen_checkpoints']==hashes
  features.append(np.load(out/f'features_{i}.npy'));rows += [json.loads(s) for s in (out/f'rows_{i}.jsonl').read_text().splitlines()]
 x=torch.from_numpy(np.concatenate(features)).cuda();results={}
 for arm in cfg['arms']:
  sd=torch.load(root/arm/'best.pt',weights_only=True,map_location='cuda');model=PrototypeClassifier(sd['prototypes'],arm).cuda();model.load_state_dict(sd);logits=predict(model,x);np.save(out/f'{arm}_logits.npy',logits.numpy());results[arm]=metrics(rows,logits)
 write_json(out/'results.json',{'results':results,'frozen_checkpoints':hashes,'note':'Source-selected model only. 2b has prior evaluation history and is included in REVE pretraining; not strict dataset-unseen pretraining.'});print(json.dumps({k:v['accuracy'] for k,v in results.items()}))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--shard',type=int);p.add_argument('--evaluate',action='store_true');a=p.parse_args();evaluate() if a.evaluate else extract(a.shard)
