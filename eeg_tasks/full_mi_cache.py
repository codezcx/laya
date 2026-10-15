"""Full audited MI source development pool; no source test or target loading."""
import argparse,hashlib,json,pickle,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel
from .data import preprocess,normalized_window

@torch.no_grad()
def main(shard):
 torch.set_num_threads(4);cfg=json.loads(Path('configs/multitask_prototypes_v1.json').read_text());base=json.loads(Path(cfg['base_config']).read_text());root=Path('artifacts/mi_full_sources_v12');root.mkdir(exist_ok=True)
 if (root/f'{shard}.npz').exists():raise FileExistsError(root)
 allrows=[]
 for split in ['train','validation']:
  for line in (Path(base['manifest_dir'])/f'{split}.jsonl').read_text().splitlines():
   r=json.loads(line);assert r['dataset'] in ['Cho2017','Dreyer2023','Lee2019_MI'];allrows.append(dict(r,split=split))
 model=AutoModel.from_pretrained(base['reve_checkpoint'],local_files_only=True,trust_remote_code=True).cuda().eval().requires_grad_(False);bank=AutoModel.from_pretrained(base['positions_checkpoint'],local_files_only=True,trust_remote_code=True);pos=bank(['C3','Cz','C4']).cuda();indices=list(range(shard,len(allrows),4));features=[];rows=[];start=time.time()
 for begin in range(0,len(indices),64):
  signals=[]
  for i in indices[begin:begin+64]:
   r=allrows[i]
   with open(r['path'],'rb') as f:d=pickle.load(f)
   assert d['freq']==200 and base['labels'].index(d['label'])==r['label']
   raw=np.stack([d['eeg'][c.upper()] for c in cfg['channels']]);signal=normalized_window(preprocess(raw,200,cfg)[:,:600]);assert signal.shape==(3,600);signals.append(signal)
   rows.append(dict(r,index=i,subject=str(r['subject']),task=1,class_id=r['label']+3,signal_sha256=hashlib.sha256(signal.tobytes()).hexdigest()))
  x=torch.from_numpy(np.stack(signals)).cuda();stride=model.config.patch_size-model.config.patch_overlap;tail=(-(600-model.config.patch_size))%stride;x=torch.nn.functional.pad(x,(0,tail),mode='replicate');z=model(x,pos[None].expand(len(x),-1,-1)).mean(2);assert z.shape[1:]==(3,512) and torch.isfinite(z).all();features.append(z.cpu().numpy().astype(np.float16))
  if begin%1024==0:print(shard,begin,len(indices),flush=True)
 np.savez(root/f'{shard}.npz',indices=np.array(indices),features=np.concatenate(features));(root/f'rows_{shard}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));(root/f'{shard}.done.json').write_text(json.dumps({'n':len(rows),'seconds':time.time()-start,'source_manifest_hashes':{split:hashlib.sha256((Path(base['manifest_dir'])/f'{split}.jsonl').read_bytes()).hexdigest() for split in ['train','validation']},'note':'Full development pool; Lee reclassified as source development. Requires cross-shard duplicate audit before training.'},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--shard',required=True,type=int,choices=range(4));main(p.parse_args().shard)
