"""Source-only 62-channel REVE cache; temporal pooling retains channel tokens."""
import argparse,json,hashlib,re,time
from pathlib import Path
import numpy as np
import torch
from scipy.io import loadmat
from transformers import AutoModel
from .data import preprocess,normalized_window

@torch.no_grad()
def main(shard):
 torch.set_num_threads(4);cfg=json.loads(Path('configs/multitask_prototypes_v1.json').read_text());base=json.loads(Path(cfg['base_config']).read_text())
 root=Path('artifacts/emotion_reve62_v5');root.mkdir(exist_ok=True);out=root/f'{shard}.npz'
 if out.exists() or (root/f'{shard}.done.json').exists():raise FileExistsError(out)
 rows=[json.loads(s) for s in Path('artifacts/emotion_source_probe_v4/rows.jsonl').read_text().splitlines()]
 assert all(r['dataset']=='SEED' and r['split'] in ('train','validation') for r in rows)
 files=sorted({r['path'] for r in rows})[shard::4];indices=[];features=[]
 names=[s.split()[-1] for s in Path(cfg['seed_channel_file']).read_text().splitlines() if s.strip()];assert len(names)==62
 model,info=AutoModel.from_pretrained(base['reve_checkpoint'],local_files_only=True,trust_remote_code=True,output_loading_info=True)
 assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
 model=model.eval().requires_grad_(False).cuda();bank=AutoModel.from_pretrained(base['positions_checkpoint'],local_files_only=True,trust_remote_code=True);pos=bank(names).detach().cuda();assert pos.shape==(62,3)
 start=time.time()
 for path in files:
  data=loadmat(path);keys={str(int(re.search(r'eeg(\d+)$',k).group(1))):k for k in data if re.search(r'eeg(\d+)$',k)};signals=[];ids=[]
  for trial,key in sorted(keys.items(),key=lambda item:int(item[0])):
   trialids=[i for i,r in enumerate(rows) if r['path']==path and r['trial']==trial]
   if not trialids:continue
   raw=data[key];assert raw.shape[0]==62;x=preprocess(raw,200,cfg)
   for i in trialids:
    s=rows[i]['start_sample'];signals.append(normalized_window(x[:,s:s+600]));ids.append(i)
  del data
  for begin in range(0,len(ids),24):
   x=torch.from_numpy(np.stack(signals[begin:begin+24])).cuda();stride=model.config.patch_size-model.config.patch_overlap;tail=(-(600-model.config.patch_size))%stride;x=torch.nn.functional.pad(x,(0,tail),mode='replicate');z=model(x,pos[None].expand(len(x),-1,-1)).mean(2)
   assert z.shape[1:]==(62,512) and torch.isfinite(z).all();features.append(z.cpu().numpy().astype(np.float16));indices.extend(ids[begin:begin+24])
  print(shard,Path(path).name,'windows',len(indices),flush=True)
 np.savez(out,indices=np.array(indices),features=np.concatenate(features))
 (root/f'{shard}.done.json').write_text(json.dumps({'n':len(indices),'seconds':time.time()-start,'channels':names,'rows_sha256':hashlib.sha256(Path('artifacts/emotion_source_probe_v4/rows.jsonl').read_bytes()).hexdigest(),'checkpoint':base['reve_checkpoint'],'shape_per_sample':[62,512],'note':'Frozen REVE fp32 compute; fp16 cached outputs; time mean only. No source test or external data.'},indent=2)+'\n')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--shard',required=True,type=int,choices=range(4));main(p.parse_args().shard)
