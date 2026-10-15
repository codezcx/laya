"""Matched 30-channel REVE features for both emotion sources, before pooling."""
import argparse,hashlib,json,re
from pathlib import Path
import numpy as np
import torch
from scipy.io import loadmat
from transformers import AutoModel
from .data import preprocess,normalized_window
from .emotion_transfer_probe import CHANNELS,picks,readrows
ROOT=Path('artifacts/emotion_fusion_v17')
@torch.no_grad()
def main(shard,prefix=False):
 global ROOT
 if prefix:ROOT=Path('artifacts/emotion_alignment_v21')
 torch.set_num_threads(4);ROOT.mkdir(exist_ok=True);out=ROOT/'features';out.mkdir(exist_ok=True)
 if (out/f'{shard}.done.json').exists() or (out/f'{shard}.npz').exists():raise FileExistsError(out)
 manifest=Path('artifacts/emotion_transfer_v16/rows.jsonl');rows=readrows(manifest);cfg=json.loads(Path('configs/multitask_prototypes_v1.json').read_text());base=json.loads(Path(cfg['base_config']).read_text());model,info=AutoModel.from_pretrained(base['reve_checkpoint'],local_files_only=True,trust_remote_code=True,output_loading_info=True);assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']);model=model.cuda().eval().requires_grad_(False);bank=AutoModel.from_pretrained(base['positions_checkpoint'],local_files_only=True,trust_remote_code=True);pos=bank(CHANNELS).cuda();assert pos.shape==(30,3);names=[s.split()[-1] for s in Path(cfg['seed_channel_file']).read_text().splitlines() if s.strip()];seedpicks=picks(names);indices=[];features=[];capture={};replay_max=0.0
 if prefix:model.transformer.layers[-2][0].register_forward_pre_hook(lambda module,args:capture.update(x=args[0].detach()))
 groups={}
 for i,r in enumerate(rows):groups.setdefault(r['path'],[]).append(i)
 for path in sorted(groups)[shard::4]:
  ids=groups[path];rr=[rows[i] for i in ids]
  if rr[0]['dataset']=='SEED':
   data=loadmat(path);keys={str(int(re.search(r'eeg(\d+)$',k).group(1))):k for k in data if re.search(r'eeg(\d+)$',k)};filtered={};signals=[]
   for r in rr:
    trial=r['trial']
    if trial not in filtered:filtered[trial]=preprocess(data[keys[trial]][seedpicks],200,cfg)
    s=r['start_sample'];signals.append(normalized_window(filtered[trial][:,s:s+600]))
   signals=np.stack(signals);del filtered,data
  else:
   p=Path('artifacts/faced_source_v14/prepared')/f"sub-{int(rr[0]['subject']):03d}";a=json.loads((p/'audit.json').read_text());sr=readrows(p/'rows.jsonl');lookup={r['signal_sha256']:i for i,r in enumerate(sr)};signals=np.load(p/'signals.npy')[[lookup[r['signal_sha256']] for r in rr]][:,picks(a['channels'])]
  assert signals.shape==(len(ids),30,600)
  for start in range(0,len(ids),32):
   x=torch.from_numpy(signals[start:start+32]).cuda();tail=(-(600-model.config.patch_size))%(model.config.patch_size-model.config.patch_overlap);x=torch.nn.functional.pad(x,(0,tail),mode='replicate');z=model(x,pos[None].expand(len(x),-1,-1)).mean(2);assert z.shape[1:]==(30,512) and torch.isfinite(z).all();
   if prefix:
    replay=capture['x']
    for attn,ff in model.transformer.layers[-2:]:replay=attn(replay)+replay;replay=ff(replay)+replay
    delta=float((replay.reshape(len(x),30,4,512).mean(2)-z).abs().max());replay_max=max(replay_max,delta)
    if delta>1e-5:raise ValueError(('Tail replay mismatch',delta))
   features.append((capture['x'] if prefix else z).cpu().numpy().astype(np.float16));indices.extend(ids[start:start+32])
  print(shard,Path(path).name,len(indices),flush=True)
 np.savez(out/f'{shard}.npz',indices=np.array(indices),features=np.concatenate(features));(out/f'{shard}.done.json').write_text(json.dumps({'n':len(indices),'channels':CHANNELS,'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'checkpoint':base['reve_checkpoint'],'shape':([120,512] if prefix else [30,512]),'prefix':prefix,'fp32_replay_max_abs':replay_max},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--shard',type=int,choices=range(4),required=True);p.add_argument('--prefix',action='store_true');a=p.parse_args();main(a.shard,a.prefix)
