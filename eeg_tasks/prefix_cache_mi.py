"""Cache frozen-prefix states for MI source train/validation only."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel

@torch.no_grad()
def main(shard):
 torch.set_num_threads(4);root=Path('artifacts/reve_prefix_v8/mi');root.mkdir(parents=True,exist_ok=True)
 if (root/f'{shard}.npz').exists():raise FileExistsError(root)
 source=Path('artifacts/multitask_prototypes_v1');rows=[json.loads(s) for s in (source/'manifest.jsonl').read_text().splitlines()];ids=[i for i,r in enumerate(rows) if r['task']==1 and r['split'] in ('train','validation')][shard::4];raw=np.load(source/'signals.npy',mmap_mode='r');base=json.loads(Path('configs/reve_qwen_mi_multi.json').read_text())
 model=AutoModel.from_pretrained(base['reve_checkpoint'],local_files_only=True,trust_remote_code=True).cuda().eval();bank=AutoModel.from_pretrained(base['positions_checkpoint'],local_files_only=True,trust_remote_code=True);pos=bank(['C3','Cz','C4']).cuda();capture={};model.transformer.layers[-2][0].register_forward_pre_hook(lambda m,args:capture.update(x=args[0].detach()));features=[]
 for start in range(0,len(ids),64):
  x=torch.from_numpy(np.array(raw[ids[start:start+64]])).cuda();stride=model.config.patch_size-model.config.patch_overlap;tail=(-(600-model.config.patch_size))%stride;x=torch.nn.functional.pad(x,(0,tail),mode='replicate');model(x,pos[None].expand(len(x),-1,-1));z=capture['x'];assert z.shape[1:]==(12,512);features.append(z.cpu().numpy().astype(np.float16))
 np.savez(root/f'{shard}.npz',indices=np.array(ids),features=np.concatenate(features));(root/f'{shard}.done.json').write_text(json.dumps({'n':len(ids),'manifest_sha256':hashlib.sha256((source/'manifest.jsonl').read_bytes()).hexdigest(),'prefix_before_last_two':True}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True);main(p.parse_args().shard)
