import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel
from eeg_align.cache_multitask_v30 import signals_for_file,ROOT,digest
from eeg_align.train_multitask_v30 import TokenCache

torch.set_num_threads(4)
c=TokenCache();root=ROOT.parent
smoke=json.loads((root/'real_smoke/data.json').read_text())
cfg=json.loads(Path('configs/multitask_prototypes_v1.json').read_text());base=json.loads(Path(cfg['base_config']).read_text())
model=AutoModel.from_pretrained(base['reve_checkpoint'],trust_remote_code=True,local_files_only=True).cuda().eval().requires_grad_(False)
bank=AutoModel.from_pretrained(base['positions_checkpoint'],trust_remote_code=True,local_files_only=True)
results=[]
for path in sorted({x['row']['path'] for x in smoke['samples']}):
 ids=[i for i,r in enumerate(c.rows) if r['path']==path];rr=[c.rows[i] for i in ids]
 signals,channels=signals_for_file(rr,cfg);pos=bank(channels).detach().cuda()[None]
 targets=[x for x in smoke['samples'] if x['row']['path']==path]
 for item in targets:
  matches=[j for j,r in enumerate(rr) if r.get('trial')==item['row'].get('trial') and r.get('start_sample')==item['row'].get('start_sample')];assert len(matches)==1;j=matches[0]
  import hashlib
  same_signal=hashlib.sha256(signals[j].tobytes()).hexdigest()==item['signal_sha256'];assert same_signal
  start=j//32*32;x=torch.from_numpy(signals[start:start+32]).cuda();tail=(-(600-model.config.patch_size))%(model.config.patch_size-model.config.patch_overlap);x=torch.nn.functional.pad(x,(0,tail),mode='replicate')
  with torch.no_grad(): z=model(x,pos.expand(len(x),-1,-1)).flatten(1,2)[j-start].cpu().numpy().astype(np.float16)
  s,k=c.location[ids[j]];cached=np.array(c.arrays[s][k,:len(z)])
  delta=float(np.abs(z.astype(np.float32)-cached.astype(np.float32)).max());assert delta==0,(ids[j],delta)
  results.append(dict(index=ids[j],dataset=rr[j]['dataset'],input_signal_hash_matches=True,matched_batch_cache_max_abs=delta))
output=dict(n=len(results),samples=results,conclusion='All selected input hashes match independent smoke; identical-batch fp16 replay is exact. Singleton comparison is not a like-for-like precision check.')
(ROOT/'matched_batch_replay.json').write_text(json.dumps(output,indent=2));print(output)
