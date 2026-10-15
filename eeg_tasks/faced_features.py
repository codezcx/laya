"""Frozen REVE and canonical spatial features for the fixed FACED source cohort."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel
from .spatial_model import spatial_pool

@torch.no_grad()
def main(shard):
 torch.set_num_threads(4);root=Path('artifacts/faced_source_v14');status=json.loads((root/'cohort_status.json').read_text());assert status['state']=='complete' and not status['failed'];plan=json.loads((root/'cohort.json').read_text());subjects=sorted(plan['train']+plan['validation']);out=root/'features';out.mkdir(exist_ok=True)
 if (out/f'{shard}.done.json').exists():raise FileExistsError(out)
 cfg=json.loads(Path('configs/spatial_prototypes_v6.json').read_text());base=json.loads(Path('configs/reve_qwen_mi_multi.json').read_text());model=AutoModel.from_pretrained(base['reve_checkpoint'],trust_remote_code=True,local_files_only=True).eval().requires_grad_(False).cuda();bank=AutoModel.from_pretrained(base['positions_checkpoint'],trust_remote_code=True,local_files_only=True);anchors=bank(cfg['anchors']).cuda();features=[];rows=[];channels={}
 for subject in subjects[shard::4]:
  p=root/'prepared'/f'sub-{subject:03d}';audit=json.loads((p/'audit.json').read_text());assert audit['split']==('train' if subject in plan['train'] else 'validation');raw_names=audit['channels']
  # These auxiliary horizontal EOG electrodes are not EEG spatial tokens.
  excluded=[c for c in raw_names if c in {'HEOR','HEOL'}]
  names=[c for c in raw_names if c not in excluded]
  unsupported=[c for c in names if c not in bank.mapping]
  if unsupported:raise ValueError(f'sub-{subject:03d}: unsupported EEG channels {unsupported}')
  positions=bank(names).cuda();signals=np.load(p/'signals.npy');rr=[json.loads(s) for s in (p/'rows.jsonl').read_text().splitlines()]
  assert len(rr)==len(signals)==280 and signals.shape[1:]==(len(raw_names),600)
  signals=signals[:,[raw_names.index(c) for c in names]]
  channels[str(subject)]=names
  print('channel_selection',subject,'included',names,'excluded_eog',excluded,flush=True)
  for start in range(0,len(signals),32):
   x=torch.from_numpy(signals[start:start+32]).cuda();stride=model.config.patch_size-model.config.patch_overlap;tail=(-(600-model.config.patch_size))%stride;x=torch.nn.functional.pad(x,(0,tail),mode='replicate');z=model(x,positions[None].expand(len(x),-1,-1)).mean(2);pooled=spatial_pool(z,positions,anchors,cfg['sigma']);assert torch.isfinite(pooled).all();features.append(pooled.cpu().numpy().astype(np.float16))
  rows.extend(rr);print(shard,subject,len(rows),flush=True)
 np.save(out/f'{shard}.npy',np.concatenate(features));(out/f'rows_{shard}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));(out/f'{shard}.done.json').write_text(json.dumps({'n':len(rows),'channels_by_subject':channels,'anchors':cfg['anchors'],'sigma':cfg['sigma'],'cohort_sha256':hashlib.sha256((root/'cohort.json').read_bytes()).hexdigest(),'source_revision':plan['source_revision']},indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True,choices=range(4));main(p.parse_args().shard)
