"""Fixed mean conditional-probability aggregation; never mix recordings/trials."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from .spatial_external import balanced


def aggregate_trials(rows,logits):
 if len(rows)!=len(logits):raise ValueError('Row/logit length mismatch')
 groups=defaultdict(list)
 for i,r in enumerate(rows):
  if not r.get('trial') or not r.get('path'):raise ValueError('Explicit trial and recording path required')
  groups[(r['dataset'],r['subject'],r['path'],r['trial'])].append(i)
 output=[];scores=[]
 for key,ids in groups.items():
  labels={rows[i]['class_id'] for i in ids};tasks={rows[i]['task'] for i in ids}
  if len(labels)!=1 or len(tasks)!=1:raise ValueError('Conflicting annotations within trial')
  starts=[rows[i]['start_sample'] for i in ids]
  if len(set(starts))!=len(starts):raise ValueError('Duplicate windows within trial')
  task=next(iter(tasks));lo,hi={0:(0,3),1:(3,5)}[task]
  p=logits[ids,lo:hi].softmax(-1).mean(0);z=torch.full((7,),-1e4);z[lo:hi]=p.clamp_min(1e-12).log();scores.append(z);output.append(dict(rows[ids[0]],windows=len(ids)))
 return output,torch.stack(scores)


def main():
 root=Path('artifacts/trial_aggregation_v9');root.mkdir(exist_ok=False);source=Path('artifacts/spatial_prototypes_v6');rows=[]
 for shard in range(4):rows.extend(json.loads(s) for s in (source/'external_seed_vii'/f'rows_{shard}.jsonl').read_text().splitlines())
 results={}
 for arm in ['text_sigmoid','learned_ce']:
  for seed in [42,43,44]:
   path=source/f'external_seed_vii_{arm}_logits.npy' if seed==42 else Path('artifacts/spatial_repeats_v7')/f'seed_vii_spatial_v7_{arm}_s{seed}_logits.npy'
   logits=torch.from_numpy(np.load(path));trialrows,triallogits=aggregate_trials(rows,logits);assert len(trialrows)==640 and all(r['windows']==10 for r in trialrows)
   m=balanced(trialrows,triallogits);results[f'{arm}_s{seed}']=m;np.save(root/f'{arm}_s{seed}_trial_logits.npy',triallogits.numpy());print(arm,seed,m['accuracy'],m['subject_macro_balanced_accuracy'],flush=True)
 (root/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in trialrows));(root/'results.json').write_text(json.dumps({'results':results,'rule':'mean task-conditional softmax probabilities of 10 windows per trial, then argmax','note':'Exploratory existing-target audit; trial is not the same prediction unit as earlier window metrics. Requires distributed windows across the trial; not a 3-second online classifier.'},indent=2))
if __name__=='__main__':main()
