"""Evaluate all source-selected task-free models on held datasets and shuffles."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from .taskfree_data import ROOT,readrows
from .taskfree_train import JointClassifier,joint_metrics,predict,split_indices

def main():
 out=Path('artifacts/taskfree_v15/evaluation');out.mkdir(exist_ok=False);jobs=json.loads(Path('configs/taskfree_v15_jobs.json').read_text());frozen={}
 for j in jobs:
  c=json.loads(Path(j['config']).read_text());r=Path(c['output']);h=hashlib.sha256((r/'best.pt').read_bytes()).hexdigest();assert h==json.loads((r/'complete.json').read_text())['sha256'];frozen[j['name']]=h
 (out/'frozen.json').write_text(json.dumps(frozen,indent=2));torch.set_num_threads(4);rows=readrows(ROOT/'rows.jsonl');x=torch.from_numpy(np.load(ROOT/'features.npy')).cuda();results={}
 for j in jobs:
  c=json.loads(Path(j['config']).read_text());state=torch.load(Path(c['output'])/'best.pt',weights_only=True,map_location='cuda');m=JointClassifier(state['prototypes'],c['arm']).cuda();m.load_state_dict(state);_,_,ids=split_indices(rows,c['heldout_emotion']);rr=[rows[i] for i in ids];z=predict(m,x[ids]);np.save(out/f"{j['name']}_logits.npy",z.numpy());scores=joint_metrics(rr,z)
  # Shuffle within each held dataset: preserves acquisition/task identity but removes label pairing.
  shuffled=np.arange(len(rr));rng=np.random.default_rng(2026)
  for ds in sorted({r['dataset'] for r in rr}):
   ii=np.array([i for i,r in enumerate(rr) if r['dataset']==ds]);shuffled[ii]=rng.permutation(ii)
  scores['within_dataset_shuffle']=joint_metrics(rr,z[shuffled]);results[j['name']]=scores;print(j['name'],scores['macro_task'],flush=True)
 (out/'results.json').write_text(json.dumps({'results':results,'frozen':frozen,'note':'Dataset held out from this run only; all are previously used development data. No target 2b or SEED-VII used. Task accuracy may still reflect acquisition differences despite common channels.'},indent=2))
if __name__=='__main__':main()
