"""Audit source-only feature cache and persist deterministic deduplication indices."""
import json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

def main():
 root=Path('artifacts/mi_full_sources_v12');rows=[];hashes=[]
 for i in range(4):
  done=json.loads((root/f'{i}.done.json').read_text());hashes.append(done['source_manifest_hashes']);rr=[json.loads(s) for s in (root/f'rows_{i}.jsonl').read_text().splitlines()];d=np.load(root/f'{i}.npz');assert len(rr)==done['n']==len(d['features']);assert d['features'].shape[1:]==(3,512);assert np.isfinite(d['features']).all();assert [r['index'] for r in rr]==d['indices'].tolist();rows+=rr
 assert all(h==hashes[0] for h in hashes);rows.sort(key=lambda r:r['index']);assert [r['index'] for r in rows]==list(range(37432))
 sets={split:{(r['dataset'],r['subject']) for r in rows if r['split']==split} for split in ['train','validation']};assert not sets['train']&sets['validation'];assert all(r['dataset'] in ['Cho2017','Dreyer2023','Lee2019_MI'] for r in rows);assert not any(r['dataset']=='Dreyer2023' and r['subject'] in ['13','14'] for r in rows)
 groups=defaultdict(list)
 for r in rows:groups[r['signal_sha256']].append(r)
 keep=[];duplicates=[];conflicts=[]
 for digest,group in groups.items():
  if len({r['split'] for r in group})>1 or len({r['class_id'] for r in group})>1:conflicts.append(group);continue
  keep.append(group[0]['index']);duplicates.extend(r['index'] for r in group[1:])
 result={'status':'failed' if conflicts else 'passed','original_n':len(rows),'kept_n':len(keep),'same_split_same_label_duplicates':duplicates,'cross_split_or_label_conflicts':conflicts,'source_manifest_hashes':hashes[0],'counts':dict(Counter(f"{r['dataset']}/{r['split']}" for r in rows))}
 (root/'audit.json').write_text(json.dumps(result,indent=2));assert not conflicts,'Conflicts found; no training authorized by this audit'
 np.save(root/'keep_indices.npy',np.array(sorted(keep)));(root/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));print({k:v for k,v in result.items() if k not in ['cross_split_or_label_conflicts','same_split_same_label_duplicates']})
if __name__=='__main__':main()
