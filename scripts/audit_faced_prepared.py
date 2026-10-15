"""Audit completed FACED preprocessing without treating partial acquisition as complete."""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('artifacts/faced_source_v14'));args=parser.parse_args()
root=args.root;plan=json.loads((root/'cohort.json').read_text());expected=set(plan['train']+plan['validation']);reports=[];seen={}
for subject in sorted(expected):
 folder=root/'prepared'/f'sub-{subject:03d}';path=folder/'audit.json'
 if not path.exists():continue
 audit=json.loads(path.read_text());rows=[json.loads(s) for s in (folder/'rows.jsonl').read_text().splitlines()];signals=np.load(folder/'signals.npy',mmap_mode='r');assert signals.shape==(280,len(audit['channels']),600) and np.isfinite(signals).all();split='train' if subject in plan['train'] else 'validation';assert audit['split']==split
 for row,signal in zip(rows,signals):
  assert row['subject']==str(subject) and row['split']==split and row['dataset']=='FACED-NEMAR';digest=hashlib.sha256(signal.tobytes()).hexdigest();assert digest==row['signal_sha256'];assert digest not in seen,('Duplicate window',subject,seen.get(digest));seen[digest]=subject
 counts=Counter(r['class_id'] for r in rows);assert counts=={0:120,1:40,2:120};trials=Counter(r['trial'] for r in rows);assert len(trials)==28 and set(trials.values())=={10};values=np.asarray(signals);reports.append({'subject':subject,'split':split,'sample_rate':audit['sampling_rate'],'channels':len(audit['channels']),'windows':len(rows),'class_counts':dict(counts),'std_quantiles':np.quantile(values.std(-1),[0,.5,1]).tolist(),'clipped_fraction':float(np.mean(abs(values)>=15))})
report={'status':'complete' if len(reports)==len(expected) else 'partial','completed_subjects':len(reports),'expected_subjects':len(expected),'windows_audited':len(seen),'subjects':reports,'note':'Partial report never authorizes training; acquisition and complete cohort checks still required.'};(root/'prepared_integrity.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k!='subjects'})
