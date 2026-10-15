"""Fetch and prepare a predeclared source-only FACED cohort, four I/O workers."""
import argparse,json,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from eeg_tasks.data import subject_split
from eeg_tasks.faced_download import download
from eeg_tasks.faced_prepare import main as prepare

parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
root=Path('artifacts/faced_source_v14');plan=root/'cohort.json'
if plan.exists() and not args.resume:raise FileExistsError('Pass --resume to preserve and resume the existing cohort')
if args.resume:
 cohort=json.loads(plan.read_text());train=cohort['train'];val=cohort['validation'];splits={int(k):v for k,v in cohort['full_subject_split'].items()}
 assert cohort['source_revision']==json.loads((root/'raw/source_revision.json').read_text())['revision']
else:
 splits=subject_split(range(123),1401);rng=np.random.default_rng(1402)
 train=[0]+rng.choice([s for s,v in splits.items() if v=='train' and s!=0],15,replace=False).tolist();val=rng.choice([s for s,v in splits.items() if v=='validation'],8,replace=False).tolist();assert splits[0]=='train'
 cohort={'train':sorted(train),'validation':sorted(val),'full_subject_split':splits,'split_seed':1401,'cohort_seed':1402,'source_revision':json.loads((root/'raw/source_revision.json').read_text())['revision'],'note':'Fixed before model fitting. Subject0 retained after format QC only; no target or reserved-test subjects included.'};plan.write_text(json.dumps(cohort,indent=2))
progress={'state':'running','completed':[],'failed':[],'started':time.time(),'resumed':args.resume}
def persist():(root/'cohort_status.json').write_text(json.dumps(progress,indent=2))
def work(subject):
 folder=root/'raw'/f'sub-{subject:03d}';out=root/'prepared'/f'sub-{subject:03d}'
 if not (folder/'download.done.json').exists():download(subject)
 if not (out/'audit.json').exists():prepare(subject)
 audit=json.loads((out/'audit.json').read_text());assert audit['split']==splits[subject] and audit['windows']==280
 return {'subject':subject,'split':audit['split'],'sample_rate':audit['sampling_rate']}
persist()
with ThreadPoolExecutor(max_workers=4) as pool:
 futures={pool.submit(work,s):s for s in train+val}
 for future in as_completed(futures):
  try:progress['completed'].append(future.result())
  except Exception as error:progress['failed'].append({'subject':futures[future],'error':repr(error)})
  persist()
progress['state']='failed' if progress['failed'] else 'complete';progress['seconds']=time.time()-progress['started'];persist();print(json.dumps(progress,indent=2))
if progress['failed']:raise RuntimeError('Inspect failed subjects before any retry or training')
