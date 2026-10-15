"""Expand the fixed FACED split to all source train/validation subjects only."""
import argparse,hashlib,json,os,shutil,subprocess,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from eeg_tasks.faced_download import download
from eeg_tasks.faced_prepare import main as prepare
ROOT=Path('artifacts/faced_full_v22');PILOT=Path('artifacts/faced_source_v14')

def initialize():
 ROOT.mkdir(exist_ok=False);(ROOT/'raw').mkdir();(ROOT/'prepared').mkdir();old=json.loads((PILOT/'cohort.json').read_text());splits={int(k):v for k,v in old['full_subject_split'].items()};train=sorted(s for s,v in splits.items() if v=='train');val=sorted(s for s,v in splits.items() if v=='validation');test=sorted(s for s,v in splits.items() if v=='test');assert len(train)==73 and len(val)==25 and len(test)==25;assert set(old['train'])<=set(train) and set(old['validation'])<=set(val)
 assert json.loads((PILOT/'prepared_integrity.json').read_text())['status']=='complete'
 for name in ['source_revision.json','README.md','dataset_description.json']:shutil.copy2(PILOT/'raw'/name,ROOT/'raw'/name)
 plan={'train':train,'validation':val,'reserved_test':test,'full_subject_split':splits,'split_seed':1401,'source_revision':old['source_revision'],'pilot_cohort_sha256':hashlib.sha256((PILOT/'cohort.json').read_bytes()).hexdigest(),'note':'All source subjects in the original fixed split. No test acquisition; old 24-person pilot preserved through read-only reuse.'};(ROOT/'cohort.json').write_text(json.dumps(plan,indent=2));reuse=[]
 for subject in sorted(old['train']+old['validation']):
  name=f'sub-{subject:03d}'
  for folder in ['raw','prepared']:(ROOT/folder/name).symlink_to((PILOT/folder/name).resolve(),target_is_directory=True)
  reuse.append({'subject':subject,'audit_sha256':hashlib.sha256((PILOT/'prepared'/name/'audit.json').read_bytes()).hexdigest()})
 (ROOT/'reused_pilot.json').write_text(json.dumps(reuse,indent=2))

def main(resume):
 if not resume:initialize()
 plan=json.loads((ROOT/'cohort.json').read_text());assert plan['source_revision']==json.loads((ROOT/'raw/source_revision.json').read_text())['revision'];subjects=plan['train']+plan['validation'];assert not set(subjects)&set(plan['reserved_test']);progress={'state':'running','completed':[],'failed':[],'started':time.time(),'resumed':resume,'expected_subjects':len(subjects)}
 def persist():
  temp=ROOT/'cohort_status.tmp';temp.write_text(json.dumps(progress,indent=2));temp.replace(ROOT/'cohort_status.json')
 def work(subject):
  raw=ROOT/'raw'/f'sub-{subject:03d}';prepared=ROOT/'prepared'/f'sub-{subject:03d}'
  if not (raw/'download.done.json').exists():download(subject,ROOT/'raw')
  if not (prepared/'audit.json').exists():
   if prepared.exists():raise FileExistsError(f'Incomplete prepared directory requires explicit inspection: {prepared}')
   prepare(subject,ROOT,quarantine_flat=True)
  a=json.loads((prepared/'audit.json').read_text());assert a['split']==('train' if subject in plan['train'] else 'validation') and a['windows']+len(a.get('quarantined_windows',[]))==280;assert a['source_revision']==plan['source_revision']
  return {'subject':subject,'split':a['split'],'sample_rate':a['sampling_rate']}
 persist()
 with ThreadPoolExecutor(max_workers=4) as pool:
  futures={pool.submit(work,s):s for s in subjects}
  for future in as_completed(futures):
   try:progress['completed'].append(future.result())
   except Exception as e:progress['failed'].append({'subject':futures[future],'error':repr(e)})
   persist()
 progress['state']='failed' if progress['failed'] else 'auditing';persist()
 if progress['failed']:raise RuntimeError('Inspect failed records; no blind retry or replacement subjects')
 with (ROOT/'prepared_audit.log').open('w') as f:subprocess.run(['laya/.venv-eeg/bin/python','scripts/audit_faced_prepared.py','--root',str(ROOT)],stdout=f,stderr=subprocess.STDOUT,check=True)
 audit=json.loads((ROOT/'prepared_integrity.json').read_text());assert audit['status']=='complete' and audit['completed_subjects']==98 and audit['windows_audited']+audit.get('windows_quarantined',0)==27440;progress['state']='complete';progress['seconds']=time.time()-progress['started'];persist()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');main(p.parse_args().resume)
