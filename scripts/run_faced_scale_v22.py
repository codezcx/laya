"""Wait for the known full FACED fetch, then run matched-validation scale controls."""
import json,subprocess,time
from pathlib import Path
ROOT=Path('artifacts/faced_scale_v22_controller')
def main():
 ROOT.mkdir(exist_ok=False);pid=int(Path('artifacts/faced_full_v22_fetch.pid').read_text());source=Path('artifacts/faced_full_v22')
 def status(**kw):(ROOT/'status.json').write_text(json.dumps(kw,indent=2))
 try:
  status(state='waiting_for_acquisition',fetch_pid=pid);deadline=time.monotonic()+12*3600
  while True:
   p=source/'cohort_status.json';s=json.loads(p.read_text()) if p.exists() else {}
   if s.get('state')=='complete':break
   if s.get('state')=='failed':raise RuntimeError('Inspect failed subject records before resuming')
   proc=Path(f'/proc/{pid}/stat')
   if not proc.exists() or proc.read_text().split()[2]=='Z':raise RuntimeError('Known fetch process ended before audited completion')
   if time.monotonic()>deadline:raise TimeoutError('Acquisition exceeded twelve hours; no restart')
   time.sleep(5)
  status(state='preparing_features')
  with (ROOT/'prepare.log').open('w') as f:subprocess.run(['laya/.venv-eeg/bin/python','-u','-m','eeg_tasks.faced_scale_probe','--prepare'],stdout=f,stderr=subprocess.STDOUT,check=True)
  ps=[];jobs=[]
  for mode in ['pilot','full']:
   for arm in ['full_linear','spatial_linear']:
    name=mode+'_'+arm
    with (ROOT/f'{name}.log').open('w') as f:p=subprocess.Popen(['laya/.venv-eeg/bin/python','-u','-m','eeg_tasks.faced_scale_probe','--mode',mode,'--arm',arm],stdout=f,stderr=subprocess.STDOUT)
    ps.append(p);jobs.append({'name':name,'pid':p.pid})
  status(state='fitting',jobs=jobs);codes=[p.wait() for p in ps]
  if any(codes):raise RuntimeError(codes)
  hashes=[]
  for job in jobs:hashes.append(json.loads((Path('artifacts/faced_scale_v22')/job['name']/'complete.json').read_text())['validation_rows_sha256'])
  assert len(set(hashes))==1;status(state='source_controls_complete',jobs=jobs,target_evaluation='not_run')
 except Exception as e:status(state='failed',error=repr(e));raise
if __name__=='__main__':main()
