"""Wait for known FACED acquisition, then extract and fit four source-only arms."""
import json,os,subprocess,time
from pathlib import Path

def main():
 root=Path('artifacts/multi_emotion_v14_controller');root.mkdir(exist_ok=False);source=Path('artifacts/faced_source_v14');fetch_pid=int((source/'cohort_fetch.pid').read_text())
 def status(**kwargs):(root/'status.json').write_text(json.dumps(kwargs,indent=2))
 def run_stage(stage,commands):
  processes=[];jobs=[]
  for gpu,(name,args) in enumerate(commands):
   env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
   with (root/f'{stage}_{name}.log').open('w') as f:p=subprocess.Popen(['laya/.venv-eeg/bin/python']+args,stdout=f,stderr=subprocess.STDOUT,env=env)
   processes.append(p);jobs.append({'name':name,'pid':p.pid,'gpu':gpu})
  status(state=stage,jobs=jobs)
  while any(p.poll() is None for p in processes):time.sleep(5)
  codes=[p.returncode for p in processes]
  if any(c!=0 for c in codes):raise RuntimeError(f'{stage} failed: {codes}')
  return jobs
 try:
  status(state='waiting_for_acquisition',fetch_pid=fetch_pid);deadline=time.monotonic()+14400
  while True:
   current=json.loads((source/'cohort_status.json').read_text())
   if current['state']=='complete':break
   if current['state']=='failed':raise RuntimeError('FACED acquisition failed; inspect cohort_status before repair')
   p=Path(f'/proc/{fetch_pid}/stat')
   if not p.exists() or p.read_text().split()[2]=='Z':raise RuntimeError('Known acquisition process ended without completion')
   if time.monotonic()>deadline:raise TimeoutError('Four-hour acquisition wait exceeded; downloader is not restarted')
   time.sleep(5)
  status(state='auditing_prepared_cohort')
  with (root/'prepared_audit.log').open('w') as f:subprocess.run(['laya/.venv-eeg/bin/python','scripts/audit_faced_prepared.py'],stdout=f,stderr=subprocess.STDOUT,check=True)
  assert json.loads((source/'prepared_integrity.json').read_text())['status']=='complete'
  run_stage('features',[(str(i),['-m','eeg_tasks.faced_features','--shard',str(i)]) for i in range(4)])
  commands=[]
  for name in ['seed_only','seed_faced','faced_only','seed_faced_text']:
   path=f'configs/multi_emotion_v14_{name}.json';cfg=json.loads(Path(path).read_text());commands.append((name,['-m','eeg_tasks.channel_train','--config',path,'--arm',cfg['arms'][0]]))
  jobs=run_stage('training',commands)
  for name,_ in commands:
   cfg=json.loads(Path(f'configs/multi_emotion_v14_{name}.json').read_text());assert (Path(cfg['output'])/cfg['arms'][0]/'complete.json').exists()
  status(state='complete',jobs=jobs)
 except Exception as e:status(state='failed',error=repr(e));raise
if __name__=='__main__':main()
