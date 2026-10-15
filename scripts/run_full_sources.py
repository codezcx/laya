"""Audit existing cache workers, then run four fixed source-development comparisons."""
import json,os,subprocess,time
from pathlib import Path

def main():
 root=Path('artifacts/full_sources_v12_controller');root.mkdir(exist_ok=False);cache=Path('artifacts/mi_full_sources_v12');jobs=json.loads((cache/'jobs.json').read_text())
 def status(**values):(root/'status.json').write_text(json.dumps(values,indent=2))
 try:
  deadline=time.monotonic()+7200;status(state='waiting_for_cache',cache_jobs=jobs)
  while not all((cache/f'{j["shard"]}.done.json').exists() for j in jobs):
   for j in jobs:
    if (cache/f'{j["shard"]}.done.json').exists():continue
    p=Path(f'/proc/{j["pid"]}/stat')
    if not p.exists() or p.read_text().split()[2]=='Z':raise RuntimeError(f'Cache worker ended without completion: {j}')
   if time.monotonic()>deadline:raise TimeoutError('Cache wait limit exceeded; do not restart automatically')
   time.sleep(5)
  status(state='auditing')
  with (root/'audit.log').open('w') as f:subprocess.run(['laya/.venv-eeg/bin/python','-m','eeg_tasks.audit_full_mi'],stdout=f,stderr=subprocess.STDOUT,check=True)
  assert json.loads((cache/'audit.json').read_text())['status']=='passed'
  processes=[];records=[]
  for gpu,name in enumerate(['capped_two','full_two','full_three','full_three_text']):
   path=f'configs/full_sources_v12_{name}.json';cfg=json.loads(Path(path).read_text());env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
   with (root/f'{name}.log').open('w') as f:p=subprocess.Popen(['laya/.venv-eeg/bin/python','-m','eeg_tasks.channel_train','--config',path,'--arm',cfg['arms'][0]],stdout=f,stderr=subprocess.STDOUT,env=env)
   processes.append(p);records.append({'name':name,'pid':p.pid,'gpu':gpu,'config':path})
  status(state='training',jobs=records)
  while any(p.poll() is None for p in processes):time.sleep(5)
  codes=[p.returncode for p in processes]
  if any(c!=0 for c in codes):raise RuntimeError(f'Training return codes {codes}')
  for r in records:
   cfg=json.loads(Path(r['config']).read_text());assert (Path(cfg['output'])/cfg['arms'][0]/'complete.json').exists()
  status(state='complete',jobs=records,return_codes=codes)
 except Exception as error:status(state='failed',error=repr(error));raise
if __name__=='__main__':main()
