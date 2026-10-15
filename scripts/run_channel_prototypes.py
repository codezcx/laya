"""Wait for known cache workers, then launch four source-only training arms."""
import json,os,subprocess,time
from pathlib import Path

def main():
 cache=Path('artifacts/emotion_reve60_v5');jobs=json.loads((cache/'jobs.json').read_text());cfg=json.loads(Path('configs/channel_prototypes_v5.json').read_text());root=Path(cfg['output']);root.mkdir(exist_ok=False)
 def status(value):(root/'controller.json').write_text(json.dumps(value,indent=2))
 try:
  status({'state':'waiting_for_cache','cache_jobs':jobs})
  deadline=time.monotonic()+3600
  while not all((cache/f'{j["shard"]}.done.json').exists() for j in jobs):
   for j in jobs:
    if (cache/f'{j["shard"]}.done.json').exists():continue
    proc=Path(f'/proc/{j["pid"]}/stat')
    if not proc.exists() or proc.read_text().split()[2]=='Z':raise RuntimeError(f'Cache worker terminated without completion: {j}')
   if time.monotonic()>deadline:raise TimeoutError('Cache wait exceeded one hour; no automatic restart')
   time.sleep(5)
  processes=[];records=[]
  for gpu,arm in enumerate(cfg['arms']):
   env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
   with (root/f'{arm}.log').open('w') as f:p=subprocess.Popen(['laya/.venv-eeg/bin/python','-m','eeg_tasks.channel_train','--arm',arm],stdout=f,stderr=subprocess.STDOUT,env=env)
   processes.append(p);records.append({'arm':arm,'gpu':gpu,'pid':p.pid})
  status({'state':'training','jobs':records})
  while any(p.poll() is None for p in processes):
   if any(p.poll() not in (None,0) for p in processes):
    status({'state':'one_or_more_failed','jobs':records,'return_codes':[p.poll() for p in processes]})
   time.sleep(5)
  codes=[p.returncode for p in processes]
  if any(c!=0 for c in codes):raise RuntimeError(f'Training return codes {codes}')
  assert all((root/arm/'complete.json').exists() for arm in cfg['arms'])
  status({'state':'complete','jobs':records,'return_codes':codes})
 except Exception as error:
  status({'state':'failed','error':repr(error)});raise

if __name__=='__main__':main()
