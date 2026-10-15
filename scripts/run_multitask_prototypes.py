"""Bounded experiment controller with explicit PID/log files and failure propagation."""
import json,os,subprocess,time
from pathlib import Path
ROOT=Path('artifacts/multitask_prototypes_v1');PY='laya/.venv-eeg/bin/python'
def state(phase,jobs=None):
 (ROOT/'status.json').write_text(json.dumps({'phase':phase,'time':time.time(),'jobs':jobs or []},indent=2))
def launch(args,gpu,name):
 log=open(ROOT/f'{name}.log','w');p=subprocess.Popen([PY,'-u','-m',*args],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4'},start_new_session=True);log.close();return p

def wait_all(jobs,phase):
 state(phase,[{'pid':p.pid,'name':n} for n,p in jobs])
 while True:
  for n,p in jobs:
   code=p.poll()
   if code not in (None,0):raise RuntimeError(f'{n} failed with {code}; inspect {n}.log')
  if all(p.poll() is not None for _,p in jobs):break
  time.sleep(5)
try:
 if not (ROOT/'data.done.json').exists():raise RuntimeError('Prepare data first')
 jobs=[(f'features_{i}',launch(['eeg_tasks.cache','--shard',str(i)],i,f'features_{i}')) for i in range(4)]
 wait_all(jobs,'extracting_features')
 wait_all([('text',launch(['eeg_tasks.cache','--text'],0,'text'))],'encoding_text')
 arms=json.loads(Path('configs/multitask_prototypes_v1.json').read_text())['arms']
 wait_all([(arm,launch(['eeg_tasks.train','--arm',arm],i,arm)) for i,arm in enumerate(arms)],'training')
 state('complete')
except Exception as e:
 state('failed:'+str(e));raise
