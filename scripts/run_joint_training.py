"""Run cached LEAF-query JEV smoke then production on GPUs0–3."""
import os,subprocess,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT)
OUT=Path('artifacts/leaf_2a_seed');OUT.mkdir(exist_ok=True,parents=True)
assert Path('datasets/leaf_2a_seed/complete.json').exists() and (OUT/'baseline.json').exists()
state=dict(controller_pid=os.getpid(),gpus=[0,1,2,3]);env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='3',TOKENIZERS_PARALLELISM='false')
def write(**kw):
 state.update(kw);p=OUT/'pipeline.tmp';p.write_text(json.dumps(state,indent=2));p.replace(OUT/'pipeline.json')
try:
 for name,smoke in [('jev_smoke',True),('jev',False)]:
  assert not (OUT/name).exists()
  cmd=[str(ROOT/'laya/.venv-eeg/bin/python'),'-u','-m','torch.distributed.run','--nproc-per-node=4','--master-port=29683','-m','eeg_align.train_joint','--out',str(OUT/name)]
  if smoke:cmd+=['--smoke']
  with (OUT/f'{name}.log').open('x') as log:
   p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env);write(status='running',stage=name,pid=p.pid,started=time.time());code=p.wait()
  if code:raise RuntimeError(f'{name}: exit {code}')
  assert (OUT/name/'complete.json').exists()
 write(status='complete',finished=time.time())
except Exception as e:write(status='failed',error=repr(e));raise
