"""Four workers; each computes emotion then MI prefix states on its GPU."""
import os,subprocess,json
from pathlib import Path
root=Path('artifacts/reve_prefix_v8');root.mkdir(exist_ok=False);jobs=[]
code='import subprocess,sys\ns=int(sys.argv[1])\nsubprocess.run([sys.executable,"-m","eeg_tasks.fullchannel_cache","--shard",str(s),"--prefix"],check=True)\nsubprocess.run([sys.executable,"-m","eeg_tasks.prefix_cache_mi","--shard",str(s)],check=True)'
for shard in range(4):
 env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(shard)
 with (root/f'{shard}.log').open('w') as f:p=subprocess.Popen(['laya/.venv-eeg/bin/python','-u','-c',code,str(shard)],stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 jobs.append({'shard':shard,'pid':p.pid,'gpu':shard})
(root/'jobs.json').write_text(json.dumps(jobs,indent=2));print(jobs)
