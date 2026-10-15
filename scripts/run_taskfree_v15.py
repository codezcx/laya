"""Run the four fixed task-free arms, then evaluate every frozen checkpoint."""
import json,os,subprocess
from pathlib import Path

def main():
 r=Path('artifacts/taskfree_v15');r.mkdir(exist_ok=False);jobs=json.loads(Path('configs/taskfree_v15_jobs.json').read_text());ps=[]
 for gpu,j in enumerate(jobs):
  with (r/(j['name']+'.log')).open('w') as f:p=subprocess.Popen(['laya/.venv-eeg/bin/python','-u','-m','eeg_tasks.taskfree_train','--config',j['config']],stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)))
  ps.append(p);j.update(pid=p.pid,gpu=gpu)
 (r/'jobs.json').write_text(json.dumps(jobs,indent=2));(r/'status.json').write_text(json.dumps({'state':'training'}));print(jobs,flush=True)
 codes=[p.wait() for p in ps]
 if any(codes):
  (r/'status.json').write_text(json.dumps({'state':'failed','returncodes':codes}));raise RuntimeError(codes)
 (r/'status.json').write_text(json.dumps({'state':'evaluating'}))
 result=subprocess.run(['laya/.venv-eeg/bin/python','-u','-m','eeg_tasks.taskfree_eval'])
 (r/'status.json').write_text(json.dumps({'state':'complete' if result.returncode==0 else 'evaluation_failed','returncodes':codes,'evaluation_returncode':result.returncode}))
 result.check_returncode()
if __name__=='__main__':main()
