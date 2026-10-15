"""Freeze source baseline, extract four target shards, evaluate all frozen arms."""
import json,os,subprocess
from pathlib import Path
ROOT=Path('artifacts/emotion_alignment_v21')
def main():
 status=ROOT/'evaluation_status.json'
 if status.exists():raise FileExistsError(status)
 def run(args,log):
  with (ROOT/log).open('w') as f:subprocess.run(['laya/.venv-eeg/bin/python','-u','-m','eeg_tasks.emotion_alignment_external']+args,stdout=f,stderr=subprocess.STDOUT,check=True)
 try:
  status.write_text(json.dumps({'state':'spectral_control'}));run(['--spectral-control'],'spectral_control.log');ps=[];jobs=[]
  for i in range(4):
   with (ROOT/f'external_{i}.log').open('w') as f:p=subprocess.Popen(['laya/.venv-eeg/bin/python','-u','-m','eeg_tasks.emotion_alignment_external','--shard',str(i)],stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i)))
   ps.append(p);jobs.append({'shard':i,'pid':p.pid,'gpu':i})
  status.write_text(json.dumps({'state':'extracting','jobs':jobs},indent=2));codes=[p.wait() for p in ps]
  if any(codes):raise RuntimeError(codes)
  status.write_text(json.dumps({'state':'evaluating'}));run(['--evaluate'],'external_evaluation.log');status.write_text(json.dumps({'state':'complete'}))
 except Exception as e:
  status.write_text(json.dumps({'state':'failed','error':repr(e)}));raise
if __name__=='__main__':main()
