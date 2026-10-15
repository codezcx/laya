"""Four CPU workers for frequency/context controls; retain failures and frozen selection."""
import json,subprocess
from pathlib import Path
ROOT=Path('artifacts/emotion_context_v20')
def main():
 ROOT.mkdir(exist_ok=False)
 def stage(name,commands):
  jobs=[];ps=[]
  for label,args in commands:
   with (ROOT/f'{name}_{label}.log').open('w') as f:p=subprocess.Popen(['laya/.venv-eeg/bin/python','-u','-m','eeg_tasks.emotion_context']+args,stdout=f,stderr=subprocess.STDOUT)
   jobs.append({'name':label,'pid':p.pid});ps.append(p)
  (ROOT/'status.json').write_text(json.dumps({'state':name,'jobs':jobs},indent=2));codes=[p.wait() for p in ps]
  if any(codes):raise RuntimeError((name,codes))
 try:
  stage('extracting',[(str(i),['--shard',str(i)]) for i in range(4)]);stage('joining',[('all',['--join'])]);stage('fitting',[(mode+'_'+arm,['--mode',mode,'--arm',arm]) for mode in ['hz30_s3','hz45_s3','hz30_s9','hz45_s9'] for arm in ['full_linear']]);stage('evaluation',[('all',['--evaluate'])]);(ROOT/'status.json').write_text(json.dumps({'state':'complete'}))
 except Exception as e:
  (ROOT/'failure.json').write_text(json.dumps({'error':repr(e)}));raise
if __name__=='__main__':main()
