"""Four GPUs: matched montage extraction, four heads, frozen transfer evaluation."""
import json,os,subprocess
from pathlib import Path
ROOT=Path('artifacts/emotion_fusion_v17')
def main():
 ROOT.mkdir(exist_ok=False)
 def stage(name,commands):
  jobs=[];ps=[]
  for gpu,(label,args) in enumerate(commands):
   with (ROOT/f'{name}_{label}.log').open('w') as f:p=subprocess.Popen(['laya/.venv-eeg/bin/python','-u']+args,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)))
   jobs.append({'name':label,'pid':p.pid,'gpu':gpu});ps.append(p)
  (ROOT/'controller.json').write_text(json.dumps({'state':name,'jobs':jobs},indent=2));codes=[p.wait() for p in ps]
  if any(codes):raise RuntimeError((name,codes))
 try:
  stage('features',[(str(i),['-m','eeg_tasks.emotion_fusion_cache','--shard',str(i)]) for i in range(4)])
  stage('training',[(arm,['-m','eeg_tasks.emotion_fusion_train','--arm',arm]) for arm in ['spectral','reve','fusion','fusion_text']])
  stage('evaluation',[('all',['-m','eeg_tasks.emotion_fusion_train','--evaluate'])]);(ROOT/'controller.json').write_text(json.dumps({'state':'complete'}))
 except Exception as e:
  (ROOT/'failure.json').write_text(json.dumps({'error':repr(e)}));raise
if __name__=='__main__':main()
