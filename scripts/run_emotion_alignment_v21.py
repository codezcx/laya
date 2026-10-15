"""Four-GPU prefix extraction and controlled last-two-block adaptation."""
import json,os,subprocess
from pathlib import Path
ROOT=Path('artifacts/emotion_alignment_v21')
def main():
 ROOT.mkdir(exist_ok=False)
 def stage(name,commands):
  ps=[];jobs=[]
  for gpu,(label,module,args) in enumerate(commands):
   with (ROOT/f'{name}_{label}.log').open('w') as f:p=subprocess.Popen(['laya/.venv-eeg/bin/python','-u','-m',module]+args,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)))
   ps.append(p);jobs.append({'name':label,'pid':p.pid,'gpu':gpu})
  (ROOT/'status.json').write_text(json.dumps({'state':name,'jobs':jobs},indent=2));codes=[p.wait() for p in ps]
  if any(codes):raise RuntimeError((name,codes))
 try:
  stage('prefix',[(str(i),'eeg_tasks.emotion_fusion_cache',['--shard',str(i),'--prefix']) for i in range(4)])
  stage('training',[(arm,'eeg_tasks.emotion_alignment',['--arm',arm]) for arm in ['frozen_ce','tail_ce','tail_align','tail_align_spectral']]);(ROOT/'status.json').write_text(json.dumps({'state':'source_training_complete','external_evaluation':'pending'}))
 except Exception as e:
  (ROOT/'failure.json').write_text(json.dumps({'error':repr(e)}));raise
if __name__=='__main__':main()
