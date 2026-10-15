"""Parallel CPU probes; freeze all eight source models before transfer scoring."""
import json,subprocess
from pathlib import Path
ROOT=Path('artifacts/emotion_transfer_v16')
def main():
 status=ROOT/'controller.json'
 if status.exists():raise FileExistsError(status)
 jobs=[];ps=[]
 for arm in ['three_linear','full_linear','spatial_linear','spatial_rbf']:
  with (ROOT/f'{arm}.log').open('w') as f:p=subprocess.Popen(['laya/.venv-eeg/bin/python','-u','-m','eeg_tasks.emotion_transfer_probe','--arm',arm],stdout=f,stderr=subprocess.STDOUT)
  jobs.append({'arm':arm,'pid':p.pid});ps.append(p)
 status.write_text(json.dumps({'state':'training','jobs':jobs},indent=2));codes=[p.wait() for p in ps]
 if any(codes):
  status.write_text(json.dumps({'state':'failed','jobs':jobs,'codes':codes},indent=2));raise RuntimeError(codes)
 status.write_text(json.dumps({'state':'evaluating','jobs':jobs},indent=2))
 with (ROOT/'evaluation.log').open('w') as f:r=subprocess.run(['laya/.venv-eeg/bin/python','-u','-m','eeg_tasks.emotion_transfer_probe','--evaluate'],stdout=f,stderr=subprocess.STDOUT)
 status.write_text(json.dumps({'state':'complete' if r.returncode==0 else 'evaluation_failed','jobs':jobs,'evaluation_code':r.returncode},indent=2));r.check_returncode()
if __name__=='__main__':main()
