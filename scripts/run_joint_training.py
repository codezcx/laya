"""Run the MI/emotion pipeline exclusively on physical GPUs 2 and 3."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
PYTHON=str(ROOT/'laya/.venv-eeg/bin/python')
OUT=ROOT/'artifacts/mi_emotion'
OUT.mkdir(exist_ok=True,parents=True)
status={}
def write(**values):
    status.update(values);temp=OUT/'pipeline.tmp';temp.write_text(json.dumps(status,indent=2));temp.replace(OUT/'pipeline.json')
def run(stage,name,checkpoint=None,smoke=False):
    dest=OUT/name
    if dest.exists():raise FileExistsError(dest)
    cmd=[PYTHON,'-u','-m','torch.distributed.run','--nproc-per-node=2','--master-port=29683','-m','eeg_align.train_joint','--stage',stage,'--out',str(dest)]
    if checkpoint:cmd+=['--encoder-checkpoint',str(checkpoint)]
    if smoke:cmd+=['--smoke']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='2,3',OMP_NUM_THREADS='3',TOKENIZERS_PARALLELISM='false',PYTHONPATH=str(ROOT))
    with (OUT/f'{name}.log').open('x') as log:
        p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
        write(stage=name,pid=p.pid,gpus=[2,3],started=time.time(),status='running')
        code=p.wait()
    if code:raise RuntimeError(f'{name} exited with code {code}')
    assert (dest/'complete.json').exists()
try:
    write(status='waiting_for_data',controller_pid=os.getpid(),gpus=[2,3])
    deadline=time.monotonic()+3600
    while not (ROOT/'datasets/mi_emotion/complete.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Dataset preparation did not finish')
        time.sleep(5)
    for name,stage,checkpoint in [('contrast_smoke','contrast',None),('jev_smoke','jev',OUT/'contrast_smoke/best.pt')]:
        if (OUT/name/'complete.json').exists():
            assert json.loads((OUT/name/'complete.json').read_text())['smoke']
            cfg=json.loads((OUT/name/'config.json').read_text())
            assert cfg['stage']==stage and cfg['smoke']
        else:run(stage,name,checkpoint,smoke=True)
    run('contrast','contrast')
    run('jev','jev',OUT/'contrast/best.pt')
    write(status='complete',finished=time.time())
except Exception as exc:
    write(status='failed',error=repr(exc),finished=time.time());raise
