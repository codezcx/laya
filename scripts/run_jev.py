"""Four-GPU full-montage MI caching, audit, smoke test, and training."""
import os
from pathlib import Path
import subprocess
import sys
import time
from eeg_align.mi_training_data import config,write_json

def main():
    c=config();out=Path(c['output']);out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',CUDA_VISIBLE_DEVICES='0,1,2,3')
    def status(phase,**kw):write_json(out/'status.json',dict(phase=phase,time=time.time(),**kw))
    def run(name,cmd):
        with (out/f'{name}.log').open('x') as log:
            p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env);status(name,pid=p.pid,command=cmd);code=p.wait()
        if code:status('failed',stage=name,returncode=code);raise RuntimeError(f'{name} failed')
    processes=[];logs=[]
    for shard in range(4):
        log=(out/f'encode_{shard}.log').open('x');logs.append(log)
        processes.append(subprocess.Popen([sys.executable,'-m','eeg_align.mi_training_data','--shard',str(shard)],stdout=log,stderr=subprocess.STDOUT,env=dict(env,CUDA_VISIBLE_DEVICES=str(shard))))
    status('encoding',pids=[p.pid for p in processes]);codes=[p.wait() for p in processes]
    for log in logs:log.close()
    if any(codes):status('failed',stage='encoding',codes=codes);raise RuntimeError(codes)
    run('audit',[sys.executable,'-m','eeg_align.mi_training_data','--finalize'])
    for name,args in [('smoke',['--smoke']),('run',[])]:
        run(name,[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=4','-m','eeg_align.train_jev','--out',str(out/name),*args])
        assert (out/name/'complete.json').exists()
    status('complete')

if __name__=='__main__':main()
