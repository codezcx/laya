"""Prepare 11 classification datasets, smoke-test, jointly SFT, then eval best.

SEED-VIG and ThingsEEG2 remain explicitly pending in the task catalog. This
controller must not label the 11-task classification run as all 13 tasks.
"""
import os,sys,json,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT)
OUT=ROOT/'artifacts/benchmark';OUT.mkdir(parents=True,exist_ok=True)
PYTHON=str(ROOT/'laya/.venv-eeg/bin/python')
LOG_MODE='a' if '--resume-preparation' in sys.argv else 'x'
state=dict(controller_pid=os.getpid(),classification_datasets=11,pending=['SEED_VIG','ThingsEEG2'])
def write(**kwargs):
    state.update(kwargs);p=OUT/'pipeline.tmp';p.write_text(json.dumps(state,indent=2));p.replace(OUT/'pipeline.json')
def run_stage(name,args):
    with (OUT/f'{name}.log').open(LOG_MODE) as log:
        child=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='3',TOKENIZERS_PARALLELISM='false'))
        write(stage=name,status='running',pid=child.pid);code=child.wait()
    if code:raise RuntimeError(f'{name} exit {code}')
def distributed(name,extra):
    run_stage(name,[PYTHON,'-u','-m','torch.distributed.run','--nproc-per-node=4','--master-port=29684','-m','eeg_align.train_joint','--benchmark','--out',str(OUT/name)]+extra)
try:
    assert (OUT/'instructions.npy').exists()
    jobs=[]
    groups=[['SleepEDF'],['CHB_MIT'],['TUAB'],['BNCI2014001','BNCI2014004','BNCI2015001','BNCI2014008','BNCI2014009','SEED','EEGMAT','Nakanishi2015']]
    for gpu,datasets in enumerate(groups):
        log=(OUT/f'prepare_gpu{gpu}.log').open(LOG_MODE)
        child=subprocess.Popen([PYTHON,'-u','scripts/prepare_benchmark.py','--gpu',str(gpu),'--datasets',*datasets],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='4'))
        jobs.append((child,log))
    write(stage='prepare',status='running',workers=[p.pid for p,_ in jobs])
    # Check every child, not just the slowest worker, so errors are surfaced early.
    while any(p.poll() is None for p,_ in jobs):
        failed=[p for p,_ in jobs if p.poll() not in (None,0)]
        if failed:raise RuntimeError(f'Preparation workers failed: {[p.pid for p in failed]}')
        time.sleep(5)
    assert all(p.returncode==0 for p,_ in jobs)
    for _,log in jobs:log.close()
    # No source changes between preparing features and the training launch.
    distributed('jev_smoke',['--smoke'])
    distributed('jev',[])
    for split in ['validation','test']:
        distributed(f'final_{split}',['--evaluate-checkpoint',str(OUT/'jev/best.pt'),'--split',split])
    write(stage='finished',status='complete',finished=time.time())
except Exception as error:
    for child,log in locals().get('jobs',[]):
        if child.poll() is None:child.terminate()
        log.close()
    write(status='failed',error=repr(error));raise
