"""Run the first joint cross-subject fold with online EEG encoding on four GPUs."""
import os,sys,json,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(ROOT))
CONFIG=json.loads(Path('configs/benchmark_train.json').read_text())
OUT=ROOT/f"artifacts/benchmark/cross_subject_fold{CONFIG['fold']}";OUT.mkdir(parents=True,exist_ok=True)
PYTHON=str(ROOT/'laya/.venv-eeg/bin/python');LOG_MODE='a' if '--resume-preparation' in sys.argv else 'x'
state=dict(controller_pid=os.getpid(),fold=CONFIG['fold'],classification_datasets=11,pending=['SEED_VIG','ThingsEEG2'],signal_input='online',encoder_warmup_steps=CONFIG['encoder_warmup_steps'])
WATCHED=list(Path('eeg_align').glob('*.py'))+[Path('configs/benchmark_tasks.json'),Path('configs/benchmark_train.json'),Path('scripts/prepare_benchmark.py')]
import hashlib
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in WATCHED}
def guard():
    assert hashes=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in WATCHED},'Active source changed; refusing next stage'
def write(**kwargs):
    state.update(kwargs);p=OUT/'pipeline.tmp';p.write_text(json.dumps(state,indent=2));p.replace(OUT/'pipeline.json')
def run_stage(name,args):
    guard()
    with (OUT/f'{name}.log').open(LOG_MODE) as log:
        child=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3',OMP_NUM_THREADS='3',TOKENIZERS_PARALLELISM='false'))
        write(stage=name,status='running',pid=child.pid);code=child.wait()
    if code:raise RuntimeError(f'{name} exit {code}')
def distributed(name,extra):
    run_stage(name,[PYTHON,'-u','-m','torch.distributed.run','--nproc-per-node=4','--master-port=29684','-m','eeg_align.train_joint','--out',str(OUT/name)]+extra)
try:
    assert Path('artifacts/benchmark/instructions.npy').exists()
    jobs=[]
    groups=[['SleepEDF'],['CHB_MIT','SEED'],['TUAB'],['BNCI2014001','BNCI2014004','BNCI2015001','BNCI2014008','BNCI2014009','EEGMAT','Nakanishi2015']]
    for worker,datasets in enumerate(groups):
        log=(OUT/f'export_worker{worker}.log').open(LOG_MODE)
        child=subprocess.Popen([PYTHON,'-u','scripts/prepare_benchmark.py','--datasets',*datasets],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='4'))
        jobs.append((child,log))
    write(stage='export_raw_arrays',status='running',workers=[p.pid for p,_ in jobs])
    while any(p.poll() is None for p,_ in jobs):
        failed=[p for p,_ in jobs if p.poll() not in (None,0)]
        if failed:raise RuntimeError(f'Export failed: {[p.pid for p in failed]}')
        time.sleep(5)
    assert all(p.returncode==0 for p,_ in jobs)
    for _,log in jobs:log.close()
    guard()
    from eeg_align.signal_data import make_manifest,ROOT as DATA
    if not (DATA/f"fold{CONFIG['fold']}"/'complete.json').exists():make_manifest(CONFIG['fold'])
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
