"""Run the four matched task-mixture controls, with smoke and startup gates."""
import os,sys,json,time,signal,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT);sys.path.insert(0,str(ROOT))
from eeg_align.leaf_runtime import CONFIG,digest
from eeg_align.monitor import update_monitor
OUT=ROOT/'artifacts/benchmark'/CONFIG['run_name']
OUT.mkdir(parents=True,exist_ok=True)
PYTHON=str(ROOT/'laya/.venv-eeg/bin/python')
BRANCHES=CONFIG['branches'];NAMES=[b['name'] for b in BRANCHES]
assert NAMES==['single_frozen','joint_frozen','single_finetuned','joint_finetuned'] and CONFIG['world_size']==1
for name in NAMES:
    assert not (OUT/name).exists() and not (OUT/f'smoke_{name}').exists(),'Existing run output must not be overwritten'
WATCHED=list(Path('eeg_align').glob('*.py'))+[Path('configs/benchmark_tasks.json'),Path('configs/benchmark_train.json'),Path(__file__),ROOT/'scripts/log_tensorboard.py']
hashes={str(p):digest(p) for p in WATCHED}
state=dict(controller_pid=os.getpid(),run_name=CONFIG['run_name'],status='starting',branches=BRANCHES,
           comparison=CONFIG['comparison'],
           world_size_per_branch=1,samples_per_task=CONFIG['samples_per_task'],stop_step=CONFIG['stop_step'],
           gpu_assignment={b['name']:b['gpu'] for b in BRANCHES})
jobs=[];handles=[];stopping=False;tb_job=None

def write(**values):
    state.update(values);p=OUT/'pipeline.tmp';p.write_text(json.dumps(state,indent=2));p.replace(OUT/'pipeline.json')

def guard():
    assert hashes=={str(p):digest(p) for p in WATCHED},'Live sources changed'

def shutdown(signum=None,frame=None):
    global stopping
    stopping=True
    for p in jobs:
        if p.poll() is None:p.terminate()

def launch(branch,gpu,smoke):
    name=f"smoke_{branch['name']}" if smoke else branch['name']
    log=(OUT/f'{name}.log').open('x');handles.append(log)
    args=[PYTHON,'-u','-m','eeg_align.train_joint','--out',str(OUT/name),'--bn-policy',branch['bn_policy'],
          '--branch',branch['name'],'--start-gate',str(OUT/('smoke_initialization_verified.json' if smoke else 'initialization_verified.json'))]
    if smoke:args+=['--smoke']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),RANK='0',LOCAL_RANK='0',WORLD_SIZE='1',
             MASTER_ADDR='127.0.0.1',MASTER_PORT=str(29685+gpu),OMP_NUM_THREADS='3',TOKENIZERS_PARALLELISM='false')
    return subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env)

signal.signal(signal.SIGTERM,shutdown);signal.signal(signal.SIGINT,shutdown)
try:
    assert Path(CONFIG['checkpoint']).is_file()
    assert Path('datasets/benchmark/signals/fold0/complete.json').exists()
    for smoke in [True,False]:
        guard();jobs=[launch(branch,branch['gpu'],smoke) for branch in BRANCHES]
        write(stage='smoke' if smoke else 'training',status='running',workers={name:p.pid for name,p in zip(NAMES,jobs)})
        if not smoke:
            tb_log=(OUT/'tensorboard_export.log').open('x');handles.append(tb_log)
            tb_job=subprocess.Popen([PYTHON,'-u',str(ROOT/'scripts/log_tensorboard.py'),
                '--run',str(OUT),'--out',str(OUT/'tensorboard'),'--branches',*NAMES,'--follow'],
                stdout=tb_log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
            write(tensorboard_pid=tb_job.pid,tensorboard_logdir=str(OUT/'tensorboard'))
        while any(p.poll() is None for p in jobs):
            failed=[p for p in jobs if p.poll() not in [None,0]]
            if failed:raise RuntimeError(f'Branch failure: {[(p.pid,p.returncode) for p in failed]}')
            if stopping:break
            guard()
            names=[f'smoke_{n}' for n in NAMES] if smoke else NAMES
            report=update_monitor(OUT,names)
            gate=OUT/('smoke_initialization_verified.json' if smoke else 'initialization_verified.json')
            if report['initial_state_match'] and report['sampling_match'] and not gate.exists():
                assert all(report['initial_state_match'].values()) and report['sampling_match']
                temp=gate.with_suffix('.tmp');temp.write_text(json.dumps(dict(
                    initial_state_match=report['initial_state_match'],sampling_match=report['sampling_match']),indent=2));temp.replace(gate)
                write(initialization_verified=True)
            time.sleep(10)
        if stopping:
            for p in jobs:p.wait(timeout=180)
            write(status='stopped_by_user');break
        assert all(p.wait()==0 for p in jobs)
        for name in NAMES:
            folder=OUT/(f'smoke_{name}' if smoke else name)
            assert (folder/'complete.json').exists()
        if not smoke:
            report=update_monitor(OUT,NAMES)
            assert all(report['initial_state_match'].values()) and report['sampling_match']
            assert tb_job.wait(timeout=60)==0,'TensorBoard export failed; authoritative JSON metrics are preserved'
            write(stage='comparison_ready',status='complete',finished=time.time())
except BaseException as error:
    shutdown()
    for p in jobs:
        try:p.wait(timeout=180)
        except subprocess.TimeoutExpired:p.kill();p.wait()
    write(status='failed',error=repr(error));raise
finally:
    if tb_job is not None and tb_job.poll() is None:
        tb_job.terminate()
        try:tb_job.wait(timeout=15)
        except subprocess.TimeoutExpired:tb_job.kill();tb_job.wait()
    for log in handles:log.close()
