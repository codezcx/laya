"""Launch two matched two-GPU amortized adaptation controls, with smoke gates."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from eeg_align.multitask_data import write_json
from eeg_align.personalization import CONFIG, ROOT, source_identity


def require_capacity():
    output = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free',
                                      '--format=csv,noheader,nounits'],text=True)
    free = {int(i):int(m) for i,m in (line.split(',') for line in output.strip().splitlines())}
    gpus = [g for group in CONFIG['gpu_groups'] for g in group]
    if len(set(gpus))!=4 or any(free.get(g,0)<CONFIG['minimum_free_gpu_mib'] for g in gpus):
        raise RuntimeError(f'Four distinct GPUs need sufficient free memory; observed {free}')
    if shutil.disk_usage(ROOT).free < CONFIG['minimum_free_disk_gib']*1024**3:
        raise RuntimeError('Insufficient free disk reserve')
    return free


def verify_paired(root,smoke=False):
    results = {}
    for arm in CONFIG['arms']:
        result = json.loads((root/('smoke' if smoke else 'train')/arm/'complete.json').read_text())
        if (result['code_sha256']!=source_identity() or result['arm']!=arm or result['smoke']!=smoke
                or result['updates']!=(CONFIG['smoke_steps'] if smoke else CONFIG['steps'])):
            raise ValueError('Mismatched run identity or budget')
        if result['encoder_updated']!=(arm=='tower_full') or not all(result['modules_updated'].values()):
            raise ValueError('Wrong modules updated')
        if not all(result[k] for k in ('frozen_server_verified','frozen_tower_buffers_verified',
                                       'rank_weights_identical','checkpoint_reload_verified')):
            raise ValueError('Freeze/distributed/checkpoint audit failed')
        if result['test_gradient_steps'] or result['test_used_for_selection']:
            raise ValueError('Target contamination')
        if not smoke and not result['target_forward_only_verified']:
            raise ValueError('Target evaluation must be forward-only')
        results[arm] = result
    reference = results[CONFIG['arms'][0]]
    for result in results.values():
        for key in ('initial_encoder_sha256','initial_adapter_sha256','episode_stream_sha256','baseline'):
            if result[key]!=reference[key]:
                raise ValueError(f'Unmatched arms: {key}')
    return dict(arms=CONFIG['arms'],smoke=smoke,paired_verified=True,
                same_initial_weights_episodes_and_baseline=True,results=results)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--start',action='store_true')
    modes.add_argument('--execute',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    os.chdir(REPO)
    root=ROOT
    if (root/'cancelled.json').exists() or (root/'complete.json').exists():
        raise RuntimeError('Cancelled/completed run will not restart')
    if args.start:
        if args.resume:
            if not root.is_dir():
                raise ValueError('No run to resume')
            with (root/'controller.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        else:
            root.mkdir(parents=True,exist_ok=False)
        stamp=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
        log_path=root/f'controller_{stamp}.log'
        env=os.environ.copy()
        env.update(OMP_NUM_THREADS='3',MKL_NUM_THREADS='3',OPENBLAS_NUM_THREADS='3',
                   TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                   PYTHONUNBUFFERED='1')
        command=[sys.executable,'-u',__file__,'--execute']+(['--resume'] if args.resume else [])
        with log_path.open('x') as log:
            child=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,
                                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(controller_pid=child.pid,root=str(root),log=str(log_path))))
        return
    lock=(root/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=source_identity()
    state=dict(controller_pid=os.getpid(),config=CONFIG,code_sha256=source)
    children=[]
    stopping=False
    def request_stop(signum,frame):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGINT,request_stop)
    def guard():
        if stopping or (root/'cancelled.json').exists() or source_identity()!=source:
            raise RuntimeError('Controller stopped/cancelled or live code changed')
    def report(**values):
        state.update(values,updated_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()))
        write_json(root/'pipeline.json',state)
        print(json.dumps(dict(status=state.get('status'),jobs=state.get('jobs'),error=state.get('error'))),flush=True)
    def launch(arm=None,smoke=False):
        guard()
        env=os.environ.copy()
        if arm is None:
            command=[sys.executable,'-u','-m','eeg_align.run_personalization','prepare']
            label='prepare'
            env['CUDA_VISIBLE_DEVICES']=''
            gpus=[]
        else:
            gpus=CONFIG['gpu_groups'][CONFIG['arms'].index(arm)]
            env['CUDA_VISIBLE_DEVICES']=','.join(map(str,gpus))
            command=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1',
                     '--nproc_per_node=2','--module','eeg_align.run_personalization','train','--arm',arm]
            if smoke:
                command.append('--smoke')
            if args.resume:
                command.append('--resume')
            label=('smoke_' if smoke else 'train_')+arm
        log_path=root/f'{label}.log'
        with log_path.open('a') as log:
            child=subprocess.Popen(command,cwd=REPO,env=env,stdin=subprocess.DEVNULL,
                                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(child)
        return dict(pid=child.pid,gpus=gpus,log=str(log_path),arm=arm,smoke=smoke)
    def wait_children():
        while any(p.poll() is None for p in children):
            guard()
            if any(p.poll() not in (None,0) for p in children):
                raise RuntimeError(f'Worker failed: {[(p.pid,p.returncode) for p in children]}')
            time.sleep(2)
        if any(p.returncode!=0 for p in children):
            raise RuntimeError(f'Worker failed: {[(p.pid,p.returncode) for p in children]}')
        children.clear()
    def run_pair(smoke):
        completed=[]
        for arm in CONFIG['arms']:
            path=root/('smoke' if smoke else 'train')/arm/'complete.json'
            if args.resume and path.exists():
                completed.append(arm)
        report(status='smoke' if smoke else 'formal',already_completed=completed,
               jobs=[launch(arm,smoke) for arm in CONFIG['arms'] if arm not in completed])
        wait_children()
        proof=verify_paired(root,smoke)
        write_json(root/('smoke_verified.json' if smoke else 'summary.json'),proof)
        return proof
    try:
        report(status='preflight',free_mib=require_capacity())
        report(status='preparing',jobs=[launch()])
        wait_children()
        run_pair(True)
        guard()
        require_capacity()
        run_pair(False)
        report(status='complete',jobs=[])
        write_json(root/'complete.json',state)
    except BaseException as error:
        # Terminate only process groups spawned by this controller, including torchrun children.
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
        deadline=time.monotonic()+25
        while any(p.poll() is None for p in children) and time.monotonic()<deadline:
            time.sleep(1)
        for child in children:
            try:
                os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
        report(status='failed_or_stopped',error=repr(error),jobs=[])
        raise


if __name__=='__main__':
    main()
