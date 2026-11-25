"""Bounded four-user matched personal curves; no source retraining."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from jev_train.runtime import ROOT, digest
from .common import CONFIG, OUT, identity, write_json


from .curves import summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'OMP_NUM_THREADS': '3', 'OPENBLAS_NUM_THREADS': '3', 'MKL_NUM_THREADS': '3',
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
           'PYTHONUNBUFFERED': '1'}
    if args.start:
        with (OUT / 'controller.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (OUT / 'controller.log').open('a') as log:
            child = subprocess.Popen([sys.executable, '-u', '-m', 'personal.run'], cwd=ROOT, env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps(dict(controller_pid=child.pid, output=str(OUT))), flush=True)
        return
    lock = (OUT / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    hashes = identity()
    state_path = OUT / 'queue.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['source_hashes'] != hashes:
            raise RuntimeError('Queue source changed; inspect before resuming')
        if state['state'] in ('complete', 'budget_exhausted', 'stopped_by_user'):
            raise RuntimeError('Queue finished; another time budget requires explicit authorization')
    else:
        state = dict(started=time.time(), deadline=time.time()+CONFIG['wall_hours']*3600,
                     source_hashes=hashes, state='preparing', parent_sha256=None)
    state['controller_pid'] = os.getpid()

    def report(**kw):
        state.update(kw, time=time.time())
        write_json(state_path, state)
        print(json.dumps({k:v for k,v in state.items() if k != 'source_hashes'}), flush=True)

    children = []
    try:
        report(state='preparing')
        if shutil.disk_usage(ROOT).free < CONFIG['minimum_free_disk_gib'] * 1024**3:
            raise RuntimeError('Insufficient disk for personal checkpoints')
        subprocess.run([sys.executable, '-u', '-m', 'personal.plan'], env=env, check=True)
        report(state='waiting_for_shared_parent')
        parent = ROOT / CONFIG['parent']
        while not (parent / 'complete.json').exists():
            if identity() != hashes:
                raise RuntimeError('Live personal source changed')
            if time.time() >= state['deadline']:
                report(state='budget_exhausted', reason='Shared parent did not finish inside budget')
                return
            pipeline = json.loads((parent.parent / 'pipeline.json').read_text())
            if pipeline['state'] == 'failed':
                raise RuntimeError('Shared parent failed; personal experiments not started')
            time.sleep(10)
        checkpoint_sha = digest(parent / 'best.pt')
        if checkpoint_sha != CONFIG['parent_sha256']:
            raise RuntimeError('Selected historical checkpoint identity changed')
        if state['parent_sha256'] not in (None, checkpoint_sha):
            raise RuntimeError('Selected parent checkpoint changed')
        state['parent_sha256'] = checkpoint_sha
        # Complete.json can precede CUDA process teardown by a few seconds.
        while True:
            free = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                       '--format=csv,noheader,nounits'], text=True)
            available = {int(a):int(b) for a,b in (r.split(',') for r in free.strip().splitlines())}
            if all(available.get(i, 0) >= CONFIG['minimum_free_gpu_mib'] for i in CONFIG['gpus']):
                break
            if time.time() >= state['deadline']:
                report(state='budget_exhausted', reason='GPU availability')
                return
            time.sleep(10)
        report(state='personal_workers')
        for worker, gpu in enumerate(CONFIG['gpus']):
            with (OUT / f'gpu{worker}.log').open('a') as log:
                child = subprocess.Popen([sys.executable, '-u', '-m', 'personal.worker', '--worker', str(worker)],
                    cwd=ROOT, env={**env, 'CUDA_VISIBLE_DEVICES': str(gpu)}, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT)
            children.append(child)
        report(worker_pids=[p.pid for p in children])
        while any(p.poll() is None for p in children):
            if identity() != hashes:
                raise RuntimeError('Live personal source changed')
            failed = [(p.pid,p.returncode) for p in children if p.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f'Personal worker failed: {failed}; see per-GPU log')
            summary()
            time.sleep(20)
        if any(p.returncode for p in children):
            raise RuntimeError('A personal worker failed')
        summary()
        plan = json.loads((OUT / 'plan.json').read_text())
        completed = sum((OUT / 'jobs' / j['key'] / 'complete.json').exists() for j in CONFIG['jobs'])
        report(state='complete' if completed == len(CONFIG['jobs']) else 'budget_exhausted',
               complete_jobs=completed, planned_jobs=len(CONFIG['jobs']))
    except BaseException as exc:
        for child in children:
            if child.poll() is None:
                child.terminate()
        report(state='failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
