"""Durable four-GPU preflight -> train-only smoke -> fresh formal training."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .runtime import CONFIG, ROOT, code_identity, write_json, backbone_files
from .evaluation_protocol import verify_evaluation_contract


def require_capacity():
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
        '--format=csv,noheader,nounits'], text=True)
    free = {int(i): int(m) for i, m in (line.split(',') for line in rows.strip().splitlines())}
    if any(free.get(i, 0) < CONFIG['minimum_free_gpu_mib'] for i in CONFIG['gpus']):
        raise RuntimeError(f'Four-GPU free memory requirement not met: {free}')
    return {str(i): free[i] for i in CONFIG['gpus']}


def training_command(out, smoke=False, resume=False):
    cmd = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone',
           '--nnodes=1', f"--nproc-per-node={CONFIG['world_size']}", '--max-restarts=0',
           '-m', 'jev_train.train', '--arm', 'leaf_tower_mlp', '--out', str(out)]
    if smoke:
        cmd.append('--smoke')
    if resume:
        cmd.append('--resume')
    return cmd


def compare_initialization(root):
    states = [json.loads((root / phase / 'initial_state.json').read_text()) for phase in ('smoke', 'train')]
    for key in ('common_connector_sha256', 'jev_adapter_head_sha256', 'initial_encoder_sha256',
                'buffer_sha256', 'world_size', 'parameter_counts'):
        if states[0][key] != states[1][key]:
            raise RuntimeError(f'Smoke/formal initialization differs: {key}')
    if not all(s['rank_initialization_verified'] for s in states):
        raise RuntimeError('Rank initialization was not verified')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start', action='store_true')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    os.chdir(ROOT)
    root = ROOT / CONFIG['output_root']
    root.mkdir(parents=True, exist_ok=True)
    if args.start:
        if (root / 'train/complete.json').exists():
            raise RuntimeError('Formal experiment is already complete; refusing duplicate training')
        with (root / 'controller.lock').open('a') as probe:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        env = {**os.environ, 'OMP_NUM_THREADS': '3', 'OPENBLAS_NUM_THREADS': '3',
               'MKL_NUM_THREADS': '3', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
               'PYTHONUNBUFFERED': '1', 'TOKENIZERS_PARALLELISM': 'false',
               'CUDA_VISIBLE_DEVICES': ','.join(map(str, CONFIG['gpus']))}
        cmd = [sys.executable, '-u', '-m', 'jev_train.run'] + (['--resume'] if args.resume else [])
        with (root / 'controller.log').open('a') as log:
            child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps(dict(controller_pid=child.pid, output=str(root))), flush=True)
        return
    lock = (root / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    hashes = code_identity()
    state = dict(controller_pid=os.getpid(), state='preflight', source_hashes=hashes,
                 gpus=CONFIG['gpus'], run_name=CONFIG['run_name'])

    def report(**kw):
        state.update(kw, time=time.time())
        write_json(root / 'pipeline.json', state)
        print(json.dumps({k: v for k, v in state.items() if k != 'source_hashes'}), flush=True)

    def guard():
        if hashes != code_identity():
            raise RuntimeError('Live source, config or manifest changed after launch')

    try:
        # The user is downloading independently. Never rename, overwrite or load partial files.
        deadline = time.time() + 24 * 3600
        while True:
            try:
                backbone_files()
                break
            except FileNotFoundError as exc:
                if time.time() >= deadline:
                    raise TimeoutError('Weights were not ready within24h') from exc
                report(state='waiting_for_weights', reason=str(exc))
                guard()
                time.sleep(30)
        report(state='checking_weights')
        subprocess.run([sys.executable, '-u', '-m', 'jev_train.weights'], check=True)
        report(free_gpu_mib=require_capacity())
        subprocess.run([sys.executable, '-u', '-m', 'jev_train.data'], check=True)
        guard()
        report(state='checking_model')
        subprocess.run([sys.executable, '-u', '-m', 'jev_train.check_model'], check=True)
        guard()
        for phase in ('smoke', 'train'):
            out = root / phase
            if (out / 'complete.json').exists():
                verify_evaluation_contract(json.loads((out / 'complete.json').read_text()))
                continue
            resume = args.resume and phase == 'train'
            if resume and not (out / 'last.pt').exists():
                raise RuntimeError('Requested resume without a complete training checkpoint')
            if out.exists() and not resume:
                raise RuntimeError(f'Unfinished {phase} directory exists; inspect before restarting')
            require_capacity()
            guard()
            with (root / f'{phase}.log').open('a') as log:
                child = subprocess.Popen(training_command(out, phase == 'smoke', resume), cwd=ROOT,
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                report(state=phase, child_pid=child.pid)
                while child.poll() is None:
                    time.sleep(5)
                    if phase == 'train' and (out / 'initial_state.json').exists():
                        compare_initialization(root)
                if child.returncode:
                    raise RuntimeError(f'{phase} exited {child.returncode}; see {phase}.log')
            guard()
            completed = json.loads((out / 'complete.json').read_text())
            verify_evaluation_contract(completed)
            for key in ('checkpoint_reload_verified', 'encoder_update_contract_verified',
                        'rank_weights_verified', 'distributed_sampling_verified'):
                if not completed[key]:
                    raise RuntimeError(f'Incomplete training verification: {key}')
            report(state=phase + '_complete')
        compare_initialization(root)
        report(state='complete', smoke_weights_not_reused=True)
    except BaseException as exc:
        report(state='failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
