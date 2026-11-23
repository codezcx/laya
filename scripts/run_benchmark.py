"""Detached four-GPU Tower-only training: input audit -> smoke -> formal run."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eeg_align.runtime import CONFIG, digest
from eeg_align.multitask_data import write_json


def require_gpu_capacity(gpus, minimum_free_mib):
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                   '--format=csv,noheader,nounits'], text=True)
    free = {int(i): int(m) for i, m in (r.split(',') for r in rows.strip().splitlines())}
    if any(free.get(g, 0) < minimum_free_mib for g in gpus):
        raise RuntimeError(f'Insufficient GPU memory: {free}; need {minimum_free_mib} MiB each')
    return {g: free[g] for g in gpus}


def training_command(python, gpus, arm, directory, smoke=False, resume=False):
    command = [python, '-u', '-m', 'torch.distributed.run', '--standalone',
               '--nnodes=1', f'--nproc-per-node={len(gpus)}', '--max-restarts=0',
               '-m', 'eeg_align.train_joint', '--arm', arm, '--out', str(directory)]
    if smoke:
        command.append('--smoke')
    if resume:
        command.append('--resume')
    return command


def verify_initialization(smoke, formal):
    for key in ('common_connector_sha256', 'jev_adapter_head_sha256',
                'initial_encoder_sha256', 'buffer_sha256', 'world_size'):
        if smoke[key] != formal[key]:
            raise RuntimeError(f'Smoke state leaked into formal initialization: {key}')
    if not smoke['rank_initialization_verified'] or not formal['rank_initialization_verified']:
        raise RuntimeError('Rank initialization was not verified')


def validate_resume(output, arm):
    directory = output / 'train' / arm
    identity = json.loads((directory / 'config.json').read_text())
    if identity['config'] != CONFIG or identity['smoke'] or identity['arm'] != arm:
        raise ValueError('Resume identity mismatch')
    if any(not Path(p).is_file() or digest(p) != h for p, h in identity['code_sha256'].items()):
        raise ValueError('Resume source hashes changed')
    if (directory / 'complete.json').exists():
        raise ValueError('Already complete; no retraining authorized')
    if not all((directory / name).exists() for name in ('best.pt', 'last.pt')):
        raise ValueError('Resume requires best.pt and last.pt')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--start', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--gpus', nargs='+', type=int)
    parser.add_argument('--recovery-dir', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    # Baseline remains reproducible; aligned arms are orchestrated by run_contrastive.py.
    arm = 'leaf_tower_mlp'
    gpus = CONFIG['gpus'] if args.gpus is None else args.gpus
    if (len(gpus) != CONFIG['world_size'] or len(set(gpus)) != len(gpus)
            or any(g < 0 for g in gpus)):
        parser.error('Use exactly world_size distinct GPU indices')
    output = Path('artifacts/benchmark') / CONFIG['run_name']
    if args.start:
        if args.resume:
            validate_resume(output, arm)
            with (output / 'controller.lock').open('a') as probe:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            args.recovery_dir = output / 'recovery' / (time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + f'_{os.getpid()}')
            args.recovery_dir.mkdir(parents=True, exist_ok=False)
            if (output / 'pipeline.json').exists():
                shutil.copy2(output / 'pipeline.json', args.recovery_dir / 'previous_pipeline.json')
            log_path = args.recovery_dir / 'controller.log'
        else:
            output.mkdir(parents=True, exist_ok=False)
            log_path = output / 'controller.log'
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='3', MKL_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3',
                   TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   PYTHONUNBUFFERED='1')
        command = [sys.executable, '-u', __file__, '--execute', '--gpus', *map(str, gpus)]
        if args.resume:
            command += ['--resume', '--recovery-dir', str(args.recovery_dir)]
        with log_path.open('x') as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True, env=env, cwd=ROOT)
        print(json.dumps(dict(controller_pid=child.pid, output=str(output), gpus=gpus, log=str(log_path))))
        return
    lock = (output / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(status='preflight', controller_pid=os.getpid(), test_evaluations=0,
                 arm=arm, world_size=len(gpus), gpus=gpus)
    def report(**values):
        state.update(values, updated_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()))
        write_json(output / 'pipeline.json', state)
        print(json.dumps(state), flush=True)
    child = None
    try:
        if args.resume:
            validate_resume(output, arm)
        watched = sorted((ROOT / 'eeg_align').glob('*.py')) + [ROOT / p for p in (
            'configs/benchmark_train.json', 'configs/benchmark_tasks.json',
            'scripts/run_benchmark.py', 'scripts/check_multitask_signals.py',
            'scripts/prepare_multitask_signals.py', 'scripts/audit_multitask_signals.py')]
        hashes = {str(p.relative_to(ROOT)): digest(p) for p in watched}
        def guard():
            if hashes != {str(p.relative_to(ROOT)): digest(p) for p in watched}:
                raise RuntimeError('Live sources/config changed')
        report(launch_free_mib=require_gpu_capacity(gpus, CONFIG['minimum_free_gpu_mib']),
               code_sha256=hashes)
        if not args.resume:
            report(status='cpu_signal_preflight')
            subprocess.run([sys.executable, '-u', 'scripts/check_multitask_signals.py',
                '--out', str(output / 'signal_preflight.json')], check=True)
        for phase in (('train',) if args.resume else ('smoke', 'train')):
            guard()
            require_gpu_capacity(gpus, CONFIG['minimum_free_gpu_mib'])
            directory = output / phase / arm
            directory.parent.mkdir(exist_ok=True)
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus))
            log_root = args.recovery_dir if args.resume else output
            command = training_command(sys.executable, gpus, arm, directory, phase == 'smoke', args.resume)
            with (log_root / f'{phase}_{arm}.log').open('x') as log:
                child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, env=env, start_new_session=True)
            report(status=phase, process_pid=child.pid)
            while child.poll() is None:
                time.sleep(10)
            if child.returncode:
                raise RuntimeError(f'{phase} exited {child.returncode}')
            guard()
            complete = json.loads((directory / 'complete.json').read_text())
            if (complete['test_evaluations'] or not complete['checkpoint_reload_verified']
                    or not complete['encoder_update_contract_verified']
                    or not complete['rank_weights_verified']
                    or not complete['distributed_sampling_verified']
                    or complete['world_size'] != len(gpus)):
                raise RuntimeError('Incomplete training contracts')
            report(status=phase + '_complete', returncode=child.returncode)
        smoke = json.loads((output / 'smoke' / arm / 'initial_state.json').read_text())
        formal = json.loads((output / 'train' / arm / 'initial_state.json').read_text())
        verify_initialization(smoke, formal)
        subprocess.run([sys.executable, 'scripts/plot_training_curves.py', '--run-dir',
                        str(output), '--arm', arm], check=True)
        report(status='complete', smoke_weights_not_reused=True)
    except BaseException as exc:
        report(status='failed', error=f'{type(exc).__name__}: {exc}',
               child_pid=child.pid if child is not None else None)
        raise


if __name__ == '__main__':
    main()
