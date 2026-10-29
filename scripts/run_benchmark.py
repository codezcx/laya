"""Persistent preparation -> smoke -> multitask SFT/eval controller."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eeg_align.runtime import CONFIG, digest
from eeg_align.multitask_data import TrialStore, write_json

PYTHON = ROOT / 'laya/.venv-eeg/bin/python'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('artifacts/benchmark') / CONFIG['run_name'])
    parser.add_argument('--start', action='store_true', help='Detach controller with persistent logs')
    parser.add_argument('--prepared', action='store_true', help='Verify existing complete caches, do not prepare again')
    args = parser.parse_args()
    os.chdir(ROOT)
    args.out = args.out.resolve()
    if args.start:
        args.out.mkdir(parents=True, exist_ok=False)
        command = [str(PYTHON), str(Path(__file__).resolve()), '--out', str(args.out)]
        if args.prepared:
            command.append('--prepared')
        with (args.out / 'controller.log').open('x') as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(args.out / 'launch.json', dict(controller_pid=process.pid, command=command))
        print(json.dumps(dict(status='launched_controller', pid=process.pid, output=str(args.out))))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'pipeline.json').exists():
        raise FileExistsError('Controller output exists; use a new run directory')
    watched = [Path('configs/benchmark_train.json'), Path('configs/benchmark_tasks.json')] + sorted(Path('eeg_align').glob('*.py'))
    hashes = {str(p): digest(p) for p in watched}
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=','.join(map(str, CONFIG['gpus'])), OMP_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3', MKL_NUM_THREADS='3', TOKENIZERS_PARALLELISM='false')
    child = None
    stop = False
    def terminate(signum, frame):
        nonlocal stop
        stop = True
        if child and child.poll() is None:
            child.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    def status(stage, **kwargs):
        write_json(args.out / 'pipeline.json', dict(stage=stage, updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), code_sha256=hashes, **kwargs))
    def run(stage, command):
        nonlocal child
        if hashes != {str(p): digest(p) for p in watched}:
            raise RuntimeError('Code changed while pipeline was running')
        if stop:
            raise InterruptedError('Controller stopped')
        status(stage, command=command)
        with (args.out / f'{stage}.log').open('x') as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            result = child.wait()
        if result != 0 or stop:
            raise RuntimeError(f'{stage} exited with {result}; stopped={stop}')
    try:
        if not args.prepared:
            run('preparation', [str(PYTHON), 'scripts/prepare_multitask_signals.py', '--workers', str(CONFIG['cpu_workers'])])
        store = TrialStore()
        status('cache_verified', datasets={store.tasks[t]['task_id']: meta['counts'] for t, meta in store.meta.items()})
        run('text', [str(PYTHON), 'scripts/prepare_benchmark_text.py'])
        command = [str(PYTHON), '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', str(len(CONFIG['gpus'])), '-m', 'eeg_align.train_joint']
        run('smoke', command + ['--out', str(args.out / 'smoke'), '--smoke'])
        smoke = json.loads((args.out / 'smoke/complete.json').read_text())
        if smoke['test_evaluations'] != 0 or not smoke['checkpoint_reload_verified']:
            raise RuntimeError('Smoke gate failed')
        run('training', command + ['--out', str(args.out / 'train')])
        if not (args.out / 'train/complete.json').exists():
            raise RuntimeError('Training child did not produce complete evaluation')
        status('complete')
    except BaseException as exc:
        status('stopped' if stop else 'failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
