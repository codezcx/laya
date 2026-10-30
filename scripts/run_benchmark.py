"""Persistent preparation -> smoke -> multitask SFT/eval controller."""
import argparse
import fcntl
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
    parser.add_argument('--continuation-config', type=Path)
    parser.add_argument('--resume', action='store_true', help='Resume only an unfinished continuation')
    args = parser.parse_args()
    os.chdir(ROOT)
    continuation = json.loads(args.continuation_config.read_text()) if args.continuation_config else None
    if args.resume and not continuation:
        raise ValueError('Controller resume is restricted to continuation mode')
    if continuation:
        args.continuation_config = args.continuation_config.resolve()
        if args.out == Path('artifacts/benchmark') / CONFIG['run_name']:
            args.out = Path('artifacts/benchmark') / continuation['run_name']
        parent = Path(continuation['parent_run']).resolve()
        if args.out.resolve() == parent or parent.is_relative_to(args.out.resolve()):
            raise ValueError('Continuation must use a separate output directory')
        if digest(parent / 'last.pt') != continuation['parent_checkpoint_sha256']:
            raise ValueError('Parent checkpoint checksum mismatch')
    args.out = args.out.resolve()
    if args.start:
        args.out.mkdir(parents=True, exist_ok=args.resume)
        command = [str(PYTHON), str(Path(__file__).resolve()), '--out', str(args.out)]
        if args.prepared:
            command.append('--prepared')
        if continuation:
            command += ['--continuation-config', str(args.continuation_config)]
        if args.resume:
            command.append('--resume')
        with (args.out / 'controller.log').open('a' if args.resume else 'x') as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(args.out / 'launch.json', dict(controller_pid=process.pid, command=command))
        print(json.dumps(dict(status='launched_controller', pid=process.pid, output=str(args.out))))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (args.out / 'train/complete.json').exists():
        raise ValueError('Run already complete')
    if (args.out / 'pipeline.json').exists() and not args.resume:
        raise FileExistsError('Controller output exists; use a new run directory')
    watched = [Path('configs/benchmark_train.json'), Path('configs/benchmark_tasks.json'),
               Path('scripts/audit_multitask_signals.py'), Path('scripts/check_multitask_signals.py'),
               Path(CONFIG['quality_root']) / 'summary.json'] + sorted(Path('eeg_align').glob('*.py'))
    hashes = {str(p): digest(p) for p in watched}
    if continuation:
        hashes[str(args.continuation_config)] = digest(args.continuation_config)
        hashes['scripts/run_benchmark.py'] = digest('scripts/run_benchmark.py')
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
        if hashes != {p: digest(p) for p in hashes}:
            raise RuntimeError('Code changed while pipeline was running')
        if stop:
            raise InterruptedError('Controller stopped')
        status(stage, command=command)
        with (args.out / f'{stage}.log').open('a' if args.resume else 'x') as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            result = child.wait()
        if result != 0 or stop:
            raise RuntimeError(f'{stage} exited with {result}; stopped={stop}')
    try:
        if continuation:
            store = TrialStore()
            status('continuation_cache_verified', parent_run=str(parent),
                   total_steps=CONFIG['steps'] + continuation['additional_steps'], test_evaluations=0)
            del store
            if not args.resume:
                run('signal_preflight', [str(PYTHON), 'scripts/check_multitask_signals.py',
                    '--out', str(args.out / 'signal_preflight.json'), '--start-step', str(CONFIG['steps'] + 1),
                    '--stop-step', str(CONFIG['steps'] + continuation['additional_steps'])])
            check = json.loads((args.out / 'signal_preflight.json').read_text())
            if (not check['passed'] or check['stop_step'] != CONFIG['steps'] + continuation['additional_steps']
                    or any(digest(p) != h for p, h in check['code_sha256'].items())):
                raise ValueError('Continuation signal preflight mismatch')
            command = [str(PYTHON), '-m', 'torch.distributed.run', '--standalone',
                '--nproc_per_node', str(len(CONFIG['gpus'])), '-m', 'eeg_align.train_joint',
                '--out', str(args.out / 'train'), '--continuation-config', str(args.continuation_config)]
            if args.resume:
                command.append('--resume')
            run('training', command)
            result = json.loads((args.out / 'train/complete.json').read_text())
            if result['test_evaluations'] != 0:
                raise RuntimeError('Continuation accessed test data')
            status('complete', selected_step=result['selected_step'], test_evaluations=0)
            return
        if not args.prepared:
            run('preparation', [str(PYTHON), 'scripts/prepare_multitask_signals.py', '--workers', str(CONFIG['cpu_workers'])])
        store = TrialStore()
        status('cache_verified', quality_summary_sha256=CONFIG['quality_summary_sha256'],
               datasets={name: report['effective_counts'] for name, report in store.quality.items()})
        del store
        run('signal_preflight', [str(PYTHON), 'scripts/check_multitask_signals.py', '--out', str(args.out / 'signal_preflight.json')])
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
