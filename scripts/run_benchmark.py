"""Detached four-GPU pilot: CPU audit -> all-arm real smoke -> matched training."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eeg_align.runtime import CONFIG
from eeg_align.multitask_data import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--start', action='store_true')
    mode.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    output = Path('artifacts/benchmark') / CONFIG['run_name']
    if args.start:
        output.mkdir(parents=True, exist_ok=False)
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='3', MKL_NUM_THREADS='3', OPENBLAS_NUM_THREADS='3',
                   TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        with (output / 'controller.log').open('x') as log:
            process = subprocess.Popen([sys.executable, '-u', __file__, '--execute'],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, env=env, cwd=ROOT)
        print(json.dumps(dict(controller_pid=process.pid, output=str(output))))
        return
    lock = (output / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = dict(status='preflight', test_evaluations=0, controller_pid=os.getpid())
    def report(**values):
        state.update(values, updated_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()))
        write_json(output / 'pipeline.json', state)
        print(json.dumps(state), flush=True)
    running = []
    try:
        rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'], text=True)
        free = {int(i):int(m) for i,m in (r.split(',') for r in rows.strip().splitlines())}
        if any(free.get(g,0) < 75000 for g in CONFIG['gpus']):
            raise RuntimeError('All four assigned GPUs must be free before launch')
        report(status='cpu_signal_preflight')
        subprocess.run([sys.executable, '-u', 'scripts/check_multitask_signals.py',
            '--out', str(output / 'signal_preflight.json')], check=True)
        for phase in ('smoke','train'):
            running = []
            pids = {}
            for gpu, arm in zip(CONFIG['gpus'], CONFIG['arms']):
                env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = str(gpu)
                directory = output / phase / arm
                directory.parent.mkdir(exist_ok=True)
                command = [sys.executable, '-u', '-m', 'eeg_align.train_joint', '--arm', arm, '--out', str(directory)]
                if phase == 'smoke': command.append('--smoke')
                with (output / f'{phase}_{arm}.log').open('x') as log:
                    child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                        stderr=subprocess.STDOUT, env=env, start_new_session=True)
                running.append((arm,child)); pids[arm] = dict(pid=child.pid,gpu=gpu)
            report(status=phase, processes=pids)
            while any(p.poll() is None for _,p in running):
                failures = {arm:p.returncode for arm,p in running if p.poll() not in (None,0)}
                if failures:
                    # Do not silently kill successful independent arms.
                    report(status=phase+'_partial_failure', failures=failures)
                time.sleep(15)
            failures = {arm:p.returncode for arm,p in running if p.returncode}
            if failures: raise RuntimeError(f'{phase} failures: {failures}')
            proofs = [json.loads((output / phase / arm / 'initial_state.json').read_text()) for arm in CONFIG['arms']]
            for key in ('common_connector_sha256','jev_adapter_head_sha256','frozen_bsr_sha256'):
                if len({p[key] for p in proofs}) != 1:
                    raise RuntimeError(f'Arm initialization mismatch: {key}')
            for arm in CONFIG['arms']:
                complete = json.loads((output / phase / arm / 'complete.json').read_text())
                if complete['test_evaluations'] or not complete['checkpoint_reload_verified']:
                    raise RuntimeError('Unsafe/incomplete preflight')
            report(status=phase+'_complete', matched_initialization=True)
        for arm in CONFIG['arms']:
            subprocess.run([sys.executable, 'scripts/plot_training_curves.py',
                            '--run-dir', str(output), '--arm', arm], check=True)
        report(status='complete', selection='Compare full validation before separately authorizing test')
    except BaseException as exc:
        report(status='failed', error=f'{type(exc).__name__}: {exc}',
               remaining_processes={arm:p.pid for arm,p in running if p.poll() is None})
        raise


if __name__ == '__main__':
    main()
