"""Detached matched pilot: CPU audit -> all-arm smoke -> queued GPU training."""
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
from eeg_align.runtime import CONFIG, digest
from eeg_align.multitask_data import write_json


def run_on_gpu_slots(arms, gpus, launch, report, *, poll_seconds=15, running=None):
    """Run every arm exactly once, at most one process per assigned GPU.

    On failure, drain live independent arms but never launch additional arms.
    ``running`` stays visible to the caller for failure reporting, not killing.
    """
    if not arms or not gpus or len(set(arms)) != len(arms) or len(set(gpus)) != len(gpus):
        raise ValueError('Arms and GPU slots must be nonempty and unique')
    pending, available = list(arms), list(gpus)
    running = [] if running is None else running
    if running:
        raise ValueError('New phase cannot inherit live children')
    processes, failures = {}, {}
    while pending or running:
        for item in list(running):
            arm, gpu, process = item
            code = process.poll()
            if code is not None:
                running.remove(item)
                available.append(gpu)
                processes[arm]['returncode'] = code
                if code:
                    failures[arm] = code
        while pending and available and not failures:
            arm, gpu = pending.pop(0), available.pop(0)
            process = launch(arm, gpu)
            running.append((arm, gpu, process))
            processes[arm] = dict(pid=process.pid, gpu=gpu, returncode=None)
        report(processes=processes, queued_arms=pending.copy(), failures=failures.copy())
        if failures and not running:
            raise RuntimeError(f'GPU job failures: {failures}; unstarted arms: {pending}')
        if running:
            time.sleep(poll_seconds)
    return processes


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
        watched = sorted((ROOT / 'eeg_align').glob('*.py')) + [ROOT / p for p in (
            'configs/benchmark_train.json', 'configs/benchmark_tasks.json',
            'scripts/run_benchmark.py', 'scripts/check_multitask_signals.py',
            'scripts/prepare_multitask_splits.py', 'scripts/prepare_multitask_signals.py',
            'scripts/audit_multitask_signals.py')]
        hashes = {str(p.relative_to(ROOT)): digest(p) for p in watched}
        def guard():
            if (hashes != {str(p.relative_to(ROOT)): digest(p) for p in watched}
                    or json.loads((ROOT / 'configs/benchmark_train.json').read_text()) != CONFIG):
                raise RuntimeError('Queued experiment code/config changed; refusing new launches')
        guard()
        report(code_sha256=hashes, configured_arms=CONFIG['arms'], assigned_gpu_slots=CONFIG['gpus'])
        rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'], text=True)
        free = {int(i):int(m) for i,m in (r.split(',') for r in rows.strip().splitlines())}
        if any(free.get(g,0) < 75000 for g in CONFIG['gpus']):
            raise RuntimeError('Every assigned GPU slot must be free before launch')
        report(status='cpu_signal_preflight')
        subprocess.run([sys.executable, '-u', 'scripts/check_multitask_signals.py',
            '--out', str(output / 'signal_preflight.json')], check=True)
        initializations = {}
        for phase in ('smoke','train'):
            running = []
            def launch(arm, gpu):
                guard()
                env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = str(gpu)
                directory = output / phase / arm
                directory.parent.mkdir(exist_ok=True)
                command = [sys.executable, '-u', '-m', 'eeg_align.train_joint', '--arm', arm, '--out', str(directory)]
                if phase == 'smoke': command.append('--smoke')
                with (output / f'{phase}_{arm}.log').open('x') as log:
                    child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                        stderr=subprocess.STDOUT, env=env, start_new_session=True)
                return child
            run_on_gpu_slots(CONFIG['arms'], CONFIG['gpus'], launch,
                lambda **values: report(status=phase + ('_partial_failure' if values['failures'] else ''),
                                        **values), running=running)
            guard()
            proofs = [json.loads((output / phase / arm / 'initial_state.json').read_text()) for arm in CONFIG['arms']]
            for key in ('jev_adapter_head_sha256',):
                if len({p[key] for p in proofs}) != 1:
                    raise RuntimeError(f'Arm initialization mismatch: {key}')
            if phase == 'smoke':
                initializations = dict(zip(CONFIG['arms'], proofs))
            else:
                for arm, proof in zip(CONFIG['arms'], proofs):
                    for key in ('common_connector_sha256', 'jev_adapter_head_sha256', 'released_encoder_sha256'):
                        if proof[key] != initializations[arm][key]:
                            raise RuntimeError(f'Smoke weights leaked into formal training: {arm}/{key}')
            for arm in CONFIG['arms']:
                complete = json.loads((output / phase / arm / 'complete.json').read_text())
                if complete['test_evaluations'] or not complete['checkpoint_reload_verified']:
                    raise RuntimeError('Unsafe/incomplete preflight')
            report(status=phase+'_complete', matched_jev_initialization=True,
                   connector_note='Native encoder widths differ; no claim of identical connector parameters')
        for arm in CONFIG['arms']:
            subprocess.run([sys.executable, 'scripts/plot_training_curves.py',
                            '--run-dir', str(output), '--arm', arm], check=True)
        report(status='complete', selection='Compare full validation before separately authorizing test')
    except BaseException as exc:
        report(status='failed', error=f'{type(exc).__name__}: {exc}',
               remaining_processes={arm:p.pid for arm,_,p in running if p.poll() is None})
        raise


if __name__ == '__main__':
    main()
